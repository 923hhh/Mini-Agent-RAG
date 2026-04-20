from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retrievers.local_kb import search_local_knowledge_base
from app.schemas.chat import ChatMessage, RetrievedReference
from app.services.settings import load_settings
from scripts.eval_retrieval import evaluate_reference_ranking


DEFAULT_INPUT = PROJECT_ROOT / "data" / "eval" / "phase0_gold_manual_seed.jsonl"
DEFAULT_JSON_OUTPUT = PROJECT_ROOT / "data" / "eval" / "phase0_bad_case_bucket_v1.json"
DEFAULT_MD_OUTPUT = PROJECT_ROOT / "data" / "eval" / "phase0_bad_case_bucket_v1.md"
TOP_K = 50
BAD_CASE_TOP1_RANK_THRESHOLD = 1
BAD_CASE_LOW_RANK_THRESHOLD = 3


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Phase 0 manual gold cases and produce first-pass bad-case buckets.")
    parser.add_argument("--input-file", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-json", default=str(DEFAULT_JSON_OUTPUT))
    parser.add_argument("--output-md", default=str(DEFAULT_MD_OUTPUT))
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    settings = load_settings(PROJECT_ROOT)
    cases = load_phase0_cases(Path(args.input_file))
    result = analyze_cases(settings, cases, top_k=args.top_k)

    output_json_path = Path(args.output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    output_md_path = Path(args.output_md)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.write_text(build_markdown_report(result), encoding="utf-8")
    return 0


def load_phase0_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Phase 0 样本文件不存在: {path}")

    cases: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        cases.append(json.loads(stripped))
    return cases


def analyze_cases(settings: Any, cases: list[dict[str, Any]], *, top_k: int) -> dict[str, Any]:
    summary_counter: Counter[str] = Counter()
    status_counter: Counter[str] = Counter()
    type_counter: Counter[str] = Counter()
    bucket_examples: dict[str, list[str]] = defaultdict(list)
    details: list[dict[str, Any]] = []

    top1_hits = 0
    recall_at_20_hits = 0
    recall_at_50_hits = 0
    mrr_at_10_sum = 0.0
    ndcg_at_10_sum = 0.0

    for case in cases:
        detail = analyze_single_case(settings, case, top_k=top_k)
        details.append(detail)

        bucket = str(detail["assigned_failure_bucket"])
        status = str(detail["status"])
        summary_counter[bucket or "passed"] += 1
        status_counter[status] += 1
        type_counter[str(case.get("question_type", "unknown"))] += 1
        if bucket:
            bucket_examples[bucket].append(str(case.get("case_id", "")))

        if detail["top1_hit"]:
            top1_hits += 1
        if detail["hit_at_20"]:
            recall_at_20_hits += 1
        if detail["hit_at_50"]:
            recall_at_50_hits += 1
        mrr_at_10_sum += float(detail["mrr_at_10"])
        ndcg_at_10_sum += float(detail["ndcg_at_10"])

    total = max(len(cases), 1)
    return {
        "input_file": str(DEFAULT_INPUT),
        "evaluated_total": len(cases),
        "protocol": {
            "top_k_retrieved": top_k,
            "reported_metrics": ["Recall@20", "Recall@50", "MRR@10", "NDCG@10", "Top1 accuracy"],
        },
        "summary": {
            "top1_accuracy": top1_hits / total,
            "recall_at_20": recall_at_20_hits / total,
            "recall_at_50": recall_at_50_hits / total,
            "mrr_at_10": mrr_at_10_sum / total,
            "ndcg_at_10": ndcg_at_10_sum / total,
            "status_counts": dict(status_counter),
            "bucket_counts": dict(summary_counter),
            "question_type_counts": dict(type_counter),
        },
        "bucket_examples": dict(bucket_examples),
        "details": details,
    }


def analyze_single_case(settings: Any, case: dict[str, Any], *, top_k: int) -> dict[str, Any]:
    case_id = str(case.get("case_id", "")).strip()
    query = str(case.get("query", "")).strip()
    knowledge_base_name = resolve_knowledge_base_name(case)
    history = build_history_messages(case.get("history_qa"))
    gold_documents = [item for item in case.get("gold_documents", []) if isinstance(item, dict)]
    gold_passages = [item for item in case.get("gold_passages", []) if isinstance(item, dict)]
    expected_references = build_expected_references(gold_documents, gold_passages)

    references = search_local_knowledge_base(
        settings=settings,
        knowledge_base_name=knowledge_base_name,
        query=query,
        top_k=top_k,
        score_threshold=0.0,
        history=history,
    )

    ranking_at_10 = evaluate_reference_ranking(
        references=references,
        expected_references=expected_references,
        cutoff=10,
    )
    ranking_at_20 = evaluate_reference_ranking(
        references=references,
        expected_references=expected_references,
        cutoff=20,
    )
    ranking_at_50 = evaluate_reference_ranking(
        references=references,
        expected_references=expected_references,
        cutoff=50,
    )

    passage_match_ranks_top5 = find_gold_passage_match_ranks(references, gold_passages, cutoff=5)
    passage_match_ranks_top10 = find_gold_passage_match_ranks(references, gold_passages, cutoff=10)
    passage_match_ranks_top20 = find_gold_passage_match_ranks(references, gold_passages, cutoff=20)

    bucket = assign_failure_bucket(
        case=case,
        ranking_at_20=ranking_at_20,
        ranking_at_50=ranking_at_50,
        passage_match_ranks_top5=passage_match_ranks_top5,
        passage_match_ranks_top10=passage_match_ranks_top10,
    )
    status = "bad_case" if bucket else "passed"

    top_titles = [reference.title or reference.source or reference.source_path for reference in references[:5]]
    top_sources = [reference.source or reference.source_path for reference in references[:5]]
    top_modalities = [reference.source_modality or "" for reference in references[:5]]

    return {
        "case_id": case_id,
        "source_dataset": case.get("source_dataset", ""),
        "knowledge_base_name": knowledge_base_name,
        "query": query,
        "question_type": case.get("question_type", ""),
        "needs_image_evidence": bool(case.get("needs_image_evidence", False)),
        "answerable_from_single_chunk": bool(case.get("answerable_from_single_chunk", False)),
        "needs_cross_passage_aggregation": bool(case.get("needs_cross_passage_aggregation", False)),
        "status": status,
        "assigned_failure_bucket": bucket,
        "failure_bucket_reason": explain_failure_bucket(
            bucket=bucket,
            ranking_at_20=ranking_at_20,
            ranking_at_50=ranking_at_50,
            passage_match_ranks_top5=passage_match_ranks_top5,
            case=case,
        ),
        "top1_hit": bool(ranking_at_10["top1_hit"]),
        "hit_at_20": bool(ranking_at_20["hit"]),
        "hit_at_50": bool(ranking_at_50["hit"]),
        "mrr_at_10": float(ranking_at_10["reciprocal_rank"]),
        "ndcg_at_10": float(ranking_at_10["ndcg_at_k"]),
        "rank_at_20": ranking_at_20["rank"],
        "rank_at_50": ranking_at_50["rank"],
        "matched_reference_count_at_20": int(ranking_at_20["matched_reference_count"]),
        "matched_ranks_at_20": list(ranking_at_20["matched_ranks"]),
        "gold_passage_match_ranks_top5": passage_match_ranks_top5,
        "gold_passage_match_ranks_top10": passage_match_ranks_top10,
        "gold_passage_match_ranks_top20": passage_match_ranks_top20,
        "top5_titles": top_titles,
        "top5_sources": top_sources,
        "top5_modalities": top_modalities,
        "gold_document_labels": [
            item.get("title") or item.get("url") or item.get("reference_id") or "unknown"
            for item in gold_documents
        ],
        "notes": str(case.get("notes", "")).strip(),
    }


def resolve_knowledge_base_name(case: dict[str, Any]) -> str:
    value = str(case.get("source_dataset", "")).strip()
    if value:
        return value
    return str(case.get("knowledge_base_name", "")).strip()


def build_history_messages(value: object) -> list[ChatMessage]:
    if not isinstance(value, list):
        return []

    history: list[ChatMessage] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = extract_answer_text(item)
        if question:
            history.append(ChatMessage(role="user", content=question))
        if answer:
            history.append(ChatMessage(role="assistant", content=answer))
    return history


def extract_answer_text(value: object) -> str:
    if isinstance(value, dict):
        if "answer" in value:
            return extract_answer_text(value.get("answer"))
        if "answers" in value:
            return extract_answer_text(value.get("answers"))
        return ""
    if isinstance(value, list):
        for item in value:
            text = extract_answer_text(item)
            if text:
                return text
        return ""
    return str(value).strip()


def build_expected_references(
    gold_documents: list[dict[str, Any]],
    gold_passages: list[dict[str, Any]],
) -> list[dict[str, str]]:
    passages_by_reference: dict[str, list[str]] = defaultdict(list)
    for passage in gold_passages:
        reference_id = str(passage.get("reference_id", "")).strip()
        text = str(passage.get("text", "")).strip()
        if reference_id and text:
            passages_by_reference[reference_id].append(text)

    expected: list[dict[str, str]] = []
    for item in gold_documents:
        reference_id = str(item.get("reference_id", "")).strip()
        contents = " ".join(passages_by_reference.get(reference_id, []))
        expected.append(
            {
                "source": "",
                "source_path": "",
                "title": str(item.get("title", "")).strip(),
                "url": str(item.get("url", "")).strip(),
                "reference_id": reference_id,
                "passage_id": "",
                "content": contents,
            }
        )
    return expected


def find_gold_passage_match_ranks(
    references: list[RetrievedReference],
    gold_passages: list[dict[str, Any]],
    *,
    cutoff: int,
) -> list[int]:
    if not gold_passages:
        return []

    fingerprints = [
        build_text_fingerprint(str(item.get("text", "")).strip())
        for item in gold_passages
    ]
    valid_fingerprints = [item for item in fingerprints if item]
    if not valid_fingerprints:
        return []

    ranks: list[int] = []
    for rank, reference in enumerate(references[:cutoff], start=1):
        haystack = normalize_text((reference.content or "") + " " + (reference.content_preview or ""))
        if any(fingerprint in haystack for fingerprint in valid_fingerprints):
            ranks.append(rank)
    return ranks


def assign_failure_bucket(
    *,
    case: dict[str, Any],
    ranking_at_20: dict[str, Any],
    ranking_at_50: dict[str, Any],
    passage_match_ranks_top5: list[int],
    passage_match_ranks_top10: list[int],
) -> str:
    needs_image_evidence = bool(case.get("needs_image_evidence", False))
    answerable_from_single_chunk = bool(case.get("answerable_from_single_chunk", False))
    needs_cross_passage_aggregation = bool(case.get("needs_cross_passage_aggregation", False))

    if needs_image_evidence:
        return "image_text_misaligned"

    if not ranking_at_50["hit"]:
        return "missed_recall"

    if not ranking_at_20["hit"]:
        return "low_rank"

    if needs_cross_passage_aggregation or not answerable_from_single_chunk:
        if len(passage_match_ranks_top10) < 2:
            return "cross_passage"

    rank_at_20 = ranking_at_20["rank"]
    if rank_at_20 and rank_at_20 > BAD_CASE_LOW_RANK_THRESHOLD:
        return "low_rank"

    if rank_at_20 and rank_at_20 > BAD_CASE_TOP1_RANK_THRESHOLD:
        return "low_rank"

    if ranking_at_20["hit"] and not passage_match_ranks_top5:
        return "chunk_noise"

    return ""


def explain_failure_bucket(
    *,
    bucket: str,
    ranking_at_20: dict[str, Any],
    ranking_at_50: dict[str, Any],
    passage_match_ranks_top5: list[int],
    case: dict[str, Any],
) -> str:
    if not bucket:
        return "当前检索结果已满足第一版人工集的 top1 / gold passage 命中要求。"
    if bucket == "missed_recall":
        return "gold 文档在 top-50 内仍未命中，属于召回层问题。"
    if bucket == "low_rank":
        if not ranking_at_20["hit"] and ranking_at_50["hit"]:
            return "gold 文档落在 top-20 外、top-50 内，属于排序深度不足。"
        return f"gold 文档已命中，但最佳排名为 {ranking_at_20['rank']}，未能稳定进入 top1。"
    if bucket == "chunk_noise":
        return "gold 文档已排到前面，但 top-5 返回块里没有直接包含 gold passage，说明 chunk 噪声较大。"
    if bucket == "cross_passage":
        return "该问题依赖跨段信息，但 top-10 内没有形成足够的 gold passage 覆盖。"
    if bucket == "image_text_misaligned":
        return "该问题需要图像或图文联合证据，当前第一版样本先归到图文关系类。"
    return str(case.get("notes", "")).strip()


def build_text_fingerprint(text: str, *, min_chars: int = 16, max_chars: int = 100) -> str:
    normalized = normalize_text(text)
    if len(normalized) < min_chars:
        return ""
    return normalized[:max_chars]


def normalize_text(value: str) -> str:
    return "".join(str(value or "").split()).casefold()


def build_markdown_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# Phase 0 第一版 Bad Case 分桶",
        "",
        "## 总体结果",
        "",
        f"- 评测样本数：`{result['evaluated_total']}`",
        f"- `Recall@20`：`{summary['recall_at_20']:.4f}`",
        f"- `Recall@50`：`{summary['recall_at_50']:.4f}`",
        f"- `MRR@10`：`{summary['mrr_at_10']:.4f}`",
        f"- `NDCG@10`：`{summary['ndcg_at_10']:.4f}`",
        f"- `Top1 accuracy`：`{summary['top1_accuracy']:.4f}`",
        "",
        "## 分桶统计",
        "",
    ]
    for bucket, count in sorted(summary["bucket_counts"].items()):
        lines.append(f"- `{bucket}`：`{count}`")

    lines.extend(["", "## 典型样本", ""])
    details = [item for item in result["details"] if item["assigned_failure_bucket"]]
    details.sort(key=lambda item: (item["assigned_failure_bucket"], item["case_id"]))
    for detail in details:
        lines.extend(
            [
                f"### {detail['case_id']}",
                f"- 分桶：`{detail['assigned_failure_bucket']}`",
                f"- 问题：{detail['query']}",
                f"- 说明：{detail['failure_bucket_reason']}",
                f"- `rank@20`：`{detail['rank_at_20']}`，`rank@50`：`{detail['rank_at_50']}`",
                f"- top5 标题：{json.dumps(detail['top5_titles'], ensure_ascii=False)}",
                "",
            ]
        )

    passed_count = sum(1 for item in result["details"] if not item["assigned_failure_bucket"])
    lines.extend(
        [
            "## 当前判断",
            "",
            f"- 当前第一版 `20` 条样本里，`{passed_count}` 条可视为检索通过，剩余样本进入初步 bad case 分桶。",
            "- 这一版分桶是为了给后续人工复核和针对性优化提供方向，不替代最终人工判定。",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
