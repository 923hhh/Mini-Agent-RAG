"""基于 Phase 0 人工集重跑检索并生成 bad case 分桶报告。"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retrievers.local_kb import search_local_knowledge_base
from app.schemas.chat import ChatMessage, RetrievedReference
from app.services.core.settings import load_settings


DOC_ID_PATTERN = re.compile(r"(?:文档ID：|doc-)(\d+)")
PASSAGE_ID_PATTERN = re.compile(r"(?:段落ID：|psg-)([\w-]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/eval/phase0/gold/phase0_gold_manual_seed.jsonl",
        help="Phase 0 人工集 jsonl 路径。",
    )
    parser.add_argument(
        "--output-json",
        default="data/eval/phase0/analysis/phase0_bad_case_bucket_v2.json",
        help="分桶结果 json 输出路径。",
    )
    parser.add_argument(
        "--output-md",
        default="data/eval/phase0/analysis/phase0_bad_case_bucket_v2.md",
        help="分桶结果 markdown 输出路径。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="检索深度，默认 50。",
    )
    return parser.parse_args()


def load_seed_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def resolve_knowledge_base_name(row: dict[str, Any]) -> str:
    source_dataset = row.get("source_dataset", "")
    if source_dataset == "domainrag_small_batch_100":
        return "domainrag_small_batch_100"
    if source_dataset == "crud_rag_3qa_full":
        return "crud_rag_3qa_full"

    for doc in row.get("gold_documents", []):
        source_path = str(doc.get("source_path", ""))
        if "domainrag_small_batch_100" in source_path:
            return "domainrag_small_batch_100"
        if "crud_rag_3qa_full" in source_path:
            return "crud_rag_3qa_full"
    raise ValueError(f"无法从样本推断 knowledge_base_name: {row.get('case_id')}")


def build_history_messages(row: dict[str, Any]) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    for item in row.get("history_qa", []):
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if question:
            messages.append(ChatMessage(role="user", content=question))
        if answer:
            messages.append(ChatMessage(role="assistant", content=answer))
    return messages


def normalize_text(value: str) -> str:
    text = str(value)
    text = re.sub(r"\s+", "", text)
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    return text


def extract_reference_id(ref: RetrievedReference) -> str:
    content = ref.content or ""
    source = ref.source or ""
    chunk_id = ref.chunk_id or ""
    for text in (content, source, chunk_id):
        match = DOC_ID_PATTERN.search(text)
        if match:
            return match.group(1)
    return ""


def extract_passage_id(ref: RetrievedReference) -> str:
    content = ref.content or ""
    source = ref.source or ""
    chunk_id = ref.chunk_id or ""
    for text in (content, source, chunk_id):
        match = PASSAGE_ID_PATTERN.search(text)
        if match:
            return match.group(1)
    return ""


def reference_matches_gold_doc(ref: RetrievedReference, gold_doc: dict[str, Any]) -> bool:
    gold_ref_id = str(gold_doc.get("reference_id", "")).strip()
    gold_title = str(gold_doc.get("title", "")).strip()
    gold_url = str(gold_doc.get("url", "")).strip()
    gold_source_path = str(gold_doc.get("source_path", "")).strip()

    if gold_source_path and ref.source_path == gold_source_path:
        return True
    if gold_title and ref.title == gold_title:
        return True
    if gold_url and gold_url in (ref.content or ""):
        return True
    if gold_ref_id:
        if gold_ref_id == extract_reference_id(ref):
            return True
        if gold_ref_id == ref.source or gold_ref_id == ref.chunk_id:
            return True
    return False


def reference_matches_gold_passage(ref: RetrievedReference, gold_passage: dict[str, Any]) -> bool:
    gold_source_path = str(gold_passage.get("source_path", "")).strip()
    gold_ref_id = str(gold_passage.get("reference_id", "")).strip()
    gold_passage_id = str(gold_passage.get("passage_id", "") or gold_passage.get("chunk_id", "")).strip()
    gold_text = normalize_text(gold_passage.get("text", ""))
    ref_content = normalize_text(ref.content or "")

    if gold_source_path and ref.source_path == gold_source_path:
        if gold_text and gold_text in ref_content:
            return True
        if gold_passage_id and gold_passage_id == extract_passage_id(ref):
            return True
    if gold_ref_id and gold_ref_id == extract_reference_id(ref):
        if not gold_text or gold_text in ref_content:
            return True
    return bool(gold_text) and gold_text in ref_content


def compute_first_match_rank(
    refs: list[RetrievedReference],
    gold_docs: list[dict[str, Any]],
    limit: int,
) -> int | None:
    for index, ref in enumerate(refs[:limit], start=1):
        if any(reference_matches_gold_doc(ref, gold_doc) for gold_doc in gold_docs):
            return index
    return None


def compute_gold_passage_match_ranks(
    refs: list[RetrievedReference],
    gold_passages: list[dict[str, Any]],
    limit: int,
) -> list[int]:
    ranks: list[int] = []
    for index, ref in enumerate(refs[:limit], start=1):
        if any(reference_matches_gold_passage(ref, gold_passage) for gold_passage in gold_passages):
            ranks.append(index)
    return ranks


def compute_ndcg_at_k(match_ranks: list[int], ideal_relevant_count: int, k: int) -> float:
    if not match_ranks or ideal_relevant_count <= 0:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank in match_ranks if rank <= k)
    ideal_count = min(ideal_relevant_count, k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def assign_bucket(
    row: dict[str, Any],
    rank_at_20: int | None,
    rank_at_50: int | None,
    gold_passage_match_ranks_top5: list[int],
    gold_passage_match_ranks_top10: list[int],
) -> tuple[str, str]:
    if rank_at_50 is None:
        return "missed_recall", "gold 文档在 top-50 内未命中，属于召回失败。"

    if row.get("needs_cross_passage_aggregation"):
        gold_count = max(1, len(row.get("gold_passages", [])))
        if len(set(gold_passage_match_ranks_top10)) < gold_count:
            return "cross_passage", "问题依赖跨段信息，但 top-10 内尚未覆盖足够 gold 片段。"

    if rank_at_20 is not None and rank_at_20 > 1:
        return "low_rank", f"gold 文档已命中，但最佳排名为 {rank_at_20}，未能稳定进入 top1。"

    if row.get("answerable_from_single_chunk", False) and not gold_passage_match_ranks_top5:
        return "chunk_noise", "gold 文档已排到前面，但 top-5 返回块没有直接承载 gold 片段。"

    return "", "当前检索结果已满足 Phase 0 人工集的通过条件。"


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Phase 0 第二版 Bad Case 分桶",
        "",
        "## 总体结果",
        "",
        f"- 评测样本数：`{report['evaluated_total']}`",
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
    detail_map = {item["case_id"]: item for item in report["details"]}
    for bucket, case_ids in sorted(report["bucket_examples"].items()):
        for case_id in case_ids[:4]:
            detail = detail_map[case_id]
            lines.extend(
                [
                    f"### {case_id}",
                    f"- 分桶：`{bucket}`",
                    f"- 问题：{detail['query']}",
                    f"- 说明：{detail['failure_bucket_reason']}",
                    f"- `rank@20`：`{detail['rank_at_20']}`，`rank@50`：`{detail['rank_at_50']}`",
                    f"- top5 标题：{json.dumps(detail['top5_titles'], ensure_ascii=False)}",
                    "",
                ]
            )

    lines.extend(
        [
            "## 当前判断",
            "",
            "- 这版分桶基于 `50` 条 Phase 0 人工集，结果更适合作为后续排序与 chunk 优化的参考基线。",
            "- 该结果仍然服务于人工复核与针对性优化，不替代最终人工判定。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    project_root = PROJECT_ROOT
    settings = load_settings(project_root)
    input_path = project_root / args.input
    output_json_path = project_root / args.output_json
    output_md_path = project_root / args.output_md
    rows = load_seed_rows(input_path)

    details: list[dict[str, Any]] = []
    bucket_examples: dict[str, list[str]] = defaultdict(list)

    top1_hits = 0
    recall20_hits = 0
    recall50_hits = 0
    mrr_values: list[float] = []
    ndcg_values: list[float] = []
    status_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    question_type_counts: Counter[str] = Counter()

    for row in rows:
        knowledge_base_name = resolve_knowledge_base_name(row)
        history = build_history_messages(row)
        refs = search_local_knowledge_base(
            settings=settings,
            knowledge_base_name=knowledge_base_name,
            query=row["query"],
            top_k=args.top_k,
            score_threshold=0.0,
            history=history,
        )
        gold_docs = row.get("gold_documents", [])
        gold_passages = row.get("gold_passages", [])

        rank_at_20 = compute_first_match_rank(refs, gold_docs, 20)
        rank_at_50 = compute_first_match_rank(refs, gold_docs, 50)
        rank_at_10 = compute_first_match_rank(refs, gold_docs, 10)

        gold_passage_match_ranks_top5 = compute_gold_passage_match_ranks(refs, gold_passages, 5)
        gold_passage_match_ranks_top10 = compute_gold_passage_match_ranks(refs, gold_passages, 10)
        gold_passage_match_ranks_top20 = compute_gold_passage_match_ranks(refs, gold_passages, 20)

        top1_hit = rank_at_20 == 1
        hit_at_20 = rank_at_20 is not None
        hit_at_50 = rank_at_50 is not None
        mrr_at_10 = 0.0 if rank_at_10 is None else 1.0 / rank_at_10
        ndcg_at_10 = compute_ndcg_at_k(
            compute_gold_passage_match_ranks(refs, gold_passages, 10) or ([] if not rank_at_10 else [rank_at_10]),
            max(len(gold_passages), len(gold_docs), 1),
            10,
        )

        assigned_bucket, reason = assign_bucket(
            row=row,
            rank_at_20=rank_at_20,
            rank_at_50=rank_at_50,
            gold_passage_match_ranks_top5=gold_passage_match_ranks_top5,
            gold_passage_match_ranks_top10=gold_passage_match_ranks_top10,
        )
        status = "passed" if not assigned_bucket else "bad_case"

        top1_hits += int(top1_hit)
        recall20_hits += int(hit_at_20)
        recall50_hits += int(hit_at_50)
        mrr_values.append(mrr_at_10)
        ndcg_values.append(ndcg_at_10)
        status_counts[status] += 1
        bucket_counts[assigned_bucket or "passed"] += 1
        question_type_counts[row.get("question_type", "")] += 1
        if assigned_bucket:
            bucket_examples[assigned_bucket].append(row["case_id"])

        details.append(
            {
                "case_id": row["case_id"],
                "source_dataset": row.get("source_dataset", ""),
                "knowledge_base_name": knowledge_base_name,
                "query": row["query"],
                "question_type": row.get("question_type", ""),
                "needs_image_evidence": row.get("needs_image_evidence", False),
                "answerable_from_single_chunk": row.get("answerable_from_single_chunk", False),
                "needs_cross_passage_aggregation": row.get("needs_cross_passage_aggregation", False),
                "status": status,
                "assigned_failure_bucket": assigned_bucket,
                "failure_bucket_reason": reason,
                "top1_hit": top1_hit,
                "hit_at_20": hit_at_20,
                "hit_at_50": hit_at_50,
                "mrr_at_10": mrr_at_10,
                "ndcg_at_10": ndcg_at_10,
                "rank_at_20": rank_at_20,
                "rank_at_50": rank_at_50,
                "matched_reference_count_at_20": len(
                    [ref for ref in refs[:20] if any(reference_matches_gold_doc(ref, gold_doc) for gold_doc in gold_docs)]
                ),
                "matched_ranks_at_20": [
                    index
                    for index, ref in enumerate(refs[:20], start=1)
                    if any(reference_matches_gold_doc(ref, gold_doc) for gold_doc in gold_docs)
                ],
                "gold_passage_match_ranks_top5": gold_passage_match_ranks_top5,
                "gold_passage_match_ranks_top10": gold_passage_match_ranks_top10,
                "gold_passage_match_ranks_top20": gold_passage_match_ranks_top20,
                "top5_titles": [ref.title for ref in refs[:5]],
                "top5_sources": [ref.source for ref in refs[:5]],
                "top5_modalities": [ref.source_modality for ref in refs[:5]],
                "gold_document_labels": [doc.get("title") or doc.get("reference_id") or doc.get("source_path") for doc in gold_docs],
                "notes": row.get("notes", ""),
            }
        )

    report = {
        "input_file": str(input_path),
        "evaluated_total": len(rows),
        "protocol": {
            "top_k_retrieved": args.top_k,
            "reported_metrics": [
                "Recall@20",
                "Recall@50",
                "MRR@10",
                "NDCG@10",
                "Top1 accuracy",
            ],
        },
        "summary": {
            "top1_accuracy": top1_hits / len(rows),
            "recall_at_20": recall20_hits / len(rows),
            "recall_at_50": recall50_hits / len(rows),
            "mrr_at_10": sum(mrr_values) / len(mrr_values),
            "ndcg_at_10": sum(ndcg_values) / len(ndcg_values),
            "status_counts": dict(status_counts),
            "bucket_counts": dict(bucket_counts),
            "question_type_counts": dict(question_type_counts),
        },
        "bucket_examples": dict(bucket_examples),
        "details": details,
    }

    output_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md_path.write_text(render_markdown(report), encoding="utf-8")

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
