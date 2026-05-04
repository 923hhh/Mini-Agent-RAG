"""Run CRUD evaluation cases against a local rebuilt knowledge base.

This script reuses the project's own retrieval pipeline (`search_local_knowledge_base`).
It supports two CRUD case styles:

1. Retrieval cases with `expected_references`:
   computes hit@k style retrieval metrics.
2. Official split local cases without gold references:
   computes proxy retrieval metrics based on whether the gold answer appears
   in retrieved context and how much character overlap exists.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from scripts.eval_common import PROJECT_ROOT, load_jsonl as load_jsonl_common, write_json_report
except ModuleNotFoundError:
    from eval_common import PROJECT_ROOT, load_jsonl as load_jsonl_common, write_json_report
from app.retrievers.local_kb import search_local_knowledge_base
from app.services.core.settings import load_settings


DEFAULT_CASES_BY_KB = {
    "crud_rag_3qa_full": PROJECT_ROOT / "data" / "eval" / "crud_rag_3qa_full_retrieval_cases_100.jsonl",
    "crud_rag_official_split_local": PROJECT_ROOT / "data" / "eval" / "crud_rag_official_split_local_official_split.jsonl",
}
SAMPLE_ID_PATTERN = re.compile(r"^[0-9a-f]{24}$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--knowledge-base-name",
        type=str,
        default="crud_rag_official_split_local",
        help="Rebuilt local knowledge base name.",
    )
    parser.add_argument(
        "--cases-file",
        type=Path,
        default=None,
        help="CRUD eval cases file. Defaults depend on knowledge base name.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "crud_local_kb_eval.json",
        help="Output JSON path.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Retriever top-k.")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.35,
        help="Retriever score threshold.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Upper bound of cases to run. <=0 means all loaded cases.",
    )
    return parser.parse_args()


def resolve_cases_file(knowledge_base_name: str, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    path = DEFAULT_CASES_BY_KB.get(knowledge_base_name)
    if path is None:
        raise ValueError(
            "未提供 --cases-file，且当前 knowledge base name 没有默认 CRUD case 文件。"
        )
    return path.resolve()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return load_jsonl_common(path, encoding="utf-8-sig")


def normalize_space(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_chars(text: object) -> list[str]:
    cleaned = normalize_space(text).replace(" ", "")
    return list(cleaned)


def lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for ch_left in left:
        current = [0]
        for idx, ch_right in enumerate(right, start=1):
            if ch_left == ch_right:
                current.append(previous[idx - 1] + 1)
            else:
                current.append(max(previous[idx], current[-1]))
        previous = current
    return previous[-1]


def char_f1(prediction: object, reference: object) -> float:
    pred_chars = normalize_chars(prediction)
    ref_chars = normalize_chars(reference)
    if not pred_chars or not ref_chars:
        return 0.0
    pred_counter = Counter(pred_chars)
    ref_counter = Counter(ref_chars)
    overlap = sum(min(pred_counter[key], ref_counter[key]) for key in pred_counter.keys() & ref_counter.keys())
    if overlap <= 0:
        return 0.0
    precision = overlap / len(pred_chars)
    recall = overlap / len(ref_chars)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def rouge_l_f1(prediction: object, reference: object) -> float:
    pred_chars = normalize_chars(prediction)
    ref_chars = normalize_chars(reference)
    if not pred_chars or not ref_chars:
        return 0.0
    overlap = lcs_length(pred_chars, ref_chars)
    if overlap <= 0:
        return 0.0
    precision = overlap / len(pred_chars)
    recall = overlap / len(ref_chars)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def infer_sample_id_from_path(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    parts = [part.strip() for part in re.split(r"[\\/]+", text) if part.strip()]
    for part in reversed(parts):
        if SAMPLE_ID_PATTERN.fullmatch(part):
            return part
    return ""


def build_query(case: dict[str, Any]) -> str:
    direct = normalize_space(case.get("query"))
    if direct:
        return direct
    event = normalize_space(case.get("event"))
    question = normalize_space(case.get("question"))
    if event and question:
        return f"{event}\n{question}"
    return question or event


def extract_answer(case: dict[str, Any]) -> str:
    return normalize_space(case.get("answer") or case.get("gold_answer"))


def has_expected_references(case: dict[str, Any]) -> bool:
    references = case.get("expected_references")
    return isinstance(references, list) and bool(references)


def build_reference_keys(reference: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    sample_id = infer_sample_id_from_path(reference.get("source_path"))
    title = normalize_space(reference.get("title")).lower()
    source = normalize_space(reference.get("source")).lower()
    if sample_id:
        keys.add(f"sample:{sample_id}")
    if title:
        keys.add(f"title:{title}")
    if source:
        keys.add(f"source:{source}")
    return keys


def build_retrieved_keys(reference) -> set[str]:
    keys: set[str] = set()
    sample_id = infer_sample_id_from_path(getattr(reference, "source_path", ""))
    title = normalize_space(getattr(reference, "title", "")).lower()
    source = normalize_space(getattr(reference, "source", "")).lower()
    if sample_id:
        keys.add(f"sample:{sample_id}")
    if title:
        keys.add(f"title:{title}")
    if source:
        keys.add(f"source:{source}")
    return keys


def reference_matches_expected(reference, expected_reference: dict[str, Any]) -> bool:
    return bool(build_retrieved_keys(reference) & build_reference_keys(expected_reference))


def eval_expected_reference_hits(references, expected_references: list[dict[str, Any]]) -> dict[str, Any]:
    matched_indexes: set[int] = set()
    hit_top1 = False
    hit_top3 = False
    hit_top5 = False

    for rank, reference in enumerate(references[:5], start=1):
        matched_here = False
        for index, expected in enumerate(expected_references):
            if reference_matches_expected(reference, expected):
                matched_indexes.add(index)
                matched_here = True
        if matched_here:
            if rank <= 1:
                hit_top1 = True
            if rank <= 3:
                hit_top3 = True
            if rank <= 5:
                hit_top5 = True

    return {
        "mode": "expected_references",
        "retrieval_non_empty": bool(references),
        "hit_top1": hit_top1,
        "hit_top3": hit_top3,
        "hit_top5": hit_top5,
        "matched_expected_count_top5": len(matched_indexes),
        "all_expected_covered_top5": len(matched_indexes) >= len(expected_references),
    }


def compute_first_match_rank(references, expected_references: list[dict[str, Any]], limit: int) -> int | None:
    for rank, reference in enumerate(references[:limit], start=1):
        if any(reference_matches_expected(reference, item) for item in expected_references):
            return rank
    return None


def compute_match_ranks(references, expected_references: list[dict[str, Any]], limit: int) -> list[int]:
    ranks: list[int] = []
    for expected_reference in expected_references:
        first_rank: int | None = None
        for rank, reference in enumerate(references[:limit], start=1):
            if reference_matches_expected(reference, expected_reference):
                first_rank = rank
                break
        if first_rank is not None:
            ranks.append(first_rank)
    return sorted(set(ranks))


def compute_ndcg_at_k(match_ranks: list[int], ideal_relevant_count: int, k: int) -> float:
    if not match_ranks or ideal_relevant_count <= 0:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank in match_ranks if rank <= k)
    ideal_count = min(ideal_relevant_count, k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def build_reference_eval_text(references) -> str:
    parts: list[str] = []
    for reference in references:
        for value in (
            getattr(reference, "content", ""),
            getattr(reference, "content_preview", ""),
            getattr(reference, "evidence_summary", ""),
        ):
            text = normalize_space(value)
            if text:
                parts.append(text)
    return "\n".join(parts)


def eval_proxy_retrieval(references, gold_answer: str) -> dict[str, Any]:
    context_text = build_reference_eval_text(references)
    answer_text = normalize_space(gold_answer)
    return {
        "mode": "proxy_answer_overlap",
        "retrieval_non_empty": bool(references),
        "context_answer_substring_hit": bool(answer_text and answer_text in normalize_space(context_text)),
        "context_char_f1": char_f1(context_text, answer_text),
        "context_rouge_l_f1": rouge_l_f1(context_text, answer_text),
        "reference_count": len(references),
    }


def serialize_reference(reference) -> dict[str, Any]:
    return {
        "source": getattr(reference, "source", ""),
        "source_path": getattr(reference, "source_path", ""),
        "sample_id": infer_sample_id_from_path(getattr(reference, "source_path", "")),
        "title": getattr(reference, "title", ""),
        "raw_score": float(getattr(reference, "raw_score", 0.0) or 0.0),
        "relevance_score": float(getattr(reference, "relevance_score", 0.0) or 0.0),
        "content_preview": normalize_space(getattr(reference, "content_preview", "")),
    }


def mean(items: list[float]) -> float:
    if not items:
        return 0.0
    return sum(items) / len(items)


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    reference_cases = [item for item in results if item["retrieval_eval"]["mode"] == "expected_references"]
    proxy_cases = [item for item in results if item["retrieval_eval"]["mode"] == "proxy_answer_overlap"]
    task_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        task_buckets[str(item["task"])].append(item)

    summary: dict[str, Any] = {
        "evaluated_total": len(results),
        "case_modes": {
            "expected_references": len(reference_cases),
            "proxy_answer_overlap": len(proxy_cases),
        },
        "reference_metrics": {},
        "proxy_metrics": {},
        "by_task": {},
    }
    if reference_cases:
        summary["reference_metrics"] = {
            "hit_at_1": round(mean([1.0 if item["retrieval_eval"]["hit_top1"] else 0.0 for item in reference_cases]), 4),
            "hit_at_3": round(mean([1.0 if item["retrieval_eval"]["hit_top3"] else 0.0 for item in reference_cases]), 4),
            "hit_at_5": round(mean([1.0 if item["retrieval_eval"]["hit_top5"] else 0.0 for item in reference_cases]), 4),
            "all_expected_covered_at_5": round(mean([1.0 if item["retrieval_eval"]["all_expected_covered_top5"] else 0.0 for item in reference_cases]), 4),
            "Recall@5": round(mean([float(item.get("ranking_metrics", {}).get("Recall@5", 0.0)) for item in reference_cases]), 4),
            "MRR": round(mean([float(item.get("ranking_metrics", {}).get("MRR", 0.0)) for item in reference_cases]), 4),
            "NDCG@5": round(mean([float(item.get("ranking_metrics", {}).get("NDCG@5", 0.0)) for item in reference_cases]), 4),
        }
    if proxy_cases:
        summary["proxy_metrics"] = {
            "retrieval_non_empty": round(mean([1.0 if item["retrieval_eval"]["retrieval_non_empty"] else 0.0 for item in proxy_cases]), 4),
            "context_answer_substring_hit": round(mean([1.0 if item["retrieval_eval"]["context_answer_substring_hit"] else 0.0 for item in proxy_cases]), 4),
            "context_char_f1": round(mean([float(item["retrieval_eval"]["context_char_f1"]) for item in proxy_cases]), 4),
            "context_rouge_l_f1": round(mean([float(item["retrieval_eval"]["context_rouge_l_f1"]) for item in proxy_cases]), 4),
            "reference_count": round(mean([float(item["retrieval_eval"]["reference_count"]) for item in proxy_cases]), 4),
        }

    for task_name, items in sorted(task_buckets.items()):
        reference_items = [item for item in items if item["retrieval_eval"]["mode"] == "expected_references"]
        proxy_items = [item for item in items if item["retrieval_eval"]["mode"] == "proxy_answer_overlap"]
        summary["by_task"][task_name] = {
            "count": len(items),
            "reference_metrics": {
                "hit_at_1": round(mean([1.0 if item["retrieval_eval"]["hit_top1"] else 0.0 for item in reference_items]), 4) if reference_items else 0.0,
                "hit_at_3": round(mean([1.0 if item["retrieval_eval"]["hit_top3"] else 0.0 for item in reference_items]), 4) if reference_items else 0.0,
                "hit_at_5": round(mean([1.0 if item["retrieval_eval"]["hit_top5"] else 0.0 for item in reference_items]), 4) if reference_items else 0.0,
                "Recall@5": round(mean([float(item.get("ranking_metrics", {}).get("Recall@5", 0.0)) for item in reference_items]), 4) if reference_items else 0.0,
                "MRR": round(mean([float(item.get("ranking_metrics", {}).get("MRR", 0.0)) for item in reference_items]), 4) if reference_items else 0.0,
                "NDCG@5": round(mean([float(item.get("ranking_metrics", {}).get("NDCG@5", 0.0)) for item in reference_items]), 4) if reference_items else 0.0,
            },
            "proxy_metrics": {
                "retrieval_non_empty": round(mean([1.0 if item["retrieval_eval"]["retrieval_non_empty"] else 0.0 for item in proxy_items]), 4) if proxy_items else 0.0,
                "context_answer_substring_hit": round(mean([1.0 if item["retrieval_eval"]["context_answer_substring_hit"] else 0.0 for item in proxy_items]), 4) if proxy_items else 0.0,
                "context_char_f1": round(mean([float(item["retrieval_eval"]["context_char_f1"]) for item in proxy_items]), 4) if proxy_items else 0.0,
                "context_rouge_l_f1": round(mean([float(item["retrieval_eval"]["context_rouge_l_f1"]) for item in proxy_items]), 4) if proxy_items else 0.0,
            },
        }
    return summary


def main() -> int:
    args = parse_args()
    settings = load_settings(PROJECT_ROOT)
    cases_file = resolve_cases_file(args.knowledge_base_name, args.cases_file)
    cases = load_jsonl(cases_file)
    if args.max_cases > 0:
        cases = cases[:args.max_cases]
    if not cases:
        raise RuntimeError("No CRUD eval cases loaded.")

    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        query = build_query(case)
        if not query:
            continue
        task = normalize_space(case.get("task") or case.get("category") or "unknown")
        print(
            f"[crud-local-eval] {index}/{len(cases)} "
            f"task={task} case_id={case.get('case_id', '')}",
            flush=True,
        )
        references = search_local_knowledge_base(
            settings=settings,
            knowledge_base_name=args.knowledge_base_name,
            query=query,
            top_k=args.top_k,
            score_threshold=args.score_threshold,
            history=None,
        )
        if has_expected_references(case):
            expected_references = list(case.get("expected_references") or [])
            retrieval_eval = eval_expected_reference_hits(
                references,
                expected_references,
            )
            first_rank = compute_first_match_rank(references, expected_references, 5)
            match_ranks = compute_match_ranks(references, expected_references, 5)
            ranking_metrics: dict[str, Any] = {
                "Recall@5": 0.0 if first_rank is None else 1.0,
                "MRR": 0.0 if first_rank is None else round(1.0 / first_rank, 6),
                "NDCG@5": round(compute_ndcg_at_k(match_ranks, max(len(expected_references), 1), 5), 6),
                "first_match_rank_at_5": first_rank,
                "matched_ranks_at_5": match_ranks,
            }
        else:
            retrieval_eval = eval_proxy_retrieval(references, extract_answer(case))
            ranking_metrics = {
                "Recall@5": None,
                "MRR": None,
                "NDCG@5": None,
                "note": "official_split_local 类 case 缺少 expected_references，无法计算标准排序指标。",
            }

        results.append(
            {
                "case_id": case.get("case_id", ""),
                "task": task,
                "source_task": case.get("source_task", ""),
                "query": query,
                "gold_answer": extract_answer(case),
                "retrieved_references": [serialize_reference(item) for item in references[: args.top_k]],
                "retrieval_eval": retrieval_eval,
                "ranking_metrics": ranking_metrics,
            }
        )

    summary = build_summary(results)
    payload = {
        "knowledge_base_name": args.knowledge_base_name,
        "cases_file": str(cases_file),
        "case_count": len(results),
        "top_k": args.top_k,
        "score_threshold": args.score_threshold,
        "summary": summary,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_report(args.output, payload)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[done] output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
