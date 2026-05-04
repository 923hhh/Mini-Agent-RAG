"""Run official DomainRAG evaluation tasks against a local rebuilt knowledge base.

This script reuses the project's own retrieval pipeline (`search_local_knowledge_base`)
instead of building a separate offline retriever. It loads the official DomainRAG
task files from a checked-out `DomainRAG-main` workspace, sends each query to the
local knowledge base, and writes per-sample retrieval results plus aggregate hit
metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from scripts.eval_common import PROJECT_ROOT, load_jsonl as load_jsonl_common, write_json_report
except ModuleNotFoundError:
    from eval_common import PROJECT_ROOT, load_jsonl as load_jsonl_common, write_json_report
from app.retrievers.local_kb import search_local_knowledge_base
from app.services.core.settings import load_settings


DEFAULT_TASK_FILES = {
    "basic": "BCM/labeled_data/extractive_qa/basic_qa.jsonl",
    "conversation": "BCM/labeled_data/conversation_qa/conversation_qa.jsonl",
    "multidoc": "BCM/labeled_data/multi-doc_qa/multidoc_qa.jsonl",
    "time": "BCM/labeled_data/time-sensitive_qa/time_sensitive.jsonl",
    "structure": "BCM/labeled_data/structured_qa/structured_qa_twopositive.jsonl",
}
TASK_NAME_MAP = {
    "basic": "extractive_qa",
    "conversation": "conversation_qa",
    "multidoc": "multi-doc_qa",
    "time": "time-sensitive_qa",
    "structure": "structured_qa",
}
DOC_PSG_PATTERN = re.compile(r"__doc-(?P<doc_id>\d+)__psg-(?P<psg_id>\d+)\.txt$", re.IGNORECASE)
DOC_ONLY_PATTERN = re.compile(r"__doc-(?P<doc_id>\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domainrag-root",
        type=Path,
        required=True,
        help="Local path of DomainRAG-main.",
    )
    parser.add_argument(
        "--knowledge-base-name",
        type=str,
        default="domainrag_full_corpus",
        help="Rebuilt local knowledge base name.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "domainrag_full_local_eval.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASK_FILES.keys()),
        choices=list(DEFAULT_TASK_FILES.keys()),
        help="Which official DomainRAG tasks to include.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Upper bound of evaluation samples to run. <=0 means all loaded samples.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Retriever top-k.")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.35,
        help="Retriever score threshold.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return load_jsonl_common(path, encoding="utf-8")


def normalize_space(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_task_alias(task: str) -> str:
    return TASK_NAME_MAP.get(task, task)


def build_query(task: str, sample: dict[str, Any]) -> str:
    if task == "time":
        year = str(sample.get("date", "")).strip()
        prefix = f"{year}年 " if year else ""
        return normalize_space(prefix + str(sample.get("question", "")))
    if task == "conversation":
        history = [
            str(item.get("question", "")).strip()
            for item in sample.get("history_qa", [])
            if str(item.get("question", "")).strip()
        ]
        history.append(str(sample.get("question", "")).strip())
        return normalize_space(" ".join(history))
    return normalize_space(str(sample.get("question", "")))


def load_eval_samples(domainrag_root: Path, tasks: list[str], max_samples: int) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for task in tasks:
        dataset_path = domainrag_root / DEFAULT_TASK_FILES[task]
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        rows = load_jsonl(dataset_path)
        for index, row in enumerate(rows):
            merged.append(
                {
                    "task": normalize_task_alias(task),
                    "task_key": task,
                    "sample_index": index,
                    "question": row.get("question", ""),
                    "query": build_query(task, row),
                    "raw": row,
                }
            )
    if max_samples > 0:
        return merged[:max_samples]
    return merged


def extract_positive_references(sample: dict[str, Any]) -> list[dict[str, Any]]:
    raw = sample["raw"]
    positives = raw.get("positive_reference")
    if isinstance(positives, list) and positives:
        return [item for item in positives if isinstance(item, dict)]
    positives = raw.get("positive_references")
    if isinstance(positives, list) and positives:
        return [item for item in positives if isinstance(item, dict)]
    return []


def extract_reference_doc_psg(source_path: str) -> tuple[str, str]:
    normalized = str(source_path or "").replace("\\", "/")
    match = DOC_PSG_PATTERN.search(normalized)
    if match:
        return match.group("doc_id"), match.group("psg_id")
    match = DOC_ONLY_PATTERN.search(normalized)
    if match:
        return match.group("doc_id"), ""
    return "", ""


def build_positive_keys(positive: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    doc_id = normalize_space(positive.get("id"))
    psg_id = normalize_space(positive.get("psg_id"))
    url = normalize_space(positive.get("url")).lower()
    title = normalize_space(positive.get("title")).lower()
    if doc_id and psg_id:
        keys.add(f"doc_psg:{doc_id}:{psg_id}")
    if doc_id:
        keys.add(f"doc:{doc_id}")
    if url:
        keys.add(f"url:{url}")
    if title:
        keys.add(f"title:{title}")
    return keys


def build_retrieved_keys(reference) -> set[str]:
    keys: set[str] = set()
    doc_id, psg_id = extract_reference_doc_psg(getattr(reference, "source_path", ""))
    if doc_id and psg_id:
        keys.add(f"doc_psg:{doc_id}:{psg_id}")
    if doc_id:
        keys.add(f"doc:{doc_id}")
    source = normalize_space(getattr(reference, "source", "")).lower()
    title = normalize_space(getattr(reference, "title", "")).lower()
    if source.startswith("http"):
        keys.add(f"url:{source}")
    if title:
        keys.add(f"title:{title}")
    return keys


def reference_matches_positive(reference, positive: dict[str, Any]) -> bool:
    return bool(build_positive_keys(positive) & build_retrieved_keys(reference))


def evaluate_retrieval(references, positives: list[dict[str, Any]]) -> dict[str, Any]:
    if not positives:
        return {
            "has_positive_annotations": False,
            "hit_top1": False,
            "hit_top3": False,
            "hit_top5": False,
            "hit_top10": False,
            "matched_positive_count_top10": 0,
            "all_positive_covered_top10": False,
        }

    matched_positive_indexes: set[int] = set()
    hit_top1 = False
    hit_top3 = False
    hit_top5 = False
    hit_top10 = False

    for rank, reference in enumerate(references[:10], start=1):
        matched_here = False
        for positive_index, positive in enumerate(positives):
            if reference_matches_positive(reference, positive):
                matched_positive_indexes.add(positive_index)
                matched_here = True
        if matched_here:
            if rank <= 1:
                hit_top1 = True
            if rank <= 3:
                hit_top3 = True
            if rank <= 5:
                hit_top5 = True
            if rank <= 10:
                hit_top10 = True

    return {
        "has_positive_annotations": True,
        "hit_top1": hit_top1,
        "hit_top3": hit_top3,
        "hit_top5": hit_top5,
        "hit_top10": hit_top10,
        "matched_positive_count_top10": len(matched_positive_indexes),
        "all_positive_covered_top10": len(matched_positive_indexes) >= len(positives),
    }


def compute_first_match_rank(references, positives: list[dict[str, Any]], limit: int) -> int | None:
    for rank, reference in enumerate(references[:limit], start=1):
        if any(reference_matches_positive(reference, positive) for positive in positives):
            return rank
    return None


def compute_match_ranks(references, positives: list[dict[str, Any]], limit: int) -> list[int]:
    ranks: list[int] = []
    for positive in positives:
        first_rank: int | None = None
        for rank, reference in enumerate(references[:limit], start=1):
            if reference_matches_positive(reference, positive):
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


def serialize_reference(reference) -> dict[str, Any]:
    doc_id, psg_id = extract_reference_doc_psg(getattr(reference, "source_path", ""))
    return {
        "source": getattr(reference, "source", ""),
        "source_path": getattr(reference, "source_path", ""),
        "title": getattr(reference, "title", ""),
        "raw_score": float(getattr(reference, "raw_score", 0.0) or 0.0),
        "relevance_score": float(getattr(reference, "relevance_score", 0.0) or 0.0),
        "doc_id": doc_id,
        "psg_id": psg_id,
        "content_preview": normalize_space(getattr(reference, "content_preview", "")),
    }


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [item for item in results if item["retrieval_eval"]["has_positive_annotations"]]
    task_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in evaluated:
        task_buckets[str(item["task"])].append(item)

    def rate(items: list[dict[str, Any]], field: str) -> float:
        if not items:
            return 0.0
        hits = sum(1 for item in items if item["retrieval_eval"].get(field))
        return round(hits / len(items), 4)

    summary = {
        "evaluated_total": len(evaluated),
        "hit_at_1": rate(evaluated, "hit_top1"),
        "hit_at_3": rate(evaluated, "hit_top3"),
        "hit_at_5": rate(evaluated, "hit_top5"),
        "hit_at_10": rate(evaluated, "hit_top10"),
        "all_positive_covered_at_10": rate(evaluated, "all_positive_covered_top10"),
        "Recall@5": round(
            sum(float(item.get("ranking_metrics", {}).get("Recall@5", 0.0)) for item in evaluated)
            / max(1, len(evaluated)),
            4,
        ),
        "MRR": round(
            sum(float(item.get("ranking_metrics", {}).get("MRR", 0.0)) for item in evaluated)
            / max(1, len(evaluated)),
            4,
        ),
        "NDCG@5": round(
            sum(float(item.get("ranking_metrics", {}).get("NDCG@5", 0.0)) for item in evaluated)
            / max(1, len(evaluated)),
            4,
        ),
        "by_task": {},
    }
    for task_name, items in sorted(task_buckets.items()):
        summary["by_task"][task_name] = {
            "count": len(items),
            "hit_at_1": rate(items, "hit_top1"),
            "hit_at_3": rate(items, "hit_top3"),
            "hit_at_5": rate(items, "hit_top5"),
            "hit_at_10": rate(items, "hit_top10"),
            "all_positive_covered_at_10": rate(items, "all_positive_covered_top10"),
            "Recall@5": round(
                sum(float(item.get("ranking_metrics", {}).get("Recall@5", 0.0)) for item in items)
                / max(1, len(items)),
                4,
            ),
            "MRR": round(
                sum(float(item.get("ranking_metrics", {}).get("MRR", 0.0)) for item in items)
                / max(1, len(items)),
                4,
            ),
            "NDCG@5": round(
                sum(float(item.get("ranking_metrics", {}).get("NDCG@5", 0.0)) for item in items)
                / max(1, len(items)),
                4,
            ),
        }
    return summary


def main() -> int:
    args = parse_args()
    domainrag_root = args.domainrag_root.resolve()
    settings = load_settings(PROJECT_ROOT)
    samples = load_eval_samples(domainrag_root, args.tasks, args.max_samples)
    if not samples:
        raise RuntimeError("No evaluation samples loaded.")

    results: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        print(
            f"[domainrag-local-eval] {index}/{len(samples)} "
            f"{sample['task']} sample={sample['sample_index']}",
            flush=True,
        )
        references = search_local_knowledge_base(
            settings=settings,
            knowledge_base_name=args.knowledge_base_name,
            query=sample["query"],
            top_k=args.top_k,
            score_threshold=args.score_threshold,
            history=None,
        )
        positives = extract_positive_references(sample)
        retrieval_eval = evaluate_retrieval(references, positives)
        first_rank = compute_first_match_rank(references, positives, 5) if positives else None
        match_ranks = compute_match_ranks(references, positives, 5) if positives else []
        ranking_metrics = {
            "Recall@5": 0.0 if first_rank is None else 1.0,
            "MRR": 0.0 if first_rank is None else round(1.0 / first_rank, 6),
            "NDCG@5": round(compute_ndcg_at_k(match_ranks, max(len(positives), 1), 5), 6) if positives else 0.0,
            "first_match_rank_at_5": first_rank,
            "matched_ranks_at_5": match_ranks,
        }
        results.append(
            {
                "task": sample["task"],
                "task_key": sample["task_key"],
                "sample_index": sample["sample_index"],
                "question": sample["question"],
                "query": sample["query"],
                "positive_reference_count": len(positives),
                "positive_references": positives,
                "retrieved_references": [serialize_reference(item) for item in references[:10]],
                "retrieval_eval": retrieval_eval,
                "ranking_metrics": ranking_metrics,
            }
        )

    summary = build_summary(results)
    payload = {
        "knowledge_base_name": args.knowledge_base_name,
        "domainrag_root": str(domainrag_root),
        "task_keys": args.tasks,
        "task_names": [normalize_task_alias(item) for item in args.tasks],
        "sample_count": len(results),
        "top_k": args.top_k,
        "score_threshold": args.score_threshold,
        "summary": summary,
        "results": results,
    }
    write_json_report(args.output, payload)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[done] output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
