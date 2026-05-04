"""在同一评测集上对 Dense only / BM25 only / Hybrid 三种检索方案进行对照评测。"""

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

from app.retrievers.local_kb import generate_hypothetical_doc
from app.services.retrieval.candidate_common_service import RetrievalCandidate, get_chunk_id
from app.services.retrieval.candidate_fusion_service import build_temporal_candidate_adjustments
from app.services.retrieval.candidate_retrieval_service import (
    build_dense_query_bundle,
    build_query_bundle,
    collect_dense_candidates,
    collect_lexical_candidates,
    filter_documents_by_metadata,
    load_all_documents,
)
from app.services.retrieval.candidate_rerank_service import (
    diversify_candidates,
    heuristic_rerank_candidates,
    rerank_candidates,
)
from app.services.retrieval.reference_assembly_service import (
    candidate_to_reference,
    group_documents_by_doc_id,
)
from app.schemas.chat import RetrievedReference
from app.services.core.settings import load_settings
from app.services.models.embedding_service import build_embeddings
from app.services.retrieval.query_profile_service import (
    infer_diversity_query_profile,
    infer_query_modality_profile,
    infer_temporal_query_profile,
)
from app.services.retrieval.query_rewrite_service import generate_multi_queries
from app.storage.bm25_index import load_bm25_index, resolve_bm25_index_path
from app.storage.vector_stores import build_vector_store_adapter


DOC_ID_PATTERN = re.compile(r"(?:文档ID：|doc-)(\d+)")
PASSAGE_ID_PATTERN = re.compile(r"(?:段落ID：|psg-)([\w-]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="data/eval/domainrag_small_batch_100_domainrag_small_batch.jsonl",
        help="评测集 jsonl 路径。",
    )
    parser.add_argument(
        "--knowledge-base-name",
        default="domainrag_small_batch_100",
        help="用于检索的 knowledge_base_name。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="单次检索深度，默认 50。",
    )
    parser.add_argument(
        "--report-top-k",
        type=int,
        default=5,
        help="用于统计 Recall@K / NDCG@K 的 K 值，默认 5。",
    )
    parser.add_argument(
        "--output-json",
        default="data/eval/shared_analysis/retrieval_baselines_domain100.json",
        help="json 输出路径。",
    )
    parser.add_argument(
        "--output-md",
        default="data/eval/shared_analysis/retrieval_baselines_domain100.md",
        help="markdown 输出路径。",
    )
    parser.add_argument(
        "--use-query-rewrite",
        action="store_true",
        help="是否启用 query rewrite / multi-query。",
    )
    parser.add_argument(
        "--use-rerank",
        action="store_true",
        help="是否启用启发式重排和模型重排。",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize_text(value: str) -> str:
    text = str(value or "")
    text = re.sub(r"\s+", "", text)
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    return text


def extract_reference_id(ref: RetrievedReference) -> str:
    for text in (ref.content or "", ref.source or "", ref.chunk_id or ""):
        match = DOC_ID_PATTERN.search(text)
        if match:
            return match.group(1)
    return ""


def extract_passage_id(ref: RetrievedReference) -> str:
    for text in (ref.content or "", ref.source or "", ref.chunk_id or ""):
        match = PASSAGE_ID_PATTERN.search(text)
        if match:
            return match.group(1)
    return ""


def reference_matches_positive_reference(ref: RetrievedReference, positive_ref: dict[str, Any]) -> bool:
    gold_title = str(positive_ref.get("title", "")).strip()
    gold_url = str(positive_ref.get("url", "")).strip()
    gold_id = str(positive_ref.get("id", "")).strip()
    gold_psg_id = str(positive_ref.get("psg_id", "")).strip()
    gold_contents = normalize_text(positive_ref.get("contents", ""))
    ref_content = normalize_text(ref.content or "")

    if gold_title and ref.title == gold_title:
        return True
    if gold_id and gold_id == extract_reference_id(ref):
        return True
    if gold_url and gold_url in (ref.content or ""):
        return True
    if gold_psg_id and gold_psg_id == extract_passage_id(ref):
        return True
    if gold_contents and ref_content and gold_contents in ref_content:
        return True
    return False


def compute_first_match_rank(
    refs: list[RetrievedReference],
    positive_refs: list[dict[str, Any]],
    limit: int,
) -> int | None:
    for index, ref in enumerate(refs[:limit], start=1):
        if any(reference_matches_positive_reference(ref, positive_ref) for positive_ref in positive_refs):
            return index
    return None


def compute_match_ranks(
    refs: list[RetrievedReference],
    positive_refs: list[dict[str, Any]],
    limit: int,
) -> list[int]:
    ranks: list[int] = []
    for positive_ref in positive_refs:
        first_rank: int | None = None
        for index, ref in enumerate(refs[:limit], start=1):
            if reference_matches_positive_reference(ref, positive_ref):
                first_rank = index
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


def build_candidates_for_mode(
    *,
    mode: str,
    settings,
    vector_store,
    all_documents: dict[str, Any],
    bm25_index,
    query: str,
    top_k: int,
) -> tuple[list[RetrievalReferenceLike], dict[str, object]]:
    raise NotImplementedError


def load_retrieval_resources(settings, knowledge_base_name: str) -> dict[str, Any]:
    vector_store_dir = settings.vector_store_dir(knowledge_base_name)
    embeddings = build_embeddings(settings)
    vector_store = build_vector_store_adapter(
        settings,
        vector_store_dir,
        embeddings,
        collection_name=vector_store_dir.name,
    )
    all_documents = filter_documents_by_metadata(load_all_documents(vector_store), None)
    bm25_index = load_bm25_index(resolve_bm25_index_path(vector_store_dir))
    grouped_documents = group_documents_by_doc_id(all_documents)
    return {
        "vector_store": vector_store,
        "all_documents": all_documents,
        "bm25_index": bm25_index,
        "grouped_documents": grouped_documents,
    }


def build_query_runtime(settings, query: str, *, use_query_rewrite: bool) -> dict[str, Any]:
    if use_query_rewrite:
        query_candidates = generate_multi_queries(settings, query, history=None)
    else:
        query_candidates = [query.strip()]
    query_bundle = build_query_bundle(query_candidates)
    dense_query_bundle = build_dense_query_bundle(
        query_bundle,
        generate_hypothetical_doc(settings, query, history=None),
    )
    query_profile = infer_query_modality_profile(query_bundle)
    temporal_profile = infer_temporal_query_profile(query_bundle)
    return {
        "query_bundle": query_bundle,
        "dense_query_bundle": dense_query_bundle,
        "query_profile": query_profile,
        "temporal_profile": temporal_profile,
    }


def search_with_mode(
    *,
    mode: str,
    settings,
    resources: dict[str, Any],
    query: str,
    query_runtime: dict[str, Any],
    top_k: int,
    use_rerank: bool,
) -> list[RetrievedReference]:
    vector_store = resources["vector_store"]
    all_documents = resources["all_documents"]
    bm25_index = resources["bm25_index"]
    grouped_documents = resources["grouped_documents"]

    query_bundle = query_runtime["query_bundle"]
    dense_query_bundle = query_runtime["dense_query_bundle"]
    query_profile = query_runtime["query_profile"]
    temporal_profile = query_runtime["temporal_profile"]
    candidate_map: dict[str, RetrievalCandidate] = {}

    if mode in {"dense", "hybrid"}:
        collect_dense_candidates(
            vector_store=vector_store,
            query_bundle=dense_query_bundle,
            dense_limit=max(top_k, settings.kb.HYBRID_DENSE_TOP_K),
            candidate_map=candidate_map,
            metadata_filters=None,
            dense_fetch_multiplier=settings.kb.METADATA_FILTER_DENSE_FETCH_MULTIPLIER,
        )

    if mode in {"bm25", "hybrid"}:
        collect_lexical_candidates(
            settings=settings,
            all_documents=all_documents,
            query_bundle=query_bundle,
            bm25_index=bm25_index,
            candidate_map=candidate_map,
            lexical_limit=max(top_k, settings.kb.HYBRID_LEXICAL_TOP_K),
        )

    candidates = list(candidate_map.values())
    if not candidates:
        return []

    max_dense = max((item.dense_relevance for item in candidates), default=1.0) or 1.0
    max_lexical = max((item.lexical_score for item in candidates), default=1.0) or 1.0
    temporal_adjustments = build_temporal_candidate_adjustments(candidates, temporal_profile)
    for item in candidates:
        score = 0.0
        if item.dense_rank is not None:
            score += 1.0 / (settings.kb.HYBRID_RRF_K + item.dense_rank)
            score += settings.kb.HYBRID_DENSE_SCORE_WEIGHT * (item.dense_relevance / max_dense)
        if item.lexical_rank is not None:
            score += 1.0 / (settings.kb.HYBRID_RRF_K + item.lexical_rank)
            score += settings.kb.HYBRID_LEXICAL_SCORE_WEIGHT * (item.lexical_score / max_lexical)
        score += temporal_adjustments.get(get_chunk_id(item.document), 0.0)
        item.fused_score = score
        item.relevance_score = max(item.dense_relevance, item.lexical_score, score)

    if use_rerank:
        heuristic_ranked = heuristic_rerank_candidates(
            settings=settings,
            candidates=candidates,
            query_bundle=query_bundle,
            top_k=top_k,
        )
        reranked = rerank_candidates(
            settings=settings,
            query=query,
            candidates=heuristic_ranked,
            query_bundle=query_bundle,
            query_profile=query_profile,
            top_k=top_k,
            diagnostics=None,
        )
    else:
        reranked = sorted(candidates, key=lambda item: item.fused_score, reverse=True)
        for candidate in reranked:
            candidate.relevance_score = max(candidate.relevance_score, candidate.fused_score)
    final_candidates = diversify_candidates(
        reranked,
        target_count=top_k,
        query_profile=query_profile,
        diversity_profile=infer_diversity_query_profile(query_bundle),
    )
    return [
        candidate_to_reference(
            settings=settings,
            candidate=item,
            grouped_documents=grouped_documents,
        )
        for item in final_candidates
    ]


def evaluate_mode(
    *,
    mode: str,
    rows: list[dict[str, Any]],
    settings,
    resources: dict[str, Any],
    query_runtimes: dict[str, dict[str, Any]],
    top_k: int,
    report_top_k: int,
    use_rerank: bool,
) -> dict[str, Any]:
    recall_hits = 0
    mrr_values: list[float] = []
    ndcg_values: list[float] = []
    task_buckets: dict[str, dict[str, list[float] | int]] = defaultdict(
        lambda: {"recall_hits": 0, "mrr_values": [], "ndcg_values": [], "count": 0}
    )

    for row in rows:
        positive_refs = row.get("positive_reference")
        if positive_refs is None:
            positive_refs = row.get("positive_references", [])
        if isinstance(positive_refs, dict):
            positive_refs = [positive_refs]

        refs = search_with_mode(
            mode=mode,
            settings=settings,
            resources=resources,
            query=row["question"],
            query_runtime=query_runtimes[row["question"]],
            top_k=top_k,
            use_rerank=use_rerank,
        )
        first_rank = compute_first_match_rank(refs, positive_refs, report_top_k)
        match_ranks = compute_match_ranks(refs, positive_refs, report_top_k)
        recall_hit = int(first_rank is not None)
        mrr = 0.0 if first_rank is None else 1.0 / first_rank
        ndcg = compute_ndcg_at_k(match_ranks, max(len(positive_refs), 1), report_top_k)

        task = str(row.get("domainrag_task", "unknown"))
        bucket = task_buckets[task]
        bucket["recall_hits"] += recall_hit
        bucket["mrr_values"].append(mrr)
        bucket["ndcg_values"].append(ndcg)
        bucket["count"] += 1

        recall_hits += recall_hit
        mrr_values.append(mrr)
        ndcg_values.append(ndcg)

    total = len(rows)
    task_metrics = {
        task: {
            "count": int(bucket["count"]),
            "Recall@5": float(bucket["recall_hits"]) / max(int(bucket["count"]), 1),
            "MRR": sum(bucket["mrr_values"]) / max(len(bucket["mrr_values"]), 1),
            "NDCG@5": sum(bucket["ndcg_values"]) / max(len(bucket["ndcg_values"]), 1),
        }
        for task, bucket in sorted(task_buckets.items())
    }
    return {
        "mode": mode,
        "count": total,
        "Recall@5": recall_hits / max(total, 1),
        "MRR": sum(mrr_values) / max(len(mrr_values), 1),
        "NDCG@5": sum(ndcg_values) / max(len(ndcg_values), 1),
        "task_metrics": task_metrics,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Domain 100 检索基线对比",
        "",
        f"- 评测集：`{report['dataset']}`",
        f"- 知识库：`{report['knowledge_base_name']}`",
        f"- 样本数：`{report['sample_count']}`",
        f"- Query Rewrite：`{'on' if report['use_query_rewrite'] else 'off'}`",
        f"- Rerank：`{'on' if report['use_rerank'] else 'off'}`",
        "",
        "## 总体结果",
        "",
        "| 方法 | Recall@5 | MRR | NDCG@5 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for mode in ("bm25", "dense", "hybrid"):
        metrics = report["modes"][mode]
        label = {
            "bm25": "BM25 only",
            "dense": "Dense only",
            "hybrid": "Hybrid",
        }[mode]
        lines.append(
            f"| {label} | {metrics['Recall@5']:.4f} | {metrics['MRR']:.4f} | {metrics['NDCG@5']:.4f} |"
        )

    lines.extend(["", "## 分任务结果", ""])
    for mode in ("bm25", "dense", "hybrid"):
        metrics = report["modes"][mode]
        label = {
            "bm25": "BM25 only",
            "dense": "Dense only",
            "hybrid": "Hybrid",
        }[mode]
        lines.extend(
            [
                f"### {label}",
                "",
                "| 任务类型 | Recall@5 | MRR | NDCG@5 |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for task, task_metrics in metrics["task_metrics"].items():
            lines.append(
                f"| {task} | {task_metrics['Recall@5']:.4f} | {task_metrics['MRR']:.4f} | {task_metrics['NDCG@5']:.4f} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    settings = load_settings(PROJECT_ROOT)
    rows = load_rows(PROJECT_ROOT / args.dataset)
    resources = load_retrieval_resources(settings, args.knowledge_base_name)
    query_runtimes = {
        row["question"]: build_query_runtime(
            settings,
            row["question"],
            use_query_rewrite=args.use_query_rewrite,
        )
        for row in rows
    }

    results = {}
    for mode in ("bm25", "dense", "hybrid"):
        results[mode] = evaluate_mode(
            mode=mode,
            rows=rows,
            settings=settings,
            resources=resources,
            query_runtimes=query_runtimes,
            top_k=args.top_k,
            report_top_k=args.report_top_k,
            use_rerank=args.use_rerank,
        )

    report = {
        "dataset": args.dataset,
        "knowledge_base_name": args.knowledge_base_name,
        "sample_count": len(rows),
        "use_query_rewrite": args.use_query_rewrite,
        "use_rerank": args.use_rerank,
        "modes": results,
    }
    output_json_path = PROJECT_ROOT / args.output_json
    output_md_path = PROJECT_ROOT / args.output_md
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md_path.write_text(render_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
