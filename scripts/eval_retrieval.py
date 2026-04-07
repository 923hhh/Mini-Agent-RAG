from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retrievers.local_kb import search_local_knowledge_base
from app.services.settings import load_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality on local JSONL cases.")
    parser.add_argument(
        "--case-file",
        default=str(PROJECT_ROOT / "data" / "eval" / "rag_eval.jsonl"),
        help="Path to the evaluation JSONL file.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Default top_k when not specified by the case.")
    parser.add_argument(
        "--compare-model-rerank",
        action="store_true",
        help="Evaluate both heuristic-only and model-rerank settings and print both summaries.",
    )
    parser.add_argument(
        "--show-cases",
        action="store_true",
        help="Print per-case retrieval results.",
    )
    args = parser.parse_args()

    settings = load_settings(PROJECT_ROOT)
    cases = load_cases(Path(args.case_file))
    if args.compare_model_rerank:
        heuristic_settings = settings.model_copy(
            update={"kb": settings.kb.model_copy(update={"ENABLE_MODEL_RERANK": False})}
        )
        model_settings = settings.model_copy(
            update={"kb": settings.kb.model_copy(update={"ENABLE_MODEL_RERANK": True})}
        )
        summary = {
            "heuristic_only": evaluate_cases(
                heuristic_settings,
                cases,
                default_top_k=args.top_k,
                show_cases=args.show_cases,
            ),
            "model_rerank": evaluate_cases(
                model_settings,
                cases,
                default_top_k=args.top_k,
                show_cases=args.show_cases,
            ),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    summary = evaluate_cases(settings, cases, default_top_k=args.top_k, show_cases=args.show_cases)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def load_cases(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"评测文件不存在: {path}")

    cases: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        cases.append(json.loads(stripped))
    return cases


def evaluate_cases(
    settings,
    cases: list[dict[str, object]],
    *,
    default_top_k: int,
    show_cases: bool,
) -> dict[str, object]:
    total = 0
    skipped = 0
    hit_count = 0
    reciprocal_rank_sum = 0.0
    top1_hit_count = 0
    evidence_eval_total = 0
    evidence_hit_count = 0
    modality_eval_total = 0
    modality_hit_count = 0
    modality_presence_eval_total = 0
    modality_presence_hit_count = 0
    details: list[dict[str, object]] = []
    category_stats: dict[str, dict[str, float]] = {}

    for case in cases:
        case_id = str(case.get("case_id", "")).strip()
        category = str(case.get("category", "uncategorized")).strip() or "uncategorized"
        knowledge_base_name = str(case.get("knowledge_base_name", "")).strip()
        query = str(case.get("query", "")).strip()
        expected_source = str(case.get("expected_source", "")).strip()
        expected_evidence_type = str(case.get("expected_evidence_type", "")).strip()
        expected_source_modality = str(case.get("expected_source_modality", "")).strip()
        expected_modalities_present = normalize_expected_list(
            case.get("expected_modalities_present")
        )
        top_k = int(case.get("top_k", default_top_k))

        if not knowledge_base_name or not query:
            skipped += 1
            continue

        total += 1
        try:
            references = search_local_knowledge_base(
                settings=settings,
                knowledge_base_name=knowledge_base_name,
                query=query,
                top_k=top_k,
                score_threshold=0.0,
            )
        except FileNotFoundError:
            skipped += 1
            continue

        ranked_sources = [item.source for item in references]
        rank = find_rank(ranked_sources, expected_source)
        top1_reference = references[0] if references else None
        top1_source = top1_reference.source if top1_reference is not None else ""
        top1_evidence_type = top1_reference.evidence_type if top1_reference is not None else ""
        top1_source_modality = top1_reference.source_modality if top1_reference is not None else ""
        hit = rank is not None if expected_source else False
        top1_hit = rank == 1 if expected_source else False
        evidence_hit = bool(expected_evidence_type) and top1_evidence_type == expected_evidence_type
        modality_hit = bool(expected_source_modality) and top1_source_modality == expected_source_modality
        returned_modalities = [
            (item.source_modality or "").strip() for item in references if (item.source_modality or "").strip()
        ]
        modality_presence_hit = (
            bool(expected_modalities_present)
            and all(expected in returned_modalities for expected in expected_modalities_present)
        )

        if expected_source and hit:
            hit_count += 1
            reciprocal_rank_sum += 1.0 / rank
        if expected_source and top1_hit:
            top1_hit_count += 1
        if expected_evidence_type:
            evidence_eval_total += 1
            if evidence_hit:
                evidence_hit_count += 1
        if expected_source_modality:
            modality_eval_total += 1
            if modality_hit:
                modality_hit_count += 1
        if expected_modalities_present:
            modality_presence_eval_total += 1
            if modality_presence_hit:
                modality_presence_hit_count += 1

        stats = category_stats.setdefault(
            category,
            {
                "total": 0.0,
                "source_hit": 0.0,
                "top1_hit": 0.0,
                "evidence_hit": 0.0,
                "evidence_total": 0.0,
                "modality_hit": 0.0,
                "modality_total": 0.0,
                "modality_presence_hit": 0.0,
                "modality_presence_total": 0.0,
            },
        )
        stats["total"] += 1
        if expected_source and hit:
            stats["source_hit"] += 1
        if expected_source and top1_hit:
            stats["top1_hit"] += 1
        if expected_evidence_type:
            stats["evidence_total"] += 1
            if evidence_hit:
                stats["evidence_hit"] += 1
        if expected_source_modality:
            stats["modality_total"] += 1
            if modality_hit:
                stats["modality_hit"] += 1
        if expected_modalities_present:
            stats["modality_presence_total"] += 1
            if modality_presence_hit:
                stats["modality_presence_hit"] += 1

        detail = {
            "case_id": case_id,
            "category": category,
            "knowledge_base_name": knowledge_base_name,
            "query": query,
            "expected_source": expected_source,
            "expected_evidence_type": expected_evidence_type,
            "expected_source_modality": expected_source_modality,
            "expected_modalities_present": expected_modalities_present,
            "ranked_sources": ranked_sources,
            "returned_source_modalities": returned_modalities,
            "returned_evidence_types": [item.evidence_type for item in references],
            "top1_source": top1_source,
            "top1_source_modality": top1_source_modality,
            "top1_evidence_type": top1_evidence_type,
            "hit": hit,
            "rank": rank,
            "top1_hit": top1_hit,
            "evidence_hit": evidence_hit,
            "modality_hit": modality_hit,
            "modality_presence_hit": modality_presence_hit,
        }
        details.append(detail)
        if show_cases:
            print(json.dumps(detail, ensure_ascii=False))

    evaluated_total = max(total, 1)
    return {
        "case_file_total": len(cases),
        "evaluated_total": total,
        "skipped": skipped,
        "hit_at_k": hit_count / evaluated_total,
        "mrr": reciprocal_rank_sum / evaluated_total,
        "top1_source_accuracy": top1_hit_count / evaluated_total,
        "top1_evidence_type_accuracy": (
            evidence_hit_count / max(evidence_eval_total, 1)
        ),
        "top1_source_modality_accuracy": (
            modality_hit_count / max(modality_eval_total, 1)
        ),
        "modality_presence_accuracy": (
            modality_presence_hit_count / max(modality_presence_eval_total, 1)
        ),
        "category_breakdown": build_category_breakdown(category_stats),
        "details": details if show_cases else [],
    }


def find_rank(ranked_sources: list[str], expected_source: str) -> int | None:
    if not expected_source:
        return None
    for index, source in enumerate(ranked_sources, start=1):
        if source == expected_source:
            return index
    return None


def normalize_expected_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def build_category_breakdown(
    category_stats: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    breakdown: dict[str, dict[str, float]] = {}
    for category, stats in category_stats.items():
        total = max(stats["total"], 1.0)
        breakdown[category] = {
            "total": stats["total"],
            "hit_at_k": stats["source_hit"] / total,
            "top1_source_accuracy": stats["top1_hit"] / total,
            "top1_evidence_type_accuracy": stats["evidence_hit"] / max(
                stats["evidence_total"], 1.0
            ),
            "top1_source_modality_accuracy": stats["modality_hit"] / max(
                stats["modality_total"], 1.0
            ),
            "modality_presence_accuracy": stats["modality_presence_hit"] / max(
                stats["modality_presence_total"], 1.0
            ),
        }
    return breakdown


if __name__ == "__main__":
    raise SystemExit(main())
