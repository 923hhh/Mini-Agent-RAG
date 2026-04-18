from __future__ import annotations

import argparse
import json
import math
import sys
import unicodedata
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retrievers.local_kb import search_local_knowledge_base
from app.schemas.chat import ChatMessage, RetrievedReference
from app.services.settings import load_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality on local JSONL cases.")
    parser.add_argument(
        "--case-file",
        default=str(PROJECT_ROOT / "data" / "eval" / "rag_eval.jsonl"),
        help="Path to the evaluation JSONL file.",
    )
    parser.add_argument(
        "--knowledge-base-name",
        default="",
        help="Override knowledge_base_name for all cases. Useful for raw DomainRAG JSONL files.",
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
    cases = load_cases(
        Path(args.case_file),
        knowledge_base_name_override=args.knowledge_base_name,
    )
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


def load_cases(
    path: Path,
    *,
    knowledge_base_name_override: str = "",
) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"评测文件不存在: {path}")

    cases: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        raw_case = json.loads(stripped)
        cases.append(
            normalize_case(
                raw_case,
                source_path=path,
                line_number=line_number,
                knowledge_base_name_override=knowledge_base_name_override,
            )
        )
    return cases


def normalize_case(
    raw_case: dict[str, object],
    *,
    source_path: Path,
    line_number: int,
    knowledge_base_name_override: str,
) -> dict[str, object]:
    if is_domainrag_case(raw_case):
        return normalize_domainrag_case(
            raw_case,
            source_path=source_path,
            line_number=line_number,
            knowledge_base_name_override=knowledge_base_name_override,
        )
    return normalize_default_case(
        raw_case,
        source_path=source_path,
        line_number=line_number,
        knowledge_base_name_override=knowledge_base_name_override,
    )


def is_domainrag_case(raw_case: dict[str, object]) -> bool:
    return "question" in raw_case and (
        "positive_reference" in raw_case or "positive_references" in raw_case
    )


def normalize_default_case(
    raw_case: dict[str, object],
    *,
    source_path: Path,
    line_number: int,
    knowledge_base_name_override: str,
) -> dict[str, object]:
    case_id = str(raw_case.get("case_id", "")).strip() or f"{source_path.stem}-{line_number}"
    category = str(raw_case.get("category", "")).strip() or source_path.parent.name or source_path.stem
    knowledge_base_name = knowledge_base_name_override.strip() or str(
        raw_case.get("knowledge_base_name", "")
    ).strip()
    query = str(raw_case.get("query", "")).strip()

    expected_references = coerce_expected_reference_list(raw_case.get("expected_references"))
    if not expected_references:
        expected_references = build_expected_references_from_flat_fields(raw_case)

    history = coerce_history_messages(raw_case.get("history"))
    return {
        "case_id": case_id,
        "category": category,
        "knowledge_base_name": knowledge_base_name,
        "query": query,
        "history": history,
        "top_k": raw_case.get("top_k"),
        "expected_references": expected_references,
        "expected_evidence_type": str(raw_case.get("expected_evidence_type", "")).strip(),
        "expected_source_modality": str(raw_case.get("expected_source_modality", "")).strip(),
        "expected_modalities_present": normalize_expected_list(
            raw_case.get("expected_modalities_present")
        ),
    }


def normalize_domainrag_case(
    raw_case: dict[str, object],
    *,
    source_path: Path,
    line_number: int,
    knowledge_base_name_override: str,
) -> dict[str, object]:
    case_id = str(raw_case.get("case_id", "")).strip()
    if not case_id:
        raw_id = raw_case.get("id")
        case_id = str(raw_id).strip() if raw_id is not None else ""
    if not case_id:
        case_id = f"{source_path.stem}-{line_number}"

    category = str(raw_case.get("domainrag_task", "")).strip() or source_path.parent.name or source_path.stem
    knowledge_base_name = knowledge_base_name_override.strip() or str(
        raw_case.get("knowledge_base_name", "")
    ).strip()
    positive_reference_value = raw_case.get("positive_reference")
    if positive_reference_value is None:
        positive_reference_value = raw_case.get("positive_references")
    return {
        "case_id": case_id,
        "category": category,
        "knowledge_base_name": knowledge_base_name,
        "query": str(raw_case.get("question", "")).strip(),
        "history": build_domainrag_history(raw_case.get("history_qa")),
        "top_k": raw_case.get("top_k"),
        "expected_references": coerce_expected_reference_list(positive_reference_value),
        "expected_evidence_type": "",
        "expected_source_modality": "",
        "expected_modalities_present": [],
    }


def coerce_expected_reference_list(value: object) -> list[dict[str, str]]:
    if isinstance(value, list):
        return [normalize_expected_reference(item) for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return [normalize_expected_reference(value)]
    return []


def normalize_expected_reference(value: dict[str, object]) -> dict[str, str]:
    return {
        "source": str(value.get("source", "")).strip(),
        "source_path": str(value.get("source_path", "")).strip(),
        "title": str(value.get("title", "")).strip(),
        "url": str(value.get("url", "")).strip(),
        "reference_id": str(value.get("id", "")).strip(),
        "passage_id": str(value.get("psg_id", "")).strip(),
        "content": str(value.get("contents", value.get("content", ""))).strip(),
    }


def build_expected_references_from_flat_fields(raw_case: dict[str, object]) -> list[dict[str, str]]:
    references: list[dict[str, str]] = []

    expected_source = str(raw_case.get("expected_source", "")).strip()
    if expected_source:
        references.append(
            {
                "source": expected_source,
                "source_path": "",
                "title": "",
                "url": "",
                "reference_id": "",
                "passage_id": "",
                "content": "",
            }
        )

    for source in normalize_expected_list(raw_case.get("expected_sources")):
        references.append(
            {
                "source": source,
                "source_path": "",
                "title": "",
                "url": "",
                "reference_id": "",
                "passage_id": "",
                "content": "",
            }
        )

    expected_title = str(raw_case.get("expected_title", "")).strip()
    if expected_title:
        references.append(
            {
                "source": "",
                "source_path": "",
                "title": expected_title,
                "url": "",
                "reference_id": "",
                "passage_id": "",
                "content": "",
            }
        )

    for title in normalize_expected_list(raw_case.get("expected_titles")):
        references.append(
            {
                "source": "",
                "source_path": "",
                "title": title,
                "url": "",
                "reference_id": "",
                "passage_id": "",
                "content": "",
            }
        )

    expected_url = str(raw_case.get("expected_url", "")).strip()
    if expected_url:
        references.append(
            {
                "source": "",
                "source_path": "",
                "title": "",
                "url": expected_url,
                "reference_id": "",
                "passage_id": "",
                "content": "",
            }
        )

    for url in normalize_expected_list(raw_case.get("expected_urls")):
        references.append(
            {
                "source": "",
                "source_path": "",
                "title": "",
                "url": url,
                "reference_id": "",
                "passage_id": "",
                "content": "",
            }
        )

    deduplicated: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in references:
        fingerprint = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduplicated.append(item)
    return deduplicated


def coerce_history_messages(value: object) -> list[ChatMessage]:
    if not isinstance(value, list):
        return []
    messages: list[ChatMessage] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
        if role not in {"system", "user", "assistant"} or not content:
            continue
        messages.append(ChatMessage(role=role, content=content))
    return messages


def build_domainrag_history(value: object) -> list[ChatMessage]:
    if not isinstance(value, list):
        return []
    history: list[ChatMessage] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = extract_first_answer_text(item.get("answers"))
        if question:
            history.append(ChatMessage(role="user", content=question))
        if answer:
            history.append(ChatMessage(role="assistant", content=answer))
    return history


def extract_first_answer_text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        for item in value:
            text = extract_first_answer_text(item)
            if text:
                return text
    return ""


def evaluate_cases(
    settings,
    cases: list[dict[str, object]],
    *,
    default_top_k: int,
    show_cases: bool,
) -> dict[str, object]:
    total = 0
    skipped = 0
    ranking_eval_total = 0
    hit_count = 0
    reciprocal_rank_sum = 0.0
    ndcg_sum = 0.0
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
        expected_references = [
            item for item in case.get("expected_references", []) if isinstance(item, dict)
        ]
        expected_evidence_type = str(case.get("expected_evidence_type", "")).strip()
        expected_source_modality = str(case.get("expected_source_modality", "")).strip()
        expected_modalities_present = normalize_expected_list(
            case.get("expected_modalities_present")
        )
        history = case.get("history", [])
        top_k = int(case.get("top_k", default_top_k) or default_top_k)

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
                history=history if isinstance(history, list) else None,
            )
        except FileNotFoundError:
            skipped += 1
            continue

        ranking = evaluate_reference_ranking(
            references=references,
            expected_references=expected_references,
            cutoff=top_k,
        )
        ranked_sources = [item.source for item in references]
        top1_reference = references[0] if references else None
        top1_source = top1_reference.source if top1_reference is not None else ""
        top1_evidence_type = top1_reference.evidence_type if top1_reference is not None else ""
        top1_source_modality = top1_reference.source_modality if top1_reference is not None else ""
        evidence_hit = bool(expected_evidence_type) and top1_evidence_type == expected_evidence_type
        modality_hit = bool(expected_source_modality) and top1_source_modality == expected_source_modality
        returned_modalities = [
            (item.source_modality or "").strip() for item in references if (item.source_modality or "").strip()
        ]
        modality_presence_hit = (
            bool(expected_modalities_present)
            and all(expected in returned_modalities for expected in expected_modalities_present)
        )

        if expected_references:
            ranking_eval_total += 1
            if ranking["hit"]:
                hit_count += 1
                reciprocal_rank_sum += float(ranking["reciprocal_rank"])
            ndcg_sum += float(ranking["ndcg_at_k"])
            if ranking["top1_hit"]:
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
                "ranking_total": 0.0,
                "source_hit": 0.0,
                "top1_hit": 0.0,
                "mrr_sum": 0.0,
                "ndcg_sum": 0.0,
                "evidence_hit": 0.0,
                "evidence_total": 0.0,
                "modality_hit": 0.0,
                "modality_total": 0.0,
                "modality_presence_hit": 0.0,
                "modality_presence_total": 0.0,
            },
        )
        stats["total"] += 1
        if expected_references:
            stats["ranking_total"] += 1
            if ranking["hit"]:
                stats["source_hit"] += 1
                stats["mrr_sum"] += float(ranking["reciprocal_rank"])
            if ranking["top1_hit"]:
                stats["top1_hit"] += 1
            stats["ndcg_sum"] += float(ranking["ndcg_at_k"])
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
            "expected_reference_count": len(expected_references),
            "expected_reference_labels": [
                summarize_expected_reference(item) for item in expected_references[:8]
            ],
            "expected_evidence_type": expected_evidence_type,
            "expected_source_modality": expected_source_modality,
            "expected_modalities_present": expected_modalities_present,
            "ranked_sources": ranked_sources,
            "returned_titles": [item.title for item in references],
            "returned_source_paths": [item.source_path for item in references],
            "returned_source_modalities": returned_modalities,
            "returned_evidence_types": [item.evidence_type for item in references],
            "top1_source": top1_source,
            "top1_source_modality": top1_source_modality,
            "top1_evidence_type": top1_evidence_type,
            "hit": ranking["hit"],
            "rank": ranking["rank"],
            "reciprocal_rank": ranking["reciprocal_rank"],
            "ndcg_at_k": ranking["ndcg_at_k"],
            "matched_reference_count": ranking["matched_reference_count"],
            "matched_ranks": ranking["matched_ranks"],
            "top1_hit": ranking["top1_hit"],
            "evidence_hit": evidence_hit,
            "modality_hit": modality_hit,
            "modality_presence_hit": modality_presence_hit,
        }
        details.append(detail)
        if show_cases:
            print(json.dumps(detail, ensure_ascii=False))

    ranking_denominator = max(ranking_eval_total, 1)
    return {
        "case_file_total": len(cases),
        "evaluated_total": total,
        "ranking_evaluated_total": ranking_eval_total,
        "skipped": skipped,
        "hit_at_k": hit_count / ranking_denominator,
        "mrr": reciprocal_rank_sum / ranking_denominator,
        "ndcg_at_k": ndcg_sum / ranking_denominator,
        "top1_hit_accuracy": top1_hit_count / ranking_denominator,
        "top1_source_accuracy": top1_hit_count / ranking_denominator,
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


def evaluate_reference_ranking(
    *,
    references: list[RetrievedReference],
    expected_references: list[dict[str, str]],
    cutoff: int,
) -> dict[str, object]:
    if not expected_references:
        return {
            "hit": False,
            "rank": None,
            "reciprocal_rank": 0.0,
            "ndcg_at_k": 0.0,
            "top1_hit": False,
            "matched_reference_count": 0,
            "matched_ranks": [],
        }

    matched_expected_indices: set[int] = set()
    matched_ranks: list[int] = []
    for rank, reference in enumerate(references[:cutoff], start=1):
        match_index = find_matching_expected_index(
            reference=reference,
            expected_references=expected_references,
            matched_expected_indices=matched_expected_indices,
        )
        if match_index is None:
            continue
        matched_expected_indices.add(match_index)
        matched_ranks.append(rank)

    best_rank = matched_ranks[0] if matched_ranks else None
    dcg = sum(1.0 / math.log2(rank + 1) for rank in matched_ranks)
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(len(expected_references), cutoff) + 1)
    )
    ndcg_at_k = dcg / idcg if idcg > 0 else 0.0
    return {
        "hit": best_rank is not None,
        "rank": best_rank,
        "reciprocal_rank": 1.0 / best_rank if best_rank else 0.0,
        "ndcg_at_k": ndcg_at_k,
        "top1_hit": best_rank == 1,
        "matched_reference_count": len(matched_ranks),
        "matched_ranks": matched_ranks,
    }


def find_matching_expected_index(
    *,
    reference: RetrievedReference,
    expected_references: list[dict[str, str]],
    matched_expected_indices: set[int],
) -> int | None:
    for index, expected in enumerate(expected_references):
        if index in matched_expected_indices:
            continue
        if reference_matches_expected(reference, expected):
            return index
    return None


def reference_matches_expected(
    reference: RetrievedReference,
    expected: dict[str, str],
) -> bool:
    expected_content = normalize_identifier(expected.get("content", ""))
    reference_content = normalize_identifier(reference.content or reference.content_preview)
    if content_fingerprint_matches(expected_content, reference_content):
        return True

    expected_strong_keys = build_identifier_set(
        expected.get("source", ""),
        expected.get("source_path", ""),
        expected.get("url", ""),
    )
    reference_strong_keys = build_identifier_set(
        reference.source,
        reference.source_path,
    )
    if expected_strong_keys & reference_strong_keys:
        return True

    expected_title_keys = build_identifier_set(expected.get("title", ""))
    reference_title_keys = build_identifier_set(reference.title or "")
    return bool(expected_title_keys & reference_title_keys)


def build_identifier_set(*values: str) -> set[str]:
    normalized: set[str] = set()
    for value in values:
        text = normalize_identifier(value)
        if text:
            normalized.add(text)
    return normalized


def content_fingerprint_matches(expected_content: str, reference_content: str) -> bool:
    if not expected_content or not reference_content:
        return False

    expected_fingerprint = build_content_fingerprint(expected_content)
    reference_fingerprint = build_content_fingerprint(reference_content)
    if not expected_fingerprint or not reference_fingerprint:
        return False

    return (
        expected_fingerprint in reference_content
        or reference_fingerprint in expected_content
    )


def build_content_fingerprint(text: str, *, min_chars: int = 24, max_chars: int = 120) -> str:
    if len(text) < min_chars:
        return ""
    return text[:max_chars]


def summarize_expected_reference(expected: dict[str, str]) -> str:
    for key in ("source", "title", "url", "source_path"):
        value = str(expected.get(key, "")).strip()
        if value:
            return value
    content = str(expected.get("content", "")).strip()
    if content:
        return content[:80]
    return "unknown"


def normalize_identifier(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    collapsed = "".join(normalized.split())
    return collapsed.casefold()


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
        ranking_total = max(stats.get("ranking_total", 0.0), 1.0)
        breakdown[category] = {
            "total": stats["total"],
            "ranking_total": stats.get("ranking_total", 0.0),
            "hit_at_k": stats["source_hit"] / ranking_total,
            "mrr": stats.get("mrr_sum", 0.0) / ranking_total,
            "ndcg_at_k": stats.get("ndcg_sum", 0.0) / ranking_total,
            "top1_hit_accuracy": stats["top1_hit"] / ranking_total,
            "top1_source_accuracy": stats["top1_hit"] / ranking_total,
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
