"""检索诊断与 trace 记录服务。"""

from __future__ import annotations

from pathlib import Path

from app.schemas.chat import RetrievedReference
from app.services.core.observability import append_jsonl_trace
from app.services.core.settings import AppSettings
from app.services.retrieval.candidate_common_service import RetrievalCandidate, get_source_modality
from app.services.retrieval.candidate_rerank_service import has_text_ts_joint_candidate_coverage
from app.services.retrieval.query_profile_service import (
    JointQueryProfile,
    TemporalQueryProfile,
    infer_temporal_query_profile,
)
from app.services.retrieval.reference_overview import build_reference_overview
from app.storage.filters import MetadataFilters


def initialize_retrieval_diagnostics(
    *,
    query_bundle: list[str],
    dense_query_bundle: list[str],
    hyde_enabled: bool,
    bm25_index_available: bool,
    bm25_backend: str,
    timeseries_query_profile: TemporalQueryProfile | object,
    joint_query_profile: JointQueryProfile,
    bm25_load_error: str = "",
) -> dict[str, object]:
    diagnostics: dict[str, object] = {
        "query_bundle_count": len(query_bundle),
        "dense_query_bundle_count": len(dense_query_bundle),
        "hyde_enabled": hyde_enabled,
        "hyde_used": len(dense_query_bundle) > len(query_bundle),
        "bm25_index_available": bm25_index_available,
        "bm25_backend": bm25_backend,
        "joint_rerank_applied": False,
    }
    if bm25_load_error:
        diagnostics["bm25_load_error"] = bm25_load_error[:200]
    if len(dense_query_bundle) > len(query_bundle):
        diagnostics["hyde_preview"] = dense_query_bundle[-1][:160]

    diagnostics["timeseries_query_detected"] = bool(
        getattr(timeseries_query_profile, "is_timeseries_related", False)
    )
    diagnostics["timeseries_query_keywords"] = list(
        getattr(timeseries_query_profile, "matched_keywords", ())
    )
    diagnostics["timeseries_window_constraint"] = bool(
        getattr(timeseries_query_profile, "has_window_constraint", False)
    )
    diagnostics["joint_query_detected"] = joint_query_profile.is_joint_query
    diagnostics["joint_requires_timeseries"] = joint_query_profile.requires_timeseries
    diagnostics["joint_requires_text_background"] = joint_query_profile.requires_text_background
    diagnostics["joint_has_guard_constraint"] = joint_query_profile.has_guard_constraint
    diagnostics["joint_location_terms"] = list(joint_query_profile.location_terms)
    diagnostics["joint_channel_terms"] = list(joint_query_profile.channel_terms)
    diagnostics["joint_event_terms"] = list(joint_query_profile.event_terms)
    diagnostics["joint_domain_terms"] = list(joint_query_profile.domain_terms)
    return diagnostics


def enrich_topk_diagnostics(
    *,
    diagnostics: dict[str, object],
    final_candidates: list[RetrievalCandidate],
    top_k: int,
) -> None:
    top_joint_cutoff = min(max(4, top_k), len(final_candidates))
    top_candidates = final_candidates[:top_joint_cutoff]
    diagnostics["topk_modality_sequence"] = [
        get_source_modality(item.document) for item in top_candidates
    ]
    diagnostics["topk_has_text_ts_joint_coverage"] = has_text_ts_joint_candidate_coverage(
        top_candidates
    )
    diagnostics["temporal_match_score_topk"] = [
        round(float(item.temporal_match_score), 3) for item in top_candidates
    ]
    diagnostics["event_type_match_score_topk"] = [
        round(float(item.event_type_match_score), 3) for item in top_candidates
    ]
    diagnostics["location_match_score_topk"] = [
        round(float(item.location_match_score), 3) for item in top_candidates
    ]
    diagnostics["channel_match_score_topk"] = [
        round(float(item.channel_match_score), 3) for item in top_candidates
    ]
    diagnostics["joint_coverage_bonus_topk"] = [
        round(float(item.joint_coverage_bonus), 3) for item in top_candidates
    ]


def append_retrieval_trace(
    *,
    settings: AppSettings,
    vector_store_dir: Path,
    query: str,
    rewritten_query: str,
    query_bundle: list[str],
    top_k: int,
    metadata_filters: MetadataFilters | None,
    diagnostics: dict[str, object],
    references: list[RetrievedReference],
) -> None:
    reference_overview = build_reference_overview(references)
    temporal_constraint_detected = (
        bool(diagnostics.get("timeseries_window_constraint", False))
        or infer_temporal_query_profile(query_bundle).is_temporal
    )
    append_jsonl_trace(
        settings,
        "retrieval_trace",
        {
            "event_type": "retrieval",
            "knowledge_base_name": vector_store_dir.name,
            "query": query,
            "rewritten_query": rewritten_query,
            "query_bundle": query_bundle[:6],
            "query_bundle_count": diagnostics.get("query_bundle_count", len(query_bundle)),
            "dense_query_bundle_count": diagnostics.get("dense_query_bundle_count", len(query_bundle)),
            "top_k": top_k,
            "metadata_filter_count": len(metadata_filters.filters) if metadata_filters else 0,
            "hyde_enabled": diagnostics.get("hyde_enabled", False),
            "hyde_used": diagnostics.get("hyde_used", False),
            "hyde_preview": diagnostics.get("hyde_preview", ""),
            "bm25_index_available": diagnostics.get("bm25_index_available", False),
            "bm25_backend": diagnostics.get("bm25_backend", "dynamic_legacy"),
            "bm25_load_error": diagnostics.get("bm25_load_error", ""),
            "query_type": diagnostics.get("query_type", "unknown"),
            "preferred_modalities": diagnostics.get("preferred_modalities", []),
            "available_modalities": diagnostics.get("available_modalities", {}),
            "modality_grouped_dense_used": diagnostics.get("modality_grouped_dense_used", False),
            "candidate_count": diagnostics.get("candidate_count", 0),
            "candidate_modality_counts": diagnostics.get("candidate_modality_counts", {}),
            "rerank_model_selected": diagnostics.get("rerank_model_selected", ""),
            "rerank_model_route": diagnostics.get("rerank_model_route", ""),
            "timeseries_query_detected": diagnostics.get("timeseries_query_detected", False),
            "timeseries_query_keywords": diagnostics.get("timeseries_query_keywords", []),
            "timeseries_window_constraint": diagnostics.get("timeseries_window_constraint", False),
            "timeseries_extension_enabled": diagnostics.get("timeseries_extension_enabled", True),
            "timeseries_extension_bypassed": diagnostics.get("timeseries_extension_bypassed", False),
            "timeseries_extension_bypass_reason": diagnostics.get("timeseries_extension_bypass_reason", ""),
            "timeseries_branch_used": diagnostics.get("timeseries_branch_used", False),
            "joint_query_detected": diagnostics.get("joint_query_detected", False),
            "joint_rerank_applied": diagnostics.get("joint_rerank_applied", False),
            "text_branch_candidate_count": diagnostics.get("text_branch_candidate_count", 0),
            "timeseries_branch_candidate_count": diagnostics.get("timeseries_branch_candidate_count", 0),
            "final_reference_count": len(references),
            "ts_reference_count": reference_overview.timeseries_count,
            "has_ts_evidence": reference_overview.timeseries_count > 0,
            "has_text_ts_joint_coverage": reference_overview.has_text_ts_joint_coverage,
            "topk_modality_sequence": diagnostics.get("topk_modality_sequence", []),
            "topk_has_text_ts_joint_coverage": diagnostics.get("topk_has_text_ts_joint_coverage", False),
            "temporal_match_score_topk": diagnostics.get("temporal_match_score_topk", []),
            "event_type_match_score_topk": diagnostics.get("event_type_match_score_topk", []),
            "location_match_score_topk": diagnostics.get("location_match_score_topk", []),
            "channel_match_score_topk": diagnostics.get("channel_match_score_topk", []),
            "joint_coverage_bonus_topk": diagnostics.get("joint_coverage_bonus_topk", []),
            "temporal_constraint_detected": temporal_constraint_detected,
            "final_source_modalities": count_reference_attributes(
                references[: settings.kb.TRACE_LOG_MAX_REFERENCES],
                "source_modality",
            ),
            "final_evidence_types": count_reference_attributes(
                references[: settings.kb.TRACE_LOG_MAX_REFERENCES],
                "evidence_type",
            ),
            "final_sources": [ref.source for ref in references[: settings.kb.TRACE_LOG_MAX_REFERENCES]],
        },
    )


def count_reference_attributes(
    references: list[RetrievedReference],
    attribute: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ref in references:
        value = getattr(ref, attribute, None)
        normalized = str(value).strip() if value is not None else ""
        key = normalized or "missing"
        counts[key] = counts.get(key, 0) + 1
    return counts
