"""候选融合与召回后初步打分服务。"""

from __future__ import annotations

from datetime import date

from langchain_core.documents import Document

from app.services.core.settings import AppSettings
from app.services.retrieval.candidate_common_service import (
    RetrievalCandidate,
    get_chunk_id,
    get_source_modality,
)
from app.services.retrieval.query_profile_service import (
    QueryModalityProfile,
    TemporalQueryProfile,
    extract_date_ordinals_from_text,
    extract_years_from_text,
    infer_temporal_query_profile,
)
from app.storage.bm25_index import build_search_text_from_parts
from app.utils.text import extract_document_headers


def apply_candidate_fusion_scores(
    *,
    settings: AppSettings,
    candidates: list[RetrievalCandidate],
    query_bundle: list[str],
    query_profile: QueryModalityProfile,
) -> list[RetrievalCandidate]:
    if not candidates:
        return []

    temporal_profile = infer_temporal_query_profile(query_bundle)
    max_dense = max((item.dense_relevance for item in candidates), default=1.0) or 1.0
    max_lexical = max((item.lexical_score for item in candidates), default=1.0) or 1.0
    temporal_adjustments = build_temporal_candidate_adjustments(candidates, temporal_profile)

    for item in candidates:
        score = 0.0
        if item.dense_rank is not None:
            score += 1.0 / (settings.kb.HYBRID_RRF_K + item.dense_rank)
        if item.lexical_rank is not None:
            score += 1.0 / (settings.kb.HYBRID_RRF_K + item.lexical_rank)
        score += settings.kb.HYBRID_DENSE_SCORE_WEIGHT * (item.dense_relevance / max_dense)
        score += settings.kb.HYBRID_LEXICAL_SCORE_WEIGHT * (item.lexical_score / max_lexical)
        score += modality_bonus_for_candidate(item.document, query_profile)
        score += temporal_adjustments.get(get_chunk_id(item.document), 0.0)
        item.fused_score = score
    return candidates


def merge_retrieval_candidate_lists(
    *,
    primary: list[RetrievalCandidate],
    secondary: list[RetrievalCandidate],
) -> list[RetrievalCandidate]:
    candidate_map: dict[str, RetrievalCandidate] = {}
    for candidate in [*primary, *secondary]:
        chunk_id = get_chunk_id(candidate.document)
        existing = candidate_map.get(chunk_id)
        if existing is None or candidate.fused_score > existing.fused_score:
            candidate_map[chunk_id] = candidate
    return sorted(candidate_map.values(), key=lambda item: item.fused_score, reverse=True)


def count_candidate_modalities(candidates: list[RetrievalCandidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in candidates:
        source_modality = get_source_modality(item.document)
        counts[source_modality] = counts.get(source_modality, 0) + 1
    return counts


def modality_bonus_for_candidate(
    document: Document,
    query_profile: QueryModalityProfile,
) -> float:
    return (
        query_profile.modality_bonus.get(get_source_modality(document), 0.0)
        + extension_bonus_for_candidate(document, query_profile)
        + path_bonus_for_candidate(document, query_profile)
    )


def extension_bonus_for_candidate(
    document: Document,
    query_profile: QueryModalityProfile,
) -> float:
    extension = str(document.metadata.get("extension", "")).strip().lower()
    return query_profile.extension_bonus.get(extension, 0.0)


def path_bonus_for_candidate(
    document: Document,
    query_profile: QueryModalityProfile,
) -> float:
    if not query_profile.path_hint_terms:
        return 0.0

    title = str(document.metadata.get("title", "")).lower()
    section_title = str(document.metadata.get("section_title", "")).lower()
    section_path = str(document.metadata.get("section_path", "")).lower()
    source = str(document.metadata.get("source", "")).lower()
    combined = " ".join(item for item in (title, section_title, section_path, source) if item)
    if not combined:
        return 0.0

    matched = sum(1 for term in query_profile.path_hint_terms if term and term in combined)
    if matched == 0:
        return 0.0
    return min(0.06, 0.02 * matched)


def build_temporal_candidate_adjustments(
    candidates: list[RetrievalCandidate],
    temporal_profile: TemporalQueryProfile,
) -> dict[str, float]:
    if not temporal_profile.is_temporal:
        return {}

    query_years = set(temporal_profile.explicit_years)
    query_dates = set(temporal_profile.explicit_dates)
    candidate_years: dict[str, set[int]] = {}
    candidate_anchors: dict[str, int | None] = {}
    anchor_values: list[int] = []

    for candidate in candidates:
        chunk_id = get_chunk_id(candidate.document)
        text = build_search_text(candidate.document)
        years = set(extract_years_from_text(text))
        anchor = extract_document_temporal_anchor(candidate.document, fallback_text=text)
        candidate_years[chunk_id] = years
        candidate_anchors[chunk_id] = anchor
        if anchor is not None:
            anchor_values.append(anchor)

    oldest_anchor = min(anchor_values) if anchor_values else None
    newest_anchor = max(anchor_values) if anchor_values else None

    adjustments: dict[str, float] = {}
    for candidate in candidates:
        chunk_id = get_chunk_id(candidate.document)
        years = candidate_years.get(chunk_id, set())
        anchor = candidate_anchors.get(chunk_id)
        score = 0.0

        if query_years:
            overlap = len(years & query_years) / max(1, len(query_years))
            if overlap > 0:
                score += 0.08 + 0.08 * overlap
            elif years:
                score -= 0.05

        if query_dates and anchor is not None:
            if any(abs(anchor - query_date) <= 3 for query_date in query_dates):
                score += 0.08

        if temporal_profile.prefers_recent and anchor is not None:
            if newest_anchor is not None and oldest_anchor is not None and newest_anchor > oldest_anchor:
                recency = (anchor - oldest_anchor) / max(1, newest_anchor - oldest_anchor)
                score += 0.10 * recency
            else:
                score += 0.04
        elif anchor is not None:
            score += 0.03
        elif years:
            score += 0.01
        else:
            score -= 0.02

        adjustments[chunk_id] = score
    return adjustments


def extract_document_temporal_anchor(
    document: Document,
    *,
    fallback_text: str = "",
) -> int | None:
    metadata_date = str(document.metadata.get("date", "")).strip()
    if metadata_date:
        ordinals = extract_date_ordinals_from_text(metadata_date)
        if ordinals:
            return max(ordinals)

    text_candidates = [
        str(document.metadata.get("title", "")).strip(),
        str(document.metadata.get("section_title", "")).strip(),
        str(document.metadata.get("source", "")).strip(),
        document.page_content[:1200],
        fallback_text[:1200],
    ]
    combined = "\n".join(item for item in text_candidates if item)
    if not combined:
        return None

    labeled_ordinals = extract_date_ordinals_from_text(combined, prefer_labeled=True)
    if labeled_ordinals:
        return max(labeled_ordinals)

    exact_ordinals = extract_date_ordinals_from_text(combined)
    if exact_ordinals:
        return max(exact_ordinals)

    years = extract_years_from_text(combined)
    if years:
        return date(max(years), 12, 31).toordinal()
    return None


def build_search_text(document: Document) -> str:
    return build_search_text_from_parts(
        page_content=document.page_content,
        metadata=document.metadata,
        headers=extract_document_headers(document),
    )
