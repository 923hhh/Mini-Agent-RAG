"""候选融合与召回后初步打分服务。"""

from __future__ import annotations

from datetime import date

import numpy as np
from langchain_core.documents import Document

from app.services.core.settings import AppSettings
from app.services.retrieval.candidate_common_service import (
    RetrievalCandidate,
    get_chunk_id,
    get_source_modality,
)
from app.services.retrieval.query_rewrite_service import (
    build_comparative_entity_profile,
    build_comparative_focus_profile,
    comparative_entity_matches_text,
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


def _compute_normalized_entropy(scores: list[float]) -> float:
    """归一化 Shannon 熵，0=完全确定，1=完全均匀。"""
    if not scores or len(scores) < 2:
        return 1.0
    arr = np.array(scores, dtype=np.float64)
    arr = arr - arr.min() + 1e-9
    probs = arr / arr.sum()
    entropy = -float(np.sum(probs * np.log2(probs)))
    max_entropy = np.log2(len(probs))
    return entropy / max_entropy if max_entropy > 0 else 1.0


def _adaptive_fusion_weights(
    candidates: list[RetrievalCandidate],
    base_dense: float,
    base_lexical: float,
) -> tuple[float, float]:
    """根据 dense/lexical score 分布的熵动态调整融合权重。"""
    dense_scores = [c.dense_relevance for c in candidates if c.dense_relevance > 0]
    lexical_scores = [c.lexical_score for c in candidates if c.lexical_score > 0]
    if not dense_scores or not lexical_scores:
        return base_dense, base_lexical
    h_dense = _compute_normalized_entropy(dense_scores)
    h_lexical = _compute_normalized_entropy(lexical_scores)
    conf_dense = 1.0 - h_dense
    conf_lexical = 1.0 - h_lexical
    total = conf_dense + conf_lexical + 1e-9
    w_dense = base_dense * (1.0 + 0.4 * (conf_dense / total - 0.5))
    w_lexical = base_lexical * (1.0 + 0.4 * (conf_lexical / total - 0.5))
    return w_dense, w_lexical


def apply_candidate_fusion_scores(
    *,
    settings: AppSettings,
    candidates: list[RetrievalCandidate],
    query_bundle: list[str],
    query_profile: QueryModalityProfile,
) -> list[RetrievalCandidate]:
    if not candidates:
        return []

    primary_query = next((item.strip() for item in query_bundle if item.strip()), "")
    comparative_profile = build_comparative_entity_profile(primary_query)
    comparative_focus_profile = build_comparative_focus_profile(primary_query)
    temporal_profile = infer_temporal_query_profile(query_bundle)
    max_dense = max((item.dense_relevance for item in candidates), default=1.0) or 1.0
    max_lexical = max((item.lexical_score for item in candidates), default=1.0) or 1.0
    temporal_adjustments = build_temporal_candidate_adjustments(candidates, temporal_profile)

    w_dense, w_lexical = _adaptive_fusion_weights(
        candidates,
        base_dense=settings.kb.HYBRID_DENSE_SCORE_WEIGHT,
        base_lexical=settings.kb.HYBRID_LEXICAL_SCORE_WEIGHT,
    )

    for item in candidates:
        score = 0.0
        if item.dense_rank is not None:
            score += 1.0 / (settings.kb.HYBRID_RRF_K + item.dense_rank)
        if item.lexical_rank is not None:
            score += 1.0 / (settings.kb.HYBRID_RRF_K + item.lexical_rank)
        score += w_dense * (item.dense_relevance / max_dense)
        score += w_lexical * (item.lexical_score / max_lexical)
        score += modality_bonus_for_candidate(item.document, query_profile)
        score += temporal_adjustments.get(get_chunk_id(item.document), 0.0)
        score += comparative_lexical_protection_bonus(
            candidate=item,
            comparative_profile=comparative_profile,
            comparative_aspect_terms=comparative_focus_profile.aspect_terms,
        )
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
        text = build_search_text(candidate.document)
        score = 0.0

        if query_years:
            overlap = len(years & query_years) / max(1, len(query_years))
            if overlap > 0:
                score += 0.12 + 0.12 * overlap
            elif years:
                score -= 0.10

        if query_dates and anchor is not None:
            if any(abs(anchor - query_date) <= 3 for query_date in query_dates):
                score += 0.12

        if temporal_profile.prefers_recent and anchor is not None:
            if newest_anchor is not None and oldest_anchor is not None and newest_anchor > oldest_anchor:
                recency = (anchor - oldest_anchor) / max(1, newest_anchor - oldest_anchor)
                score += 0.15 * recency
            else:
                score += 0.06
        elif anchor is not None:
            score += 0.03
        elif years:
            score += 0.01
        else:
            score -= 0.04

        score += compute_temporal_role_alignment_adjustment(
            candidate.document,
            temporal_profile=temporal_profile,
            search_text=text,
            anchor=anchor,
        )

        adjustments[chunk_id] = score
    return adjustments


def apply_temporal_prefilter(
    candidates: list[RetrievalCandidate],
    temporal_profile: TemporalQueryProfile,
    *,
    min_keep_ratio: float = 0.5,
) -> list[RetrievalCandidate]:
    if not temporal_profile.is_temporal or not temporal_profile.explicit_years:
        return candidates

    query_years = set(temporal_profile.explicit_years)
    matched: list[RetrievalCandidate] = []
    unmatched: list[RetrievalCandidate] = []

    for candidate in candidates:
        text = build_search_text(candidate.document)
        years = set(extract_years_from_text(text))
        if years & query_years:
            matched.append(candidate)
        else:
            unmatched.append(candidate)

    min_keep = max(1, int(len(candidates) * min_keep_ratio))
    if len(matched) >= min_keep:
        return matched
    return matched + unmatched[: min_keep - len(matched)]


def compute_temporal_role_alignment_adjustment(
    document: Document,
    *,
    temporal_profile: TemporalQueryProfile,
    search_text: str,
    anchor: int | None,
) -> float:
    if not temporal_profile.is_current_role_query or not temporal_profile.asked_role_terms:
        return 0.0

    title = str(document.metadata.get("title", "") or "").strip().lower()
    section_title = str(document.metadata.get("section_title", "") or "").strip().lower()
    source = str(document.metadata.get("source", "") or "").strip().lower()
    combined = "\n".join(item for item in (title, section_title, source, search_text.lower()) if item)
    if not combined:
        return 0.0

    asked_roles = tuple(term.lower() for term in temporal_profile.asked_role_terms if term)
    conflicting_roles = resolve_conflicting_role_terms(asked_roles)
    asked_role_hit = any(term in combined for term in asked_roles)
    conflicting_role_hit = any(term in combined for term in conflicting_roles)
    has_temporal_signal = anchor is not None or bool(extract_years_from_text(combined))

    score = 0.0
    if asked_role_hit:
        score += 0.05
        if has_temporal_signal or temporal_profile.prefers_recent:
            score += 0.05
    if conflicting_role_hit and not asked_role_hit:
        score -= 0.05
        if has_temporal_signal or temporal_profile.prefers_recent:
            score -= 0.04
    return score


def resolve_conflicting_role_terms(asked_roles: tuple[str, ...]) -> tuple[str, ...]:
    role_conflicts = {
        "校长": ("书记",),
        "书记": ("校长",),
        "院长": ("主任", "负责人"),
        "主任": ("院长", "负责人"),
        "负责人": ("主任", "院长"),
    }
    conflicts: list[str] = []
    for role in asked_roles:
        conflicts.extend(role_conflicts.get(role, ()))
    unique_conflicts = []
    for role in conflicts:
        if role not in unique_conflicts and role not in asked_roles:
            unique_conflicts.append(role)
    return tuple(unique_conflicts)


def comparative_lexical_protection_bonus(
    *,
    candidate: RetrievalCandidate,
    comparative_profile,
    comparative_aspect_terms: tuple[str, ...],
) -> float:
    if not comparative_profile.is_multi_entity_comparative:
        return 0.0
    if candidate.lexical_rank is None or candidate.lexical_rank > 20:
        return 0.0

    title = str(candidate.document.metadata.get("title", "") or "").strip().lower()
    source = str(candidate.document.metadata.get("source", "") or "").strip().lower()
    section_title = str(candidate.document.metadata.get("section_title", "") or "").strip().lower()
    search_text = build_search_text(candidate.document).lower()
    combined = " ".join(item for item in (title, section_title, source, search_text) if item)
    if not combined:
        return 0.0

    entity_hits = sum(
        1
        for name in comparative_profile.entity_names
        if comparative_entity_matches_text(
            name,
            title=title,
            section_title=section_title,
            source=source,
            search_text=search_text,
        )
    )
    if entity_hits <= 0:
        return 0.0
    aspect_hits = sum(1 for term in comparative_aspect_terms if term and term.lower() in combined)

    title_hits = sum(
        1
        for name in comparative_profile.entity_names
        if name.lower().replace("专业", "") in title or name.lower() in title
    )
    lexical_rank_bonus = max(0.0, 0.08 - 0.003 * max(candidate.lexical_rank - 1, 0))
    entity_bonus = 0.03 * min(entity_hits, 2)
    title_bonus = 0.02 * min(title_hits, 2)
    lexical_only_bonus = 0.04 if candidate.dense_rank is None and candidate.lexical_rank <= 5 else 0.0
    aspect_bonus = 0.025 * min(aspect_hits, 2)
    lexical_entity_aspect_bonus = (
        0.03
        if candidate.dense_rank is None and candidate.lexical_rank <= 8 and entity_hits > 0 and aspect_hits > 0
        else 0.0
    )
    return min(
        0.24,
        lexical_rank_bonus
        + entity_bonus
        + title_bonus
        + lexical_only_bonus
        + aspect_bonus
        + lexical_entity_aspect_bonus,
    )


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
