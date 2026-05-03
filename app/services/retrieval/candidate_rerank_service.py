"""候选重排与最终选择服务。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re

from langchain_core.documents import Document

from app.services.core.settings import AppSettings
from app.services.retrieval.candidate_common_service import (
    RetrievalCandidate,
    get_chunk_id,
    get_source_modality,
)
from app.services.retrieval.candidate_fusion_service import (
    build_search_text,
    build_temporal_candidate_adjustments,
    modality_bonus_for_candidate,
)
from app.services.retrieval.query_profile_service import (
    DiversityQueryProfile,
    JointQueryProfile,
    QueryModalityProfile,
    extract_date_ordinals_from_text,
    extract_years_from_text,
    infer_query_modality_profile,
    infer_temporal_query_profile,
    resolve_required_modalities_for_query,
    resolve_rerank_cutoff,
    resolve_rerank_model_selection,
    should_focus_answer_window,
)
from app.services.retrieval.rerank_service import RerankTextInput, rerank_texts
from app.utils.text import extract_document_headers


SAMPLE_ID_PATTERN = re.compile(r"^[0-9a-f]{24}$", re.IGNORECASE)
SAMPLE_GROUP_AGGREGATION_LIMIT = 3
SAMPLE_GROUP_MIN_COUNT_FOR_RERANK = 2
SAMPLE_GROUP_MIN_COUNT_FOR_TRIM = 3
SAMPLE_GROUP_DOMINANCE_RATIO_FOR_TRIM = 0.15
ANSWER_NUMERIC_PATTERN = re.compile(
    r"(?:\d+|[零一二三四五六七八九十百千万两〇]+)\s*(?:人|项|家|篇|分|个|所|名|种|门|届|年|月|日|级)"
)
ANSWER_GRADE_PATTERN = re.compile(r"(?:A\+|A-|A|B\+|B-|B|C\+|C-|C|甲类|乙类|丙类)", re.IGNORECASE)
ANSWER_SEEKING_HINTS = (
    "多少",
    "几",
    "谁",
    "哪位",
    "何人",
    "哪一年",
    "哪年",
    "何时",
    "什么时候",
    "开始时间",
    "什么时间",
    "什么等级",
    "被评为",
    "哪三个",
    "哪几",
    "哪些",
    "哪个",
    "是什么",
    "是怎样",
    "理念",
    "愿景",
    "定位",
    "目标",
    "对象",
    "政策",
)
ANSWER_LABEL_TERMS = (
    "办学理念",
    "办学愿景",
    "发展定位",
    "人才培养目标",
    "降分政策",
    "招生人数",
    "招生计划",
    "开始时间",
    "招生对象",
    "定向招收对象",
    "办学目标",
    "等级",
)
LIST_QUERY_HINTS = ("哪些", "哪三个", "哪几", "包括哪些")
LANGUAGE_TERMS = ("汉语", "中文", "英语", "英文", "法语", "法文", "德语", "日语", "俄语")
ANSWER_ROLE_TERMS = ("校长", "院长", "主任", "书记", "负责人", "创办", "成立")
ANSWER_WINDOW_SPLIT_PATTERN = re.compile(r"(?:\r?\n+|[。！？!?；;])")
RERANK_FIELD_MAX_CHARS = 80
RERANK_SUMMARY_MAX_CHARS = 120
RERANK_BODY_MAX_CHARS = 280


@dataclass(frozen=True)
class SampleGroupStats:
    sample_id: str
    aggregate_score: float
    max_score: float
    candidate_count: int


def rerank_candidates(
    settings: AppSettings,
    query: str,
    candidates: list[RetrievalCandidate],
    query_bundle: list[str],
    query_profile: QueryModalityProfile,
    joint_query_profile: JointQueryProfile,
    top_k: int,
    diagnostics: dict[str, object] | None = None,
) -> list[RetrievalCandidate]:
    heuristic_ranked = heuristic_rerank_candidates(
        settings=settings,
        candidates=candidates,
        query_bundle=query_bundle,
        joint_query_profile=joint_query_profile,
        top_k=top_k,
    )
    if not heuristic_ranked:
        return []

    top_n = max(settings.kb.RERANK_CANDIDATES_TOP_N, top_k)
    query_term_set = set(build_match_terms(query_bundle))
    plain_queries = [item.lower().strip() for item in query_bundle if item.strip()]
    primary_query = plain_queries[0] if plain_queries else ""
    rerank_model_selection = resolve_rerank_model_selection(
        settings=settings,
        query_bundle=query_bundle,
        query_profile=query_profile,
    )
    if diagnostics is not None:
        diagnostics["rerank_model_selected"] = rerank_model_selection.model_name
        diagnostics["rerank_model_route"] = rerank_model_selection.route
        diagnostics["joint_rerank_applied"] = joint_query_profile.is_joint_query
    rerank_inputs = [
        RerankTextInput(
            candidate_id=get_chunk_id(candidate.document),
            text=build_candidate_rerank_text(candidate.document, primary_query, query_term_set),
        )
        for candidate in heuristic_ranked[:top_n]
    ]
    rerank_outcome = rerank_texts(
        settings=settings,
        query=query.strip(),
        items=rerank_inputs,
        top_n=top_n,
        model_name_override=rerank_model_selection.model_name,
    )
    if not rerank_outcome.applied:
        if settings.kb.RERANK_FALLBACK_TO_HEURISTIC:
            return heuristic_ranked
        return []

    max_heuristic = max((item.rerank_score for item in heuristic_ranked), default=1.0) or 1.0
    reranked: list[RetrievalCandidate] = []
    for candidate in heuristic_ranked:
        chunk_id = get_chunk_id(candidate.document)
        model_score = rerank_outcome.scores.get(chunk_id)
        if model_score is None:
            reranked.append(candidate)
            continue

        candidate.model_rerank_score = model_score
        heuristic_component = candidate.rerank_score / max_heuristic
        candidate.rerank_score = model_score + 0.03 * heuristic_component + 0.02 * candidate.fused_score
        if model_score < settings.kb.RERANK_SCORE_THRESHOLD:
            candidate.relevance_score = min(candidate.relevance_score, model_score)
        else:
            candidate.relevance_score = min(
                1.0,
                0.85 * model_score + 0.15 * candidate.relevance_score,
            )
        reranked.append(candidate)

    apply_same_sample_group_rerank_adjustments(reranked)
    apply_joint_query_rerank_adjustments(reranked, joint_query_profile=joint_query_profile)
    reranked.sort(key=lambda item: item.rerank_score, reverse=True)
    cutoff = resolve_rerank_cutoff(settings, query_profile, top_k)
    return ensure_modality_coverage(
        reranked_candidates=reranked,
        required_modalities=resolve_required_modalities_for_query(query_profile),
        cutoff=cutoff,
    )


def heuristic_rerank_candidates(
    settings: AppSettings,
    candidates: list[RetrievalCandidate],
    query_bundle: list[str],
    joint_query_profile: JointQueryProfile,
    top_k: int,
) -> list[RetrievalCandidate]:
    if not candidates:
        return []

    query_profile = infer_query_modality_profile(query_bundle)
    temporal_profile = infer_temporal_query_profile(query_bundle)
    query_terms = build_match_terms(query_bundle)
    max_fused = max((item.fused_score for item in candidates), default=1.0) or 1.0
    max_dense = max((item.dense_relevance for item in candidates), default=1.0) or 1.0
    max_lexical = max((item.lexical_score for item in candidates), default=1.0) or 1.0
    normalized_queries = [normalize_search_text(item) for item in query_bundle if item.strip()]
    plain_queries = [item.lower().strip() for item in query_bundle if item.strip()]
    primary_query = plain_queries[0] if plain_queries else ""
    temporal_adjustments = build_temporal_candidate_adjustments(candidates, temporal_profile)
    joint_adjustments = build_joint_candidate_adjustments(
        candidates,
        joint_query_profile=joint_query_profile,
        temporal_adjustments=temporal_adjustments,
    )

    query_term_set = set(query_terms)
    reranked: list[RetrievalCandidate] = []
    for candidate in candidates:
        search_text = build_search_text(candidate.document)
        normalized_text = normalize_search_text(search_text)
        doc_terms = set(build_match_terms([search_text]))
        page_text = str(candidate.document.page_content or "")
        overlap_ratio = (
            len(doc_terms & query_term_set) / len(query_term_set)
            if query_term_set
            else 0.0
        )
        body_overlap_ratio = compute_body_overlap_ratio(page_text, query_term_set)
        answer_window_text = build_answer_window_text(primary_query, page_text, query_term_set)
        sentence_hint = candidate.sentence_text.strip()
        preferred_answer_text = answer_window_text
        if sentence_hint:
            preferred_answer_text = (
                f"{sentence_hint}\n{answer_window_text}"
                if answer_window_text and answer_window_text != page_text
                else sentence_hint
            )
            candidate.document.metadata["sentence_query_hit"] = sentence_hint
        answer_window_overlap_ratio = (
            compute_body_overlap_ratio(preferred_answer_text, query_term_set)
            if preferred_answer_text and preferred_answer_text != page_text
            else 0.0
        )
        phrase_bonus = 1.0 if any(query_text and query_text in search_text.lower() for query_text in plain_queries) else 0.0
        normalized_bonus = (
            1.0 if any(query_text and query_text in normalized_text for query_text in normalized_queries) else 0.0
        )
        source_text = f"{candidate.document.metadata.get('title', '')} {candidate.document.metadata.get('source', '')}".lower()
        source_bonus = 0.4 if any(term in source_text for term in query_terms if len(term) >= 2) else 0.0
        answer_support_bonus = compute_answer_support_bonus(primary_query, preferred_answer_text)
        answer_focus_score = min(
            1.0,
            0.35 * body_overlap_ratio
            + 0.45 * answer_window_overlap_ratio
            + 1.80 * answer_support_bonus
            + (0.10 if sentence_hint else 0.0),
        )

        fused_component = candidate.fused_score / max_fused
        dense_component = candidate.dense_relevance / max_dense
        lexical_component = candidate.lexical_score / max_lexical
        modality_bonus = modality_bonus_for_candidate(candidate.document, query_profile)
        temporal_bonus = temporal_adjustments.get(get_chunk_id(candidate.document), 0.0)
        joint_feature = joint_adjustments.get(get_chunk_id(candidate.document), {})
        temporal_match_score = float(joint_feature.get("temporal_match_score", max(0.0, temporal_bonus)))
        event_type_match_score = float(joint_feature.get("event_type_match_score", 0.0))
        location_match_score = float(joint_feature.get("location_match_score", 0.0))
        channel_match_score = float(joint_feature.get("channel_match_score", 0.0))
        joint_coverage_bonus = float(joint_feature.get("joint_coverage_bonus", 0.0))
        same_series_or_same_event_group = float(joint_feature.get("same_series_or_same_event_group", 0.0))

        if settings.kb.ENABLE_HEURISTIC_RERANK:
            candidate.rerank_score = (
                0.35 * fused_component
                + 0.25 * dense_component
                + 0.20 * lexical_component
                + 0.10 * overlap_ratio
                + 0.06 * body_overlap_ratio
                + 0.05 * answer_window_overlap_ratio
                + 0.06 * phrase_bonus
                + 0.10 * normalized_bonus
                + 0.04 * source_bonus
                + answer_support_bonus
                + modality_bonus
                + temporal_bonus
                + 0.05 * temporal_match_score
                + 0.04 * event_type_match_score
                + 0.04 * location_match_score
                + 0.04 * channel_match_score
                + joint_coverage_bonus
                + 0.03 * same_series_or_same_event_group
            )
        else:
            candidate.rerank_score = (
                0.55 * fused_component
                + 0.30 * dense_component
                + 0.15 * lexical_component
                + 0.04 * body_overlap_ratio
                + 0.04 * answer_window_overlap_ratio
                + answer_support_bonus
                + modality_bonus
                + temporal_bonus
                + 0.03 * temporal_match_score
                + 0.02 * event_type_match_score
                + 0.02 * location_match_score
                + 0.02 * channel_match_score
                + joint_coverage_bonus
            )

        candidate.relevance_score = min(
            1.0,
            candidate.dense_relevance
            + 0.22 * lexical_component
            + 0.18 * overlap_ratio
            + 0.08 * body_overlap_ratio
            + 0.08 * answer_window_overlap_ratio
            + 0.10 * phrase_bonus
            + 0.15 * normalized_bonus
            + 0.05 * source_bonus
            + 0.50 * answer_support_bonus,
        )
        candidate.relevance_score = max(
            -0.25,
            min(1.0, candidate.relevance_score + temporal_bonus),
        )
        candidate.body_overlap_ratio = body_overlap_ratio
        candidate.answer_window_overlap_ratio = answer_window_overlap_ratio
        candidate.answer_support_bonus = answer_support_bonus
        candidate.answer_focus_score = answer_focus_score
        candidate.temporal_match_score = temporal_match_score
        candidate.event_type_match_score = event_type_match_score
        candidate.location_match_score = location_match_score
        candidate.channel_match_score = channel_match_score
        candidate.joint_coverage_bonus = joint_coverage_bonus
        candidate.same_series_or_same_event_group = same_series_or_same_event_group
        reranked.append(candidate)

    reranked.sort(key=lambda item: item.rerank_score, reverse=True)
    apply_same_sample_group_rerank_adjustments(reranked)
    apply_joint_query_rerank_adjustments(reranked, joint_query_profile=joint_query_profile)
    reranked.sort(key=lambda item: item.rerank_score, reverse=True)
    cutoff = resolve_rerank_cutoff(settings, query_profile, top_k)
    return ensure_modality_coverage(
        reranked_candidates=reranked,
        required_modalities=resolve_required_modalities_for_query(query_profile),
        cutoff=cutoff,
    )


def diversify_candidates(
    candidates: list[RetrievalCandidate],
    target_count: int,
    query_profile: QueryModalityProfile,
    joint_query_profile: JointQueryProfile,
    diversity_profile: DiversityQueryProfile,
) -> list[RetrievalCandidate]:
    if joint_query_profile.is_joint_query:
        joint_candidates = select_joint_query_candidates(
            candidates,
            target_count=target_count,
            joint_query_profile=joint_query_profile,
        )
        if joint_candidates:
            return joint_candidates

    dominant_group_candidates = select_dominant_sample_group_candidates(
        candidates,
        target_count=target_count,
    )
    if dominant_group_candidates:
        return dominant_group_candidates

    if diversity_profile.prefer_family_diversity:
        diversified_family_candidates = select_family_diverse_candidates(
            candidates,
            target_count=target_count,
            query_profile=query_profile,
        )
        if diversified_family_candidates:
            return diversified_family_candidates

    selected: list[RetrievalCandidate] = []
    reserve: list[RetrievalCandidate] = []
    seen_doc_ids: set[str] = set()

    required_modalities = resolve_required_modalities_for_query(query_profile)
    for required_modality in required_modalities:
        for item in candidates:
            doc_id = get_document_doc_id(item.document)
            source_modality = get_source_modality(item.document)
            if source_modality != required_modality:
                continue
            if doc_id and doc_id in seen_doc_ids:
                continue
            selected.append(item)
            if doc_id:
                seen_doc_ids.add(doc_id)
            break
        if len(selected) >= target_count:
            return selected[:target_count]

    for item in candidates:
        doc_id = get_document_doc_id(item.document)
        if doc_id and doc_id not in seen_doc_ids:
            selected.append(item)
            seen_doc_ids.add(doc_id)
        else:
            reserve.append(item)
        if len(selected) >= target_count:
            return selected[:target_count]

    for item in reserve:
        selected.append(item)
        if len(selected) >= target_count:
            break
    return selected[:target_count]


def select_family_diverse_candidates(
    candidates: list[RetrievalCandidate],
    *,
    target_count: int,
    query_profile: QueryModalityProfile,
) -> list[RetrievalCandidate]:
    selected: list[RetrievalCandidate] = []
    reserve: list[RetrievalCandidate] = []
    seen_doc_ids: set[str] = set()
    seen_family_ids: set[str] = set()

    required_modalities = resolve_required_modalities_for_query(query_profile)
    for required_modality in required_modalities:
        for item in candidates:
            doc_id = get_document_doc_id(item.document)
            family_id = get_document_family_id(item.document)
            source_modality = get_source_modality(item.document)
            if source_modality != required_modality:
                continue
            if doc_id and doc_id in seen_doc_ids:
                continue
            if family_id and family_id in seen_family_ids:
                continue
            selected.append(item)
            if doc_id:
                seen_doc_ids.add(doc_id)
            if family_id:
                seen_family_ids.add(family_id)
            break
        if len(selected) >= target_count:
            return selected[:target_count]

    for item in candidates:
        doc_id = get_document_doc_id(item.document)
        family_id = get_document_family_id(item.document)
        if doc_id and doc_id in seen_doc_ids:
            reserve.append(item)
            continue
        if family_id and family_id in seen_family_ids:
            reserve.append(item)
            continue
        selected.append(item)
        if doc_id:
            seen_doc_ids.add(doc_id)
        if family_id:
            seen_family_ids.add(family_id)
        if len(selected) >= target_count:
            return selected[:target_count]

    for item in candidates:
        if item in selected or item in reserve:
            continue
        reserve.append(item)

    for item in reserve:
        doc_id = get_document_doc_id(item.document)
        if doc_id and doc_id in seen_doc_ids:
            continue
        selected.append(item)
        if doc_id:
            seen_doc_ids.add(doc_id)
        if len(selected) >= target_count:
            break
    return selected[:target_count]


def compute_body_overlap_ratio(page_text: str, query_term_set: set[str]) -> float:
    if not query_term_set:
        return 0.0
    page_terms = set(build_match_terms([page_text]))
    if not page_terms:
        return 0.0
    return len(page_terms & query_term_set) / len(query_term_set)


def compute_answer_support_bonus(primary_query: str, page_text: str) -> float:
    query = str(primary_query or "").strip().lower()
    if not query or not any(hint in query for hint in ANSWER_SEEKING_HINTS):
        return 0.0

    text = str(page_text or "")
    lowered = text.lower()
    bonus = 0.0

    matched_label_term = next((term for term in ANSWER_LABEL_TERMS if term in query), "")
    if matched_label_term:
        if any(
            marker in lowered
            for marker in (
                f"{matched_label_term.lower()}：",
                f"{matched_label_term.lower()}:",
                f"{matched_label_term.lower()}是",
                f"{matched_label_term.lower()}为",
            )
        ):
            bonus += 0.06

    if any(hint in query for hint in ("多少", "几", "几项", "几家", "几篇", "多少人", "招收多少", "降多少")):
        if ANSWER_NUMERIC_PATTERN.search(text):
            bonus += 0.05

    if any(hint in query for hint in ("哪一年", "哪年", "何时", "什么时候", "开始时间", "什么时间")):
        if extract_years_from_text(text) or extract_date_ordinals_from_text(text):
            bonus += 0.05

    if "等级" in query and ANSWER_GRADE_PATTERN.search(text):
        bonus += 0.05
        if any(marker in lowered for marker in ("被评为", "评为", "获评")):
            bonus += 0.03

    if any(hint in query for hint in ("谁", "哪位", "何人")):
        if any(role in text for role in ANSWER_ROLE_TERMS):
            bonus += 0.05

    if "语言" in query:
        language_hits = sum(1 for term in LANGUAGE_TERMS if term in text)
        if language_hits >= 2:
            bonus += 0.06

    if any(hint in query for hint in LIST_QUERY_HINTS):
        if any(separator in text for separator in ("、", "；", ";", "以及", "分别")):
            bonus += 0.03

    return min(0.14, bonus)


def build_candidate_rerank_text(
    document: Document,
    primary_query: str,
    query_term_set: set[str],
) -> str:
    page_text = str(document.page_content or "")
    sentence_hint = str(document.metadata.get("sentence_query_hit", "") or "").strip()
    answer_window_text = build_answer_window_text(primary_query, page_text, query_term_set)
    body_text = select_rerank_body_text(
        primary_query=primary_query,
        page_text=page_text,
        query_term_set=query_term_set,
        sentence_hint=sentence_hint,
        answer_window_text=answer_window_text,
    )

    title = pick_rerank_title(document)
    doc_type = infer_rerank_doc_type(document)
    section_text = pick_rerank_section(document)
    date_text = sanitize_rerank_text(document.metadata.get("date", ""), max_chars=32)
    summary_text = pick_rerank_summary(document, page_text)
    sentence_text = sanitize_rerank_text(sentence_hint, max_chars=RERANK_SUMMARY_MAX_CHARS)

    lines: list[str] = []
    if title:
        lines.append(f"[标题] {title}")
    if doc_type:
        lines.append(f"[类型] {doc_type}")
    if section_text:
        lines.append(f"[章节] {section_text}")
    if date_text:
        lines.append(f"[日期] {date_text}")
    if summary_text:
        lines.append(f"[摘要] {summary_text}")
    if sentence_text and sentence_text != body_text:
        lines.append(f"[命中句] {sentence_text}")
    if body_text:
        lines.append(f"[正文] {body_text}")

    if lines:
        return "\n".join(lines)

    return sanitize_rerank_text(page_text, max_chars=RERANK_BODY_MAX_CHARS)


def pick_rerank_title(document: Document) -> str:
    title = sanitize_rerank_text(document.metadata.get("title", ""), max_chars=RERANK_FIELD_MAX_CHARS)
    if title:
        return title
    section_title = sanitize_rerank_text(document.metadata.get("section_title", ""), max_chars=RERANK_FIELD_MAX_CHARS)
    if section_title:
        return section_title
    headers = extract_document_headers(document)
    for value in headers.values():
        candidate = sanitize_rerank_text(value, max_chars=RERANK_FIELD_MAX_CHARS)
        if candidate:
            return candidate
    return ""


def pick_rerank_section(document: Document) -> str:
    section_path = sanitize_rerank_text(document.metadata.get("section_path", ""), max_chars=RERANK_SUMMARY_MAX_CHARS)
    section_title = sanitize_rerank_text(document.metadata.get("section_title", ""), max_chars=RERANK_SUMMARY_MAX_CHARS)
    title = sanitize_rerank_text(document.metadata.get("title", ""), max_chars=RERANK_SUMMARY_MAX_CHARS)
    for candidate in (section_path, section_title):
        if candidate and candidate != title:
            return candidate
    return ""


def pick_rerank_summary(document: Document, page_text: str) -> str:
    summary = sanitize_rerank_text(document.metadata.get("evidence_summary", ""), max_chars=RERANK_SUMMARY_MAX_CHARS)
    title = sanitize_rerank_text(document.metadata.get("title", ""), max_chars=RERANK_SUMMARY_MAX_CHARS)
    section_title = sanitize_rerank_text(document.metadata.get("section_title", ""), max_chars=RERANK_SUMMARY_MAX_CHARS)
    if summary and summary not in {title, section_title}:
        return summary
    if not page_text:
        return ""
    return sanitize_rerank_text(page_text, max_chars=RERANK_SUMMARY_MAX_CHARS)


def infer_rerank_doc_type(document: Document) -> str:
    content_type = sanitize_rerank_text(document.metadata.get("content_type", ""), max_chars=48).lower()
    source_modality = sanitize_rerank_text(document.metadata.get("source_modality", ""), max_chars=24).lower()

    content_type_map = {
        "document_text": "文档正文",
        "image_evidence": "图像证据",
        "image_region_evidence": "图像区域证据",
        "instruction_text_evidence": "操作说明文本",
        "instruction_parts_evidence": "零件清单",
        "instruction_figure_evidence": "图示说明",
        "instruction_arrow_evidence": "图示标注",
        "web_search_result": "网络结果",
    }
    modality_map = {
        "text": "文本证据",
        "ocr": "OCR 文本",
        "vision": "视觉描述",
        "image": "图像内容",
        "ocr+vision": "图文联合证据",
    }

    if content_type in content_type_map:
        return content_type_map[content_type]
    if source_modality in modality_map:
        return modality_map[source_modality]
    if content_type:
        return content_type
    if source_modality:
        return source_modality
    return "文本片段"


def select_rerank_body_text(
    *,
    primary_query: str,
    page_text: str,
    query_term_set: set[str],
    sentence_hint: str,
    answer_window_text: str,
) -> str:
    focused_answer_text = ""
    if answer_window_text and answer_window_text != page_text:
        focused_answer_text = answer_window_text

    focused_window_text = build_query_focused_window_text(
        primary_query=primary_query,
        page_text=page_text,
        query_term_set=query_term_set,
        max_chars=RERANK_BODY_MAX_CHARS,
    )

    body_text = focused_answer_text or focused_window_text or sanitize_rerank_text(
        page_text,
        max_chars=RERANK_BODY_MAX_CHARS,
    )
    sentence_text = sanitize_rerank_text(sentence_hint, max_chars=RERANK_SUMMARY_MAX_CHARS)
    if sentence_text and sentence_text not in body_text:
        combined = f"{sentence_text} {body_text}".strip()
        return sanitize_rerank_text(combined, max_chars=RERANK_BODY_MAX_CHARS)
    return body_text


def build_query_focused_window_text(
    *,
    primary_query: str,
    page_text: str,
    query_term_set: set[str],
    max_chars: int,
) -> str:
    text = str(page_text or "").strip()
    if not text:
        return ""

    segments = split_answer_segments(text)
    if not segments:
        return sanitize_rerank_text(text, max_chars=max_chars)

    scored_segments = [
        (index, segment, score_rerank_segment(primary_query, segment, query_term_set))
        for index, segment in enumerate(segments)
    ]
    best_index, _, best_score = max(scored_segments, key=lambda item: item[2], default=(-1, "", 0.0))
    if best_index < 0:
        return sanitize_rerank_text(text, max_chars=max_chars)
    if best_score <= 0:
        return sanitize_rerank_text(text, max_chars=max_chars)

    selected_indices = {best_index}
    current_length = len(segments[best_index])
    left = best_index - 1
    right = best_index + 1
    score_map = {index: score for index, _, score in scored_segments}

    while current_length < max_chars and (left >= 0 or right < len(segments)):
        candidates: list[tuple[float, int]] = []
        if left >= 0:
            candidates.append((score_map.get(left, 0.0), left))
        if right < len(segments):
            candidates.append((score_map.get(right, 0.0), right))
        if not candidates:
            break
        candidates.sort(key=lambda item: (item[0], -abs(item[1] - best_index)), reverse=True)
        neighbor_score, neighbor_index = candidates[0]
        if current_length >= 180 and neighbor_score <= 0:
            break
        neighbor_text = segments[neighbor_index]
        projected_length = current_length + len(neighbor_text) + 1
        if projected_length > max_chars and current_length >= 180:
            break
        selected_indices.add(neighbor_index)
        current_length = projected_length
        if neighbor_index == left:
            left -= 1
        elif neighbor_index == right:
            right += 1

    ordered_segments = [segments[index] for index in sorted(selected_indices)]
    combined = " ".join(item for item in ordered_segments if item).strip()
    return sanitize_rerank_text(combined or text, max_chars=max_chars)


def score_rerank_segment(primary_query: str, segment: str, query_term_set: set[str]) -> float:
    text = str(segment or "").strip()
    if not text:
        return 0.0
    overlap_ratio = compute_body_overlap_ratio(text, query_term_set)
    answer_support_bonus = compute_answer_support_bonus(primary_query, text)
    return overlap_ratio + 0.8 * answer_support_bonus


def sanitize_rerank_text(text: object, max_chars: int) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    compact = re.sub(r"\s+", " ", raw)
    compact = compact.strip(" -:：|")
    if len(compact) <= max_chars:
        return compact
    clipped = compact[: max(1, max_chars - 1)].rstrip()
    return f"{clipped}…"


def build_answer_window_text(
    primary_query: str,
    page_text: str,
    query_term_set: set[str],
    max_chars: int = 320,
) -> str:
    text = str(page_text or "").strip()
    if not text or not should_focus_answer_window(primary_query):
        return text

    segments = split_answer_segments(text)
    if not segments:
        return text

    scored_segments = [
        (index, segment, score_answer_segment(primary_query, segment, query_term_set))
        for index, segment in enumerate(segments)
    ]
    best_index, _, best_score = max(scored_segments, key=lambda item: item[2], default=(-1, "", 0.0))
    if best_index < 0 or best_score < 0.08:
        return text

    selected_indices = {best_index}
    current_length = len(segments[best_index])
    left = best_index - 1
    right = best_index + 1
    score_map = {index: score for index, _, score in scored_segments}

    while current_length < max_chars and (left >= 0 or right < len(segments)):
        candidates: list[tuple[float, int]] = []
        if left >= 0:
            candidates.append((score_map.get(left, 0.0), left))
        if right < len(segments):
            candidates.append((score_map.get(right, 0.0), right))
        if not candidates:
            break
        candidates.sort(key=lambda item: (item[0], -abs(item[1] - best_index)), reverse=True)
        neighbor_score, neighbor_index = candidates[0]
        neighbor_text = segments[neighbor_index]
        if current_length >= 160 and neighbor_score <= 0:
            break
        projected_length = current_length + len(neighbor_text) + 1
        if projected_length > max_chars and current_length >= 160:
            break
        selected_indices.add(neighbor_index)
        current_length = projected_length
        if neighbor_index == left:
            left -= 1
        elif neighbor_index == right:
            right += 1

    ordered_segments = [segments[index] for index in sorted(selected_indices)]
    return "\n".join(item for item in ordered_segments if item).strip() or text


def split_answer_segments(text: str) -> list[str]:
    parts = [part.strip() for part in ANSWER_WINDOW_SPLIT_PATTERN.split(str(text or ""))]
    return [part for part in parts if len(part) >= 2]


def score_answer_segment(primary_query: str, segment: str, query_term_set: set[str]) -> float:
    text = str(segment or "").strip()
    if not text:
        return 0.0
    overlap_ratio = compute_body_overlap_ratio(text, query_term_set)
    answer_support_bonus = compute_answer_support_bonus(primary_query, text)
    return overlap_ratio + answer_support_bonus


def build_joint_candidate_adjustments(
    candidates: list[RetrievalCandidate],
    *,
    joint_query_profile: JointQueryProfile,
    temporal_adjustments: dict[str, float],
) -> dict[str, dict[str, float]]:
    sample_groups = build_sample_group_stats(candidates)
    dominant_sample_ids = {item.sample_id for item in sample_groups[:2] if item.sample_id}
    adjustments: dict[str, dict[str, float]] = {}
    for candidate in candidates:
        document = candidate.document
        chunk_id = get_chunk_id(document)
        source_modality = get_source_modality(document)
        channel_names = [
            str(item).strip().lower()
            for item in (document.metadata.get("channel_names") or [])
            if str(item).strip()
        ]
        location_text = " ".join(
            str(document.metadata.get(key, "") or "").strip()
            for key in ("location", "title", "source")
        ).lower()
        event_text = " ".join(
            str(document.metadata.get(key, "") or "").strip()
            for key in ("event_type", "title", "source")
        ).lower()
        page_text = str(document.page_content or "").lower()
        domain_text = " ".join(
            str(document.metadata.get(key, "") or "").strip()
            for key in ("event_type", "title", "source", "location")
        ).lower()
        location_match_score = (
            1.0
            if joint_query_profile.location_terms
            and any(term.lower() in location_text or term.lower() in page_text for term in joint_query_profile.location_terms)
            else 0.0
        )
        channel_match_score = (
            len([term for term in joint_query_profile.channel_terms if term.lower() in channel_names or term.lower() in page_text])
            / max(1, len(joint_query_profile.channel_terms))
            if joint_query_profile.channel_terms
            else 0.0
        )
        event_type_match_score = (
            len([term for term in joint_query_profile.event_terms if term.lower() in event_text or term.lower() in page_text])
            / max(1, len(joint_query_profile.event_terms))
            if joint_query_profile.event_terms
            else 0.0
        )
        domain_match_score = (
            len([term for term in joint_query_profile.domain_terms if term.lower() in domain_text or term.lower() in page_text])
            / max(1, len(joint_query_profile.domain_terms))
            if joint_query_profile.domain_terms
            else 0.0
        )
        joint_coverage_bonus = 0.0
        if joint_query_profile.is_joint_query:
            if source_modality == "timeseries" and joint_query_profile.requires_timeseries:
                joint_coverage_bonus += 0.07
            if source_modality == "text" and joint_query_profile.requires_text_background:
                joint_coverage_bonus += 0.06
            if joint_query_profile.has_explicit_window:
                joint_coverage_bonus += 0.015
            if joint_query_profile.has_guard_constraint and source_modality == "timeseries":
                joint_coverage_bonus += 0.05
            if location_match_score > 0:
                joint_coverage_bonus += 0.04
            elif joint_query_profile.location_terms:
                joint_coverage_bonus -= 0.05 if source_modality == "timeseries" else 0.03
            if domain_match_score > 0:
                joint_coverage_bonus += 0.04
            elif joint_query_profile.domain_terms:
                joint_coverage_bonus -= 0.05 if source_modality == "timeseries" else 0.03
            if event_type_match_score > 0:
                joint_coverage_bonus += 0.02
            elif joint_query_profile.event_terms and source_modality == "timeseries":
                joint_coverage_bonus -= 0.03
            if channel_match_score > 0:
                joint_coverage_bonus += 0.02
            elif joint_query_profile.channel_terms and source_modality == "timeseries":
                joint_coverage_bonus -= 0.02
            if (
                joint_query_profile.has_guard_constraint
                and source_modality == "timeseries"
                and (location_match_score <= 0 or domain_match_score <= 0)
                and (joint_query_profile.location_terms or joint_query_profile.domain_terms)
            ):
                joint_coverage_bonus -= 0.04
        sample_id = get_document_sample_id(document)
        same_series_or_same_event_group = 1.0 if sample_id and sample_id in dominant_sample_ids else 0.0
        adjustments[chunk_id] = {
            "temporal_match_score": max(0.0, float(temporal_adjustments.get(chunk_id, 0.0))),
            "event_type_match_score": min(1.0, event_type_match_score),
            "location_match_score": min(1.0, location_match_score),
            "channel_match_score": min(1.0, channel_match_score),
            "joint_coverage_bonus": joint_coverage_bonus,
            "same_series_or_same_event_group": same_series_or_same_event_group,
        }
    return adjustments


def apply_joint_query_rerank_adjustments(
    candidates: list[RetrievalCandidate],
    *,
    joint_query_profile: JointQueryProfile,
) -> None:
    if not joint_query_profile.is_joint_query:
        return
    for candidate in candidates:
        source_modality = get_source_modality(candidate.document)
        if source_modality == "timeseries" and joint_query_profile.requires_timeseries:
            candidate.rerank_score += 0.015
        elif source_modality == "text" and joint_query_profile.requires_text_background:
            candidate.rerank_score += 0.015


def select_joint_query_candidates(
    candidates: list[RetrievalCandidate],
    *,
    target_count: int,
    joint_query_profile: JointQueryProfile,
) -> list[RetrievalCandidate]:
    if not joint_query_profile.is_joint_query:
        return []

    best_text = next((item for item in candidates if get_source_modality(item.document) == "text"), None)
    best_timeseries = next((item for item in candidates if get_source_modality(item.document) == "timeseries"), None)
    if best_text is None or best_timeseries is None:
        return []

    anchor_candidates = sorted(
        [best_text, best_timeseries],
        key=lambda item: item.rerank_score,
        reverse=True,
    )
    selected: list[RetrievalCandidate] = []
    seen_chunk_ids: set[str] = set()
    for item in anchor_candidates:
        chunk_id = get_chunk_id(item.document)
        if chunk_id in seen_chunk_ids:
            continue
        selected.append(item)
        seen_chunk_ids.add(chunk_id)
    for item in candidates:
        chunk_id = get_chunk_id(item.document)
        if chunk_id in seen_chunk_ids:
            continue
        selected.append(item)
        seen_chunk_ids.add(chunk_id)
        if len(selected) >= target_count:
            break
    return selected[:target_count]


def has_text_ts_joint_candidate_coverage(
    candidates: list[RetrievalCandidate],
) -> bool:
    has_text = any(get_source_modality(item.document) == "text" for item in candidates)
    has_timeseries = any(get_source_modality(item.document) == "timeseries" for item in candidates)
    return has_text and has_timeseries


def get_document_doc_id(document: Document) -> str:
    value = document.metadata.get("doc_id") or document.metadata.get("relative_path") or document.metadata.get("source")
    return str(value or "")


def get_document_family_id(document: Document) -> str:
    for key in ("reference_id", "url", "title", "source"):
        value = str(document.metadata.get(key, "") or "").strip()
        normalized = normalize_search_text(value)
        if normalized:
            return normalized
    return ""


def get_document_sample_id(document: Document) -> str:
    value = document.metadata.get("sample_id")
    if isinstance(value, str) and value.strip():
        return value.strip()

    for key in ("doc_id", "relative_path", "source_path"):
        candidate = infer_sample_id_from_text(document.metadata.get(key))
        if candidate:
            return candidate
    return ""


def infer_sample_id_from_text(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""

    parts = [part.strip() for part in re.split(r"[\\/]+", text) if part.strip()]
    for part in reversed(parts):
        if SAMPLE_ID_PATTERN.fullmatch(part):
            return part
    return ""


def ensure_modality_coverage(
    *,
    reranked_candidates: list[RetrievalCandidate],
    required_modalities: tuple[str, ...],
    cutoff: int,
) -> list[RetrievalCandidate]:
    if not required_modalities:
        return reranked_candidates[:cutoff]

    selected: list[RetrievalCandidate] = []
    seen_chunk_ids: set[str] = set()
    for required_modality in required_modalities:
        for candidate in reranked_candidates:
            chunk_id = get_chunk_id(candidate.document)
            if chunk_id in seen_chunk_ids:
                continue
            if get_source_modality(candidate.document) != required_modality:
                continue
            selected.append(candidate)
            seen_chunk_ids.add(chunk_id)
            break
        if len(selected) >= cutoff:
            return selected[:cutoff]

    for candidate in reranked_candidates:
        chunk_id = get_chunk_id(candidate.document)
        if chunk_id in seen_chunk_ids:
            continue
        selected.append(candidate)
        seen_chunk_ids.add(chunk_id)
        if len(selected) >= cutoff:
            break
    return selected[:cutoff]


def apply_same_sample_group_rerank_adjustments(
    candidates: list[RetrievalCandidate],
) -> None:
    group_stats = build_sample_group_stats(candidates)
    if len(group_stats) < 2:
        return

    dominant_group = group_stats[0]
    if dominant_group.candidate_count < SAMPLE_GROUP_MIN_COUNT_FOR_RERANK:
        return

    dominant_score = max(dominant_group.aggregate_score, 1e-6)
    second_score = group_stats[1].aggregate_score if len(group_stats) > 1 else 0.0
    dominance_ratio = max(0.0, (dominant_score - second_score) / dominant_score)
    if dominance_ratio <= 0:
        return

    group_lookup = {item.sample_id: item for item in group_stats}
    for candidate in candidates:
        sample_id = get_document_sample_id(candidate.document)
        if not sample_id:
            continue

        stats = group_lookup.get(sample_id)
        if stats is None:
            continue

        group_ratio = stats.aggregate_score / dominant_score
        count_bonus = 0.02 * min(max(stats.candidate_count - 1, 0), 2)
        if sample_id == dominant_group.sample_id:
            boost = 0.06 + 0.08 * group_ratio + count_bonus + 0.10 * dominance_ratio
            candidate.rerank_score += boost
            candidate.relevance_score = min(
                1.0,
                candidate.relevance_score + 0.04 + 0.06 * dominance_ratio,
            )
            continue

        penalty = 0.04 + 0.08 * (1.0 - group_ratio) + 0.12 * dominance_ratio
        candidate.rerank_score -= penalty
        candidate.relevance_score = max(-0.25, candidate.relevance_score - penalty)


def select_dominant_sample_group_candidates(
    candidates: list[RetrievalCandidate],
    *,
    target_count: int,
) -> list[RetrievalCandidate]:
    if target_count < SAMPLE_GROUP_MIN_COUNT_FOR_TRIM:
        return []

    group_stats = build_sample_group_stats(candidates)
    if len(group_stats) < 2:
        return []

    dominant_group = group_stats[0]
    if dominant_group.candidate_count < min(SAMPLE_GROUP_MIN_COUNT_FOR_TRIM, target_count):
        return []

    dominant_score = max(dominant_group.aggregate_score, 1e-6)
    second_score = group_stats[1].aggregate_score if len(group_stats) > 1 else 0.0
    dominance_ratio = max(0.0, (dominant_score - second_score) / dominant_score)
    if dominance_ratio < SAMPLE_GROUP_DOMINANCE_RATIO_FOR_TRIM:
        return []

    selected = [
        item
        for item in candidates
        if get_document_sample_id(item.document) == dominant_group.sample_id
    ]
    return selected[:target_count]


def build_sample_group_stats(
    candidates: list[RetrievalCandidate],
) -> list[SampleGroupStats]:
    grouped_scores: dict[str, list[float]] = defaultdict(list)
    for candidate in candidates:
        sample_id = get_document_sample_id(candidate.document)
        if not sample_id:
            continue
        grouped_scores[sample_id].append(max(float(candidate.rerank_score), 0.0))

    stats: list[SampleGroupStats] = []
    for sample_id, scores in grouped_scores.items():
        if not scores:
            continue
        sorted_scores = sorted(scores, reverse=True)
        stats.append(
            SampleGroupStats(
                sample_id=sample_id,
                aggregate_score=sum(sorted_scores[:SAMPLE_GROUP_AGGREGATION_LIMIT]),
                max_score=sorted_scores[0],
                candidate_count=len(sorted_scores),
            )
        )
    stats.sort(
        key=lambda item: (item.aggregate_score, item.candidate_count, item.max_score),
        reverse=True,
    )
    return stats


def build_match_terms(texts: list[str], deduplicate: bool = True) -> list[str]:
    from app.services.retrieval.candidate_retrieval_service import build_match_terms as _build_match_terms

    return _build_match_terms(texts, deduplicate=deduplicate)


def normalize_search_text(text: str) -> str:
    from app.services.retrieval.candidate_retrieval_service import normalize_search_text as _normalize_search_text

    return _normalize_search_text(text)
