"""RAG 证据块组装与统计服务。"""

from __future__ import annotations

import re

from app.schemas.chat import RetrievedReference
from app.services.retrieval.query_answer_policy_service import QueryAnswerPolicy


def resolve_reference_context_group(ref: RetrievedReference) -> str:
    source_modality = (ref.source_modality or "").strip().lower()
    if source_modality == "timeseries":
        return "timeseries"
    if source_modality == "ocr":
        return "ocr"
    if source_modality in {"vision", "image"}:
        return "vision"
    if source_modality == "ocr+vision":
        return "vision"
    if ref.ocr_text and not ref.image_caption:
        return "ocr"
    if ref.image_caption:
        return "vision"
    return "text"


def resolve_reference_content_limit(
    *,
    context_group: str,
    policy: QueryAnswerPolicy,
) -> int:
    if context_group == "timeseries":
        return 260
    if context_group in {"ocr", "vision"}:
        return 220
    if policy.is_procedural:
        return 320
    if policy.is_symbol_explanation:
        return 360
    if policy.is_numeric_fact:
        return 220
    if policy.is_multi_doc_comparative:
        return 260
    if policy.should_direct_answer:
        return 120
    if policy.requirement_count > 1:
        return 220
    return 160


def format_reference_block(
    index: int,
    ref: RetrievedReference,
    *,
    snippet_limit: int = 160,
) -> str:
    metadata_parts = [f"source={ref.source}"]
    if ref.page is not None:
        metadata_parts.append(f"page={ref.page}")
    if ref.section_title:
        metadata_parts.append(f"section={ref.section_title}")
    if ref.source_modality:
        metadata_parts.append(f"modality={ref.source_modality}")
    if ref.content_type:
        metadata_parts.append(f"type={ref.content_type}")
    metadata_parts.append(f"relevance={ref.relevance_score:.3f}")

    evidence_lines: list[str] = [f"[{index}] " + " | ".join(metadata_parts)]
    if ref.evidence_summary:
        evidence_lines.append(f"evidence_summary: {ref.evidence_summary}")
    if (ref.source_modality or "").strip().lower() == "timeseries":
        if ref.series_id:
            evidence_lines.append(f"series_id: {ref.series_id}")
        if ref.start_time or ref.end_time:
            evidence_lines.append(
                f"time_range: {ref.start_time or '?'} -> {ref.end_time or '?'}"
            )
        if ref.location:
            evidence_lines.append(f"location: {ref.location}")
        if ref.event_type:
            evidence_lines.append(f"event_type: {ref.event_type}")
        if ref.channel_names:
            evidence_lines.append(f"channel_names: {', '.join(ref.channel_names)}")
        if ref.ts_summary:
            evidence_lines.append(
                f"ts_summary: {clip_prompt_snippet(ref.ts_summary, min(280, snippet_limit + 80))}"
            )
    if ref.ocr_text:
        evidence_lines.append(
            f"ocr_text: {clip_prompt_snippet(ref.ocr_text, min(240, snippet_limit))}"
        )
    if ref.image_caption:
        evidence_lines.append(
            f"image_caption: {clip_prompt_snippet(ref.image_caption, min(240, snippet_limit))}"
        )
    content_source = ref.content or ref.content_preview
    if ref.evidence_summary:
        content_text = clip_prompt_snippet(
            content_source,
            max(120, snippet_limit),
        )
    else:
        content_text = clip_prompt_snippet(
            content_source,
            snippet_limit,
        )
    evidence_lines.append(f"content:\n{content_text}")
    return "\n".join(evidence_lines)


def clip_prompt_snippet(text: str, limit: int) -> str:
    normalized = str(text or "").strip()
    if not normalized or len(normalized) <= limit:
        return normalized

    snippet = normalized[:limit]
    preferred_boundary = max(24, limit // 2)
    boundary_chars = "。！？；\n，,、"
    last_boundary = max(snippet.rfind(char) for char in boundary_chars)
    if last_boundary >= preferred_boundary:
        candidate = snippet[:last_boundary].rstrip(" ，,、；;:：")
        if candidate:
            return candidate
    return snippet.rstrip(" ，,、；;:：")


def count_reference_attribute(
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


def count_context_groups(
    references: list[RetrievedReference],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ref in references:
        key = resolve_reference_context_group(ref)
        counts[key] = counts.get(key, 0) + 1
    return counts


def build_prompt_reference_fingerprint(
    ref: RetrievedReference,
    *,
    prefer_content_detail: bool = False,
) -> str:
    if prefer_content_detail:
        base = ref.content_preview or ref.content or ref.evidence_summary
    else:
        base = ref.evidence_summary or ref.content_preview or ref.content
    normalized = re.sub(r"\s+", " ", str(base or "")).strip().lower()
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", "", normalized)
    if normalized:
        limit = 260 if prefer_content_detail else 180
        return normalized[:limit]
    fallback = ref.source_path or ref.chunk_id or ref.source
    return str(fallback)
