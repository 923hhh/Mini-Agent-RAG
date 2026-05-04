"""RAG 上下文构建服务。"""

from __future__ import annotations

from app.schemas.chat import RetrievedReference
from app.services.retrieval.evidence_packing_service import (
    build_prompt_reference_fingerprint,
    format_reference_block,
    resolve_reference_content_limit,
    resolve_reference_context_group,
)
from app.services.retrieval.query_answer_policy_service import QueryAnswerPolicy


def build_context(
    query: str,
    references: list[RetrievedReference],
    *,
    policy: QueryAnswerPolicy,
) -> str:
    if not references:
        return ""

    prompt_references = deduplicate_references_for_prompt(
        references,
        prefer_content_detail=policy.is_multi_doc_comparative,
        max_items=resolve_prompt_reference_limit(query, references),
    )

    grouped_blocks = {
        "text": [],
        "timeseries": [],
        "ocr": [],
        "vision": [],
    }

    for index, ref in enumerate(prompt_references, start=1):
        context_group = resolve_reference_context_group(ref)
        grouped_blocks[context_group].append(
            format_reference_block(
                index,
                ref,
                snippet_limit=resolve_reference_content_limit(
                    context_group=context_group,
                    policy=policy,
                ),
            )
        )

    sections: list[str] = []
    section_order = (
        ("text", "文本证据"),
        ("timeseries", "时间序列证据"),
        ("ocr", "OCR 证据"),
        ("vision", "视觉描述证据"),
    )
    for key, title in section_order:
        blocks = grouped_blocks[key]
        if not blocks:
            continue
        sections.append(f"## {title}\n" + "\n\n".join(blocks))

    return "\n\n".join(sections)


def deduplicate_references_for_prompt(
    references: list[RetrievedReference],
    *,
    prefer_content_detail: bool = False,
    max_items: int = 5,
) -> list[RetrievedReference]:
    ranked_references = sort_references_for_prompt(references)
    selected: list[RetrievedReference] = []
    seen_fingerprints: set[str] = set()
    for ref in ranked_references:
        fingerprint = build_prompt_reference_fingerprint(
            ref,
            prefer_content_detail=prefer_content_detail,
        )
        if fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)
        selected.append(ref)
        if len(selected) >= max_items:
            break
    return selected or references[:max_items]


def resolve_prompt_reference_limit(
    query: str,
    references: list[RetrievedReference],
) -> int:
    normalized = str(query or "").strip().lower()
    if not normalized:
        return 5 if len(references) >= 5 else 3

    multi_part_markers = (
        "分别",
        "同时",
        "以及",
        "并且",
        "共同",
        "区别",
        "不同",
        "相同",
        "哪些",
        "哪几",
        "包括",
        "原因",
        "措施",
        "步骤",
    )
    if any(marker in normalized for marker in multi_part_markers):
        return 5

    answer_seeking_markers = (
        "多少",
        "几",
        "哪一年",
        "哪年",
        "何时",
        "什么时候",
        "什么时间",
        "哪个",
        "是什么",
    )
    if any(marker in normalized for marker in answer_seeking_markers):
        return 3

    return 5 if len(references) >= 5 else 3


def sort_references_for_prompt(
    references: list[RetrievedReference],
) -> list[RetrievedReference]:
    return sorted(
        references,
        key=lambda ref: (
            0 if ref.evidence_summary else 1,
            -float(ref.relevance_score),
            len(ref.content_preview or ref.content or ""),
        ),
    )
