"""检索结果引用组装服务。"""

from __future__ import annotations

from collections import defaultdict

from langchain_core.documents import Document

from app.schemas.chat import RetrievedReference
from app.services.core.settings import AppSettings
from app.services.retrieval.candidate_common_service import RetrievalCandidate, get_source_modality
from app.utils.text import coerce_optional_text, extract_document_headers


def candidate_to_reference(
    *,
    settings: AppSettings,
    candidate: RetrievalCandidate,
    grouped_documents: dict[str, dict[int, Document]],
) -> RetrievedReference:
    expanded_content = build_expanded_content(
        settings=settings,
        document=candidate.document,
        grouped_documents=grouped_documents,
    )
    raw_score = (
        float(candidate.dense_distance)
        if candidate.dense_distance is not None
        else float(max(0.0, 1.0 - candidate.relevance_score))
    )

    return RetrievedReference(
        chunk_id=str(candidate.document.metadata.get("chunk_id", "")),
        source=str(candidate.document.metadata.get("source", "")),
        source_path=str(candidate.document.metadata.get("source_path", "")),
        extension=str(candidate.document.metadata.get("extension", "")),
        page=_coerce_page(candidate.document.metadata.get("page")),
        page_end=_coerce_page(candidate.document.metadata.get("page_end")),
        title=candidate.document.metadata.get("title"),
        section_title=candidate.document.metadata.get("section_title"),
        section_path=candidate.document.metadata.get("section_path"),
        section_index=_coerce_page(candidate.document.metadata.get("section_index")),
        content_type=coerce_optional_text(candidate.document.metadata.get("content_type")),
        source_modality=coerce_optional_text(candidate.document.metadata.get("source_modality")),
        evidence_type=resolve_reference_evidence_type(candidate.document),
        used_for_answer=True,
        original_file_type=coerce_optional_text(candidate.document.metadata.get("original_file_type")),
        ocr_text=coerce_optional_text(candidate.document.metadata.get("ocr_text")),
        image_caption=coerce_optional_text(candidate.document.metadata.get("image_caption")),
        evidence_summary=coerce_optional_text(candidate.document.metadata.get("evidence_summary")),
        series_id=coerce_optional_text(candidate.document.metadata.get("series_id")),
        start_time=coerce_optional_text(candidate.document.metadata.get("start_time")),
        end_time=coerce_optional_text(candidate.document.metadata.get("end_time")),
        ts_summary=coerce_optional_text(candidate.document.metadata.get("ts_summary")),
        event_type=coerce_optional_text(candidate.document.metadata.get("event_type")),
        location=coerce_optional_text(candidate.document.metadata.get("location")),
        channel_names=[
            str(item).strip()
            for item in (candidate.document.metadata.get("channel_names") or [])
            if str(item).strip()
        ],
        headers=extract_document_headers(candidate.document),
        content=expanded_content,
        content_preview=expanded_content[:200],
        raw_score=raw_score,
        relevance_score=float(candidate.relevance_score),
    )


def build_expanded_content(
    *,
    settings: AppSettings,
    document: Document,
    grouped_documents: dict[str, dict[int, Document]],
) -> str:
    if not settings.kb.ENABLE_SMALL_TO_BIG_CONTEXT:
        return document.page_content

    doc_id = get_document_doc_id(document)
    chunk_index = _coerce_page(document.metadata.get("chunk_index"))
    if not doc_id or chunk_index is None:
        return document.page_content

    available = grouped_documents.get(doc_id, {})
    if not available:
        return document.page_content

    pieces: list[str] = []
    seen_texts: set[str] = set()
    expand_chunks = settings.kb.SMALL_TO_BIG_EXPAND_CHUNKS
    for index in range(chunk_index - expand_chunks, chunk_index + expand_chunks + 1):
        item = available.get(index)
        if item is None:
            continue
        text = item.page_content.strip()
        if not text or text in seen_texts:
            continue
        pieces.append(text)
        seen_texts.add(text)

    return "\n".join(pieces) if pieces else document.page_content


def group_documents_by_doc_id(all_documents: dict[str, Document]) -> dict[str, dict[int, Document]]:
    grouped: dict[str, dict[int, Document]] = defaultdict(dict)
    for document in all_documents.values():
        doc_id = get_document_doc_id(document)
        chunk_index = _coerce_page(document.metadata.get("chunk_index"))
        if not doc_id or chunk_index is None:
            continue
        grouped[doc_id][chunk_index] = document
    return grouped


def get_document_doc_id(document: Document) -> str:
    value = document.metadata.get("doc_id") or document.metadata.get("relative_path") or document.metadata.get("source")
    return str(value or "")


def resolve_reference_evidence_type(document: Document) -> str:
    source_modality = get_source_modality(document)
    if source_modality == "ocr":
        return "ocr"
    if source_modality in {"vision", "image"}:
        return "vision"
    if source_modality == "ocr+vision":
        return "multimodal"
    return "text"


def _coerce_page(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
