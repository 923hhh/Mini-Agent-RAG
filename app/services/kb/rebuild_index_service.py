"""知识库重建中的索引产物写入服务。"""

from __future__ import annotations

import json
from pathlib import Path

from app.schemas.kb import DocumentChunkRecord
from app.services.kb.embedding_assembler import extract_header_metadata
from app.services.kb.rebuild_cache_service import (
    CachedChunkEntry,
    FileChunkCache,
    chunk_cache_embedding_path,
    flatten_chunk_entries,
)
from app.services.kb.rebuild_planning_service import BuildManifestEntry
from app.storage.bm25_index import (
    build_persisted_bm25_document,
    delete_bm25_index,
    write_bm25_index,
)
from app.utils.text import coerce_optional_text


def chunk_entry_to_record(entry: CachedChunkEntry) -> DocumentChunkRecord:
    page = entry.metadata.get("page")
    page_value = int(page) if page is not None else None
    page_end = entry.metadata.get("page_end")
    page_end_value = int(page_end) if page_end is not None else None
    section_index = entry.metadata.get("section_index")
    section_index_value = int(section_index) if section_index is not None else None
    return DocumentChunkRecord(
        chunk_id=entry.chunk_id,
        doc_id=str(entry.metadata.get("doc_id", "")),
        source=str(entry.metadata.get("source", "")),
        source_path=str(entry.metadata.get("source_path", "")),
        extension=str(entry.metadata.get("extension", "")),
        chunk_index=int(entry.metadata.get("chunk_index", 0)),
        page=page_value,
        page_end=page_end_value,
        title=entry.metadata.get("title"),
        section_title=entry.metadata.get("section_title"),
        section_path=entry.metadata.get("section_path"),
        section_index=section_index_value,
        content_type=coerce_optional_text(entry.metadata.get("content_type")),
        source_modality=coerce_optional_text(entry.metadata.get("source_modality")),
        original_file_type=coerce_optional_text(entry.metadata.get("original_file_type")),
        ocr_text=coerce_optional_text(entry.metadata.get("ocr_text")),
        ocr_language=coerce_optional_text(entry.metadata.get("ocr_language")),
        image_caption=coerce_optional_text(entry.metadata.get("image_caption")),
        evidence_summary=coerce_optional_text(entry.metadata.get("evidence_summary")),
        series_id=coerce_optional_text(entry.metadata.get("series_id")),
        start_time=coerce_optional_text(entry.metadata.get("start_time")),
        end_time=coerce_optional_text(entry.metadata.get("end_time")),
        ts_summary=coerce_optional_text(entry.metadata.get("ts_summary")),
        event_type=coerce_optional_text(entry.metadata.get("event_type")),
        location=coerce_optional_text(entry.metadata.get("location")),
        channel_names=[
            str(item).strip()
            for item in (entry.metadata.get("channel_names") or [])
            if str(item).strip()
        ],
        headers=extract_header_metadata(entry.metadata),
        content_length=len(entry.page_content),
        content_preview=entry.page_content[:120],
    )


def load_metadata_records(path: Path) -> list[DocumentChunkRecord]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [DocumentChunkRecord.model_validate(item) for item in payload]


def write_metadata_records(path: Path, records: list[DocumentChunkRecord]) -> None:
    path.write_text(
        json.dumps([record.model_dump() for record in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_bm25_index_for_caches(path: Path, caches: list[FileChunkCache]) -> None:
    write_bm25_index_for_chunk_entries(path, flatten_chunk_entries(caches))


def write_bm25_index_for_chunk_entries(path: Path, chunk_entries: list[CachedChunkEntry]) -> None:
    documents = [
        build_persisted_bm25_document(
            chunk_id=entry.chunk_id,
            page_content=entry.page_content,
            metadata=entry.metadata,
            headers=extract_header_metadata(entry.metadata),
        )
        for entry in chunk_entries
    ]
    if not documents:
        delete_bm25_index(path)
        return
    write_bm25_index(path, documents)


def cleanup_deleted_caches(chunk_cache_dir: Path, deleted_entries: list[BuildManifestEntry]) -> None:
    for entry in deleted_entries:
        cache_path = chunk_cache_dir / entry.cache_file
        if cache_path.exists():
            cache_path.unlink(missing_ok=True)
        embedding_path = chunk_cache_embedding_path(cache_path)
        if embedding_path.exists():
            embedding_path.unlink(missing_ok=True)
