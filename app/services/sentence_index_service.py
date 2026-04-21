from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from app.services.embedding_service import embed_texts_batched
from app.services.settings import AppSettings
from app.storage.bm25_index import (
    build_persisted_bm25_document,
    build_search_text_from_parts,
    delete_bm25_index,
    resolve_bm25_index_path,
    write_bm25_index,
)
from app.storage.vector_stores import VectorStoreEntry, build_vector_store_adapter, vector_store_index_exists
from app.utils.text import extract_header_metadata


SENTENCE_INDEX_DIRNAME = "sentence_index"
SENTENCE_INDEX_SUMMARY_FILENAME = "sentence_index_summary.json"
SENTENCE_SPLIT_PATTERN = re.compile(r"(?:\r?\n+|(?<=[。！？!?；;]))")
CLAUSE_SPLIT_PATTERN = re.compile(r"(?<=[，、,:：])")


@dataclass(frozen=True)
class SentenceIndexBuildSummary:
    sentence_count: int
    source_chunk_count: int
    indexed_chunk_count: int


def resolve_sentence_index_dir(vector_store_dir: Path) -> Path:
    return vector_store_dir / SENTENCE_INDEX_DIRNAME


def resolve_sentence_index_summary_path(vector_store_dir: Path) -> Path:
    return vector_store_dir / SENTENCE_INDEX_SUMMARY_FILENAME


def sentence_index_exists(vector_store_dir: Path, vector_store_type: str | None = None) -> bool:
    return vector_store_index_exists(resolve_sentence_index_dir(vector_store_dir), vector_store_type)


def rebuild_sentence_index(
    *,
    settings: AppSettings,
    vector_store_dir: Path,
    knowledge_name: str,
    chunk_entries: list[VectorStoreEntry],
    embeddings: object,
    vector_store_type: str,
) -> SentenceIndexBuildSummary:
    sentence_entries = build_sentence_index_entries(settings=settings, chunk_entries=chunk_entries, embeddings=embeddings)
    sentence_dir = resolve_sentence_index_dir(vector_store_dir)
    summary_path = resolve_sentence_index_summary_path(vector_store_dir)

    if not settings.kb.ENABLE_SENTENCE_INDEX or not sentence_entries:
        clear_sentence_index(vector_store_dir)
        return SentenceIndexBuildSummary(
            sentence_count=0,
            source_chunk_count=len(chunk_entries),
            indexed_chunk_count=0,
        )

    adapter = build_vector_store_adapter(
        settings,
        sentence_dir,
        embeddings,
        collection_name=f"{knowledge_name}-sentence-index",
        vector_store_type=vector_store_type,
    )
    adapter.build(sentence_entries)
    bm25_index_path = resolve_bm25_index_path(sentence_dir)
    if settings.kb.ENABLE_HYBRID_RETRIEVAL:
        write_bm25_index(
            bm25_index_path,
            [
                build_persisted_bm25_document(
                    chunk_id=entry.chunk_id,
                    page_content=str(entry.metadata.get("sentence_text", "") or entry.page_content),
                    metadata=entry.metadata,
                    headers=extract_header_metadata(entry.metadata),
                )
                for entry in sentence_entries
            ],
        )
    else:
        delete_bm25_index(bm25_index_path)

    indexed_chunk_ids = {
        str(entry.metadata.get("parent_chunk_id", "")).strip()
        for entry in sentence_entries
        if str(entry.metadata.get("parent_chunk_id", "")).strip()
    }
    summary = SentenceIndexBuildSummary(
        sentence_count=len(sentence_entries),
        source_chunk_count=len(chunk_entries),
        indexed_chunk_count=len(indexed_chunk_ids),
    )
    summary_path.write_text(
        json.dumps(
            {
                "sentence_count": summary.sentence_count,
                "source_chunk_count": summary.source_chunk_count,
                "indexed_chunk_count": summary.indexed_chunk_count,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def clear_sentence_index(vector_store_dir: Path) -> None:
    sentence_dir = resolve_sentence_index_dir(vector_store_dir)
    if sentence_dir.exists():
        shutil.rmtree(sentence_dir, ignore_errors=True)
    summary_path = resolve_sentence_index_summary_path(vector_store_dir)
    if summary_path.exists():
        summary_path.unlink(missing_ok=True)


def build_sentence_index_entries(
    *,
    settings: AppSettings,
    chunk_entries: list[VectorStoreEntry],
    embeddings: object,
) -> list[VectorStoreEntry]:
    sentence_entries: list[tuple[str, str, dict[str, object]]] = []
    max_sentences_per_chunk = settings.kb.SENTENCE_INDEX_MAX_SENTENCES_PER_CHUNK
    min_chars = settings.kb.SENTENCE_INDEX_MIN_CHARS
    max_chars = settings.kb.SENTENCE_INDEX_MAX_CHARS

    for chunk_entry in chunk_entries:
        source_modality = str(chunk_entry.metadata.get("source_modality", "") or "").strip().lower()
        if source_modality in {"vision", "image"}:
            continue

        parent_chunk_id = str(chunk_entry.metadata.get("chunk_id") or chunk_entry.chunk_id or "").strip()
        if not parent_chunk_id:
            continue

        sentence_units = split_text_into_sentence_units(
            str(chunk_entry.page_content or ""),
            min_chars=min_chars,
            max_chars=max_chars,
        )
        if not sentence_units:
            continue

        if max_sentences_per_chunk > 0:
            sentence_units = sentence_units[:max_sentences_per_chunk]

        for sentence_index, sentence_text in enumerate(sentence_units):
            metadata = dict(chunk_entry.metadata)
            metadata["parent_chunk_id"] = parent_chunk_id
            metadata["parent_doc_id"] = str(chunk_entry.metadata.get("doc_id") or "")
            metadata["sentence_text"] = sentence_text
            metadata["sentence_index"] = sentence_index
            metadata["sentence_total"] = len(sentence_units)
            metadata["chunk_id"] = f"{parent_chunk_id}::sentence-{sentence_index:03d}"
            search_text = build_search_text_from_parts(
                page_content=sentence_text,
                metadata=metadata,
                headers=extract_header_metadata(metadata),
            )
            sentence_entries.append((str(metadata["chunk_id"]), search_text, metadata))

    if not sentence_entries:
        return []

    vectors = embed_texts_batched(
        embeddings,
        [item[1] for item in sentence_entries],
        settings.kb.EMBEDDING_BATCH_SIZE,
    )
    return [
        VectorStoreEntry(
            chunk_id=sentence_id,
            page_content=search_text,
            metadata=dict(metadata),
            embedding=vector,
        )
        for (sentence_id, search_text, metadata), vector in zip(sentence_entries, vectors, strict=False)
    ]


def split_text_into_sentence_units(
    text: str,
    *,
    min_chars: int,
    max_chars: int,
) -> list[str]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return []

    raw_segments = [segment.strip() for segment in SENTENCE_SPLIT_PATTERN.split(cleaned) if segment.strip()]
    if not raw_segments:
        return []

    normalized_segments: list[str] = []
    for segment in raw_segments:
        if len(segment) <= max_chars:
            normalized_segments.append(segment)
            continue
        normalized_segments.extend(split_long_sentence(segment, max_chars=max_chars))

    merged: list[str] = []
    buffer = ""
    for segment in normalized_segments:
        candidate = f"{buffer}{segment}".strip() if buffer else segment
        if len(candidate) < min_chars:
            buffer = candidate
            continue
        if len(candidate) <= max_chars:
            merged.append(candidate)
            buffer = ""
            continue
        if buffer:
            merged.append(buffer.strip())
            buffer = ""
        merged.append(segment)
    if buffer:
        if merged and len(buffer) < min_chars:
            merged[-1] = f"{merged[-1]} {buffer}".strip()
        else:
            merged.append(buffer)

    return [item for item in merged if len(item.strip()) >= min_chars]


def split_long_sentence(text: str, *, max_chars: int) -> list[str]:
    clauses = [clause.strip() for clause in CLAUSE_SPLIT_PATTERN.split(text) if clause.strip()]
    if len(clauses) <= 1:
        return [text.strip()]

    chunks: list[str] = []
    current = ""
    for clause in clauses:
        candidate = f"{current}{clause}".strip() if current else clause
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current.strip())
        current = clause
    if current:
        chunks.append(current.strip())
    return chunks or [text.strip()]
