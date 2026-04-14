from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from app.chains.text_splitter import split_documents
from app.loaders.documents import load_documents, load_file
from app.schemas.kb import DocumentChunkRecord
from app.services.embedding_service import build_embeddings, embed_texts_batched
from app.services.settings import AppSettings
from app.storage.vector_stores import VectorStoreEntry, build_vector_store_adapter
from app.utils.text import coerce_optional_text, extract_header_metadata


@dataclass(frozen=True)
class AssembledEmbeddings:
    chunks: list[Document]
    chunk_records: list[DocumentChunkRecord]
    entries: list[VectorStoreEntry]


class EmbeddingAssembler:
    def __init__(
        self,
        settings: AppSettings,
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        embedding_model: str | None = None,
        text_splitter_name: str | None = None,
        vector_store_type: str | None = None,
    ) -> None:
        self.settings = settings
        self.chunk_size = chunk_size or settings.kb.CHUNK_SIZE
        self.chunk_overlap = (
            settings.kb.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
        )
        self.embedding_model = embedding_model or settings.model.DEFAULT_EMBEDDING_MODEL
        self.text_splitter_name = text_splitter_name or settings.kb.TEXT_SPLITTER_NAME
        self.vector_store_type = vector_store_type or settings.kb.DEFAULT_VS_TYPE
        self.embeddings = build_embeddings(settings, model_name=self.embedding_model)

    def load_content_dir(
        self,
        content_dir: Path,
        supported_extensions: list[str] | None = None,
    ) -> tuple[list[Document], list[Path]]:
        return load_documents(
            content_dir,
            supported_extensions or self.settings.kb.SUPPORTED_EXTENSIONS,
            settings=self.settings,
        )

    def load_paths(
        self,
        *,
        content_dir: Path,
        paths: list[Path],
        workers: int | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, list[Document]]:
        if not paths:
            return {}

        max_workers = workers or self.settings.kb.DOC_PARSE_WORKERS
        relative_paths = {
            path.relative_to(content_dir).as_posix(): path
            for path in paths
        }
        if max_workers <= 1 or len(relative_paths) <= 1:
            results: dict[str, list[Document]] = {}
            total = len(relative_paths)
            for index, (relative_path, path) in enumerate(relative_paths.items(), start=1):
                results[relative_path] = load_file(path, content_dir, settings=self.settings)
                if progress_callback is not None:
                    progress_callback(index, total, relative_path)
            return results

        results: dict[str, list[Document]] = {}
        total = len(relative_paths)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(load_file, path, content_dir, self.settings): relative_path
                for relative_path, path in relative_paths.items()
            }
            completed = 0
            for future in as_completed(future_map):
                relative_path = future_map[future]
                results[relative_path] = future.result()
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, total, relative_path)
        return results

    def split_loaded_documents(
        self,
        documents: list[Document],
    ) -> tuple[list[Document], list[DocumentChunkRecord]]:
        chunks = split_documents(
            documents,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            splitter_name=self.text_splitter_name,
        )
        chunk_records = attach_chunk_metadata(chunks)
        return chunks, chunk_records

    def assemble_documents(
        self,
        documents: list[Document],
    ) -> AssembledEmbeddings:
        chunks, chunk_records = self.split_loaded_documents(documents)
        entries = self.embed_chunks(chunks)
        return AssembledEmbeddings(
            chunks=chunks,
            chunk_records=chunk_records,
            entries=entries,
        )

    def embed_chunks(self, chunks: list[Document]) -> list[VectorStoreEntry]:
        texts = [chunk.page_content for chunk in chunks]
        vectors = embed_texts_batched(
            self.embeddings,
            texts,
            self.settings.kb.EMBEDDING_BATCH_SIZE,
        )
        return [
            VectorStoreEntry(
                chunk_id=str(chunk.metadata.get("chunk_id", "")),
                page_content=chunk.page_content,
                metadata=dict(chunk.metadata),
                embedding=vector,
            )
            for chunk, vector in zip(chunks, vectors, strict=False)
        ]

    def persist_entries(
        self,
        *,
        vector_store_dir: Path,
        knowledge_name: str,
        entries: list[VectorStoreEntry],
        mode: str = "full",
    ) -> None:
        adapter = build_vector_store_adapter(
            self.settings,
            vector_store_dir,
            self.embeddings,
            collection_name=knowledge_name,
            vector_store_type=self.vector_store_type,
        )
        if mode == "append":
            adapter.append(entries)
            return
        adapter.build(entries)


def attach_chunk_metadata(chunks: list[Document]) -> list[DocumentChunkRecord]:
    counters: dict[str, int] = defaultdict(int)
    records: list[DocumentChunkRecord] = []
    for chunk in chunks:
        doc_id = (
            chunk.metadata.get("doc_id")
            or chunk.metadata.get("relative_path")
            or chunk.metadata.get("source")
        )
        chunk_index = counters[str(doc_id)]
        counters[str(doc_id)] += 1
        chunk_id = f"{doc_id}::chunk-{chunk_index:04d}"
        chunk.metadata["chunk_id"] = chunk_id
        chunk.metadata["chunk_index"] = chunk_index
        headers = extract_header_metadata(chunk.metadata)
        page = chunk.metadata.get("page")
        page_end = chunk.metadata.get("page_end")
        section_index = chunk.metadata.get("section_index")
        records.append(
            DocumentChunkRecord(
                chunk_id=chunk_id,
                doc_id=str(doc_id),
                source=str(chunk.metadata.get("source", "")),
                source_path=str(chunk.metadata.get("source_path", "")),
                extension=str(chunk.metadata.get("extension", "")),
                chunk_index=chunk_index,
                page=int(page) if page is not None else None,
                page_end=int(page_end) if page_end is not None else None,
                title=chunk.metadata.get("title"),
                section_title=chunk.metadata.get("section_title"),
                section_path=chunk.metadata.get("section_path"),
                section_index=int(section_index) if section_index is not None else None,
                content_type=coerce_optional_text(chunk.metadata.get("content_type")),
                source_modality=coerce_optional_text(chunk.metadata.get("source_modality")),
                original_file_type=coerce_optional_text(chunk.metadata.get("original_file_type")),
                ocr_text=coerce_optional_text(chunk.metadata.get("ocr_text")),
                ocr_language=coerce_optional_text(chunk.metadata.get("ocr_language")),
                image_caption=coerce_optional_text(chunk.metadata.get("image_caption")),
                evidence_summary=coerce_optional_text(chunk.metadata.get("evidence_summary")),
                headers=headers,
                content_length=len(chunk.page_content),
                content_preview=chunk.page_content[:120],
            )
        )
    return records
