"""知识库上传编排辅助服务。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.schemas.kb import KnowledgeBaseUploadResponse, RebuildKnowledgeBaseResult
from app.services.kb.embedding_assembler import EmbeddingAssembler
from app.services.kb.sentence_index_service import rebuild_sentence_index
from app.services.core.settings import AppSettings
from app.services.runtime.temp_kb_service import create_temp_manifest, write_temp_manifest
from app.storage.bm25_index import (
    build_persisted_bm25_document,
    delete_bm25_index,
    resolve_bm25_index_path,
    write_bm25_index,
)
from app.utils.text import extract_header_metadata


@dataclass(frozen=True)
class TempUploadArtifacts:
    metadata_path: Path
    raw_documents: int
    chunks: int
    saved_files: list[str]
    manifest: object


def persist_uploaded_temp_knowledge(
    *,
    settings: AppSettings,
    knowledge_id: str,
    content_dir: Path,
    vector_store_dir: Path,
    saved_files: list[str],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    embedding_model: str | None = None,
) -> TempUploadArtifacts:
    assembler = EmbeddingAssembler(
        settings,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        text_splitter_name=settings.kb.TEXT_SPLITTER_NAME,
        vector_store_type=settings.kb.DEFAULT_VS_TYPE,
    )
    raw_documents, loaded_files = assembler.load_content_dir(content_dir, settings.kb.SUPPORTED_EXTENSIONS)
    if not loaded_files:
        raise ValueError("上传文件已保存，但没有生成任何可处理文档。")

    assembled = assembler.assemble_documents(raw_documents)
    chunks = assembled.chunks
    if not chunks:
        raise ValueError("上传文件已解析，但未生成任何切片。")
    chunk_records = assembled.chunk_records
    assembler.persist_entries(
        vector_store_dir=vector_store_dir,
        knowledge_name=knowledge_id,
        entries=assembled.entries,
        mode="full",
    )

    metadata_path = vector_store_dir / "metadata.json"
    metadata_payload = [record.model_dump() for record in chunk_records]
    metadata_path.write_text(
        json.dumps(metadata_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    bm25_index_path = resolve_bm25_index_path(vector_store_dir)
    if settings.kb.ENABLE_HYBRID_RETRIEVAL:
        write_bm25_index(
            bm25_index_path,
            [
                build_persisted_bm25_document(
                    chunk_id=entry.chunk_id,
                    page_content=entry.page_content,
                    metadata=entry.metadata,
                    headers=extract_header_metadata(entry.metadata),
                )
                for entry in assembled.entries
            ],
        )
    else:
        delete_bm25_index(bm25_index_path)

    rebuild_sentence_index(
        settings=settings,
        vector_store_dir=vector_store_dir,
        knowledge_name=knowledge_id,
        chunk_entries=assembled.entries,
        embeddings=assembler.embeddings,
        vector_store_type=assembler.vector_store_type,
    )

    manifest = create_temp_manifest(
        settings=settings,
        knowledge_id=knowledge_id,
        saved_files=saved_files,
    )
    write_temp_manifest(settings, manifest)
    return TempUploadArtifacts(
        metadata_path=metadata_path,
        raw_documents=len(raw_documents),
        chunks=len(chunks),
        saved_files=saved_files,
        manifest=manifest,
    )


def build_temp_upload_response(
    *,
    knowledge_id: str,
    content_dir: Path,
    vector_store_dir: Path,
    save_result: dict[str, list[str]],
    artifacts: TempUploadArtifacts,
) -> KnowledgeBaseUploadResponse:
    manifest = artifacts.manifest
    return KnowledgeBaseUploadResponse(
        knowledge_id=knowledge_id,
        scope="temp",
        content_dir=content_dir,
        vector_store_dir=vector_store_dir,
        metadata_path=artifacts.metadata_path,
        saved_files=artifacts.saved_files,
        overwritten_files=save_result["overwritten_files"],
        skipped_files=save_result["skipped_files"],
        files_processed=len(artifacts.saved_files),
        raw_documents=artifacts.raw_documents,
        chunks=artifacts.chunks,
        auto_rebuild=False,
        requires_rebuild=False,
        created_at=manifest.created_at,
        last_accessed_at=manifest.last_accessed_at,
        expires_at=manifest.expires_at,
        ttl_minutes=manifest.ttl_minutes,
        touch_on_access=manifest.touch_on_access,
        cleanup_policy=manifest.cleanup_policy,
    )


def build_local_upload_response(
    *,
    knowledge_base_name: str,
    content_dir: Path,
    vector_store_dir: Path,
    metadata_path: Path | None,
    save_result: dict[str, list[str]],
    files_processed: int,
    auto_rebuild: bool,
    requires_rebuild: bool,
    rebuild_result: RebuildKnowledgeBaseResult | None,
) -> KnowledgeBaseUploadResponse:
    return KnowledgeBaseUploadResponse(
        scope="local",
        knowledge_base_name=knowledge_base_name,
        content_dir=content_dir,
        vector_store_dir=vector_store_dir,
        metadata_path=metadata_path,
        saved_files=save_result["saved_files"],
        overwritten_files=save_result["overwritten_files"],
        skipped_files=save_result["skipped_files"],
        files_processed=files_processed,
        raw_documents=rebuild_result.raw_documents if rebuild_result else None,
        chunks=rebuild_result.chunks if rebuild_result else None,
        auto_rebuild=auto_rebuild,
        requires_rebuild=requires_rebuild,
        rebuild_result=rebuild_result,
    )
