"""负责知识库导入、重建与结果汇总。"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.schemas.kb import (
    KnowledgeBaseUploadResponse,
    KnowledgeBaseSummary,
    RebuildKnowledgeBaseResult,
)
from app.services.kb.embedding_assembler import EmbeddingAssembler
from app.services.kb.kb_incremental_rebuild import rebuild_incremental_knowledge_base
from app.services.kb.sentence_index_service import rebuild_sentence_index
from app.services.core.settings import AppSettings
from app.services.runtime.temp_kb_service import create_temp_manifest, write_temp_manifest
from app.storage.bm25_index import (
    build_persisted_bm25_document,
    delete_bm25_index,
    resolve_bm25_index_path,
    write_bm25_index,
)
from app.storage.vector_stores import vector_store_index_exists
from app.utils.text import extract_header_metadata


def ensure_knowledge_base_layout(settings: AppSettings, knowledge_base_name: str) -> tuple[Path, Path]:
    content_dir = settings.knowledge_base_content_dir(knowledge_base_name)
    vector_store_dir = settings.vector_store_dir(knowledge_base_name)
    content_dir.mkdir(parents=True, exist_ok=True)
    vector_store_dir.mkdir(parents=True, exist_ok=True)
    return content_dir, vector_store_dir


def rebuild_knowledge_base(
    settings: AppSettings,
    knowledge_base_name: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    embedding_model: str | None = None,
    enable_image_vlm_for_build: bool = False,
    force_full_rebuild: bool = False,
    progress_callback: Callable[[float, str], None] | None = None,
) -> RebuildKnowledgeBaseResult:
    effective_settings = build_rebuild_settings(
        settings,
        enable_image_vlm_for_build=enable_image_vlm_for_build,
        force_full_rebuild=force_full_rebuild,
    )
    content_dir, vector_store_dir = ensure_knowledge_base_layout(effective_settings, knowledge_base_name)
    resolved_chunk_size = chunk_size or effective_settings.kb.CHUNK_SIZE
    resolved_chunk_overlap = (
        chunk_overlap if chunk_overlap is not None else effective_settings.kb.CHUNK_OVERLAP
    )
    resolved_embedding_model = embedding_model or effective_settings.model.DEFAULT_EMBEDDING_MODEL

    result = rebuild_incremental_knowledge_base(
        settings=effective_settings,
        knowledge_base_name=knowledge_base_name,
        content_dir=content_dir,
        vector_store_dir=vector_store_dir,
        chunk_size=resolved_chunk_size,
        chunk_overlap=resolved_chunk_overlap,
        embedding_model=resolved_embedding_model,
        progress_callback=progress_callback,
    )
    return result.model_copy(
        update={
            "image_vlm_enabled_for_build": enable_image_vlm_for_build,
            "force_full_rebuild": force_full_rebuild,
        }
    )


def build_rebuild_settings(
    settings: AppSettings,
    *,
    enable_image_vlm_for_build: bool = False,
    force_full_rebuild: bool = False,
) -> AppSettings:
    kb_updates: dict[str, object] = {}
    model_updates: dict[str, object] = {}

    if force_full_rebuild or enable_image_vlm_for_build:
        kb_updates["ENABLE_INCREMENTAL_REBUILD"] = False

    if enable_image_vlm_for_build:
        model_updates["IMAGE_VLM_ENABLED"] = True
        model_updates["IMAGE_VLM_AUTO_CAPTION_ENABLED"] = True
        model_updates["IMAGE_VLM_AUTO_TRIGGER_BY_OCR"] = False
        model_updates["IMAGE_VLM_ONLY_WHEN_OCR_EMPTY"] = False

    if not kb_updates and not model_updates:
        return settings

    updates: dict[str, object] = {}
    if kb_updates:
        updates["kb"] = settings.kb.model_copy(update=kb_updates)
    if model_updates:
        updates["model"] = settings.model.model_copy(update=model_updates)
    return settings.model_copy(update=updates)


def list_knowledge_bases(settings: AppSettings) -> list[KnowledgeBaseSummary]:
    root = settings.knowledge_base_root
    root.mkdir(parents=True, exist_ok=True)

    summaries: list[KnowledgeBaseSummary] = []
    for entry in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if not entry.is_dir():
            continue

        content_dir = settings.knowledge_base_content_dir(entry.name)
        vector_store_dir = settings.vector_store_dir(entry.name)
        content_dir.mkdir(parents=True, exist_ok=True)
        vector_store_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(
            [
                file.relative_to(content_dir).as_posix()
                for file in content_dir.rglob("*")
                if file.is_file()
            ]
        )
        summaries.append(
            KnowledgeBaseSummary(
                knowledge_base_name=entry.name,
                content_dir=content_dir,
                vector_store_dir=vector_store_dir,
                files=files,
                file_count=len(files),
                vector_store_type=settings.kb.DEFAULT_VS_TYPE,
                index_exists=vector_store_index_exists(vector_store_dir),
                metadata_exists=(vector_store_dir / "metadata.json").exists(),
            )
        )
    return summaries


def upload_temp_files(
    settings: AppSettings,
    files: list[UploadFile],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    embedding_model: str | None = None,
) -> KnowledgeBaseUploadResponse:
    if not files:
        raise ValueError("未上传任何文件。")

    knowledge_id = f"temp-{uuid4().hex[:12]}"
    content_dir = settings.temp_content_dir(knowledge_id)
    vector_store_dir = settings.temp_vector_store_dir(knowledge_id)
    content_dir.mkdir(parents=True, exist_ok=True)
    vector_store_dir.mkdir(parents=True, exist_ok=True)

    save_result = _save_uploaded_files(
        files=files,
        target_dir=content_dir,
        supported_extensions=settings.kb.SUPPORTED_EXTENSIONS,
        overwrite_existing=True,
    )
    saved_files = save_result["saved_files"]

    if not saved_files:
        raise ValueError("上传文件为空或文件名无效。")

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

    return KnowledgeBaseUploadResponse(
        knowledge_id=knowledge_id,
        scope="temp",
        content_dir=content_dir,
        vector_store_dir=vector_store_dir,
        metadata_path=metadata_path,
        saved_files=saved_files,
        overwritten_files=save_result["overwritten_files"],
        skipped_files=save_result["skipped_files"],
        files_processed=len(saved_files),
        raw_documents=len(raw_documents),
        chunks=len(chunks),
        auto_rebuild=False,
        requires_rebuild=False,
        created_at=manifest.created_at,
        last_accessed_at=manifest.last_accessed_at,
        expires_at=manifest.expires_at,
        ttl_minutes=manifest.ttl_minutes,
        touch_on_access=manifest.touch_on_access,
        cleanup_policy=manifest.cleanup_policy,
    )


def upload_local_files(
    settings: AppSettings,
    knowledge_base_name: str,
    files: list[UploadFile],
    *,
    overwrite_existing: bool = False,
    auto_rebuild: bool = False,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    embedding_model: str | None = None,
    enable_image_vlm_for_build: bool = False,
    force_full_rebuild: bool = False,
) -> KnowledgeBaseUploadResponse:
    normalized_name = knowledge_base_name.strip()
    if not normalized_name:
        raise ValueError("scope=local 时 knowledge_base_name 不能为空。")
    if not files:
        raise ValueError("未上传任何文件。")

    content_dir, vector_store_dir = ensure_knowledge_base_layout(settings, normalized_name)
    save_result = _save_uploaded_files(
        files=files,
        target_dir=content_dir,
        supported_extensions=settings.kb.SUPPORTED_EXTENSIONS,
        overwrite_existing=overwrite_existing,
    )

    files_processed = len(save_result["saved_files"]) + len(save_result["overwritten_files"])
    existing_metadata_path = vector_store_dir / "metadata.json"
    metadata_path = existing_metadata_path if existing_metadata_path.exists() else None
    rebuild_result: RebuildKnowledgeBaseResult | None = None

    if auto_rebuild and files_processed > 0:
        rebuild_result = rebuild_knowledge_base(
            settings=settings,
            knowledge_base_name=normalized_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            enable_image_vlm_for_build=enable_image_vlm_for_build,
            force_full_rebuild=force_full_rebuild,
        )
        metadata_path = rebuild_result.metadata_path

    requires_rebuild = _requires_local_rebuild(
        content_dir=content_dir,
        vector_store_dir=vector_store_dir,
        files_processed=files_processed,
        auto_rebuild=auto_rebuild,
    )

    return KnowledgeBaseUploadResponse(
        scope="local",
        knowledge_base_name=normalized_name,
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
def render_rebuild_summary(result: RebuildKnowledgeBaseResult) -> str:
    lines = [
        "知识库重建完成",
        f"知识库名称: {result.knowledge_base_name}",
        f"知识库内容目录: {result.content_dir}",
        f"向量库存储目录: {result.vector_store_dir}",
        f"元数据文件: {result.metadata_path}",
        f"构建清单文件: {result.build_manifest_path or '未生成'}",
        f"处理文件数: {result.files_processed}",
        f"原始文档数: {result.raw_documents}",
        f"最终切片数: {result.chunks}",
        f"增量重建: {'是' if result.incremental_rebuild else '否'}",
        f"索引构建模式: {result.index_mode}",
        f"复用文件数: {result.files_reused}",
        f"重建文件数: {result.files_rebuilt}",
        f"删除文件数: {result.files_deleted}",
        f"复用切片数: {result.chunks_reused}",
        f"新增向量切片数: {result.chunks_embedded}",
        f"向量存储类型: {result.vector_store_type}",
        f"本次启用图片 VLM: {'是' if result.image_vlm_enabled_for_build else '否'}",
        f"本次强制全量重建: {'是' if result.force_full_rebuild else '否'}",
    ]
    if result.stage_timings_seconds:
        lines.append("阶段耗时(秒):")
        for name, value in result.stage_timings_seconds.items():
            lines.append(f"  - {name}: {value:.4f}")
    return "\n".join(lines)


def _save_uploaded_files(
    *,
    files: list[UploadFile],
    target_dir: Path,
    supported_extensions: list[str],
    overwrite_existing: bool,
) -> dict[str, list[str]]:
    target_dir.mkdir(parents=True, exist_ok=True)
    supported = {ext.lower() for ext in supported_extensions}
    saved_files: list[str] = []
    overwritten_files: list[str] = []
    skipped_files: list[str] = []

    for upload in files:
        try:
            filename = Path(upload.filename or "").name
            if not filename:
                continue

            suffix = Path(filename).suffix.lower()
            if suffix not in supported:
                raise ValueError(
                    f"不支持的文件类型: {filename}。支持格式: {', '.join(supported_extensions)}"
                )

            target = _safe_upload_target(target_dir, filename)
            file_bytes = upload.file.read()
            existed_before_write = target.exists()
            if existed_before_write and not overwrite_existing:
                skipped_files.append(filename)
                continue

            target.write_bytes(file_bytes)
            if existed_before_write:
                overwritten_files.append(filename)
            else:
                saved_files.append(filename)
        finally:
            upload.file.close()

    return {
        "saved_files": saved_files,
        "overwritten_files": overwritten_files,
        "skipped_files": skipped_files,
    }


def _safe_upload_target(target_dir: Path, filename: str) -> Path:
    target = (target_dir / filename).resolve()
    target_root = target_dir.resolve()
    try:
        target.relative_to(target_root)
    except ValueError as exc:
        raise ValueError(f"非法文件名: {filename}") from exc
    return target


def _requires_local_rebuild(
    *,
    content_dir: Path,
    vector_store_dir: Path,
    files_processed: int,
    auto_rebuild: bool,
) -> bool:
    if auto_rebuild:
        return False
    if files_processed > 0:
        return True
    has_content = any(path.is_file() for path in content_dir.rglob("*"))
    index_exists = vector_store_index_exists(vector_store_dir)
    return has_content and not index_exists

