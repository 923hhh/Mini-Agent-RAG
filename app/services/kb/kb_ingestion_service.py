"""负责知识库导入入口、重建入口与结果汇总。"""

from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

from fastapi import UploadFile

from app.schemas.kb import (
    KnowledgeBaseUploadResponse,
    RebuildKnowledgeBaseResult,
)
from app.services.kb.rebuild_execution_plan_service import build_rebuild_execution_plan
from app.services.kb.kb_incremental_rebuild import rebuild_incremental_knowledge_base
from app.services.kb.upload_storage_service import (
    requires_local_rebuild,
    save_uploaded_files,
)
from app.services.kb.upload_runtime_service import (
    build_local_upload_response,
    build_temp_upload_response,
    persist_uploaded_temp_knowledge,
)
from app.services.core.settings import AppSettings


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
    execution_plan = build_rebuild_execution_plan(
        settings=settings,
        knowledge_base_name=knowledge_base_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        enable_image_vlm_for_build=enable_image_vlm_for_build,
        force_full_rebuild=force_full_rebuild,
    )
    result = rebuild_incremental_knowledge_base(
        settings=execution_plan.settings,
        knowledge_base_name=knowledge_base_name,
        content_dir=execution_plan.content_dir,
        vector_store_dir=execution_plan.vector_store_dir,
        chunk_size=execution_plan.runtime.chunk_size,
        chunk_overlap=execution_plan.runtime.chunk_overlap,
        embedding_model=execution_plan.runtime.embedding_model,
        progress_callback=progress_callback,
    )
    return result.model_copy(
        update={
            "image_vlm_enabled_for_build": enable_image_vlm_for_build,
            "force_full_rebuild": force_full_rebuild,
        }
    )


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

    save_result = save_uploaded_files(
        files=files,
        target_dir=content_dir,
        supported_extensions=settings.kb.SUPPORTED_EXTENSIONS,
        overwrite_existing=True,
    )
    saved_files = save_result["saved_files"]

    if not saved_files:
        raise ValueError("上传文件为空或文件名无效。")

    artifacts = persist_uploaded_temp_knowledge(
        settings=settings,
        knowledge_id=knowledge_id,
        content_dir=content_dir,
        vector_store_dir=vector_store_dir,
        saved_files=saved_files,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
    )
    return build_temp_upload_response(
        knowledge_id=knowledge_id,
        content_dir=content_dir,
        vector_store_dir=vector_store_dir,
        save_result=save_result,
        artifacts=artifacts,
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
    save_result = save_uploaded_files(
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

    requires_rebuild = requires_local_rebuild(
        content_dir=content_dir,
        vector_store_dir=vector_store_dir,
        files_processed=files_processed,
        auto_rebuild=auto_rebuild,
    )

    return build_local_upload_response(
        knowledge_base_name=normalized_name,
        content_dir=content_dir,
        vector_store_dir=vector_store_dir,
        metadata_path=metadata_path,
        save_result=save_result,
        files_processed=files_processed,
        auto_rebuild=auto_rebuild,
        requires_rebuild=requires_rebuild,
        rebuild_result=rebuild_result,
    )

