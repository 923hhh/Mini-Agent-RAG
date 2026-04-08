from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.api.errors import error_payload
from app.schemas.kb import (
    KnowledgeBaseUploadResponse,
    KnowledgeBaseSummary,
    RebuildKnowledgeBaseRequest,
    RebuildKnowledgeBaseResult,
)
from app.services.kb_ingestion_service import (
    list_knowledge_bases,
    rebuild_knowledge_base,
    upload_local_files,
    upload_temp_files,
)
from app.services.settings import load_settings


router = APIRouter(prefix="/knowledge_base", tags=["knowledge_base"])
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@router.post("/upload", response_model=KnowledgeBaseUploadResponse)
def upload_knowledge_base_files(
    files: list[UploadFile] = File(...),
    scope: str = Form(...),
    knowledge_base_name: str = Form(""),
    overwrite_existing: bool = Form(default=False),
    auto_rebuild: bool = Form(default=False),
    chunk_size: int | None = Form(default=None),
    chunk_overlap: int | None = Form(default=None),
    enable_image_vlm_for_build: bool = Form(default=False),
    force_full_rebuild: bool = Form(default=False),
) -> KnowledgeBaseUploadResponse:
    if scope not in {"temp", "local"}:
        raise HTTPException(
            status_code=400,
            detail=error_payload(
                code="unsupported_scope",
                message="仅支持 scope=temp 或 scope=local。",
            ),
        )
    if scope == "temp" and knowledge_base_name:
        raise HTTPException(
            status_code=400,
            detail=error_payload(
                code="invalid_knowledge_base_name",
                message="scope=temp 时不需要 knowledge_base_name，请直接使用返回的 knowledge_id 进行问答。",
            ),
        )

    settings = load_settings(PROJECT_ROOT)
    try:
        if scope == "temp":
            return upload_temp_files(
                settings=settings,
                files=files,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

        return upload_local_files(
            settings=settings,
            knowledge_base_name=knowledge_base_name,
            files=files,
            overwrite_existing=overwrite_existing,
            auto_rebuild=auto_rebuild,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            enable_image_vlm_for_build=enable_image_vlm_for_build,
            force_full_rebuild=force_full_rebuild,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=error_payload(code="upload_validation_error", message=str(exc)),
        ) from exc
    except Exception as exc:
        failure_code = "temp_upload_failed" if scope == "temp" else "local_upload_failed"
        failure_message = "临时知识库创建失败" if scope == "temp" else "本地知识库上传失败"
        raise HTTPException(
            status_code=500,
            detail=error_payload(code=failure_code, message=f"{failure_message}: {exc}"),
        ) from exc


@router.post("/rebuild", response_model=RebuildKnowledgeBaseResult)
def rebuild_local_knowledge_base(
    request: RebuildKnowledgeBaseRequest,
) -> RebuildKnowledgeBaseResult:
    settings = load_settings(PROJECT_ROOT)
    try:
        return rebuild_knowledge_base(
            settings=settings,
            knowledge_base_name=request.knowledge_base_name,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            embedding_model=request.embedding_model,
            enable_image_vlm_for_build=request.enable_image_vlm_for_build,
            force_full_rebuild=request.force_full_rebuild,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=error_payload(code="rebuild_validation_error", message=str(exc)),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=error_payload(code="rebuild_failed", message=f"知识库重建失败: {exc}"),
        ) from exc


@router.get("/list", response_model=list[KnowledgeBaseSummary])
def get_knowledge_base_list() -> list[KnowledgeBaseSummary]:
    settings = load_settings(PROJECT_ROOT)
    try:
        return list_knowledge_bases(settings)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=error_payload(code="list_knowledge_bases_failed", message=f"知识库列表获取失败: {exc}"),
        ) from exc
