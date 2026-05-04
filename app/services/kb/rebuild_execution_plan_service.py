"""统一解析知识库重建入口所需的执行计划。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.services.core.settings import AppSettings
from app.services.kb.rebuild_runtime_service import (
    RebuildRuntimeConfig,
    resolve_rebuild_runtime_config,
)
from app.services.kb.rebuild_settings_service import build_rebuild_settings
from app.services.kb.upload_storage_service import ensure_knowledge_base_layout


@dataclass(frozen=True)
class RebuildExecutionPlan:
    settings: AppSettings
    runtime: RebuildRuntimeConfig
    content_dir: Path
    vector_store_dir: Path


def build_rebuild_execution_plan(
    settings: AppSettings,
    knowledge_base_name: str,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    embedding_model: str | None = None,
    enable_image_vlm_for_build: bool = False,
    force_full_rebuild: bool = False,
) -> RebuildExecutionPlan:
    effective_settings = build_rebuild_settings(
        settings,
        enable_image_vlm_for_build=enable_image_vlm_for_build,
        force_full_rebuild=force_full_rebuild,
    )
    runtime = resolve_rebuild_runtime_config(
        effective_settings,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
    )
    content_dir, vector_store_dir = ensure_knowledge_base_layout(
        effective_settings,
        knowledge_base_name,
    )
    return RebuildExecutionPlan(
        settings=effective_settings,
        runtime=runtime,
        content_dir=content_dir,
        vector_store_dir=vector_store_dir,
    )


__all__ = [
    "RebuildExecutionPlan",
    "build_rebuild_execution_plan",
]
