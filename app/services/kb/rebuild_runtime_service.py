"""解析知识库重建时的运行参数。"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.core.settings import AppSettings


@dataclass(frozen=True)
class RebuildRuntimeConfig:
    chunk_size: int
    chunk_overlap: int
    embedding_model: str


def resolve_rebuild_runtime_config(
    settings: AppSettings,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    embedding_model: str | None = None,
) -> RebuildRuntimeConfig:
    return RebuildRuntimeConfig(
        chunk_size=chunk_size or settings.kb.CHUNK_SIZE,
        chunk_overlap=(
            chunk_overlap if chunk_overlap is not None else settings.kb.CHUNK_OVERLAP
        ),
        embedding_model=embedding_model or settings.model.DEFAULT_EMBEDDING_MODEL,
    )


__all__ = [
    "RebuildRuntimeConfig",
    "resolve_rebuild_runtime_config",
]
