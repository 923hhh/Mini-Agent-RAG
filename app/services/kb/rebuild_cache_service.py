"""知识库重建中的切片缓存构建与读写服务。"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha1, sha256
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from app.services.core.settings import AppSettings
from app.services.kb.embedding_assembler import EmbeddingAssembler
from app.services.kb.rebuild_planning_service import BuildManifestEntry, FileBuildPlan
from app.storage.vector_stores import VectorStoreEntry


class CachedChunkEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    page_content: str
    metadata: dict[str, Any]
    embedding: list[float] | None = None
    content_sha256: str


class FileChunkCache(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relative_path: str
    source_path: str
    extension: str
    size_bytes: int
    modified_at: float
    sha256: str
    raw_documents: int
    chunks: int
    chunk_entries: list[CachedChunkEntry]


@dataclass
class BuildCachesResult:
    outputs: dict[str, tuple[BuildManifestEntry, FileChunkCache]]
    stage_timings: dict[str, float]


ProgressCallback = Callable[[float, str], None]


def build_caches_for_plans(
    *,
    settings: AppSettings,
    content_dir: Path,
    plans: list[FileBuildPlan],
    chunk_size: int,
    chunk_overlap: int,
    chunk_cache_dir: Path,
    embedding_model: str,
    compute_file_sha256: Callable[[Path], str],
    progress_callback: ProgressCallback | None = None,
) -> BuildCachesResult:
    if not plans:
        return BuildCachesResult(
            outputs={},
            stage_timings={"document_load": 0.0, "text_split": 0.0, "embedding": 0.0},
        )
    assembler = EmbeddingAssembler(
        settings,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        text_splitter_name=settings.kb.TEXT_SPLITTER_NAME,
        vector_store_type=settings.kb.DEFAULT_VS_TYPE,
    )
    started_at = perf_counter()
    emit_progress(
        progress_callback,
        0.24,
        f"[rebuild] 开始解析 {len(plans)} 个待重建文件",
    )
    file_documents = assembler.load_paths(
        content_dir=content_dir,
        paths=[plan.snapshot.path for plan in plans],
        workers=settings.kb.DOC_PARSE_WORKERS,
        progress_callback=lambda completed, total, relative_path: emit_file_stage_progress(
            progress_callback,
            start=0.24,
            end=0.40,
            stage_label="文档解析",
            completed=completed,
            total=total,
            suffix=relative_path,
        ),
    )
    document_load_seconds = round(perf_counter() - started_at, 4)
    chunk_batches: dict[str, list] = {}
    entries_by_owner: dict[str, list[VectorStoreEntry]] = {}
    started_at = perf_counter()
    total_split_chunks = 0
    for index, plan in enumerate(plans, start=1):
        documents = file_documents[plan.snapshot.relative_path]
        chunks, _ = assembler.split_loaded_documents(documents)
        chunk_batches[plan.snapshot.relative_path] = chunks
        total_split_chunks += len(chunks)
        emit_file_stage_progress(
            progress_callback,
            start=0.40,
            end=0.56,
            stage_label="文本切片",
            completed=index,
            total=len(plans),
            suffix=f"{plan.snapshot.relative_path} | 累计 chunks={total_split_chunks}",
        )
    text_split_seconds = round(perf_counter() - started_at, 4)
    started_at = perf_counter()
    total_embedded_chunks = 0
    for index, plan in enumerate(plans, start=1):
        entries_by_owner[plan.snapshot.relative_path] = assembler.embed_chunks(
            chunk_batches[plan.snapshot.relative_path]
        )
        total_embedded_chunks += len(chunk_batches[plan.snapshot.relative_path])
        emit_file_stage_progress(
            progress_callback,
            start=0.56,
            end=0.84,
            stage_label="向量构建",
            completed=index,
            total=len(plans),
            suffix=f"{plan.snapshot.relative_path} | 累计 embedded chunks={total_embedded_chunks}",
        )
    embedding_seconds = round(perf_counter() - started_at, 4) if chunk_batches else 0.0
    built: dict[str, tuple[BuildManifestEntry, FileChunkCache]] = {}
    for plan in plans:
        chunks = chunk_batches[plan.snapshot.relative_path]
        snapshot = plan.snapshot
        snapshot.sha256 = snapshot.sha256 or compute_file_sha256(snapshot.path)
        cache_file = chunk_cache_filename(snapshot.relative_path)
        chunk_entries = [
            CachedChunkEntry(
                chunk_id=entry.chunk_id,
                page_content=entry.page_content,
                metadata=dict(entry.metadata),
                embedding=entry.embedding,
                content_sha256=hash_text(entry.page_content),
            )
            for entry in entries_by_owner.get(snapshot.relative_path, [])
        ]
        cache = FileChunkCache(
            relative_path=snapshot.relative_path,
            source_path=str(snapshot.path.resolve()),
            extension=snapshot.extension,
            size_bytes=snapshot.size_bytes,
            modified_at=snapshot.modified_at,
            sha256=snapshot.sha256,
            raw_documents=len(file_documents[snapshot.relative_path]),
            chunks=len(chunks),
            chunk_entries=chunk_entries,
        )
        if settings.kb.ENABLE_CHUNK_CACHE:
            write_chunk_cache(chunk_cache_dir / cache_file, cache)
        manifest_entry = BuildManifestEntry(
            relative_path=snapshot.relative_path,
            source_path=str(snapshot.path.resolve()),
            extension=snapshot.extension,
            size_bytes=snapshot.size_bytes,
            modified_at=snapshot.modified_at,
            sha256=snapshot.sha256,
            raw_documents=cache.raw_documents,
            chunks=cache.chunks,
            cache_file=cache_file,
        )
        built[snapshot.relative_path] = (manifest_entry, cache)
    return BuildCachesResult(
        outputs=built,
        stage_timings={
            "document_load": document_load_seconds,
            "text_split": text_split_seconds,
            "embedding": embedding_seconds,
        },
    )


def emit_progress(
    callback: ProgressCallback | None,
    progress: float,
    message: str,
) -> None:
    if callback is None:
        return
    callback(max(0.0, min(1.0, progress)), message)


def emit_file_stage_progress(
    callback: ProgressCallback | None,
    *,
    start: float,
    end: float,
    stage_label: str,
    completed: int,
    total: int,
    suffix: str = "",
) -> None:
    if callback is None or total <= 0 or not should_emit_progress_update(completed, total):
        return
    fraction = completed / total
    progress = start + (end - start) * fraction
    message = f"[rebuild] {stage_label}: {completed}/{total}"
    trimmed_suffix = suffix.strip()
    if trimmed_suffix:
        message += f" | {trimmed_suffix}"
    emit_progress(callback, progress, message)


def should_emit_progress_update(completed: int, total: int) -> bool:
    if completed <= 1 or completed >= total:
        return True
    if total <= 20:
        return True
    interval = max(1, total // 20)
    return completed % interval == 0


def load_cached_caches_for_plans(
    plans: list[FileBuildPlan],
    chunk_cache_dir: Path,
) -> list[FileChunkCache]:
    caches: list[FileChunkCache] = []
    for plan in plans:
        if plan.manifest_entry is None:
            continue
        cache_path = chunk_cache_dir / plan.manifest_entry.cache_file
        if not cache_path.exists():
            raise FileNotFoundError(f"切片缓存不存在: {cache_path}")
        caches.append(load_chunk_cache(cache_path))
    return caches


def flatten_chunk_entries(caches: list[FileChunkCache]) -> list[CachedChunkEntry]:
    entries: list[CachedChunkEntry] = []
    for cache in caches:
        entries.extend(cache.chunk_entries)
    return entries


def load_chunk_cache(path: Path) -> FileChunkCache:
    cache = FileChunkCache.model_validate_json(path.read_text(encoding="utf-8"))
    embedding_path = chunk_cache_embedding_path(path)
    if embedding_path.exists():
        embeddings = load_chunk_cache_embeddings(embedding_path)
        if len(cache.chunk_entries) != embeddings.shape[0]:
            raise ValueError(
                f"切片缓存向量数量与 metadata 不一致: {path} "
                f"(entries={len(cache.chunk_entries)}, embeddings={embeddings.shape[0]})"
            )
        for index, entry in enumerate(cache.chunk_entries):
            entry.embedding = normalize_embedding_vector(embeddings[index])
        return cache

    missing_embedding_entries = [
        entry.chunk_id for entry in cache.chunk_entries if entry.embedding is None
    ]
    if missing_embedding_entries:
        raise FileNotFoundError(
            f"切片向量缓存不存在: {embedding_path}，且 metadata 中也没有内嵌 embedding。"
        )
    return cache


def write_chunk_cache(path: Path, cache: FileChunkCache) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    embedding_path = chunk_cache_embedding_path(path)
    np.save(
        embedding_path,
        build_chunk_cache_embedding_matrix(cache),
        allow_pickle=False,
    )
    payload = cache.model_dump()
    for entry in payload.get("chunk_entries", []):
        if isinstance(entry, dict):
            entry.pop("embedding", None)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def chunk_cache_filename(relative_path: str) -> str:
    return f"{sha1(relative_path.encode('utf-8')).hexdigest()[:20]}.json"


def chunk_cache_embedding_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".npy")


def is_chunk_cache_available(cache_path: Path) -> bool:
    if not cache_path.exists():
        return False
    if chunk_cache_embedding_path(cache_path).exists():
        return True
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    chunk_entries = payload.get("chunk_entries")
    if not isinstance(chunk_entries, list):
        return False
    if not chunk_entries:
        return True
    return all(
        isinstance(entry, dict) and entry.get("embedding") is not None
        for entry in chunk_entries
    )


def build_chunk_cache_embedding_matrix(cache: FileChunkCache) -> np.ndarray:
    if not cache.chunk_entries:
        return np.empty((0, 0), dtype=np.float32)

    rows: list[list[float]] = []
    for entry in cache.chunk_entries:
        if entry.embedding is None:
            raise ValueError(f"切片缓存缺少 embedding: {entry.chunk_id}")
        rows.append(entry.embedding)
    return np.asarray(rows, dtype=np.float32)


def load_chunk_cache_embeddings(path: Path) -> np.ndarray:
    embeddings = np.load(path, allow_pickle=False)
    if embeddings.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    if embeddings.ndim == 1:
        return embeddings.reshape(1, -1).astype(np.float32, copy=False)
    if embeddings.ndim != 2:
        raise ValueError(f"不支持的切片向量缓存维度: {path} -> {embeddings.shape}")
    return embeddings.astype(np.float32, copy=False)


def normalize_embedding_vector(vector: np.ndarray) -> list[float]:
    if vector.ndim != 1:
        raise ValueError(f"期望单条 embedding 为一维向量，实际维度: {vector.shape}")
    return [float(item) for item in vector.tolist()]


def hash_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def cached_entry_to_vector_entry(entry: CachedChunkEntry) -> VectorStoreEntry:
    if entry.embedding is None:
        raise ValueError(f"切片缓存缺少 embedding，无法写入向量库: {entry.chunk_id}")
    return VectorStoreEntry(
        chunk_id=entry.chunk_id,
        page_content=entry.page_content,
        metadata=dict(entry.metadata),
        embedding=list(entry.embedding),
    )
