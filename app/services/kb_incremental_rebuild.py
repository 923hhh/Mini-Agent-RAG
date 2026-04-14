from __future__ import annotations

from collections.abc import Callable
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha1, sha256
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from app.loaders.documents import list_supported_files
from app.schemas.kb import DocumentChunkRecord, RebuildKnowledgeBaseResult
from app.services.embedding_assembler import EmbeddingAssembler, extract_header_metadata
from app.services.settings import AppSettings
from app.storage.bm25_index import (
    build_persisted_bm25_document,
    delete_bm25_index,
    resolve_bm25_index_path,
    write_bm25_index,
)
from app.storage.vector_stores import (
    VectorStoreEntry,
    vector_store_index_exists,
)
from app.utils.text import coerce_optional_text


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


class BuildManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relative_path: str
    source_path: str
    extension: str
    size_bytes: int
    modified_at: float
    sha256: str
    raw_documents: int
    chunks: int
    cache_file: str


class KnowledgeBaseBuildManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    knowledge_base_name: str
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    text_splitter_name: str = "ChineseRecursiveTextSplitter"
    vector_store_type: str = "faiss"
    built_at: datetime
    files_total: int
    raw_documents_total: int
    chunks_total: int
    files: list[BuildManifestEntry]


@dataclass
class FileSnapshot:
    path: Path
    relative_path: str
    size_bytes: int
    modified_at: float
    extension: str
    sha256: str | None = None


@dataclass
class FileBuildPlan:
    snapshot: FileSnapshot
    change_kind: Literal["reuse", "new", "modified"]
    manifest_entry: BuildManifestEntry | None = None


@dataclass
class BuildCachesResult:
    outputs: dict[str, tuple[BuildManifestEntry, FileChunkCache]]
    stage_timings: dict[str, float]


ProgressCallback = Callable[[float, str], None]


def rebuild_incremental_knowledge_base(
    *,
    settings: AppSettings,
    knowledge_base_name: str,
    content_dir: Path,
    vector_store_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
    progress_callback: ProgressCallback | None = None,
) -> RebuildKnowledgeBaseResult:
    total_started_at = perf_counter()
    stage_timings: dict[str, float] = {}
    metadata_path = vector_store_dir / "metadata.json"
    manifest_path = settings.vector_store_manifest_path(knowledge_base_name)
    bm25_index_path = resolve_bm25_index_path(vector_store_dir)
    chunk_cache_dir = settings.vector_store_chunk_cache_dir(knowledge_base_name)
    chunk_cache_dir.mkdir(parents=True, exist_ok=True)

    started_at = perf_counter()
    emit_progress(
        progress_callback,
        0.02,
        f"[rebuild] 扫描知识库文件: {knowledge_base_name}",
    )
    files = list_supported_files(content_dir, settings.kb.SUPPORTED_EXTENSIONS)
    stage_timings["scan_files"] = round(perf_counter() - started_at, 4)
    if not files:
        supported = ", ".join(settings.kb.SUPPORTED_EXTENSIONS)
        raise ValueError(
            f"知识库目录中未找到可处理文件: {content_dir}\n"
            f"请将文档放入该目录后重试。支持格式: {supported}"
        )
    emit_progress(
        progress_callback,
        0.08,
        f"[rebuild] 已发现 {len(files)} 个可处理文件，开始规划重建方式",
    )

    existing_manifest = load_build_manifest(manifest_path)
    started_at = perf_counter()
    plans, deleted_entries, index_mode = plan_rebuild(
        settings=settings,
        files=files,
        content_dir=content_dir,
        existing_manifest=existing_manifest,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        metadata_path=metadata_path,
        vector_store_dir=vector_store_dir,
        chunk_cache_dir=chunk_cache_dir,
    )
    stage_timings["plan_rebuild"] = round(perf_counter() - started_at, 4)
    rebuilt_plans = [plan for plan in plans if plan.change_kind in {"new", "modified"}]
    reused_plans = [plan for plan in plans if plan.change_kind == "reuse"]
    files_rebuilt = len(rebuilt_plans)
    files_reused = len(reused_plans)
    files_deleted = len(deleted_entries)
    emit_progress(
        progress_callback,
        0.16,
        (
            f"[rebuild] 重建计划完成: mode={index_mode}, "
            f"重建 {files_rebuilt} 个文件, 复用 {files_reused} 个文件, 删除 {files_deleted} 个文件"
        ),
    )

    if index_mode == "reuse":
        started_at = perf_counter()
        if settings.kb.ENABLE_HYBRID_RETRIEVAL and not bm25_index_path.exists():
            reused_caches = load_cached_caches_for_plans(reused_plans, chunk_cache_dir)
            write_bm25_index_for_caches(bm25_index_path, reused_caches)
        elif not settings.kb.ENABLE_HYBRID_RETRIEVAL:
            delete_bm25_index(bm25_index_path)
        stage_timings["bm25_index_write"] = round(perf_counter() - started_at, 4)
        emit_progress(progress_callback, 0.92, "[rebuild] 复用现有索引，正在写入 manifest")
        final_manifest = build_manifest_from_plans(
            knowledge_base_name=knowledge_base_name,
            plans=plans,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            text_splitter_name=settings.kb.TEXT_SPLITTER_NAME,
            vector_store_type=settings.kb.DEFAULT_VS_TYPE,
        )
        started_at = perf_counter()
        write_build_manifest(manifest_path, final_manifest)
        stage_timings["manifest_write"] = round(perf_counter() - started_at, 4)
        stage_timings["total"] = round(perf_counter() - total_started_at, 4)
        emit_progress(progress_callback, 1.0, "[rebuild] 重建完成")
        return RebuildKnowledgeBaseResult(
            knowledge_base_name=knowledge_base_name,
            content_dir=content_dir,
            vector_store_dir=vector_store_dir,
            metadata_path=metadata_path,
            build_manifest_path=manifest_path,
            files_processed=len(files),
            raw_documents=final_manifest.raw_documents_total,
            chunks=final_manifest.chunks_total,
            incremental_rebuild=settings.kb.ENABLE_INCREMENTAL_REBUILD,
            index_mode=index_mode,
            files_total=len(files),
            files_reused=files_reused,
            files_rebuilt=files_rebuilt,
            files_deleted=files_deleted,
            chunks_reused=final_manifest.chunks_total,
            chunks_embedded=0,
            vector_store_type=settings.kb.DEFAULT_VS_TYPE,
            stage_timings_seconds=stage_timings,
        )

    started_at = perf_counter()
    emit_progress(progress_callback, 0.20, "[rebuild] 初始化 embedding 模型")
    assembler = EmbeddingAssembler(
        settings,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        text_splitter_name=settings.kb.TEXT_SPLITTER_NAME,
        vector_store_type=settings.kb.DEFAULT_VS_TYPE,
    )
    stage_timings["init_embeddings"] = round(perf_counter() - started_at, 4)

    build_caches_result = build_caches_for_plans(
        settings=settings,
        content_dir=content_dir,
        plans=rebuilt_plans,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunk_cache_dir=chunk_cache_dir,
        embedding_model=embedding_model,
        progress_callback=progress_callback,
    )
    rebuilt_outputs = build_caches_result.outputs
    stage_timings.update(build_caches_result.stage_timings)
    chunks_embedded = sum(cache.chunks for _, cache in rebuilt_outputs.values())
    chunks_reused = sum((plan.manifest_entry.chunks if plan.manifest_entry else 0) for plan in reused_plans)
    all_chunk_entries_for_bm25: list[CachedChunkEntry] = []

    if index_mode == "append":
        started_at = perf_counter()
        emit_progress(progress_callback, 0.88, "[rebuild] 追加写入向量索引")
        existing_records = load_metadata_records(metadata_path)
        new_chunk_entries = flatten_chunk_entries([cache for _, cache in rebuilt_outputs.values()])
        if new_chunk_entries:
            assembler.persist_entries(
                vector_store_dir=vector_store_dir,
                knowledge_name=knowledge_base_name,
                entries=[cached_entry_to_vector_entry(item) for item in new_chunk_entries],
                mode="append",
            )
        metadata_records = existing_records + [chunk_entry_to_record(item) for item in new_chunk_entries]
        stage_timings["index_build"] = round(perf_counter() - started_at, 4)
        if settings.kb.ENABLE_HYBRID_RETRIEVAL:
            reused_caches = load_cached_caches_for_plans(reused_plans, chunk_cache_dir)
            all_chunk_entries_for_bm25 = flatten_chunk_entries(
                reused_caches + [cache for _, cache in rebuilt_outputs.values()]
            )
    else:
        reused_caches = load_cached_caches_for_plans(reused_plans, chunk_cache_dir)
        all_chunk_entries = flatten_chunk_entries(reused_caches + [cache for _, cache in rebuilt_outputs.values()])
        if not all_chunk_entries:
            raise ValueError(f"文档已加载，但未生成任何切片: {content_dir}")
        started_at = perf_counter()
        emit_progress(
            progress_callback,
            0.88,
            f"[rebuild] 构建完整向量索引，共 {len(all_chunk_entries)} 个 chunks",
        )
        assembler.persist_entries(
            vector_store_dir=vector_store_dir,
            knowledge_name=knowledge_base_name,
            entries=[cached_entry_to_vector_entry(item) for item in all_chunk_entries],
            mode="full",
        )
        metadata_records = [chunk_entry_to_record(item) for item in all_chunk_entries]
        stage_timings["index_build"] = round(perf_counter() - started_at, 4)
        all_chunk_entries_for_bm25 = all_chunk_entries

    started_at = perf_counter()
    emit_progress(progress_callback, 0.93, "[rebuild] 写入 metadata")
    write_metadata_records(metadata_path, metadata_records)
    stage_timings["metadata_write"] = round(perf_counter() - started_at, 4)
    started_at = perf_counter()
    emit_progress(progress_callback, 0.96, "[rebuild] 写入 BM25 索引")
    if settings.kb.ENABLE_HYBRID_RETRIEVAL:
        write_bm25_index_for_chunk_entries(bm25_index_path, all_chunk_entries_for_bm25)
    else:
        delete_bm25_index(bm25_index_path)
    stage_timings["bm25_index_write"] = round(perf_counter() - started_at, 4)
    final_manifest = build_manifest_from_plans(
        knowledge_base_name=knowledge_base_name,
        plans=plans,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        text_splitter_name=settings.kb.TEXT_SPLITTER_NAME,
        vector_store_type=settings.kb.DEFAULT_VS_TYPE,
        rebuilt_outputs=rebuilt_outputs,
    )
    started_at = perf_counter()
    emit_progress(progress_callback, 0.98, "[rebuild] 写入 build manifest 并清理缓存")
    write_build_manifest(manifest_path, final_manifest)
    stage_timings["manifest_write"] = round(perf_counter() - started_at, 4)
    started_at = perf_counter()
    cleanup_deleted_caches(chunk_cache_dir, deleted_entries)
    stage_timings["cache_cleanup"] = round(perf_counter() - started_at, 4)
    stage_timings["total"] = round(perf_counter() - total_started_at, 4)
    emit_progress(progress_callback, 1.0, "[rebuild] 重建完成")

    return RebuildKnowledgeBaseResult(
        knowledge_base_name=knowledge_base_name,
        content_dir=content_dir,
        vector_store_dir=vector_store_dir,
        metadata_path=metadata_path,
        build_manifest_path=manifest_path,
        files_processed=len(files),
        raw_documents=final_manifest.raw_documents_total,
        chunks=final_manifest.chunks_total,
        incremental_rebuild=settings.kb.ENABLE_INCREMENTAL_REBUILD,
        index_mode=index_mode,
        files_total=len(files),
        files_reused=files_reused,
        files_rebuilt=files_rebuilt,
        files_deleted=files_deleted,
        chunks_reused=chunks_reused,
        chunks_embedded=chunks_embedded,
        vector_store_type=settings.kb.DEFAULT_VS_TYPE,
        stage_timings_seconds=stage_timings,
    )


def plan_rebuild(
    *,
    settings: AppSettings,
    files: list[Path],
    content_dir: Path,
    existing_manifest: KnowledgeBaseBuildManifest | None,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
    metadata_path: Path,
    vector_store_dir: Path,
    chunk_cache_dir: Path,
) -> tuple[list[FileBuildPlan], list[BuildManifestEntry], str]:
    previous_entries = {entry.relative_path: entry for entry in (existing_manifest.files if existing_manifest else [])}
    current_relative_paths = {path.relative_to(content_dir).as_posix() for path in files}
    deleted_entries = [
        entry for relative_path, entry in previous_entries.items() if relative_path not in current_relative_paths
    ]
    config_changed = (
        existing_manifest is None
        or existing_manifest.chunk_size != chunk_size
        or existing_manifest.chunk_overlap != chunk_overlap
        or existing_manifest.embedding_model != embedding_model
        or existing_manifest.text_splitter_name != settings.kb.TEXT_SPLITTER_NAME
        or existing_manifest.vector_store_type != settings.kb.DEFAULT_VS_TYPE
    )
    can_use_incremental = (
        settings.kb.ENABLE_INCREMENTAL_REBUILD
        and existing_manifest is not None
        and vector_store_index_exists(vector_store_dir, settings.kb.DEFAULT_VS_TYPE)
        and metadata_path.exists()
        and not config_changed
    )
    plans: list[FileBuildPlan] = []
    for path in files:
        relative_path = path.relative_to(content_dir).as_posix()
        stat = path.stat()
        snapshot = FileSnapshot(
            path=path,
            relative_path=relative_path,
            size_bytes=stat.st_size,
            modified_at=stat.st_mtime,
            extension=path.suffix.lower(),
        )
        previous_entry = previous_entries.get(relative_path)
        if not can_use_incremental or previous_entry is None:
            change_kind: Literal["new", "modified"] = "new" if previous_entry is None else "modified"
            plans.append(FileBuildPlan(snapshot=snapshot, change_kind=change_kind, manifest_entry=previous_entry))
            continue
        cache_path = chunk_cache_dir / previous_entry.cache_file
        if not is_chunk_cache_available(cache_path) or not settings.kb.ENABLE_FILE_HASH_CACHE:
            plans.append(FileBuildPlan(snapshot=snapshot, change_kind="modified", manifest_entry=previous_entry))
            continue
        if previous_entry.size_bytes == snapshot.size_bytes and abs(previous_entry.modified_at - snapshot.modified_at) < 1e-6:
            plans.append(
                FileBuildPlan(
                    snapshot=snapshot,
                    change_kind="reuse",
                    manifest_entry=previous_entry.model_copy(update={"source_path": str(path.resolve())}),
                )
            )
            continue
        snapshot.sha256 = compute_file_sha256(path)
        if snapshot.sha256 == previous_entry.sha256:
            plans.append(
                FileBuildPlan(
                    snapshot=snapshot,
                    change_kind="reuse",
                    manifest_entry=previous_entry.model_copy(
                        update={
                            "source_path": str(path.resolve()),
                            "size_bytes": snapshot.size_bytes,
                            "modified_at": snapshot.modified_at,
                        }
                    ),
                )
            )
            continue
        plans.append(FileBuildPlan(snapshot=snapshot, change_kind="modified", manifest_entry=previous_entry))
    if not can_use_incremental:
        return plans, deleted_entries, "full"
    rebuild_kinds = {plan.change_kind for plan in plans if plan.change_kind != "reuse"}
    if not rebuild_kinds and not deleted_entries:
        return plans, deleted_entries, "reuse"
    if deleted_entries or "modified" in rebuild_kinds or not settings.kb.ENABLE_APPEND_INDEX:
        return plans, deleted_entries, "full"
    return plans, deleted_entries, "append"


def build_caches_for_plans(
    *,
    settings: AppSettings,
    content_dir: Path,
    plans: list[FileBuildPlan],
    chunk_size: int,
    chunk_overlap: int,
    chunk_cache_dir: Path,
    embedding_model: str,
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


def build_manifest_from_plans(
    *,
    knowledge_base_name: str,
    plans: list[FileBuildPlan],
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
    text_splitter_name: str,
    vector_store_type: str,
    rebuilt_outputs: dict[str, tuple[BuildManifestEntry, FileChunkCache]] | None = None,
) -> KnowledgeBaseBuildManifest:
    entries: list[BuildManifestEntry] = []
    for plan in sorted(plans, key=lambda item: item.snapshot.relative_path):
        if rebuilt_outputs and plan.snapshot.relative_path in rebuilt_outputs:
            entries.append(rebuilt_outputs[plan.snapshot.relative_path][0])
        elif plan.manifest_entry is not None:
            entries.append(plan.manifest_entry)
    return KnowledgeBaseBuildManifest(
        knowledge_base_name=knowledge_base_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        text_splitter_name=text_splitter_name,
        vector_store_type=vector_store_type,
        built_at=datetime.now(UTC),
        files_total=len(entries),
        raw_documents_total=sum(entry.raw_documents for entry in entries),
        chunks_total=sum(entry.chunks for entry in entries),
        files=entries,
    )


def flatten_chunk_entries(caches: list[FileChunkCache]) -> list[CachedChunkEntry]:
    entries: list[CachedChunkEntry] = []
    for cache in caches:
        entries.extend(cache.chunk_entries)
    return entries


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
        headers=extract_header_metadata(entry.metadata),
        content_length=len(entry.page_content),
        content_preview=entry.page_content[:120],
    )


def load_build_manifest(path: Path) -> KnowledgeBaseBuildManifest | None:
    if not path.exists():
        return None
    return KnowledgeBaseBuildManifest.model_validate_json(path.read_text(encoding="utf-8"))


def write_build_manifest(path: Path, manifest: KnowledgeBaseBuildManifest) -> None:
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


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


def compute_file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


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
