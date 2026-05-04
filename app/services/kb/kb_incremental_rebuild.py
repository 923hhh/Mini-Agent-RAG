"""执行知识库的增量重建与索引刷新。"""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from time import perf_counter

from app.loaders.documents import list_supported_files
from app.schemas.kb import RebuildKnowledgeBaseResult
from app.services.kb.embedding_assembler import EmbeddingAssembler
from app.services.kb.rebuild_cache_service import (
    CachedChunkEntry,
    build_caches_for_plans,
    cached_entry_to_vector_entry,
    emit_progress,
    flatten_chunk_entries,
    is_chunk_cache_available,
    load_cached_caches_for_plans,
)
from app.services.kb.rebuild_index_service import (
    chunk_entry_to_record,
    cleanup_deleted_caches,
    load_metadata_records,
    write_bm25_index_for_chunk_entries,
    write_metadata_records,
)
from app.services.kb.rebuild_planning_service import (
    FileBuildPlan,
    build_manifest_from_plans,
    load_build_manifest,
    plan_rebuild,
    write_build_manifest,
)
from app.services.kb.sentence_index_service import rebuild_sentence_index
from app.services.core.settings import AppSettings
from app.storage.bm25_index import (
    delete_bm25_index,
    resolve_bm25_index_path,
)


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
        is_chunk_cache_available=is_chunk_cache_available,
        compute_file_sha256=compute_file_sha256,
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
        reused_caches = load_cached_caches_for_plans(reused_plans, chunk_cache_dir)
        if settings.kb.ENABLE_HYBRID_RETRIEVAL and not bm25_index_path.exists():
            write_bm25_index_for_caches(bm25_index_path, reused_caches)
        elif not settings.kb.ENABLE_HYBRID_RETRIEVAL:
            delete_bm25_index(bm25_index_path)
        stage_timings["bm25_index_write"] = round(perf_counter() - started_at, 4)
        if settings.kb.ENABLE_SENTENCE_INDEX:
            started_at = perf_counter()
            emit_progress(progress_callback, 0.90, "[rebuild] 复用缓存并补建句级索引")
            sentence_assembler = EmbeddingAssembler(
                settings,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embedding_model=embedding_model,
                text_splitter_name=settings.kb.TEXT_SPLITTER_NAME,
                vector_store_type=settings.kb.DEFAULT_VS_TYPE,
            )
            rebuild_sentence_index(
                settings=settings,
                vector_store_dir=vector_store_dir,
                knowledge_name=knowledge_base_name,
                chunk_entries=[
                    cached_entry_to_vector_entry(item)
                    for item in flatten_chunk_entries(reused_caches)
                ],
                embeddings=sentence_assembler.embeddings,
                vector_store_type=settings.kb.DEFAULT_VS_TYPE,
            )
            stage_timings["sentence_index_build"] = round(perf_counter() - started_at, 4)
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
        compute_file_sha256=compute_file_sha256,
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
    if settings.kb.ENABLE_SENTENCE_INDEX:
        started_at = perf_counter()
        emit_progress(progress_callback, 0.97, "[rebuild] 构建句级副索引")
        rebuild_sentence_index(
            settings=settings,
            vector_store_dir=vector_store_dir,
            knowledge_name=knowledge_base_name,
            chunk_entries=[
                cached_entry_to_vector_entry(item)
                for item in all_chunk_entries_for_bm25
            ],
            embeddings=assembler.embeddings,
            vector_store_type=settings.kb.DEFAULT_VS_TYPE,
        )
        stage_timings["sentence_index_build"] = round(perf_counter() - started_at, 4)
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


def compute_file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()



