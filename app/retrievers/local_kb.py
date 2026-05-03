"""执行本地知识库的混合检索与结果编排。"""

from __future__ import annotations

from pathlib import Path

from app.schemas.chat import ChatMessage, RetrievedReference
from app.services.core.settings import AppSettings
from app.services.kb.sentence_index_service import sentence_index_exists, resolve_sentence_index_dir
from app.services.models.embedding_service import build_embeddings
from app.services.retrieval.candidate_retrieval_service import (
    build_dense_query_bundle,
    build_query_bundle,
    filter_documents_by_metadata,
    load_all_documents,
)
from app.services.retrieval.candidate_rerank_service import diversify_candidates, rerank_candidates
from app.services.retrieval.query_profile_service import (
    infer_diversity_query_profile,
    infer_query_modality_profile,
)
from app.services.retrieval.query_rewrite_service import generate_hypothetical_doc, generate_multi_queries
from app.services.retrieval.reference_assembly_service import (
    candidate_to_reference,
    group_documents_by_doc_id,
)
from app.services.retrieval.retrieval_diagnostics_service import (
    append_retrieval_trace,
    enrich_topk_diagnostics,
    initialize_retrieval_diagnostics,
)
from app.services.retrieval.timeseries_extension_service import (
    build_timeseries_extension_plan,
    retrieve_candidates_with_timeseries_extension,
)
from app.services.runtime.temp_kb_service import ensure_temp_knowledge_available
from app.storage.bm25_index import LoadedBM25Index, load_bm25_index, resolve_bm25_index_path
from app.storage.filters import MetadataFilters
from app.storage.vector_stores import BaseVectorStoreAdapter, build_vector_store_adapter


def search_local_knowledge_base(
    settings: AppSettings,
    knowledge_base_name: str,
    query: str,
    top_k: int,
    score_threshold: float,
    history: list[ChatMessage] | None = None,
    metadata_filters: MetadataFilters | None = None,
) -> list[RetrievedReference]:
    vector_store_dir = settings.vector_store_dir(knowledge_base_name)
    return search_vector_store(
        settings=settings,
        vector_store_dir=vector_store_dir,
        query=query,
        top_k=top_k,
        score_threshold=score_threshold,
        not_found_hint="请先执行 rebuild_kb 构建知识库索引。",
        history=history,
        metadata_filters=metadata_filters,
    )


def search_local_knowledge_base_second_pass(
    settings: AppSettings,
    knowledge_base_name: str,
    query: str,
    top_k: int,
    score_threshold: float,
    history: list[ChatMessage] | None = None,
    metadata_filters: MetadataFilters | None = None,
) -> list[RetrievedReference]:
    return search_local_knowledge_base(
        settings=settings,
        knowledge_base_name=knowledge_base_name,
        query=query,
        top_k=top_k,
        score_threshold=score_threshold,
        history=history,
        metadata_filters=metadata_filters,
    )


def search_temp_knowledge_base(
    settings: AppSettings,
    knowledge_id: str,
    query: str,
    top_k: int,
    score_threshold: float,
    history: list[ChatMessage] | None = None,
    metadata_filters: MetadataFilters | None = None,
) -> list[RetrievedReference]:
    ensure_temp_knowledge_available(settings, knowledge_id)
    vector_store_dir = settings.temp_vector_store_dir(knowledge_id)
    return search_vector_store(
        settings=settings,
        vector_store_dir=vector_store_dir,
        query=query,
        top_k=top_k,
        score_threshold=score_threshold,
        not_found_hint="请先通过 /knowledge_base/upload 上传临时文件。",
        history=history,
        metadata_filters=metadata_filters,
    )


def search_temp_knowledge_base_second_pass(
    settings: AppSettings,
    knowledge_id: str,
    query: str,
    top_k: int,
    score_threshold: float,
    history: list[ChatMessage] | None = None,
    metadata_filters: MetadataFilters | None = None,
) -> list[RetrievedReference]:
    return search_temp_knowledge_base(
        settings=settings,
        knowledge_id=knowledge_id,
        query=query,
        top_k=top_k,
        score_threshold=score_threshold,
        history=history,
        metadata_filters=metadata_filters,
    )


def search_vector_store(
    settings: AppSettings,
    vector_store_dir: Path,
    query: str,
    top_k: int,
    score_threshold: float,
    not_found_hint: str,
    history: list[ChatMessage] | None = None,
    metadata_filters: MetadataFilters | None = None,
) -> list[RetrievedReference]:
    if not vector_store_dir.exists():
        raise FileNotFoundError(f"知识库索引不存在: {vector_store_dir}\n{not_found_hint}")

    embeddings = build_embeddings(settings)
    vector_store = build_vector_store_adapter(
        settings,
        vector_store_dir,
        embeddings,
        collection_name=vector_store_dir.name,
    )
    if not vector_store.exists():
        raise FileNotFoundError(f"知识库索引不存在: {vector_store_dir}\n{not_found_hint}")

    sentence_vector_store: BaseVectorStoreAdapter | None = None
    if settings.kb.ENABLE_SENTENCE_INDEX and sentence_index_exists(
        vector_store_dir,
        settings.kb.DEFAULT_VS_TYPE,
    ):
        sentence_dir = resolve_sentence_index_dir(vector_store_dir)
        sentence_vector_store = build_vector_store_adapter(
            settings,
            sentence_dir,
            embeddings,
            collection_name=f"{vector_store_dir.name}-sentence-index",
        )

    all_documents = load_all_documents(vector_store)
    if not all_documents:
        return []

    filtered_documents = filter_documents_by_metadata(all_documents, metadata_filters)
    if not filtered_documents:
        return []

    bm25_index: LoadedBM25Index | None = None
    bm25_load_error = ""
    try:
        bm25_index = load_bm25_index(resolve_bm25_index_path(vector_store_dir))
    except Exception as exc:
        bm25_load_error = str(exc)

    query_candidates = generate_multi_queries(settings, query, history)
    rewritten_query = query_candidates[1] if len(query_candidates) > 1 else query.strip()
    query_bundle = build_query_bundle(query_candidates)
    dense_query_bundle = build_dense_query_bundle(
        query_bundle,
        generate_hypothetical_doc(settings, query, history),
    )
    timeseries_extension_plan = build_timeseries_extension_plan(query_bundle)
    query_profile = infer_query_modality_profile(
        query_bundle,
        timeseries_query_profile=timeseries_extension_plan.timeseries_query_profile,
    )
    retrieval_diagnostics = initialize_retrieval_diagnostics(
        query_bundle=query_bundle,
        dense_query_bundle=dense_query_bundle,
        hyde_enabled=settings.kb.ENABLE_HYDE,
        bm25_index_available=bm25_index is not None,
        bm25_backend=bm25_index.backend if bm25_index is not None else "dynamic_legacy",
        timeseries_query_profile=timeseries_extension_plan.timeseries_query_profile,
        joint_query_profile=timeseries_extension_plan.joint_query_profile,
        bm25_load_error=bm25_load_error,
    )
    candidates = retrieve_candidates_with_timeseries_extension(
        settings=settings,
        vector_store=vector_store,
        sentence_vector_store=sentence_vector_store,
        all_documents=filtered_documents,
        query_bundle=query_bundle,
        dense_query_bundle=dense_query_bundle,
        bm25_index=bm25_index,
        top_k=top_k,
        metadata_filters=metadata_filters,
        query_profile=query_profile,
        extension_plan=timeseries_extension_plan,
        diagnostics=retrieval_diagnostics,
    )
    if not candidates:
        append_retrieval_trace(
            settings=settings,
            vector_store_dir=vector_store_dir,
            query=query,
            rewritten_query=rewritten_query,
            query_bundle=query_bundle,
            top_k=top_k,
            metadata_filters=metadata_filters,
            diagnostics=retrieval_diagnostics,
            references=[],
        )
        return []

    reranked = rerank_candidates(
        settings=settings,
        query=query,
        candidates=candidates,
        query_bundle=query_bundle,
        query_profile=query_profile,
        joint_query_profile=timeseries_extension_plan.joint_query_profile,
        top_k=top_k,
        diagnostics=retrieval_diagnostics,
    )
    filtered = [item for item in reranked if item.relevance_score >= score_threshold]
    if not filtered:
        append_retrieval_trace(
            settings=settings,
            vector_store_dir=vector_store_dir,
            query=query,
            rewritten_query=rewritten_query,
            query_bundle=query_bundle,
            top_k=top_k,
            metadata_filters=metadata_filters,
            diagnostics=retrieval_diagnostics,
            references=[],
        )
        return []

    grouped_documents = group_documents_by_doc_id(filtered_documents)
    final_candidates = diversify_candidates(
        filtered,
        target_count=top_k,
        query_profile=query_profile,
        joint_query_profile=timeseries_extension_plan.joint_query_profile,
        diversity_profile=infer_diversity_query_profile(query_bundle),
    )
    enrich_topk_diagnostics(
        diagnostics=retrieval_diagnostics,
        final_candidates=final_candidates,
        top_k=top_k,
    )
    references = [
        candidate_to_reference(
            settings=settings,
            candidate=item,
            grouped_documents=grouped_documents,
        )
        for item in final_candidates
    ]
    append_retrieval_trace(
        settings=settings,
        vector_store_dir=vector_store_dir,
        query=query,
        rewritten_query=rewritten_query,
        query_bundle=query_bundle,
        top_k=top_k,
        metadata_filters=metadata_filters,
        diagnostics=retrieval_diagnostics,
        references=references,
    )
    return references
