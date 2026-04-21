"""执行本地知识库的混合检索、重排与结果组装。"""

from __future__ import annotations

from datetime import date
from math import ceil
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log
from pathlib import Path
import re

from langchain_core.documents import Document

from app.constants import IMAGE_QUERY_HINTS
from app.schemas.chat import ChatMessage, RetrievedReference
from app.services.models.embedding_service import build_embeddings
from app.services.core.observability import append_jsonl_trace
from app.services.retrieval.query_rewrite_service import generate_hypothetical_doc, generate_multi_queries
from app.services.retrieval.rerank_service import RerankTextInput, rerank_texts
from app.services.kb.sentence_index_service import sentence_index_exists, resolve_sentence_index_dir
from app.services.core.settings import AppSettings
from app.services.runtime.temp_kb_service import ensure_temp_knowledge_available
from app.storage.bm25_index import (
    LoadedBM25Index,
    build_match_terms as build_bm25_match_terms,
    build_search_text_from_parts,
    load_bm25_index,
    normalize_search_text as normalize_bm25_search_text,
    resolve_bm25_index_path,
    score_bm25_index,
)
from app.storage.filters import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    matches_metadata_filters,
)
from app.storage.vector_stores import BaseVectorStoreAdapter, build_vector_store_adapter
from app.utils.text import coerce_optional_text, deduplicate_strings, extract_document_headers


TEXT_QUERY_HINTS = (
    "文档",
    "章节",
    "文章",
    "这本书",
    "说明书",
    "资料",
    "参数",
    "配置",
)
MULTI_DOC_QUERY_HINTS = (
    "共同",
    "分别",
    "各自",
    "区别",
    "不同",
    "相同",
    "比较",
    "相比",
    "对比",
    "优势",
    "特点",
    "目标",
    "哪些专业",
    "哪个专业",
    "哪几个专业",
)
MULTI_DOC_CONNECTORS = (
    "与",
    "和",
    "及",
    "以及",
    "、",
)
SAMPLE_ID_PATTERN = re.compile(r"^[0-9a-f]{24}$", re.IGNORECASE)
SAMPLE_GROUP_AGGREGATION_LIMIT = 3
SAMPLE_GROUP_MIN_COUNT_FOR_RERANK = 2
SAMPLE_GROUP_MIN_COUNT_FOR_TRIM = 3
SAMPLE_GROUP_DOMINANCE_RATIO_FOR_TRIM = 0.15
TEMPORAL_QUERY_HINTS = (
    "时间",
    "日期",
    "哪一年",
    "哪年",
    "何时",
    "什么时候",
    "几月",
    "几号",
    "多久",
    "截至",
    "截止",
    "开始",
    "结束",
    "报名",
    "查询",
)
TEMPORAL_RECENCY_HINTS = (
    "最新",
    "当前",
    "目前",
    "现任",
    "最近",
    "今年",
    "本年",
    "本年度",
)
YEAR_PATTERN = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
DATE_PATTERN = re.compile(
    r"(?<!\d)((?:19|20)\d{2})\s*(?:-|/|\.|年)\s*(\d{1,2})\s*(?:-|/|\.|月)\s*(\d{1,2})\s*(?:日|号)?(?!\d)"
)
LABELED_DATE_PATTERN = re.compile(
    r"(?:日期|发布时间|更新(?:时间)?|发布(?:时间)?)\s*[：: ]+\s*"
    r"((?:19|20)\d{2})\s*(?:-|/|\.|年)\s*(\d{1,2})\s*(?:-|/|\.|月)\s*(\d{1,2})\s*(?:日|号)?"
)
ANSWER_NUMERIC_PATTERN = re.compile(
    r"(?:\d+|[零一二三四五六七八九十百千万两〇]+)\s*(?:人|项|家|篇|分|个|所|名|种|门|届|年|月|日|级)"
)
ANSWER_GRADE_PATTERN = re.compile(r"(?:A\+|A-|A|B\+|B-|B|C\+|C-|C|甲类|乙类|丙类)", re.IGNORECASE)
ANSWER_SEEKING_HINTS = (
    "多少",
    "几",
    "谁",
    "哪位",
    "何人",
    "哪一年",
    "哪年",
    "何时",
    "什么时候",
    "开始时间",
    "什么时间",
    "什么等级",
    "被评为",
    "哪三个",
    "哪几",
    "哪些",
    "哪个",
    "是什么",
    "是怎样",
    "理念",
    "愿景",
    "定位",
    "目标",
    "对象",
    "政策",
)
ANSWER_LABEL_TERMS = (
    "办学理念",
    "办学愿景",
    "发展定位",
    "人才培养目标",
    "降分政策",
    "招生人数",
    "招生计划",
    "开始时间",
    "招生对象",
    "定向招收对象",
    "办学目标",
    "等级",
)
LIST_QUERY_HINTS = ("哪些", "哪三个", "哪几", "包括哪些")
LANGUAGE_TERMS = ("汉语", "中文", "英语", "英文", "法语", "法文", "德语", "日语", "俄语")
ANSWER_ROLE_TERMS = ("校长", "院长", "主任", "书记", "负责人", "创办", "成立")
ANSWER_WINDOW_SPLIT_PATTERN = re.compile(r"(?:\r?\n+|[。！？!?；;])")
RERANK_FIELD_MAX_CHARS = 80
RERANK_SUMMARY_MAX_CHARS = 120
RERANK_BODY_MAX_CHARS = 280


@dataclass
class RetrievalCandidate:
    document: Document
    dense_rank: int | None = None
    dense_distance: float | None = None
    dense_relevance: float = 0.0
    sentence_rank: int | None = None
    sentence_distance: float | None = None
    sentence_relevance: float = 0.0
    sentence_text: str = ""
    lexical_rank: int | None = None
    lexical_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float = 0.0
    model_rerank_score: float = 0.0
    relevance_score: float = 0.0
    body_overlap_ratio: float = 0.0
    answer_window_overlap_ratio: float = 0.0
    answer_support_bonus: float = 0.0
    answer_focus_score: float = 0.0


@dataclass(frozen=True)
class QueryModalityProfile:
    query_type: str
    preferred_modalities: tuple[str, ...]
    modality_bonus: dict[str, float]
    preferred_extensions: tuple[str, ...]
    extension_bonus: dict[str, float]
    path_hint_terms: tuple[str, ...]


@dataclass(frozen=True)
class SampleGroupStats:
    sample_id: str
    aggregate_score: float
    max_score: float
    candidate_count: int


@dataclass(frozen=True)
class TemporalQueryProfile:
    is_temporal: bool
    prefers_recent: bool
    explicit_years: tuple[int, ...]
    explicit_dates: tuple[int, ...]


@dataclass(frozen=True)
class DiversityQueryProfile:
    prefer_family_diversity: bool


@dataclass(frozen=True)
class RerankModelSelection:
    model_name: str
    route: str


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
        raise FileNotFoundError(
            f"知识库索引不存在: {vector_store_dir}\n{not_found_hint}"
        )

    embeddings = build_embeddings(settings)
    vector_store = build_vector_store_adapter(
        settings,
        vector_store_dir,
        embeddings,
        collection_name=vector_store_dir.name,
    )
    if not vector_store.exists():
        raise FileNotFoundError(
            f"知识库索引不存在: {vector_store_dir}\n{not_found_hint}"
        )
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
    retrieval_diagnostics: dict[str, object] = {
        "query_bundle_count": len(query_bundle),
        "dense_query_bundle_count": len(dense_query_bundle),
        "hyde_enabled": settings.kb.ENABLE_HYDE,
        "hyde_used": len(dense_query_bundle) > len(query_bundle),
        "bm25_index_available": bm25_index is not None,
        "bm25_backend": bm25_index.backend if bm25_index is not None else "dynamic_legacy",
    }
    if bm25_load_error:
        retrieval_diagnostics["bm25_load_error"] = bm25_load_error[:200]
    if len(dense_query_bundle) > len(query_bundle):
        retrieval_diagnostics["hyde_preview"] = dense_query_bundle[-1][:160]
    query_profile = infer_query_modality_profile(query_bundle)
    candidates = retrieve_candidates(
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
        diversity_profile=infer_diversity_query_profile(query_bundle),
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


def retrieve_candidates(
    settings: AppSettings,
    vector_store: BaseVectorStoreAdapter,
    sentence_vector_store: BaseVectorStoreAdapter | None,
    all_documents: dict[str, Document],
    query_bundle: list[str],
    dense_query_bundle: list[str] | None,
    bm25_index: LoadedBM25Index | None,
    top_k: int,
    metadata_filters: MetadataFilters | None = None,
    query_profile: QueryModalityProfile | None = None,
    diagnostics: dict[str, object] | None = None,
) -> list[RetrievalCandidate]:
    candidate_map: dict[str, RetrievalCandidate] = {}
    dense_limit = max(top_k, settings.kb.HYBRID_DENSE_TOP_K)
    dense_queries = dense_query_bundle or query_bundle
    query_profile = query_profile or infer_query_modality_profile(query_bundle)
    temporal_profile = infer_temporal_query_profile(query_bundle)
    modality_groups = group_documents_by_source_modality(all_documents)
    modality_grouped_dense_used = should_use_modality_grouped_dense_retrieval(
        modality_groups,
        all_documents,
    )

    if diagnostics is not None:
        diagnostics.update(
            {
                "query_type": resolve_query_type_label(query_profile),
                "preferred_modalities": list(query_profile.preferred_modalities),
                "preferred_extensions": list(query_profile.preferred_extensions),
                "path_hint_terms": list(query_profile.path_hint_terms),
                "temporal_query": temporal_profile.is_temporal,
                "temporal_prefers_recent": temporal_profile.prefers_recent,
                "temporal_explicit_years": list(temporal_profile.explicit_years),
                "available_modalities": {
                    key: len(value) for key, value in modality_groups.items()
                },
                "modality_grouped_dense_used": modality_grouped_dense_used,
                "sentence_index_available": sentence_vector_store is not None,
            }
        )

    if modality_grouped_dense_used:
        per_modality_dense_limit = max(1, ceil(dense_limit / max(1, len(modality_groups)))) + 2
        for source_modality in select_modalities_for_query(modality_groups, query_profile):
            collect_dense_candidates(
                vector_store=vector_store,
                query_bundle=dense_queries,
                dense_limit=per_modality_dense_limit,
                candidate_map=candidate_map,
                metadata_filters=merge_metadata_filters_with_source_modality(
                    metadata_filters,
                    source_modality,
                ),
                dense_fetch_multiplier=settings.kb.METADATA_FILTER_DENSE_FETCH_MULTIPLIER,
            )
    else:
        collect_dense_candidates(
            vector_store=vector_store,
            query_bundle=dense_queries,
            dense_limit=dense_limit,
            candidate_map=candidate_map,
            metadata_filters=metadata_filters,
            dense_fetch_multiplier=settings.kb.METADATA_FILTER_DENSE_FETCH_MULTIPLIER,
        )

    if settings.kb.ENABLE_HYBRID_RETRIEVAL:
        if modality_groups:
            per_modality_lexical_limit = max(
                top_k,
                ceil(settings.kb.HYBRID_LEXICAL_TOP_K / max(1, len(modality_groups))),
            )
            for source_modality in select_modalities_for_query(modality_groups, query_profile):
                collect_lexical_candidates(
                    settings=settings,
                    all_documents=modality_groups[source_modality],
                    query_bundle=query_bundle,
                    bm25_index=bm25_index,
                    candidate_map=candidate_map,
                    lexical_limit=per_modality_lexical_limit,
                )
        else:
            collect_lexical_candidates(
                settings=settings,
                all_documents=all_documents,
                query_bundle=query_bundle,
                bm25_index=bm25_index,
                candidate_map=candidate_map,
                lexical_limit=settings.kb.HYBRID_LEXICAL_TOP_K,
            )

    sentence_index_used = False
    if sentence_vector_store is not None and should_use_sentence_index(query_bundle):
        collect_sentence_dense_candidates(
            settings=settings,
            sentence_vector_store=sentence_vector_store,
            all_documents=all_documents,
            query_bundle=query_bundle,
            sentence_limit=max(top_k, settings.kb.SENTENCE_INDEX_TOP_K),
            candidate_map=candidate_map,
            metadata_filters=metadata_filters,
            dense_fetch_multiplier=settings.kb.METADATA_FILTER_DENSE_FETCH_MULTIPLIER,
        )
        sentence_index_used = True

    fused = list(candidate_map.values())
    if not fused:
        if diagnostics is not None:
            diagnostics["candidate_count"] = 0
        return []

    max_dense = max((item.dense_relevance for item in fused), default=1.0) or 1.0
    max_lexical = max((item.lexical_score for item in fused), default=1.0) or 1.0
    temporal_adjustments = build_temporal_candidate_adjustments(fused, temporal_profile)
    for item in fused:
        score = 0.0
        if item.dense_rank is not None:
            score += 1.0 / (settings.kb.HYBRID_RRF_K + item.dense_rank)
        if item.lexical_rank is not None:
            score += 1.0 / (settings.kb.HYBRID_RRF_K + item.lexical_rank)
        score += settings.kb.HYBRID_DENSE_SCORE_WEIGHT * (item.dense_relevance / max_dense)
        score += settings.kb.HYBRID_LEXICAL_SCORE_WEIGHT * (item.lexical_score / max_lexical)
        score += modality_bonus_for_candidate(item.document, query_profile)
        score += temporal_adjustments.get(get_chunk_id(item.document), 0.0)
        item.fused_score = score

    if diagnostics is not None:
        diagnostics["candidate_count"] = len(fused)
        diagnostics["candidate_modality_counts"] = count_candidate_modalities(fused)
        diagnostics["sentence_index_used"] = sentence_index_used

    return fused


def collect_dense_candidates(
    *,
    vector_store: BaseVectorStoreAdapter,
    query_bundle: list[str],
    dense_limit: int,
    candidate_map: dict[str, RetrievalCandidate],
    metadata_filters: MetadataFilters | None,
    dense_fetch_multiplier: int,
) -> None:
    dense_hits: dict[str, tuple[int, float, Document]] = {}
    fetch_k = dense_limit * max(1, dense_fetch_multiplier) if metadata_filters else dense_limit
    for dense_query in query_bundle:
        docs_with_scores = vector_store.similarity_search_with_score(
            dense_query,
            k=dense_limit,
            metadata_filters=metadata_filters,
            fetch_k=fetch_k,
        )
        for rank, (document, raw_score) in enumerate(docs_with_scores, start=1):
            chunk_id = get_chunk_id(document)
            previous = dense_hits.get(chunk_id)
            if previous is None or raw_score < previous[1]:
                dense_hits[chunk_id] = (rank, float(raw_score), document)

    for rank, (chunk_id, (_, raw_score, document)) in enumerate(
        sorted(dense_hits.items(), key=lambda item: item[1][1]),
        start=1,
    ):
        candidate = candidate_map.setdefault(chunk_id, RetrievalCandidate(document=document))
        candidate.dense_rank = rank
        candidate.dense_distance = raw_score
        candidate.dense_relevance = distance_to_relevance(raw_score)


def collect_sentence_dense_candidates(
    *,
    settings: AppSettings,
    sentence_vector_store: BaseVectorStoreAdapter,
    all_documents: dict[str, Document],
    query_bundle: list[str],
    sentence_limit: int,
    candidate_map: dict[str, RetrievalCandidate],
    metadata_filters: MetadataFilters | None,
    dense_fetch_multiplier: int,
) -> None:
    sentence_hits: dict[str, tuple[int, float, Document]] = {}
    fetch_k = sentence_limit * max(1, dense_fetch_multiplier) if metadata_filters else sentence_limit
    for dense_query in query_bundle:
        docs_with_scores = sentence_vector_store.similarity_search_with_score(
            dense_query,
            k=sentence_limit,
            metadata_filters=metadata_filters,
            fetch_k=fetch_k,
        )
        for rank, (document, raw_score) in enumerate(docs_with_scores, start=1):
            parent_chunk_id = str(document.metadata.get("parent_chunk_id", "") or "").strip()
            if not parent_chunk_id or parent_chunk_id not in all_documents:
                continue
            previous = sentence_hits.get(parent_chunk_id)
            if previous is None or raw_score < previous[1]:
                sentence_hits[parent_chunk_id] = (rank, float(raw_score), document)

    for rank, (parent_chunk_id, (_, raw_score, sentence_document)) in enumerate(
        sorted(sentence_hits.items(), key=lambda item: item[1][1]),
        start=1,
    ):
        candidate = candidate_map.get(parent_chunk_id)
        if candidate is None:
            continue
        candidate.sentence_rank = rank
        candidate.sentence_distance = raw_score
        candidate.sentence_relevance = distance_to_relevance(raw_score)
        sentence_text = str(
            sentence_document.metadata.get("sentence_text")
            or sentence_document.page_content
            or ""
        ).strip()
        if sentence_text:
            candidate.sentence_text = sentence_text


def collect_lexical_candidates(
    *,
    settings: AppSettings,
    all_documents: dict[str, Document],
    query_bundle: list[str],
    bm25_index: LoadedBM25Index | None,
    candidate_map: dict[str, RetrievalCandidate],
    lexical_limit: int,
) -> None:
    query_terms = build_match_terms(query_bundle)
    if not query_terms:
        return

    normalized_queries = [normalize_search_text(item) for item in query_bundle if item.strip()]
    plain_queries = [item.lower().strip() for item in query_bundle if item.strip()]
    allowed_chunk_ids = set(all_documents)
    if bm25_index is not None:
        lexical_scores = score_bm25_index(
            index=bm25_index,
            query_terms=query_terms,
            normalized_queries=normalized_queries,
            plain_queries=plain_queries,
            allowed_chunk_ids=allowed_chunk_ids,
        )
    else:
        query_counter = Counter(query_terms)
        doc_infos = build_lexical_doc_infos(all_documents)
        total_docs = len(doc_infos)
        average_length = (
            sum(item["length"] for item in doc_infos.values()) / total_docs if total_docs else 1.0
        )
        document_frequency = Counter()
        unique_query_terms = set(query_counter)
        for info in doc_infos.values():
            terms = info["unique_terms"]
            for term in unique_query_terms:
                if term in terms:
                    document_frequency[term] += 1

        lexical_scores: list[tuple[str, float]] = []
        for chunk_id, info in doc_infos.items():
            term_counter: Counter[str] = info["term_counter"]
            doc_length = max(1, info["length"])
            score = 0.0
            for term, query_tf in query_counter.items():
                term_tf = term_counter.get(term, 0)
                if term_tf == 0:
                    continue
                df = document_frequency.get(term, 0)
                idf = log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
                numerator = term_tf * (1.5 + 1.0)
                denominator = term_tf + 1.5 * (1.0 - 0.75 + 0.75 * (doc_length / average_length))
                score += query_tf * idf * (numerator / denominator)

            normalized_text = info["normalized_text"]
            raw_text = info["raw_text"]
            if any(query_text and query_text in raw_text for query_text in plain_queries):
                score += 0.8
            if any(normalized_query and normalized_query in normalized_text for normalized_query in normalized_queries):
                score += 1.2

            if score <= 0:
                continue
            lexical_scores.append((chunk_id, score))

        lexical_scores.sort(key=lambda item: item[1], reverse=True)
    for rank, (chunk_id, score) in enumerate(
        lexical_scores[: lexical_limit],
        start=1,
    ):
        document = all_documents[chunk_id]
        candidate = candidate_map.setdefault(chunk_id, RetrievalCandidate(document=document))
        candidate.lexical_rank = rank
        candidate.lexical_score = score


def rerank_candidates(
    settings: AppSettings,
    query: str,
    candidates: list[RetrievalCandidate],
    query_bundle: list[str],
    query_profile: QueryModalityProfile,
    top_k: int,
    diagnostics: dict[str, object] | None = None,
) -> list[RetrievalCandidate]:
    heuristic_ranked = heuristic_rerank_candidates(
        settings=settings,
        candidates=candidates,
        query_bundle=query_bundle,
        top_k=top_k,
    )
    if not heuristic_ranked:
        return []

    top_n = max(settings.kb.RERANK_CANDIDATES_TOP_N, top_k)
    query_term_set = set(build_match_terms(query_bundle))
    plain_queries = [item.lower().strip() for item in query_bundle if item.strip()]
    primary_query = plain_queries[0] if plain_queries else ""
    rerank_model_selection = resolve_rerank_model_selection(
        settings=settings,
        query_bundle=query_bundle,
        query_profile=query_profile,
    )
    if diagnostics is not None:
        diagnostics["rerank_model_selected"] = rerank_model_selection.model_name
        diagnostics["rerank_model_route"] = rerank_model_selection.route
    rerank_inputs = [
        RerankTextInput(
            candidate_id=get_chunk_id(candidate.document),
            text=build_candidate_rerank_text(candidate.document, primary_query, query_term_set),
        )
        for candidate in heuristic_ranked[:top_n]
    ]
    rerank_outcome = rerank_texts(
        settings=settings,
        query=query.strip(),
        items=rerank_inputs,
        top_n=top_n,
        model_name_override=rerank_model_selection.model_name,
    )
    if not rerank_outcome.applied:
        if settings.kb.RERANK_FALLBACK_TO_HEURISTIC:
            return heuristic_ranked
        return []

    max_heuristic = max((item.rerank_score for item in heuristic_ranked), default=1.0) or 1.0
    reranked: list[RetrievalCandidate] = []
    for candidate in heuristic_ranked:
        chunk_id = get_chunk_id(candidate.document)
        model_score = rerank_outcome.scores.get(chunk_id)
        if model_score is None:
            reranked.append(candidate)
            continue

        candidate.model_rerank_score = model_score
        heuristic_component = candidate.rerank_score / max_heuristic
        candidate.rerank_score = model_score + 0.03 * heuristic_component + 0.02 * candidate.fused_score
        if model_score < settings.kb.RERANK_SCORE_THRESHOLD:
            candidate.relevance_score = min(candidate.relevance_score, model_score)
        else:
            candidate.relevance_score = min(
                1.0,
                0.85 * model_score + 0.15 * candidate.relevance_score,
            )
        reranked.append(candidate)

    apply_same_sample_group_rerank_adjustments(reranked)
    reranked.sort(key=lambda item: item.rerank_score, reverse=True)
    cutoff = resolve_rerank_cutoff(settings, query_profile, top_k)
    return ensure_modality_coverage(
        reranked_candidates=reranked,
        required_modalities=resolve_required_modalities_for_query(query_profile),
        cutoff=cutoff,
    )


def heuristic_rerank_candidates(
    settings: AppSettings,
    candidates: list[RetrievalCandidate],
    query_bundle: list[str],
    top_k: int,
) -> list[RetrievalCandidate]:
    if not candidates:
        return []

    query_profile = infer_query_modality_profile(query_bundle)
    temporal_profile = infer_temporal_query_profile(query_bundle)
    query_terms = build_match_terms(query_bundle)
    max_fused = max((item.fused_score for item in candidates), default=1.0) or 1.0
    max_dense = max((item.dense_relevance for item in candidates), default=1.0) or 1.0
    max_lexical = max((item.lexical_score for item in candidates), default=1.0) or 1.0
    normalized_queries = [normalize_search_text(item) for item in query_bundle if item.strip()]
    plain_queries = [item.lower().strip() for item in query_bundle if item.strip()]
    primary_query = plain_queries[0] if plain_queries else ""
    temporal_adjustments = build_temporal_candidate_adjustments(candidates, temporal_profile)

    query_term_set = set(query_terms)
    reranked: list[RetrievalCandidate] = []
    for candidate in candidates:
        search_text = build_search_text(candidate.document)
        normalized_text = normalize_search_text(search_text)
        doc_terms = set(build_match_terms([search_text]))
        page_text = str(candidate.document.page_content or "")
        overlap_ratio = (
            len(doc_terms & query_term_set) / len(query_term_set)
            if query_term_set
            else 0.0
        )
        body_overlap_ratio = compute_body_overlap_ratio(page_text, query_term_set)
        answer_window_text = build_answer_window_text(primary_query, page_text, query_term_set)
        sentence_hint = candidate.sentence_text.strip()
        preferred_answer_text = answer_window_text
        if sentence_hint:
            preferred_answer_text = (
                f"{sentence_hint}\n{answer_window_text}"
                if answer_window_text and answer_window_text != page_text
                else sentence_hint
            )
            candidate.document.metadata["sentence_query_hit"] = sentence_hint
        answer_window_overlap_ratio = (
            compute_body_overlap_ratio(preferred_answer_text, query_term_set)
            if preferred_answer_text and preferred_answer_text != page_text
            else 0.0
        )
        phrase_bonus = 1.0 if any(query_text and query_text in search_text.lower() for query_text in plain_queries) else 0.0
        normalized_bonus = (
            1.0 if any(query_text and query_text in normalized_text for query_text in normalized_queries) else 0.0
        )
        source_text = f"{candidate.document.metadata.get('title', '')} {candidate.document.metadata.get('source', '')}".lower()
        source_bonus = 0.4 if any(term in source_text for term in query_terms if len(term) >= 2) else 0.0
        answer_support_bonus = compute_answer_support_bonus(primary_query, preferred_answer_text)
        answer_focus_score = min(
            1.0,
            0.35 * body_overlap_ratio
            + 0.45 * answer_window_overlap_ratio
            + 1.80 * answer_support_bonus
            + (0.10 if sentence_hint else 0.0),
        )

        fused_component = candidate.fused_score / max_fused
        dense_component = candidate.dense_relevance / max_dense
        lexical_component = candidate.lexical_score / max_lexical
        modality_bonus = modality_bonus_for_candidate(candidate.document, query_profile)
        temporal_bonus = temporal_adjustments.get(get_chunk_id(candidate.document), 0.0)

        if settings.kb.ENABLE_HEURISTIC_RERANK:
            candidate.rerank_score = (
                0.35 * fused_component
                + 0.25 * dense_component
                + 0.20 * lexical_component
                + 0.10 * overlap_ratio
                + 0.06 * body_overlap_ratio
                + 0.05 * answer_window_overlap_ratio
                + 0.06 * phrase_bonus
                + 0.10 * normalized_bonus
                + 0.04 * source_bonus
                + answer_support_bonus
                + modality_bonus
                + temporal_bonus
            )
        else:
            candidate.rerank_score = (
                0.55 * fused_component
                + 0.30 * dense_component
                + 0.15 * lexical_component
                + 0.04 * body_overlap_ratio
                + 0.04 * answer_window_overlap_ratio
                + answer_support_bonus
                + modality_bonus
                + temporal_bonus
            )

        candidate.relevance_score = min(
            1.0,
            candidate.dense_relevance
            + 0.22 * lexical_component
            + 0.18 * overlap_ratio
            + 0.08 * body_overlap_ratio
            + 0.08 * answer_window_overlap_ratio
            + 0.10 * phrase_bonus
            + 0.15 * normalized_bonus
            + 0.05 * source_bonus
            + 0.50 * answer_support_bonus,
        )
        candidate.relevance_score = max(
            -0.25,
            min(1.0, candidate.relevance_score + temporal_bonus),
        )
        candidate.body_overlap_ratio = body_overlap_ratio
        candidate.answer_window_overlap_ratio = answer_window_overlap_ratio
        candidate.answer_support_bonus = answer_support_bonus
        candidate.answer_focus_score = answer_focus_score
        reranked.append(candidate)

    reranked.sort(key=lambda item: item.rerank_score, reverse=True)
    apply_same_sample_group_rerank_adjustments(reranked)
    reranked.sort(key=lambda item: item.rerank_score, reverse=True)
    cutoff = resolve_rerank_cutoff(settings, query_profile, top_k)
    return ensure_modality_coverage(
        reranked_candidates=reranked,
        required_modalities=resolve_required_modalities_for_query(query_profile),
        cutoff=cutoff,
    )


def diversify_candidates(
    candidates: list[RetrievalCandidate],
    target_count: int,
    query_profile: QueryModalityProfile,
    diversity_profile: DiversityQueryProfile,
) -> list[RetrievalCandidate]:
    dominant_group_candidates = select_dominant_sample_group_candidates(
        candidates,
        target_count=target_count,
    )
    if dominant_group_candidates:
        return dominant_group_candidates

    if diversity_profile.prefer_family_diversity:
        diversified_family_candidates = select_family_diverse_candidates(
            candidates,
            target_count=target_count,
            query_profile=query_profile,
        )
        if diversified_family_candidates:
            return diversified_family_candidates

    selected: list[RetrievalCandidate] = []
    reserve: list[RetrievalCandidate] = []
    seen_doc_ids: set[str] = set()

    required_modalities = resolve_required_modalities_for_query(query_profile)
    for required_modality in required_modalities:
        for item in candidates:
            doc_id = get_document_doc_id(item.document)
            source_modality = get_source_modality(item.document)
            if source_modality != required_modality:
                continue
            if doc_id and doc_id in seen_doc_ids:
                continue
            selected.append(item)
            if doc_id:
                seen_doc_ids.add(doc_id)
            break
        if len(selected) >= target_count:
            return selected[:target_count]

    for item in candidates:
        doc_id = get_document_doc_id(item.document)
        if doc_id and doc_id not in seen_doc_ids:
            selected.append(item)
            seen_doc_ids.add(doc_id)
        else:
            reserve.append(item)
        if len(selected) >= target_count:
            return selected[:target_count]

    for item in reserve:
        selected.append(item)
        if len(selected) >= target_count:
            break
    return selected[:target_count]


def select_family_diverse_candidates(
    candidates: list[RetrievalCandidate],
    *,
    target_count: int,
    query_profile: QueryModalityProfile,
) -> list[RetrievalCandidate]:
    selected: list[RetrievalCandidate] = []
    reserve: list[RetrievalCandidate] = []
    seen_doc_ids: set[str] = set()
    seen_family_ids: set[str] = set()

    required_modalities = resolve_required_modalities_for_query(query_profile)
    for required_modality in required_modalities:
        for item in candidates:
            doc_id = get_document_doc_id(item.document)
            family_id = get_document_family_id(item.document)
            source_modality = get_source_modality(item.document)
            if source_modality != required_modality:
                continue
            if doc_id and doc_id in seen_doc_ids:
                continue
            if family_id and family_id in seen_family_ids:
                continue
            selected.append(item)
            if doc_id:
                seen_doc_ids.add(doc_id)
            if family_id:
                seen_family_ids.add(family_id)
            break
        if len(selected) >= target_count:
            return selected[:target_count]

    for item in candidates:
        doc_id = get_document_doc_id(item.document)
        family_id = get_document_family_id(item.document)
        if doc_id and doc_id in seen_doc_ids:
            reserve.append(item)
            continue
        if family_id and family_id in seen_family_ids:
            reserve.append(item)
            continue
        selected.append(item)
        if doc_id:
            seen_doc_ids.add(doc_id)
        if family_id:
            seen_family_ids.add(family_id)
        if len(selected) >= target_count:
            return selected[:target_count]

    for item in candidates:
        if item in selected or item in reserve:
            continue
        reserve.append(item)

    for item in reserve:
        doc_id = get_document_doc_id(item.document)
        if doc_id and doc_id in seen_doc_ids:
            continue
        selected.append(item)
        if doc_id:
            seen_doc_ids.add(doc_id)
        if len(selected) >= target_count:
            break
    return selected[:target_count]


def candidate_to_reference(
    *,
    settings: AppSettings,
    candidate: RetrievalCandidate,
    grouped_documents: dict[str, dict[int, Document]],
) -> RetrievedReference:
    expanded_content = build_expanded_content(
        settings=settings,
        document=candidate.document,
        grouped_documents=grouped_documents,
    )
    raw_score = (
        float(candidate.dense_distance)
        if candidate.dense_distance is not None
        else float(max(0.0, 1.0 - candidate.relevance_score))
    )

    return RetrievedReference(
        chunk_id=str(candidate.document.metadata.get("chunk_id", "")),
        source=str(candidate.document.metadata.get("source", "")),
        source_path=str(candidate.document.metadata.get("source_path", "")),
        extension=str(candidate.document.metadata.get("extension", "")),
        page=_coerce_page(candidate.document.metadata.get("page")),
        page_end=_coerce_page(candidate.document.metadata.get("page_end")),
        title=candidate.document.metadata.get("title"),
        section_title=candidate.document.metadata.get("section_title"),
        section_path=candidate.document.metadata.get("section_path"),
        section_index=_coerce_page(candidate.document.metadata.get("section_index")),
        content_type=coerce_optional_text(candidate.document.metadata.get("content_type")),
        source_modality=coerce_optional_text(candidate.document.metadata.get("source_modality")),
        evidence_type=resolve_reference_evidence_type(candidate.document),
        used_for_answer=True,
        original_file_type=coerce_optional_text(candidate.document.metadata.get("original_file_type")),
        ocr_text=coerce_optional_text(candidate.document.metadata.get("ocr_text")),
        image_caption=coerce_optional_text(candidate.document.metadata.get("image_caption")),
        evidence_summary=coerce_optional_text(candidate.document.metadata.get("evidence_summary")),
        headers=extract_document_headers(candidate.document),
        content=expanded_content,
        content_preview=expanded_content[:200],
        raw_score=raw_score,
        relevance_score=float(candidate.relevance_score),
    )


def build_expanded_content(
    *,
    settings: AppSettings,
    document: Document,
    grouped_documents: dict[str, dict[int, Document]],
) -> str:
    if not settings.kb.ENABLE_SMALL_TO_BIG_CONTEXT:
        return document.page_content

    doc_id = get_document_doc_id(document)
    chunk_index = _coerce_page(document.metadata.get("chunk_index"))
    if not doc_id or chunk_index is None:
        return document.page_content

    available = grouped_documents.get(doc_id, {})
    if not available:
        return document.page_content

    pieces: list[str] = []
    seen_texts: set[str] = set()
    expand_chunks = settings.kb.SMALL_TO_BIG_EXPAND_CHUNKS
    for index in range(chunk_index - expand_chunks, chunk_index + expand_chunks + 1):
        item = available.get(index)
        if item is None:
            continue
        text = item.page_content.strip()
        if not text or text in seen_texts:
            continue
        pieces.append(text)
        seen_texts.add(text)

    return "\n".join(pieces) if pieces else document.page_content


def load_all_documents(vector_store: BaseVectorStoreAdapter) -> dict[str, Document]:
    return vector_store.load_all_documents()


def filter_documents_by_metadata(
    all_documents: dict[str, Document],
    metadata_filters: MetadataFilters | None,
) -> dict[str, Document]:
    if metadata_filters is None or not metadata_filters.filters:
        return all_documents
    return {
        chunk_id: document
        for chunk_id, document in all_documents.items()
        if matches_metadata_filters(document.metadata, metadata_filters)
    }


def group_documents_by_source_modality(
    all_documents: dict[str, Document],
) -> dict[str, dict[str, Document]]:
    grouped: dict[str, dict[str, Document]] = defaultdict(dict)
    for chunk_id, document in all_documents.items():
        grouped[get_source_modality(document)][chunk_id] = document
    return dict(grouped)


def group_documents_by_doc_id(all_documents: dict[str, Document]) -> dict[str, dict[int, Document]]:
    grouped: dict[str, dict[int, Document]] = defaultdict(dict)
    for document in all_documents.values():
        doc_id = get_document_doc_id(document)
        chunk_index = _coerce_page(document.metadata.get("chunk_index"))
        if not doc_id or chunk_index is None:
            continue
        grouped[doc_id][chunk_index] = document
    return grouped


def build_lexical_doc_infos(all_documents: dict[str, Document]) -> dict[str, dict[str, object]]:
    infos: dict[str, dict[str, object]] = {}
    for chunk_id, document in all_documents.items():
        raw_text = build_search_text(document)
        terms = build_match_terms([raw_text], deduplicate=False)
        term_counter = Counter(terms)
        infos[chunk_id] = {
            "raw_text": raw_text.lower(),
            "normalized_text": normalize_search_text(raw_text),
            "term_counter": term_counter,
            "unique_terms": set(term_counter),
            "length": sum(term_counter.values()) or 1,
        }
    return infos


def build_query_bundle(query_candidates: list[str]) -> list[str]:
    queries: list[str] = []
    for item in query_candidates:
        normalized = item.strip()
        if normalized and normalized not in queries:
            queries.append(normalized)
    for expanded in build_image_query_expansions(queries):
        if expanded not in queries:
            queries.append(expanded)
    return queries


def build_dense_query_bundle(query_bundle: list[str], hypothetical_doc: str) -> list[str]:
    dense_queries = list(query_bundle)
    normalized_hypothetical_doc = hypothetical_doc.strip()
    if normalized_hypothetical_doc and normalized_hypothetical_doc not in dense_queries:
        dense_queries.append(normalized_hypothetical_doc)
    return dense_queries


def build_image_query_expansions(query_bundle: list[str]) -> list[str]:
    profile = infer_query_modality_profile(query_bundle)
    if profile.query_type not in {"image_related", "multimodal_joint"}:
        return []

    expansions: list[str] = []
    base_queries = [item.strip() for item in query_bundle if item.strip()]
    suffixes = (
        "图片内容",
        "图像描述",
        "图片文字",
        "ocr文字",
        "画面信息",
    )
    for query in base_queries[:2]:
        lowered = query.lower()
        if any(suffix.lower() in lowered for suffix in suffixes):
            continue
        expansions.extend(f"{query} {suffix}" for suffix in suffixes[:3])
        if profile.query_type == "multimodal_joint":
            expansions.append(f"{query} 文档内容")
            expansions.append(f"{query} 图片内容")
    return deduplicate_strings(expansions)


def infer_query_modality_profile(query_bundle: list[str]) -> QueryModalityProfile:
    combined = " ".join(item.strip().lower() for item in query_bundle if item.strip())
    primary_query = next((item.strip().lower() for item in query_bundle if item.strip()), "")
    if not combined:
        return QueryModalityProfile(
            query_type="text_related",
            preferred_modalities=("text",),
            modality_bonus={"text": 0.04, "ocr": 0.0, "vision": -0.02, "ocr+vision": -0.01, "image": -0.02},
            preferred_extensions=(".txt", ".md", ".pdf", ".docx", ".epub"),
            extension_bonus={".txt": 0.02, ".md": 0.02, ".pdf": 0.02, ".docx": 0.02, ".epub": 0.02},
            path_hint_terms=(),
        )

    image_hits = sum(1 for keyword in IMAGE_QUERY_HINTS if keyword in combined)
    text_hits = sum(1 for keyword in TEXT_QUERY_HINTS if keyword in combined)
    if looks_like_multimodal_joint_query(primary_query):
        return QueryModalityProfile(
            query_type="multimodal_joint",
            preferred_modalities=("text", "vision", "image", "ocr", "ocr+vision"),
            modality_bonus={"text": 0.04, "ocr": 0.04, "ocr+vision": 0.05, "vision": 0.04, "image": 0.04},
            preferred_extensions=(".txt", ".md", ".pdf", ".docx", ".epub", ".png", ".jpg", ".jpeg", ".bmp", ".webp"),
            extension_bonus={".txt": 0.02, ".md": 0.02, ".pdf": 0.02, ".docx": 0.02, ".epub": 0.02, ".png": 0.02, ".jpg": 0.02, ".jpeg": 0.02, ".bmp": 0.02, ".webp": 0.02},
            path_hint_terms=("图片", "图像", "照片", "文档", "书籍", "内容"),
        )
    if image_hits > text_hits:
        return QueryModalityProfile(
            query_type="image_related",
            preferred_modalities=("ocr", "vision", "ocr+vision", "image", "text"),
            modality_bonus={"ocr": 0.05, "vision": 0.05, "ocr+vision": 0.06, "image": 0.03, "text": 0.0},
            preferred_extensions=(".png", ".jpg", ".jpeg", ".bmp", ".webp", ".pdf"),
            extension_bonus={".png": 0.05, ".jpg": 0.05, ".jpeg": 0.05, ".bmp": 0.05, ".webp": 0.05, ".pdf": 0.01, ".epub": -0.03, ".docx": -0.03},
            path_hint_terms=("图片", "图像", "截图", "照片", "ocr", "图中", "画面"),
        )
    return QueryModalityProfile(
        query_type="text_related",
        preferred_modalities=("text", "ocr", "ocr+vision", "vision", "image"),
        modality_bonus={"text": 0.05, "ocr": 0.02, "ocr+vision": 0.01, "vision": -0.02, "image": -0.02},
        preferred_extensions=(".txt", ".md", ".pdf", ".docx", ".epub"),
        extension_bonus={".txt": 0.02, ".md": 0.03, ".pdf": 0.03, ".docx": 0.03, ".epub": 0.03, ".png": -0.03, ".jpg": -0.03, ".jpeg": -0.03, ".bmp": -0.03, ".webp": -0.03},
        path_hint_terms=extract_path_hint_terms_from_queries(query_bundle),
    )


def select_modalities_for_query(
    modality_groups: dict[str, dict[str, Document]],
    query_profile: QueryModalityProfile,
) -> list[str]:
    selected: list[str] = []
    for source_modality in query_profile.preferred_modalities:
        if source_modality in modality_groups and source_modality not in selected:
            selected.append(source_modality)
    for source_modality in modality_groups:
        if source_modality not in selected:
            selected.append(source_modality)
    return selected


def should_use_modality_grouped_dense_retrieval(
    modality_groups: dict[str, dict[str, Document]],
    all_documents: dict[str, Document],
) -> bool:
    if not modality_groups or len(modality_groups) <= 1:
        return False
    explicit_count = sum(
        1
        for document in all_documents.values()
        if isinstance(document.metadata.get("source_modality"), str)
        and str(document.metadata.get("source_modality")).strip()
    )
    return explicit_count / max(1, len(all_documents)) >= 0.9


def merge_metadata_filters_with_source_modality(
    metadata_filters: MetadataFilters | None,
    source_modality: str,
) -> MetadataFilters:
    modality_filter = MetadataFilter(
        key="source_modality",
        operator=FilterOperator.EQ,
        value=source_modality,
    )
    if metadata_filters is None or not metadata_filters.filters:
        return MetadataFilters(filters=[modality_filter])
    return MetadataFilters(
        condition=FilterCondition.AND,
        filters=[*metadata_filters.filters, modality_filter],
    )


def modality_bonus_for_candidate(
    document: Document,
    query_profile: QueryModalityProfile,
) -> float:
    return (
        query_profile.modality_bonus.get(get_source_modality(document), 0.0)
        + extension_bonus_for_candidate(document, query_profile)
        + path_bonus_for_candidate(document, query_profile)
    )


def extension_bonus_for_candidate(
    document: Document,
    query_profile: QueryModalityProfile,
) -> float:
    extension = str(document.metadata.get("extension", "")).strip().lower()
    return query_profile.extension_bonus.get(extension, 0.0)


def path_bonus_for_candidate(
    document: Document,
    query_profile: QueryModalityProfile,
) -> float:
    if not query_profile.path_hint_terms:
        return 0.0

    title = str(document.metadata.get("title", "")).lower()
    section_title = str(document.metadata.get("section_title", "")).lower()
    section_path = str(document.metadata.get("section_path", "")).lower()
    source = str(document.metadata.get("source", "")).lower()
    combined = " ".join(item for item in (title, section_title, section_path, source) if item)
    if not combined:
        return 0.0

    matched = sum(1 for term in query_profile.path_hint_terms if term and term in combined)
    if matched == 0:
        return 0.0
    return min(0.06, 0.02 * matched)


def build_search_text(document: Document) -> str:
    return build_search_text_from_parts(
        page_content=document.page_content,
        metadata=document.metadata,
        headers=extract_document_headers(document),
    )


def build_match_terms(texts: list[str], deduplicate: bool = True) -> list[str]:
    return build_bm25_match_terms(texts, deduplicate=deduplicate)


def compute_body_overlap_ratio(page_text: str, query_term_set: set[str]) -> float:
    if not query_term_set:
        return 0.0
    page_terms = set(build_match_terms([page_text]))
    if not page_terms:
        return 0.0
    return len(page_terms & query_term_set) / len(query_term_set)


def compute_answer_support_bonus(primary_query: str, page_text: str) -> float:
    query = str(primary_query or "").strip().lower()
    if not query or not any(hint in query for hint in ANSWER_SEEKING_HINTS):
        return 0.0

    text = str(page_text or "")
    lowered = text.lower()
    bonus = 0.0

    matched_label_term = next((term for term in ANSWER_LABEL_TERMS if term in query), "")
    if matched_label_term:
        if any(
            marker in lowered
            for marker in (
                f"{matched_label_term.lower()}：",
                f"{matched_label_term.lower()}:",
                f"{matched_label_term.lower()}是",
                f"{matched_label_term.lower()}为",
            )
        ):
            bonus += 0.06

    if any(hint in query for hint in ("多少", "几", "几项", "几家", "几篇", "多少人", "招收多少", "降多少")):
        if ANSWER_NUMERIC_PATTERN.search(text):
            bonus += 0.05

    if any(hint in query for hint in ("哪一年", "哪年", "何时", "什么时候", "开始时间", "什么时间")):
        if extract_years_from_text(text) or extract_date_ordinals_from_text(text):
            bonus += 0.05

    if "等级" in query and ANSWER_GRADE_PATTERN.search(text):
        bonus += 0.05
        if any(marker in lowered for marker in ("被评为", "评为", "获评")):
            bonus += 0.03

    if any(hint in query for hint in ("谁", "哪位", "何人")):
        if any(role in text for role in ANSWER_ROLE_TERMS):
            bonus += 0.05

    if "语言" in query:
        language_hits = sum(1 for term in LANGUAGE_TERMS if term in text)
        if language_hits >= 2:
            bonus += 0.06

    if any(hint in query for hint in LIST_QUERY_HINTS):
        if any(separator in text for separator in ("、", "；", ";", "以及", "分别")):
            bonus += 0.03

    return min(0.14, bonus)


def should_focus_answer_window(primary_query: str) -> bool:
    query = str(primary_query or "").strip().lower()
    if not query:
        return False
    return any(hint in query for hint in ANSWER_SEEKING_HINTS)


def should_use_sentence_index(query_bundle: list[str]) -> bool:
    primary_query = next((item.lower().strip() for item in query_bundle if item.strip()), "")
    if not should_focus_answer_window(primary_query):
        return False
    if any(term in primary_query for term in TEMPORAL_QUERY_HINTS + TEMPORAL_RECENCY_HINTS):
        return False
    if any(hint in primary_query for hint in MULTI_DOC_QUERY_HINTS):
        return False
    connector_hits = sum(primary_query.count(connector) for connector in MULTI_DOC_CONNECTORS)
    if connector_hits >= 2:
        return False
    if any(term in primary_query for term in ANSWER_LABEL_TERMS):
        return True
    if "语言" in primary_query:
        return True
    strong_extractive_hints = (
        "多少",
        "几",
        "哪一年",
        "哪年",
        "何时",
        "什么时候",
        "开始时间",
        "什么时间",
        "什么等级",
        "哪三个",
        "哪几",
        "哪些",
        "哪个",
    )
    return any(hint in primary_query for hint in strong_extractive_hints)


def build_candidate_rerank_text(
    document: Document,
    primary_query: str,
    query_term_set: set[str],
) -> str:
    page_text = str(document.page_content or "")
    sentence_hint = str(document.metadata.get("sentence_query_hit", "") or "").strip()
    answer_window_text = build_answer_window_text(primary_query, page_text, query_term_set)
    body_text = select_rerank_body_text(
        primary_query=primary_query,
        page_text=page_text,
        query_term_set=query_term_set,
        sentence_hint=sentence_hint,
        answer_window_text=answer_window_text,
    )

    title = pick_rerank_title(document)
    doc_type = infer_rerank_doc_type(document)
    section_text = pick_rerank_section(document)
    date_text = sanitize_rerank_text(document.metadata.get("date", ""), max_chars=32)
    summary_text = pick_rerank_summary(document, page_text)
    sentence_text = sanitize_rerank_text(sentence_hint, max_chars=RERANK_SUMMARY_MAX_CHARS)

    lines: list[str] = []
    if title:
        lines.append(f"[标题] {title}")
    if doc_type:
        lines.append(f"[类型] {doc_type}")
    if section_text:
        lines.append(f"[章节] {section_text}")
    if date_text:
        lines.append(f"[日期] {date_text}")
    if summary_text:
        lines.append(f"[摘要] {summary_text}")
    if sentence_text and sentence_text != body_text:
        lines.append(f"[命中句] {sentence_text}")
    if body_text:
        lines.append(f"[正文] {body_text}")

    if lines:
        return "\n".join(lines)

    return sanitize_rerank_text(page_text, max_chars=RERANK_BODY_MAX_CHARS)


def pick_rerank_title(document: Document) -> str:
    title = sanitize_rerank_text(document.metadata.get("title", ""), max_chars=RERANK_FIELD_MAX_CHARS)
    if title:
        return title
    section_title = sanitize_rerank_text(document.metadata.get("section_title", ""), max_chars=RERANK_FIELD_MAX_CHARS)
    if section_title:
        return section_title
    headers = extract_document_headers(document)
    for value in headers.values():
        candidate = sanitize_rerank_text(value, max_chars=RERANK_FIELD_MAX_CHARS)
        if candidate:
            return candidate
    return ""


def pick_rerank_section(document: Document) -> str:
    section_path = sanitize_rerank_text(document.metadata.get("section_path", ""), max_chars=RERANK_SUMMARY_MAX_CHARS)
    section_title = sanitize_rerank_text(document.metadata.get("section_title", ""), max_chars=RERANK_SUMMARY_MAX_CHARS)
    title = sanitize_rerank_text(document.metadata.get("title", ""), max_chars=RERANK_SUMMARY_MAX_CHARS)
    for candidate in (section_path, section_title):
        if candidate and candidate != title:
            return candidate
    return ""


def pick_rerank_summary(document: Document, page_text: str) -> str:
    summary = sanitize_rerank_text(document.metadata.get("evidence_summary", ""), max_chars=RERANK_SUMMARY_MAX_CHARS)
    title = sanitize_rerank_text(document.metadata.get("title", ""), max_chars=RERANK_SUMMARY_MAX_CHARS)
    section_title = sanitize_rerank_text(document.metadata.get("section_title", ""), max_chars=RERANK_SUMMARY_MAX_CHARS)
    if summary and summary not in {title, section_title}:
        return summary
    if not page_text:
        return ""
    return sanitize_rerank_text(page_text, max_chars=RERANK_SUMMARY_MAX_CHARS)


def infer_rerank_doc_type(document: Document) -> str:
    content_type = sanitize_rerank_text(document.metadata.get("content_type", ""), max_chars=48).lower()
    source_modality = sanitize_rerank_text(document.metadata.get("source_modality", ""), max_chars=24).lower()

    content_type_map = {
        "document_text": "文档正文",
        "image_evidence": "图像证据",
        "image_region_evidence": "图像区域证据",
        "instruction_text_evidence": "操作说明文本",
        "instruction_parts_evidence": "零件清单",
        "instruction_figure_evidence": "图示说明",
        "instruction_arrow_evidence": "图示标注",
        "web_search_result": "网络结果",
    }
    modality_map = {
        "text": "文本证据",
        "ocr": "OCR 文本",
        "vision": "视觉描述",
        "image": "图像内容",
        "ocr+vision": "图文联合证据",
    }

    if content_type in content_type_map:
        return content_type_map[content_type]
    if source_modality in modality_map:
        return modality_map[source_modality]
    if content_type:
        return content_type
    if source_modality:
        return source_modality
    return "文本片段"


def select_rerank_body_text(
    *,
    primary_query: str,
    page_text: str,
    query_term_set: set[str],
    sentence_hint: str,
    answer_window_text: str,
) -> str:
    focused_answer_text = ""
    if answer_window_text and answer_window_text != page_text:
        focused_answer_text = answer_window_text

    focused_window_text = build_query_focused_window_text(
        primary_query=primary_query,
        page_text=page_text,
        query_term_set=query_term_set,
        max_chars=RERANK_BODY_MAX_CHARS,
    )

    body_text = focused_answer_text or focused_window_text or sanitize_rerank_text(
        page_text,
        max_chars=RERANK_BODY_MAX_CHARS,
    )
    sentence_text = sanitize_rerank_text(sentence_hint, max_chars=RERANK_SUMMARY_MAX_CHARS)
    if sentence_text and sentence_text not in body_text:
        combined = f"{sentence_text} {body_text}".strip()
        return sanitize_rerank_text(combined, max_chars=RERANK_BODY_MAX_CHARS)
    return body_text


def build_query_focused_window_text(
    *,
    primary_query: str,
    page_text: str,
    query_term_set: set[str],
    max_chars: int,
) -> str:
    text = str(page_text or "").strip()
    if not text:
        return ""

    segments = split_answer_segments(text)
    if not segments:
        return sanitize_rerank_text(text, max_chars=max_chars)

    scored_segments = [
        (index, segment, score_rerank_segment(primary_query, segment, query_term_set))
        for index, segment in enumerate(segments)
    ]
    best_index, _, best_score = max(scored_segments, key=lambda item: item[2], default=(-1, "", 0.0))
    if best_index < 0:
        return sanitize_rerank_text(text, max_chars=max_chars)
    if best_score <= 0:
        return sanitize_rerank_text(text, max_chars=max_chars)

    selected_indices = {best_index}
    current_length = len(segments[best_index])
    left = best_index - 1
    right = best_index + 1
    score_map = {index: score for index, _, score in scored_segments}

    while current_length < max_chars and (left >= 0 or right < len(segments)):
        candidates: list[tuple[float, int]] = []
        if left >= 0:
            candidates.append((score_map.get(left, 0.0), left))
        if right < len(segments):
            candidates.append((score_map.get(right, 0.0), right))
        if not candidates:
            break
        candidates.sort(key=lambda item: (item[0], -abs(item[1] - best_index)), reverse=True)
        neighbor_score, neighbor_index = candidates[0]
        if current_length >= 180 and neighbor_score <= 0:
            break
        neighbor_text = segments[neighbor_index]
        projected_length = current_length + len(neighbor_text) + 1
        if projected_length > max_chars and current_length >= 180:
            break
        selected_indices.add(neighbor_index)
        current_length = projected_length
        if neighbor_index == left:
            left -= 1
        elif neighbor_index == right:
            right += 1

    ordered_segments = [segments[index] for index in sorted(selected_indices)]
    combined = " ".join(item for item in ordered_segments if item).strip()
    return sanitize_rerank_text(combined or text, max_chars=max_chars)


def score_rerank_segment(primary_query: str, segment: str, query_term_set: set[str]) -> float:
    text = str(segment or "").strip()
    if not text:
        return 0.0
    overlap_ratio = compute_body_overlap_ratio(text, query_term_set)
    answer_support_bonus = compute_answer_support_bonus(primary_query, text)
    return overlap_ratio + 0.8 * answer_support_bonus


def sanitize_rerank_text(text: object, max_chars: int) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    compact = re.sub(r"\s+", " ", raw)
    compact = compact.strip(" -:：|")
    if len(compact) <= max_chars:
        return compact
    clipped = compact[: max(1, max_chars - 1)].rstrip()
    return f"{clipped}…"


def build_answer_window_text(
    primary_query: str,
    page_text: str,
    query_term_set: set[str],
    max_chars: int = 320,
) -> str:
    text = str(page_text or "").strip()
    if not text or not should_focus_answer_window(primary_query):
        return text

    segments = split_answer_segments(text)
    if not segments:
        return text

    scored_segments = [
        (index, segment, score_answer_segment(primary_query, segment, query_term_set))
        for index, segment in enumerate(segments)
    ]
    best_index, _, best_score = max(scored_segments, key=lambda item: item[2], default=(-1, "", 0.0))
    if best_index < 0 or best_score < 0.08:
        return text

    selected_indices = {best_index}
    current_length = len(segments[best_index])
    left = best_index - 1
    right = best_index + 1
    score_map = {index: score for index, _, score in scored_segments}

    while current_length < max_chars and (left >= 0 or right < len(segments)):
        candidates: list[tuple[float, int]] = []
        if left >= 0:
            candidates.append((score_map.get(left, 0.0), left))
        if right < len(segments):
            candidates.append((score_map.get(right, 0.0), right))
        if not candidates:
            break
        candidates.sort(key=lambda item: (item[0], -abs(item[1] - best_index)), reverse=True)
        neighbor_score, neighbor_index = candidates[0]
        neighbor_text = segments[neighbor_index]
        if current_length >= 160 and neighbor_score <= 0:
            break
        projected_length = current_length + len(neighbor_text) + 1
        if projected_length > max_chars and current_length >= 160:
            break
        selected_indices.add(neighbor_index)
        current_length = projected_length
        if neighbor_index == left:
            left -= 1
        elif neighbor_index == right:
            right += 1

    ordered_segments = [segments[index] for index in sorted(selected_indices)]
    return "\n".join(item for item in ordered_segments if item).strip() or text


def split_answer_segments(text: str) -> list[str]:
    parts = [part.strip() for part in ANSWER_WINDOW_SPLIT_PATTERN.split(str(text or ""))]
    return [part for part in parts if len(part) >= 2]


def score_answer_segment(primary_query: str, segment: str, query_term_set: set[str]) -> float:
    text = str(segment or "").strip()
    if not text:
        return 0.0
    overlap_ratio = compute_body_overlap_ratio(text, query_term_set)
    answer_support_bonus = compute_answer_support_bonus(primary_query, text)
    return overlap_ratio + answer_support_bonus


def normalize_search_text(text: str) -> str:
    return normalize_bm25_search_text(text)


def infer_diversity_query_profile(query_bundle: list[str]) -> DiversityQueryProfile:
    combined = " ".join(item.strip().lower() for item in query_bundle if item.strip())
    connector_hits = sum(combined.count(connector) for connector in MULTI_DOC_CONNECTORS)
    hint_hits = sum(1 for hint in MULTI_DOC_QUERY_HINTS if hint in combined)
    prefer_family_diversity = hint_hits > 0 and connector_hits > 0
    return DiversityQueryProfile(prefer_family_diversity=prefer_family_diversity)


def resolve_rerank_model_selection(
    settings: AppSettings,
    query_bundle: list[str],
    query_profile: QueryModalityProfile,
) -> RerankModelSelection:
    default_model = settings.model.RERANK_MODEL.strip()
    primary_query = next((item.lower().strip() for item in query_bundle if item.strip()), "")
    temporal_profile = infer_temporal_query_profile(query_bundle)
    diversity_profile = infer_diversity_query_profile(query_bundle)
    answer_focused_model = settings.model.RERANK_MODEL_ANSWER_FOCUSED.strip()
    multi_doc_model = settings.model.RERANK_MODEL_MULTI_DOC.strip()
    temporal_model = settings.model.RERANK_MODEL_TEMPORAL.strip()

    if temporal_profile.is_temporal and temporal_model:
        return RerankModelSelection(model_name=temporal_model, route="temporal")
    if diversity_profile.prefer_family_diversity and multi_doc_model:
        return RerankModelSelection(model_name=multi_doc_model, route="multi_doc")
    if should_focus_answer_window(primary_query) and answer_focused_model:
        return RerankModelSelection(model_name=answer_focused_model, route="answer_focused")
    return RerankModelSelection(
        model_name=default_model,
        route=query_profile.query_type or "default",
    )


def infer_temporal_query_profile(query_bundle: list[str]) -> TemporalQueryProfile:
    combined = " ".join(item.strip().lower() for item in query_bundle if item.strip())
    explicit_years = tuple(sorted(set(extract_years_from_text(combined))))
    explicit_dates = tuple(sorted(set(extract_date_ordinals_from_text(combined))))
    prefers_recent = any(term in combined for term in TEMPORAL_RECENCY_HINTS)
    is_temporal = (
        prefers_recent
        or bool(explicit_years)
        or bool(explicit_dates)
        or any(term in combined for term in TEMPORAL_QUERY_HINTS)
    )
    return TemporalQueryProfile(
        is_temporal=is_temporal,
        prefers_recent=prefers_recent,
        explicit_years=explicit_years,
        explicit_dates=explicit_dates,
    )


def build_temporal_candidate_adjustments(
    candidates: list[RetrievalCandidate],
    temporal_profile: TemporalQueryProfile,
) -> dict[str, float]:
    if not temporal_profile.is_temporal:
        return {}

    query_years = set(temporal_profile.explicit_years)
    query_dates = set(temporal_profile.explicit_dates)
    candidate_years: dict[str, set[int]] = {}
    candidate_anchors: dict[str, int | None] = {}
    anchor_values: list[int] = []

    for candidate in candidates:
        chunk_id = get_chunk_id(candidate.document)
        text = build_search_text(candidate.document)
        years = set(extract_years_from_text(text))
        anchor = extract_document_temporal_anchor(candidate.document, fallback_text=text)
        candidate_years[chunk_id] = years
        candidate_anchors[chunk_id] = anchor
        if anchor is not None:
            anchor_values.append(anchor)

    oldest_anchor = min(anchor_values) if anchor_values else None
    newest_anchor = max(anchor_values) if anchor_values else None

    adjustments: dict[str, float] = {}
    for candidate in candidates:
        chunk_id = get_chunk_id(candidate.document)
        years = candidate_years.get(chunk_id, set())
        anchor = candidate_anchors.get(chunk_id)
        score = 0.0

        if query_years:
            overlap = len(years & query_years) / max(1, len(query_years))
            if overlap > 0:
                score += 0.08 + 0.08 * overlap
            elif years:
                score -= 0.05

        if query_dates and anchor is not None:
            if any(abs(anchor - query_date) <= 3 for query_date in query_dates):
                score += 0.08

        if temporal_profile.prefers_recent and anchor is not None:
            if newest_anchor is not None and oldest_anchor is not None and newest_anchor > oldest_anchor:
                recency = (anchor - oldest_anchor) / max(1, newest_anchor - oldest_anchor)
                score += 0.10 * recency
            else:
                score += 0.04
        elif anchor is not None:
            score += 0.03
        elif years:
            score += 0.01
        else:
            score -= 0.02

        adjustments[chunk_id] = score
    return adjustments


def extract_document_temporal_anchor(
    document: Document,
    *,
    fallback_text: str = "",
) -> int | None:
    metadata_date = str(document.metadata.get("date", "")).strip()
    if metadata_date:
        ordinals = extract_date_ordinals_from_text(metadata_date)
        if ordinals:
            return max(ordinals)

    text_candidates = [
        str(document.metadata.get("title", "")).strip(),
        str(document.metadata.get("section_title", "")).strip(),
        str(document.metadata.get("source", "")).strip(),
        document.page_content[:1200],
        fallback_text[:1200],
    ]
    combined = "\n".join(item for item in text_candidates if item)
    if not combined:
        return None

    labeled_ordinals = extract_date_ordinals_from_text(combined, prefer_labeled=True)
    if labeled_ordinals:
        return max(labeled_ordinals)

    exact_ordinals = extract_date_ordinals_from_text(combined)
    if exact_ordinals:
        return max(exact_ordinals)

    years = extract_years_from_text(combined)
    if years:
        return date(max(years), 12, 31).toordinal()
    return None


def extract_years_from_text(text: str) -> list[int]:
    return [int(match.group(1)) for match in YEAR_PATTERN.finditer(str(text or ""))]


def extract_date_ordinals_from_text(
    text: str,
    *,
    prefer_labeled: bool = False,
) -> list[int]:
    source = str(text or "")
    pattern = LABELED_DATE_PATTERN if prefer_labeled else DATE_PATTERN
    ordinals: list[int] = []
    for match in pattern.finditer(source):
        ordinal = coerce_date_ordinal(
            match.group(1),
            match.group(2),
            match.group(3),
        )
        if ordinal is not None:
            ordinals.append(ordinal)
    return ordinals


def coerce_date_ordinal(year_text: str, month_text: str, day_text: str) -> int | None:
    try:
        year = int(year_text)
        month = int(month_text)
        day = int(day_text)
        return date(year, month, day).toordinal()
    except (TypeError, ValueError):
        return None


def get_chunk_id(document: Document, fallback: str = "") -> str:
    return str(document.metadata.get("chunk_id") or fallback)


def get_document_doc_id(document: Document) -> str:
    value = document.metadata.get("doc_id") or document.metadata.get("relative_path") or document.metadata.get("source")
    return str(value or "")


def get_document_family_id(document: Document) -> str:
    for key in ("reference_id", "url", "title", "source"):
        value = str(document.metadata.get(key, "") or "").strip()
        normalized = normalize_search_text(value)
        if normalized:
            return normalized
    return ""


def get_document_sample_id(document: Document) -> str:
    value = document.metadata.get("sample_id")
    if isinstance(value, str) and value.strip():
        return value.strip()

    for key in ("doc_id", "relative_path", "source_path"):
        candidate = infer_sample_id_from_text(document.metadata.get(key))
        if candidate:
            return candidate
    return ""


def infer_sample_id_from_text(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""

    parts = [part.strip() for part in re.split(r"[\\/]+", text) if part.strip()]
    for part in reversed(parts):
        if SAMPLE_ID_PATTERN.fullmatch(part):
            return part
    return ""


def get_source_modality(document: Document) -> str:
    value = document.metadata.get("source_modality")
    if isinstance(value, str):
        normalized = value.strip()
        if normalized:
            return normalized
    return "text"


def resolve_reference_evidence_type(document: Document) -> str:
    source_modality = get_source_modality(document)
    if source_modality == "ocr":
        return "ocr"
    if source_modality in {"vision", "image"}:
        return "vision"
    if source_modality == "ocr+vision":
        return "multimodal"
    return "text"


def resolve_query_type_label(query_profile: QueryModalityProfile) -> str:
    return query_profile.query_type


def looks_like_multimodal_joint_query(combined_query: str) -> bool:
    joint_markers = ("既有", "也有", "同时", "一起", "以及", "又有", "既包含", "也包含")
    image_markers = ("图片", "图像", "照片", "截图")
    text_markers = ("书籍", "文档", "内容", "正文", "资料")
    if any(marker in combined_query for marker in joint_markers):
        if any(marker in combined_query for marker in image_markers):
            return True
    return any(marker in combined_query for marker in image_markers) and any(
        marker in combined_query for marker in text_markers
    )


def resolve_rerank_cutoff(
    settings: AppSettings,
    query_profile: QueryModalityProfile,
    top_k: int,
) -> int:
    base_cutoff = max(settings.kb.HYBRID_RERANK_TOP_K, top_k, 1)
    if query_profile.query_type == "multimodal_joint":
        return max(base_cutoff, top_k * 4, 12)
    if query_profile.query_type == "image_related":
        return max(base_cutoff, top_k * 3, 10)
    return base_cutoff


def resolve_required_modalities_for_query(
    query_profile: QueryModalityProfile,
) -> tuple[str, ...]:
    if query_profile.query_type == "image_related":
        return ("ocr", "vision", "ocr+vision", "image")
    if query_profile.query_type == "multimodal_joint":
        return ("text", "vision", "image", "ocr", "ocr+vision")
    return ()


def ensure_modality_coverage(
    *,
    reranked_candidates: list[RetrievalCandidate],
    required_modalities: tuple[str, ...],
    cutoff: int,
) -> list[RetrievalCandidate]:
    if not required_modalities:
        return reranked_candidates[:cutoff]

    selected: list[RetrievalCandidate] = []
    seen_chunk_ids: set[str] = set()
    for required_modality in required_modalities:
        for candidate in reranked_candidates:
            chunk_id = get_chunk_id(candidate.document)
            if chunk_id in seen_chunk_ids:
                continue
            if get_source_modality(candidate.document) != required_modality:
                continue
            selected.append(candidate)
            seen_chunk_ids.add(chunk_id)
            break
        if len(selected) >= cutoff:
            return selected[:cutoff]

    for candidate in reranked_candidates:
        chunk_id = get_chunk_id(candidate.document)
        if chunk_id in seen_chunk_ids:
            continue
        selected.append(candidate)
        seen_chunk_ids.add(chunk_id)
        if len(selected) >= cutoff:
            break
    return selected[:cutoff]


def apply_same_sample_group_rerank_adjustments(
    candidates: list[RetrievalCandidate],
) -> None:
    group_stats = build_sample_group_stats(candidates)
    if len(group_stats) < 2:
        return

    dominant_group = group_stats[0]
    if dominant_group.candidate_count < SAMPLE_GROUP_MIN_COUNT_FOR_RERANK:
        return

    dominant_score = max(dominant_group.aggregate_score, 1e-6)
    second_score = group_stats[1].aggregate_score if len(group_stats) > 1 else 0.0
    dominance_ratio = max(0.0, (dominant_score - second_score) / dominant_score)
    if dominance_ratio <= 0:
        return

    group_lookup = {item.sample_id: item for item in group_stats}
    for candidate in candidates:
        sample_id = get_document_sample_id(candidate.document)
        if not sample_id:
            continue

        stats = group_lookup.get(sample_id)
        if stats is None:
            continue

        group_ratio = stats.aggregate_score / dominant_score
        count_bonus = 0.02 * min(max(stats.candidate_count - 1, 0), 2)
        if sample_id == dominant_group.sample_id:
            boost = 0.06 + 0.08 * group_ratio + count_bonus + 0.10 * dominance_ratio
            candidate.rerank_score += boost
            candidate.relevance_score = min(
                1.0,
                candidate.relevance_score + 0.04 + 0.06 * dominance_ratio,
            )
            continue

        penalty = 0.04 + 0.08 * (1.0 - group_ratio) + 0.12 * dominance_ratio
        candidate.rerank_score -= penalty
        candidate.relevance_score = max(-0.25, candidate.relevance_score - penalty)


def select_dominant_sample_group_candidates(
    candidates: list[RetrievalCandidate],
    *,
    target_count: int,
) -> list[RetrievalCandidate]:
    if target_count < SAMPLE_GROUP_MIN_COUNT_FOR_TRIM:
        return []

    group_stats = build_sample_group_stats(candidates)
    if len(group_stats) < 2:
        return []

    dominant_group = group_stats[0]
    if dominant_group.candidate_count < min(SAMPLE_GROUP_MIN_COUNT_FOR_TRIM, target_count):
        return []

    dominant_score = max(dominant_group.aggregate_score, 1e-6)
    second_score = group_stats[1].aggregate_score if len(group_stats) > 1 else 0.0
    dominance_ratio = max(0.0, (dominant_score - second_score) / dominant_score)
    if dominance_ratio < SAMPLE_GROUP_DOMINANCE_RATIO_FOR_TRIM:
        return []

    selected = [
        item
        for item in candidates
        if get_document_sample_id(item.document) == dominant_group.sample_id
    ]
    return selected[:target_count]


def build_sample_group_stats(
    candidates: list[RetrievalCandidate],
) -> list[SampleGroupStats]:
    grouped_scores: dict[str, list[float]] = defaultdict(list)
    for candidate in candidates:
        sample_id = get_document_sample_id(candidate.document)
        if not sample_id:
            continue
        grouped_scores[sample_id].append(max(float(candidate.rerank_score), 0.0))

    stats: list[SampleGroupStats] = []
    for sample_id, scores in grouped_scores.items():
        if not scores:
            continue
        sorted_scores = sorted(scores, reverse=True)
        stats.append(
            SampleGroupStats(
                sample_id=sample_id,
                aggregate_score=sum(sorted_scores[:SAMPLE_GROUP_AGGREGATION_LIMIT]),
                max_score=sorted_scores[0],
                candidate_count=len(sorted_scores),
            )
        )
    stats.sort(
        key=lambda item: (item.aggregate_score, item.candidate_count, item.max_score),
        reverse=True,
    )
    return stats


def count_candidate_modalities(candidates: list[RetrievalCandidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in candidates:
        source_modality = get_source_modality(item.document)
        counts[source_modality] = counts.get(source_modality, 0) + 1
    return counts


def count_reference_attributes(
    references: list[RetrievedReference],
    attribute: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ref in references:
        value = getattr(ref, attribute, None)
        normalized = str(value).strip() if value is not None else ""
        key = normalized or "missing"
        counts[key] = counts.get(key, 0) + 1
    return counts


def append_retrieval_trace(
    *,
    settings: AppSettings,
    vector_store_dir: Path,
    query: str,
    rewritten_query: str,
    query_bundle: list[str],
    top_k: int,
    metadata_filters: MetadataFilters | None,
    diagnostics: dict[str, object],
    references: list[RetrievedReference],
) -> None:
    append_jsonl_trace(
        settings,
        "retrieval_trace",
        {
            "event_type": "retrieval",
            "knowledge_base_name": vector_store_dir.name,
            "query": query,
            "rewritten_query": rewritten_query,
            "query_bundle": query_bundle[:6],
            "query_bundle_count": diagnostics.get("query_bundle_count", len(query_bundle)),
            "dense_query_bundle_count": diagnostics.get("dense_query_bundle_count", len(query_bundle)),
            "top_k": top_k,
            "metadata_filter_count": len(metadata_filters.filters) if metadata_filters else 0,
            "hyde_enabled": diagnostics.get("hyde_enabled", False),
            "hyde_used": diagnostics.get("hyde_used", False),
            "hyde_preview": diagnostics.get("hyde_preview", ""),
            "bm25_index_available": diagnostics.get("bm25_index_available", False),
            "bm25_backend": diagnostics.get("bm25_backend", "dynamic_legacy"),
            "bm25_load_error": diagnostics.get("bm25_load_error", ""),
            "query_type": diagnostics.get("query_type", "unknown"),
            "preferred_modalities": diagnostics.get("preferred_modalities", []),
            "available_modalities": diagnostics.get("available_modalities", {}),
            "modality_grouped_dense_used": diagnostics.get("modality_grouped_dense_used", False),
            "candidate_count": diagnostics.get("candidate_count", 0),
            "candidate_modality_counts": diagnostics.get("candidate_modality_counts", {}),
            "rerank_model_selected": diagnostics.get("rerank_model_selected", ""),
            "rerank_model_route": diagnostics.get("rerank_model_route", ""),
            "final_reference_count": len(references),
            "final_source_modalities": count_reference_attributes(
                references[: settings.kb.TRACE_LOG_MAX_REFERENCES],
                "source_modality",
            ),
            "final_evidence_types": count_reference_attributes(
                references[: settings.kb.TRACE_LOG_MAX_REFERENCES],
                "evidence_type",
            ),
            "final_sources": [ref.source for ref in references[: settings.kb.TRACE_LOG_MAX_REFERENCES]],
        },
    )


def distance_to_relevance(distance: float) -> float:
    return 1.0 / (1.0 + float(distance))


def _coerce_page(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_path_hint_terms_from_queries(query_bundle: list[str]) -> tuple[str, ...]:
    hints: list[str] = []
    for query in query_bundle:
        stripped = query.strip()
        if not stripped:
            continue
        hints.extend(
            token
            for token in re.findall(r"[\u4e00-\u9fff]{2,}", stripped)
            if len(token) >= 2
        )
    return tuple(deduplicate_strings(hints)[:8])

