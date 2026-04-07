from __future__ import annotations

import re
from math import ceil
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log
from pathlib import Path

from langchain_core.documents import Document

from app.schemas.chat import ChatMessage, RetrievedReference
from app.services.embedding_service import build_embeddings
from app.services.observability import append_jsonl_trace
from app.services.query_rewrite_service import rewrite_query_for_retrieval
from app.services.rerank_service import RerankTextInput, rerank_texts
from app.services.settings import AppSettings
from app.services.temp_kb_service import ensure_temp_knowledge_available
from app.storage.filters import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    matches_metadata_filters,
)
from app.storage.vector_stores import BaseVectorStoreAdapter, build_vector_store_adapter


ASCII_TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9._:/-]*", re.IGNORECASE)
CJK_SEQUENCE_PATTERN = re.compile(r"[\u4e00-\u9fff]+")
NON_ALNUM_PATTERN = re.compile(r"[^0-9a-z\u4e00-\u9fff]+", re.IGNORECASE)
IMAGE_QUERY_HINTS = (
    "图片",
    "图像",
    "照片",
    "截图",
    "看图",
    "图中",
    "画面",
    "ocr",
    "文字识别",
    "识别图",
    "内容描述",
    "视觉描述",
    "画了什么",
    "有什么内容",
)
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


@dataclass
class RetrievalCandidate:
    document: Document
    dense_rank: int | None = None
    dense_distance: float | None = None
    dense_relevance: float = 0.0
    lexical_rank: int | None = None
    lexical_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float = 0.0
    model_rerank_score: float = 0.0
    relevance_score: float = 0.0


@dataclass(frozen=True)
class QueryModalityProfile:
    query_type: str
    preferred_modalities: tuple[str, ...]
    modality_bonus: dict[str, float]
    preferred_extensions: tuple[str, ...]
    extension_bonus: dict[str, float]
    path_hint_terms: tuple[str, ...]


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
    all_documents = load_all_documents(vector_store)
    if not all_documents:
        return []
    filtered_documents = filter_documents_by_metadata(all_documents, metadata_filters)
    if not filtered_documents:
        return []

    rewritten_query = rewrite_query_for_retrieval(settings, query, history)
    query_bundle = build_query_bundle(query, rewritten_query)
    retrieval_diagnostics: dict[str, object] = {}
    query_profile = infer_query_modality_profile(query_bundle)
    candidates = retrieve_candidates(
        settings=settings,
        vector_store=vector_store,
        all_documents=filtered_documents,
        query_bundle=query_bundle,
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
    all_documents: dict[str, Document],
    query_bundle: list[str],
    top_k: int,
    metadata_filters: MetadataFilters | None = None,
    query_profile: QueryModalityProfile | None = None,
    diagnostics: dict[str, object] | None = None,
) -> list[RetrievalCandidate]:
    candidate_map: dict[str, RetrievalCandidate] = {}
    dense_limit = max(top_k, settings.kb.HYBRID_DENSE_TOP_K)
    query_profile = query_profile or infer_query_modality_profile(query_bundle)
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
                "available_modalities": {
                    key: len(value) for key, value in modality_groups.items()
                },
                "modality_grouped_dense_used": modality_grouped_dense_used,
            }
        )

    if modality_grouped_dense_used:
        per_modality_dense_limit = max(1, ceil(dense_limit / max(1, len(modality_groups)))) + 2
        for source_modality in select_modalities_for_query(modality_groups, query_profile):
            collect_dense_candidates(
                vector_store=vector_store,
                query_bundle=query_bundle,
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
            query_bundle=query_bundle,
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
                    candidate_map=candidate_map,
                    lexical_limit=per_modality_lexical_limit,
                )
        else:
            collect_lexical_candidates(
                settings=settings,
                all_documents=all_documents,
                query_bundle=query_bundle,
                candidate_map=candidate_map,
                lexical_limit=settings.kb.HYBRID_LEXICAL_TOP_K,
            )

    fused = list(candidate_map.values())
    if not fused:
        if diagnostics is not None:
            diagnostics["candidate_count"] = 0
        return []

    max_dense = max((item.dense_relevance for item in fused), default=1.0) or 1.0
    max_lexical = max((item.lexical_score for item in fused), default=1.0) or 1.0
    for item in fused:
        score = 0.0
        if item.dense_rank is not None:
            score += 1.0 / (settings.kb.HYBRID_RRF_K + item.dense_rank)
        if item.lexical_rank is not None:
            score += 1.0 / (settings.kb.HYBRID_RRF_K + item.lexical_rank)
        score += 0.35 * (item.dense_relevance / max_dense)
        score += 0.25 * (item.lexical_score / max_lexical)
        score += modality_bonus_for_candidate(item.document, query_profile)
        item.fused_score = score

    if diagnostics is not None:
        diagnostics["candidate_count"] = len(fused)
        diagnostics["candidate_modality_counts"] = count_candidate_modalities(fused)

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


def collect_lexical_candidates(
    *,
    settings: AppSettings,
    all_documents: dict[str, Document],
    query_bundle: list[str],
    candidate_map: dict[str, RetrievalCandidate],
    lexical_limit: int,
) -> None:
    query_terms = build_match_terms(query_bundle)
    if not query_terms:
        return

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
    normalized_queries = [normalize_search_text(item) for item in query_bundle if item.strip()]
    plain_queries = [item.lower().strip() for item in query_bundle if item.strip()]
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
    rerank_inputs = [
        RerankTextInput(
            candidate_id=get_chunk_id(candidate.document),
            text=build_search_text(candidate.document),
        )
        for candidate in heuristic_ranked[:top_n]
    ]
    rerank_outcome = rerank_texts(
        settings=settings,
        query=query.strip(),
        items=rerank_inputs,
        top_n=top_n,
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
    query_terms = build_match_terms(query_bundle)
    max_fused = max((item.fused_score for item in candidates), default=1.0) or 1.0
    max_dense = max((item.dense_relevance for item in candidates), default=1.0) or 1.0
    max_lexical = max((item.lexical_score for item in candidates), default=1.0) or 1.0
    normalized_queries = [normalize_search_text(item) for item in query_bundle if item.strip()]
    plain_queries = [item.lower().strip() for item in query_bundle if item.strip()]

    query_term_set = set(query_terms)
    reranked: list[RetrievalCandidate] = []
    for candidate in candidates:
        search_text = build_search_text(candidate.document)
        normalized_text = normalize_search_text(search_text)
        doc_terms = set(build_match_terms([search_text]))
        overlap_ratio = (
            len(doc_terms & query_term_set) / len(query_term_set)
            if query_term_set
            else 0.0
        )
        phrase_bonus = 1.0 if any(query_text and query_text in search_text.lower() for query_text in plain_queries) else 0.0
        normalized_bonus = (
            1.0 if any(query_text and query_text in normalized_text for query_text in normalized_queries) else 0.0
        )
        source_text = f"{candidate.document.metadata.get('title', '')} {candidate.document.metadata.get('source', '')}".lower()
        source_bonus = 0.4 if any(term in source_text for term in query_terms if len(term) >= 2) else 0.0

        fused_component = candidate.fused_score / max_fused
        dense_component = candidate.dense_relevance / max_dense
        lexical_component = candidate.lexical_score / max_lexical
        modality_bonus = modality_bonus_for_candidate(candidate.document, query_profile)

        if settings.kb.ENABLE_HEURISTIC_RERANK:
            candidate.rerank_score = (
                0.35 * fused_component
                + 0.25 * dense_component
                + 0.20 * lexical_component
                + 0.10 * overlap_ratio
                + 0.06 * phrase_bonus
                + 0.10 * normalized_bonus
                + 0.04 * source_bonus
                + modality_bonus
            )
        else:
            candidate.rerank_score = (
                0.55 * fused_component + 0.30 * dense_component + 0.15 * lexical_component + modality_bonus
            )

        candidate.relevance_score = min(
            1.0,
            candidate.dense_relevance
            + 0.22 * lexical_component
            + 0.18 * overlap_ratio
            + 0.10 * phrase_bonus
            + 0.15 * normalized_bonus
            + 0.05 * source_bonus,
        )
        reranked.append(candidate)

    reranked.sort(key=lambda item: item.rerank_score, reverse=True)
    return reranked[: max(settings.kb.HYBRID_RERANK_TOP_K, top_k, 1)]


def diversify_candidates(
    candidates: list[RetrievalCandidate],
    target_count: int,
    query_profile: QueryModalityProfile,
) -> list[RetrievalCandidate]:
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
        content_type=_coerce_optional_text(candidate.document.metadata.get("content_type")),
        source_modality=_coerce_optional_text(candidate.document.metadata.get("source_modality")),
        evidence_type=resolve_reference_evidence_type(candidate.document),
        used_for_answer=True,
        original_file_type=_coerce_optional_text(candidate.document.metadata.get("original_file_type")),
        ocr_text=_coerce_optional_text(candidate.document.metadata.get("ocr_text")),
        image_caption=_coerce_optional_text(candidate.document.metadata.get("image_caption")),
        evidence_summary=_coerce_optional_text(candidate.document.metadata.get("evidence_summary")),
        headers=extract_headers(candidate.document),
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


def build_query_bundle(query: str, rewritten_query: str) -> list[str]:
    queries: list[str] = []
    for item in (query, rewritten_query):
        normalized = item.strip()
        if normalized and normalized not in queries:
            queries.append(normalized)
    for expanded in build_image_query_expansions(queries):
        if expanded not in queries:
            queries.append(expanded)
    return queries


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
    title = str(document.metadata.get("title", "")).strip()
    section_title = str(document.metadata.get("section_title", "")).strip()
    section_path = str(document.metadata.get("section_path", "")).strip()
    source = str(document.metadata.get("source", "")).strip()
    header_text = " ".join(extract_headers(document).values()).strip()
    content = document.page_content.strip()
    parts: list[str] = []
    for item in (title, section_title, section_path, header_text, source, content):
        if not item or item in parts:
            continue
        parts.append(item)
    return "\n".join(parts)


def build_match_terms(texts: list[str], deduplicate: bool = True) -> list[str]:
    terms: list[str] = []
    for text in texts:
        lowered = text.lower()
        for match in ASCII_TOKEN_PATTERN.findall(lowered):
            token = match.strip()
            if not token:
                continue
            terms.append(token)
            compact = re.sub(r"[^0-9a-z]+", "", token)
            if len(compact) >= 3 and compact != token:
                terms.append(compact)

        for sequence in CJK_SEQUENCE_PATTERN.findall(text):
            if len(sequence) == 1:
                terms.append(sequence)
                continue
            for index in range(len(sequence) - 1):
                terms.append(sequence[index : index + 2])

    if not deduplicate:
        return [term for term in terms if term]

    return deduplicate_strings(terms)


def normalize_search_text(text: str) -> str:
    return NON_ALNUM_PATTERN.sub("", text.lower())


def get_chunk_id(document: Document, fallback: str = "") -> str:
    return str(document.metadata.get("chunk_id") or fallback)


def get_document_doc_id(document: Document) -> str:
    value = document.metadata.get("doc_id") or document.metadata.get("relative_path") or document.metadata.get("source")
    return str(value or "")


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
            "query_bundle": query_bundle[:3],
            "top_k": top_k,
            "metadata_filter_count": len(metadata_filters.filters) if metadata_filters else 0,
            "query_type": diagnostics.get("query_type", "unknown"),
            "preferred_modalities": diagnostics.get("preferred_modalities", []),
            "available_modalities": diagnostics.get("available_modalities", {}),
            "modality_grouped_dense_used": diagnostics.get("modality_grouped_dense_used", False),
            "candidate_count": diagnostics.get("candidate_count", 0),
            "candidate_modality_counts": diagnostics.get("candidate_modality_counts", {}),
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


def _coerce_optional_text(value: object) -> str | None:
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    return None


def deduplicate_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


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


def extract_headers(document: Document) -> dict[str, str]:
    headers: dict[str, str] = {}
    for key in ("Header1", "Header2", "Header3"):
        value = document.metadata.get(key)
        if isinstance(value, str) and value.strip():
            headers[key] = value.strip()
    return headers
