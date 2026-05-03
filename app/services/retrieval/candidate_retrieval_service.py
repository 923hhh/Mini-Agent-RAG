"""候选召回服务。"""

from __future__ import annotations

from collections import Counter, defaultdict
from math import ceil, log

from langchain_core.documents import Document

from app.services.core.settings import AppSettings
from app.services.retrieval.candidate_common_service import (
    RetrievalCandidate,
    get_chunk_id,
    get_source_modality,
)
from app.services.retrieval.candidate_fusion_service import (
    apply_candidate_fusion_scores,
    build_search_text,
    count_candidate_modalities,
    merge_retrieval_candidate_lists,
)
from app.services.retrieval.query_profile_service import (
    JointQueryProfile,
    QueryModalityProfile,
    build_image_query_expansions,
    infer_query_modality_profile,
    infer_temporal_query_profile,
    should_use_sentence_index,
)
from app.services.retrieval.timeseries_retrieval_service import infer_timeseries_query_profile
from app.storage.bm25_index import (
    LoadedBM25Index,
    build_match_terms as build_bm25_match_terms,
    normalize_search_text as normalize_bm25_search_text,
    score_bm25_index,
)
from app.storage.filters import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    matches_metadata_filters,
)
from app.storage.vector_stores import BaseVectorStoreAdapter


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

    apply_candidate_fusion_scores(
        settings=settings,
        candidates=fused,
        query_bundle=query_bundle,
        query_profile=query_profile,
    )

    if diagnostics is not None:
        diagnostics["candidate_count"] = len(fused)
        diagnostics["candidate_modality_counts"] = count_candidate_modalities(fused)
        diagnostics["sentence_index_used"] = sentence_index_used

    return fused


def retrieve_candidates_with_timeseries_branching(
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
    joint_query_profile: JointQueryProfile | None = None,
    timeseries_query_profile=None,
    diagnostics: dict[str, object] | None = None,
) -> list[RetrievalCandidate]:
    query_profile = query_profile or infer_query_modality_profile(query_bundle)
    timeseries_query_profile = timeseries_query_profile or infer_timeseries_query_profile(query_bundle)
    has_timeseries_documents = any(
        get_source_modality(document) == "timeseries"
        for document in all_documents.values()
    )
    should_force_timeseries_branch = bool(
        joint_query_profile
        and (
            joint_query_profile.requires_timeseries
            or joint_query_profile.has_guard_constraint
            or joint_query_profile.location_terms
            or joint_query_profile.domain_terms
            or joint_query_profile.channel_terms
            or joint_query_profile.event_terms
        )
    )
    if (
        not has_timeseries_documents
        or (
            not timeseries_query_profile.is_timeseries_related
            and not should_force_timeseries_branch
        )
    ):
        if diagnostics is not None:
            diagnostics["timeseries_branch_used"] = False
        return retrieve_candidates(
            settings=settings,
            vector_store=vector_store,
            sentence_vector_store=sentence_vector_store,
            all_documents=all_documents,
            query_bundle=query_bundle,
            dense_query_bundle=dense_query_bundle,
            bm25_index=bm25_index,
            top_k=top_k,
            metadata_filters=metadata_filters,
            query_profile=query_profile,
            diagnostics=diagnostics,
        )

    text_documents = {
        chunk_id: document
        for chunk_id, document in all_documents.items()
        if get_source_modality(document) != "timeseries"
    }
    timeseries_documents = {
        chunk_id: document
        for chunk_id, document in all_documents.items()
        if get_source_modality(document) == "timeseries"
    }
    text_candidates = retrieve_candidates(
        settings=settings,
        vector_store=vector_store,
        sentence_vector_store=sentence_vector_store,
        all_documents=text_documents,
        query_bundle=query_bundle,
        dense_query_bundle=dense_query_bundle,
        bm25_index=bm25_index,
        top_k=top_k,
        metadata_filters=merge_metadata_filters_excluding_source_modality(
            metadata_filters,
            "timeseries",
        ),
        query_profile=query_profile,
        diagnostics=None,
    )
    timeseries_candidates = retrieve_candidates(
        settings=settings,
        vector_store=vector_store,
        sentence_vector_store=None,
        all_documents=timeseries_documents,
        query_bundle=query_bundle,
        dense_query_bundle=dense_query_bundle,
        bm25_index=bm25_index,
        top_k=(
            max(5, top_k + 2)
            if should_force_timeseries_branch
            else max(3, top_k)
        ),
        metadata_filters=merge_metadata_filters_with_source_modality(
            metadata_filters,
            "timeseries",
        ),
        query_profile=query_profile,
        diagnostics=None,
    )
    merged_candidates = merge_retrieval_candidate_lists(
        primary=text_candidates,
        secondary=timeseries_candidates,
    )
    if diagnostics is not None:
        diagnostics["timeseries_branch_used"] = True
        diagnostics["text_branch_candidate_count"] = len(text_candidates)
        diagnostics["timeseries_branch_candidate_count"] = len(timeseries_candidates)
        diagnostics["merged_candidate_count"] = len(merged_candidates)
        diagnostics["candidate_count"] = len(merged_candidates)
        diagnostics["joint_rerank_applied"] = bool(joint_query_profile and joint_query_profile.is_joint_query)
    return merged_candidates


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
        lexical_scores[:lexical_limit],
        start=1,
    ):
        document = all_documents[chunk_id]
        candidate = candidate_map.setdefault(chunk_id, RetrievalCandidate(document=document))
        candidate.lexical_rank = rank
        candidate.lexical_score = score


def distance_to_relevance(distance: float) -> float:
    return 1.0 / (1.0 + float(distance))


def resolve_query_type_label(query_profile: QueryModalityProfile) -> str:
    return query_profile.query_type


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


def select_modalities_for_query(
    modality_groups: dict[str, dict[str, Document]],
    query_profile: QueryModalityProfile,
) -> list[str]:
    selected: list[str] = []
    for source_modality in query_profile.preferred_modalities:
        if source_modality in modality_groups and source_modality not in selected:
            selected.append(source_modality)
    for source_modality in modality_groups:
        if source_modality == "timeseries" and query_profile.query_type != "timeseries_related":
            continue
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


def merge_metadata_filters_excluding_source_modality(
    metadata_filters: MetadataFilters | None,
    source_modality: str,
) -> MetadataFilters:
    modality_filter = MetadataFilter(
        key="source_modality",
        operator=FilterOperator.NE,
        value=source_modality,
    )
    if metadata_filters is None or not metadata_filters.filters:
        return MetadataFilters(filters=[modality_filter])
    return MetadataFilters(
        condition=FilterCondition.AND,
        filters=[*metadata_filters.filters, modality_filter],
    )


def build_match_terms(texts: list[str], deduplicate: bool = True) -> list[str]:
    return build_bm25_match_terms(texts, deduplicate=deduplicate)


def normalize_search_text(text: str) -> str:
    return normalize_bm25_search_text(text)
