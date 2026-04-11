from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from math import log
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict

from app.utils.text import deduplicate_strings

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - optional runtime dependency
    BM25Okapi = None


ASCII_TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9._:/-]*", re.IGNORECASE)
CJK_SEQUENCE_PATTERN = re.compile(r"[\u4e00-\u9fff]+")
NON_ALNUM_PATTERN = re.compile(r"[^0-9a-z\u4e00-\u9fff]+", re.IGNORECASE)
BM25_INDEX_FILENAME = "bm25_index.json"
EMPTY_BM25_TOKEN = "__bm25_empty__"


class PersistedBM25Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    search_text: str
    terms: list[str]


class PersistedBM25IndexPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    chunk_count: int
    documents: list[PersistedBM25Document]


@dataclass(frozen=True)
class LoadedBM25Index:
    chunk_ids: list[str]
    search_texts: list[str]
    raw_texts_lower: list[str]
    normalized_texts: list[str]
    tokenized_corpus: list[list[str]]
    term_counters: list[Counter[str]]
    document_frequency: Counter[str]
    doc_lengths: list[int]
    average_length: float
    bm25: Any | None
    backend: str


def resolve_bm25_index_path(vector_store_dir: Path) -> Path:
    return vector_store_dir / BM25_INDEX_FILENAME


def build_search_text_from_parts(
    *,
    page_content: str,
    metadata: Mapping[str, object],
    headers: Mapping[str, str] | None = None,
) -> str:
    title = str(metadata.get("title", "")).strip()
    section_title = str(metadata.get("section_title", "")).strip()
    section_path = str(metadata.get("section_path", "")).strip()
    source = str(metadata.get("source", "")).strip()
    header_text = " ".join((headers or {}).values()).strip()
    content = page_content.strip()
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


def build_persisted_bm25_document(
    *,
    chunk_id: str,
    page_content: str,
    metadata: Mapping[str, object],
    headers: Mapping[str, str] | None = None,
) -> PersistedBM25Document:
    search_text = build_search_text_from_parts(
        page_content=page_content,
        metadata=metadata,
        headers=headers,
    )
    return PersistedBM25Document(
        chunk_id=chunk_id,
        search_text=search_text,
        terms=build_match_terms([search_text], deduplicate=False),
    )


def write_bm25_index(path: Path, documents: list[PersistedBM25Document]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = PersistedBM25IndexPayload(
        chunk_count=len(documents),
        documents=documents,
    )
    path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")


def delete_bm25_index(path: Path) -> None:
    if path.exists():
        path.unlink(missing_ok=True)


def load_bm25_index(path: Path) -> LoadedBM25Index | None:
    if not path.exists():
        return None
    stat = path.stat()
    return _load_bm25_index_cached(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=16)
def _load_bm25_index_cached(
    path_str: str,
    modified_at_ns: int,
    size_bytes: int,
) -> LoadedBM25Index:
    del modified_at_ns, size_bytes
    payload = PersistedBM25IndexPayload.model_validate_json(Path(path_str).read_text(encoding="utf-8"))
    chunk_ids = [item.chunk_id for item in payload.documents]
    search_texts = [item.search_text for item in payload.documents]
    raw_texts_lower = [item.search_text.lower() for item in payload.documents]
    normalized_texts = [normalize_search_text(item.search_text) for item in payload.documents]
    tokenized_corpus = [item.terms or [EMPTY_BM25_TOKEN] for item in payload.documents]
    term_counters = [Counter(item.terms) for item in payload.documents]
    document_frequency: Counter[str] = Counter()
    for counter in term_counters:
        for term in counter:
            document_frequency[term] += 1
    doc_lengths = [sum(counter.values()) or 1 for counter in term_counters]
    average_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 1.0
    bm25 = BM25Okapi(tokenized_corpus) if BM25Okapi is not None and tokenized_corpus else None
    backend = "rank_bm25" if bm25 is not None else "fallback"
    return LoadedBM25Index(
        chunk_ids=chunk_ids,
        search_texts=search_texts,
        raw_texts_lower=raw_texts_lower,
        normalized_texts=normalized_texts,
        tokenized_corpus=tokenized_corpus,
        term_counters=term_counters,
        document_frequency=document_frequency,
        doc_lengths=doc_lengths,
        average_length=average_length,
        bm25=bm25,
        backend=backend,
    )


def score_bm25_index(
    *,
    index: LoadedBM25Index,
    query_terms: list[str],
    normalized_queries: list[str],
    plain_queries: list[str],
    allowed_chunk_ids: set[str] | None = None,
) -> list[tuple[str, float]]:
    if not query_terms or not index.chunk_ids:
        return []

    if index.bm25 is not None:
        base_scores = [float(item) for item in index.bm25.get_scores(query_terms)]
    else:
        base_scores = _compute_legacy_bm25_scores(index=index, query_terms=query_terms)

    lexical_scores: list[tuple[str, float]] = []
    for idx, chunk_id in enumerate(index.chunk_ids):
        if allowed_chunk_ids is not None and chunk_id not in allowed_chunk_ids:
            continue
        score = base_scores[idx]
        raw_text = index.raw_texts_lower[idx]
        normalized_text = index.normalized_texts[idx]
        if any(query_text and query_text in raw_text for query_text in plain_queries):
            score += 0.8
        if any(normalized_query and normalized_query in normalized_text for normalized_query in normalized_queries):
            score += 1.2
        if score <= 0:
            continue
        lexical_scores.append((chunk_id, score))

    lexical_scores.sort(key=lambda item: item[1], reverse=True)
    return lexical_scores


def _compute_legacy_bm25_scores(
    *,
    index: LoadedBM25Index,
    query_terms: list[str],
) -> list[float]:
    total_docs = len(index.chunk_ids)
    if total_docs == 0:
        return []
    query_counter = Counter(query_terms)
    average_length = index.average_length or 1.0
    scores = [0.0] * total_docs
    for idx, term_counter in enumerate(index.term_counters):
        doc_length = index.doc_lengths[idx]
        score = 0.0
        for term, query_tf in query_counter.items():
            term_tf = term_counter.get(term, 0)
            if term_tf == 0:
                continue
            df = index.document_frequency.get(term, 0)
            idf = log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
            numerator = term_tf * (1.5 + 1.0)
            denominator = term_tf + 1.5 * (1.0 - 0.75 + 0.75 * (doc_length / average_length))
            score += query_tf * idf * (numerator / denominator)
        scores[idx] = score
    return scores
