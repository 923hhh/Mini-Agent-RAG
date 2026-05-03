"""候选对象与公共标识 helper。"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document


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
    temporal_match_score: float = 0.0
    event_type_match_score: float = 0.0
    location_match_score: float = 0.0
    channel_match_score: float = 0.0
    joint_coverage_bonus: float = 0.0
    same_series_or_same_event_group: float = 0.0


def get_chunk_id(document: Document, fallback: str = "") -> str:
    return str(document.metadata.get("chunk_id") or fallback)


def get_source_modality(document: Document) -> str:
    value = document.metadata.get("source_modality")
    if isinstance(value, str):
        normalized = value.strip()
        if normalized:
            return normalized
    return "text"
