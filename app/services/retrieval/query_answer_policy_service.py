"""统一管理 RAG 回答阶段的题型判定与上下文策略。"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.chat import RetrievedReference


@dataclass(frozen=True)
class QueryAnswerPolicy:
    query: str
    requirement_count: int
    should_direct_answer: bool
    is_multi_doc_comparative: bool
    is_procedural: bool
    is_numeric_fact: bool
    is_symbol_explanation: bool


def build_query_answer_policy(
    query: str,
    references: list[RetrievedReference] | None = None,
) -> QueryAnswerPolicy:
    del references
    from app.services.retrieval.answer_guard_service import (
        is_multi_doc_comparative_query,
        is_numeric_fact_query,
        is_procedural_query,
        is_symbol_explanation_query,
        should_directly_answer_query,
        split_query_into_requirements,
    )

    normalized_query = str(query or "").strip()
    return QueryAnswerPolicy(
        query=normalized_query,
        requirement_count=len(split_query_into_requirements(normalized_query)),
        should_direct_answer=should_directly_answer_query(normalized_query),
        is_multi_doc_comparative=is_multi_doc_comparative_query(normalized_query),
        is_procedural=is_procedural_query(normalized_query),
        is_numeric_fact=is_numeric_fact_query(normalized_query),
        is_symbol_explanation=is_symbol_explanation_query(normalized_query),
    )


__all__ = [
    "QueryAnswerPolicy",
    "build_query_answer_policy",
]
