from __future__ import annotations

import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.schemas.chat import ChatMessage
from app.services.llm_service import build_chat_model
from app.services.settings import AppSettings


LIST_ITEM_PREFIX_PATTERN = re.compile(r"^(?:[-*]\s*|\d+[\.\)]\s*|[一二三四五六七八九十]+[、.]\s*)")

QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是企业级 RAG 系统的检索查询改写器。"
            "你的任务是把用户问题重写成更适合检索的短查询。"
            "必须保留专有名词、货号、型号、缩写、数字和关键约束。"
            "不要回答问题，不要解释，只输出一行重写后的检索查询。",
        ),
        (
            "human",
            "对话历史：\n{history_text}\n\n当前问题：\n{query}\n\n"
            "请输出适合混合检索的检索查询：",
        ),
    ]
)

MULTI_QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是企业级 RAG 系统的多查询规划器。"
            "你的任务是围绕同一个问题，生成若干条适合检索的短查询。"
            "必须保留专有名词、货号、型号、缩写、数字和关键约束。"
            "至少覆盖两个角度："
            "1. 更适合检索的直接改写；"
            "2. 围绕可能答案关键词的补充检索。"
            "不要回答问题，不要解释，不要编号，每行只输出一条查询。",
        ),
        (
            "human",
            "对话历史：\n{history_text}\n\n当前问题：\n{query}\n\n"
            "请输出 {variant_count} 条不同角度的检索查询：",
        ),
    ]
)

HYDE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是企业级 RAG 系统的 HyDE 假设文档生成器。"
            "你的任务是根据用户问题，写一小段可能出现在目标文档中的说明性文本，"
            "用于向量检索，不是给用户看的最终答案。"
            "必须尽量包含可能命中的术语、参数名、部件名、章节线索、步骤线索或故障线索。"
            "不要解释你的思路，不要分点，不要使用引号。",
        ),
        (
            "human",
            "对话历史：\n{history_text}\n\n当前问题：\n{query}\n\n"
            "请输出一段 2 到 4 句的假设文档片段：",
        ),
    ]
)


def rewrite_query_for_retrieval(
    settings: AppSettings,
    query: str,
    history: list[ChatMessage] | None = None,
) -> str:
    normalized_query = query.strip()
    if not normalized_query:
        return normalized_query
    query_bundle = generate_multi_queries(settings, normalized_query, history)
    if len(query_bundle) >= 2:
        return query_bundle[1]
    return normalized_query


def generate_multi_queries(
    settings: AppSettings,
    query: str,
    history: list[ChatMessage] | None = None,
) -> list[str]:
    normalized_query = query.strip()
    if not normalized_query:
        return []
    if not settings.kb.ENABLE_QUERY_REWRITE or _should_skip_rewrite(normalized_query):
        return [normalized_query]

    max_queries = settings.kb.MULTI_QUERY_MAX_QUERIES
    if max_queries <= 1:
        return [normalized_query]

    if settings.kb.ENABLE_MULTI_QUERY_RETRIEVAL:
        generated_queries = _invoke_multi_query_rewrite(
            settings=settings,
            query=normalized_query,
            history=history,
            max_queries=max_queries,
        )
        if generated_queries:
            return deduplicate_query_candidates(
                normalized_query,
                generated_queries,
                limit=max_queries,
            )

    rewritten = _invoke_single_query_rewrite(settings, normalized_query, history)
    if not rewritten:
        return [normalized_query]
    return deduplicate_query_candidates(normalized_query, [rewritten], limit=max_queries)


def generate_hypothetical_doc(
    settings: AppSettings,
    query: str,
    history: list[ChatMessage] | None = None,
) -> str:
    normalized_query = query.strip()
    if not normalized_query or not settings.kb.ENABLE_HYDE:
        return ""

    hypothetical_doc = _invoke_hypothetical_doc_generation(settings, normalized_query, history)
    return sanitize_hypothetical_doc(hypothetical_doc)


def _invoke_single_query_rewrite(
    settings: AppSettings,
    query: str,
    history: list[ChatMessage] | None = None,
) -> str:
    normalized_query = query.strip()
    if not normalized_query:
        return ""

    model_name = settings.model.QUERY_REWRITE_MODEL.strip() or settings.model.DEFAULT_LLM_MODEL
    try:
        llm = build_chat_model(settings, model_name=model_name, temperature=0.0)
        chain = QUERY_REWRITE_PROMPT | llm | StrOutputParser()
        rewritten = chain.invoke(
            {
                "query": normalized_query,
                "history_text": format_history(history or []),
            }
        )
    except Exception:
        return ""

    return normalize_candidate_query(rewritten, normalized_query)


def _invoke_multi_query_rewrite(
    *,
    settings: AppSettings,
    query: str,
    history: list[ChatMessage] | None = None,
    max_queries: int,
) -> list[str]:
    normalized_query = query.strip()
    if not normalized_query:
        return []

    model_name = settings.model.QUERY_REWRITE_MODEL.strip() or settings.model.DEFAULT_LLM_MODEL
    variant_count = max(1, max_queries - 1)
    try:
        llm = build_chat_model(settings, model_name=model_name, temperature=0.0)
        chain = MULTI_QUERY_REWRITE_PROMPT | llm | StrOutputParser()
        rewritten = chain.invoke(
            {
                "query": normalized_query,
                "history_text": format_history(history or []),
                "variant_count": variant_count,
            }
        )
    except Exception:
        return []

    return parse_multi_query_output(rewritten, normalized_query, limit=variant_count)


def _invoke_hypothetical_doc_generation(
    settings: AppSettings,
    query: str,
    history: list[ChatMessage] | None = None,
) -> str:
    normalized_query = query.strip()
    if not normalized_query:
        return ""

    model_name = settings.model.QUERY_REWRITE_MODEL.strip() or settings.model.DEFAULT_LLM_MODEL
    try:
        llm = build_chat_model(settings, model_name=model_name, temperature=0.0)
        chain = HYDE_PROMPT | llm | StrOutputParser()
        return chain.invoke(
            {
                "query": normalized_query,
                "history_text": format_history(history or []),
            }
        )
    except Exception:
        return ""


def format_history(history: list[ChatMessage]) -> str:
    if not history:
        return "无"

    lines: list[str] = []
    for item in history[-6:]:
        lines.append(f"{item.role}: {item.content.strip()}")
    return "\n".join(lines)


def sanitize_rewritten_query(text: str) -> str:
    cleaned = (text or "").strip().strip("`").strip()
    if not cleaned:
        return ""

    first_line = cleaned.splitlines()[0].strip()
    first_line = LIST_ITEM_PREFIX_PATTERN.sub("", first_line).strip()
    prefixes = (
        "检索查询：",
        "检索词：",
        "重写后：",
        "改写后：",
        "query:",
        "Query:",
    )
    for prefix in prefixes:
        if first_line.startswith(prefix):
            first_line = first_line[len(prefix) :].strip()
            break

    return first_line.strip("。 ").strip()


def parse_multi_query_output(
    text: str,
    original_query: str,
    *,
    limit: int,
) -> list[str]:
    cleaned = (text or "").strip().strip("`").strip()
    if not cleaned:
        return []

    raw_lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if len(raw_lines) == 1:
        raw_lines = split_candidate_line(raw_lines[0])

    candidates: list[str] = []
    for line in raw_lines:
        normalized = normalize_candidate_query(line, original_query)
        if normalized:
            candidates.append(normalized)
        if len(candidates) >= limit:
            break
    return deduplicate_query_candidates(original_query, candidates, limit=limit + 1)[1:]


def split_candidate_line(text: str) -> list[str]:
    normalized = text.replace("；", "\n").replace(";", "\n")
    return [item.strip() for item in normalized.splitlines() if item.strip()]


def normalize_candidate_query(text: str, original_query: str) -> str:
    cleaned = sanitize_rewritten_query(text)
    if not cleaned:
        return ""
    if len(cleaned) > max(96, len(original_query) * 2):
        return ""
    return cleaned


def deduplicate_query_candidates(
    original_query: str,
    candidates: list[str],
    *,
    limit: int,
) -> list[str]:
    queries = [original_query]
    for item in candidates:
        normalized = normalize_candidate_query(item, original_query)
        if not normalized or normalized in queries:
            continue
        queries.append(normalized)
        if len(queries) >= limit:
            break
    return queries


def sanitize_hypothetical_doc(text: str) -> str:
    cleaned = " ".join((text or "").strip().strip("`").split())
    if not cleaned:
        return ""
    if len(cleaned) > 480:
        cleaned = cleaned[:480].rstrip()
    return cleaned


def _should_skip_rewrite(query: str) -> bool:
    compact = query.replace(" ", "")
    if len(compact) <= 6:
        return True
    if "\n" not in query and compact.isascii() and len(compact) <= 24:
        return True
    return False
