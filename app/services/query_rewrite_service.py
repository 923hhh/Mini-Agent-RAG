from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.schemas.chat import ChatMessage
from app.services.llm_service import build_chat_model
from app.services.settings import AppSettings


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


def rewrite_query_for_retrieval(
    settings: AppSettings,
    query: str,
    history: list[ChatMessage] | None = None,
) -> str:
    normalized_query = query.strip()
    if not settings.kb.ENABLE_QUERY_REWRITE or not normalized_query:
        return normalized_query

    if _should_skip_rewrite(normalized_query):
        return normalized_query

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
        return normalized_query

    cleaned = sanitize_rewritten_query(rewritten)
    if not cleaned:
        return normalized_query
    if len(cleaned) > max(96, len(normalized_query) * 2):
        return normalized_query
    return cleaned


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


def _should_skip_rewrite(query: str) -> bool:
    compact = query.replace(" ", "")
    if len(compact) <= 6:
        return True
    if "\n" not in query and compact.isascii() and len(compact) <= 24:
        return True
    return False
