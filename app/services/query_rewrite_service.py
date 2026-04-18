from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.schemas.chat import ChatMessage
from app.services.llm_service import build_chat_model
from app.services.settings import AppSettings
from app.utils.text import deduplicate_strings


LIST_ITEM_PREFIX_PATTERN = re.compile(r"^(?:[-*]\s*|\d+[\.\)]\s*|[一二三四五六七八九十]+[、.]\s*)")
YEAR_PATTERN = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
TEMPORAL_RECENCY_TERMS = (
    "最新",
    "当前",
    "目前",
    "现任",
    "最近",
    "今年",
    "本年",
    "本年度",
)
TEMPORAL_SLOT_PHRASES = (
    "报名时间",
    "截止时间",
    "查询时间",
    "开始时间",
    "结束时间",
    "工作时间",
    "测试时间",
    "发布时间",
    "更新时间",
    "录取时间",
    "成绩查询",
    "确认志愿",
)
TEMPORAL_KEYWORDS = (
    "时间",
    "日期",
    "何时",
    "什么时候",
    "哪一年",
    "哪年",
    "几月",
    "几号",
    "截至",
    "截止",
    "开始",
    "结束",
    "报名",
    "查询",
)


@dataclass(frozen=True)
class TemporalConstraintProfile:
    is_temporal: bool
    explicit_years: tuple[str, ...]
    recency_terms: tuple[str, ...]
    slot_phrases: tuple[str, ...]
    keep_keywords: tuple[str, ...]

QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是企业级 RAG 系统的检索查询改写器。"
            "你的任务是把用户问题重写成更适合检索的短查询。"
            "必须保留专有名词、货号、型号、缩写、数字和关键约束。"
            "如果问题涉及时间、年份、当前/最新、报名/截止/查询时间，必须保留这些时间条件，不得省略或泛化。"
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
            "如果原问题包含时间、年份、当前/最新、报名/截止/查询时间等约束，每一条查询都必须保留这些约束。"
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
    cleaned = enforce_temporal_constraints(cleaned, original_query)
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
    normalized_original = normalize_candidate_query(original_query, original_query) or original_query.strip()
    queries = [normalized_original]
    for item in candidates:
        normalized = normalize_candidate_query(item, original_query)
        if normalized:
            queries.append(normalized)
    return deduplicate_strings(queries)[:limit]


def enforce_temporal_constraints(candidate: str, original_query: str) -> str:
    cleaned_candidate = candidate.strip()
    cleaned_original = original_query.strip()
    if not cleaned_candidate or not cleaned_original:
        return cleaned_candidate

    profile = build_temporal_constraint_profile(cleaned_original)
    if not profile.is_temporal:
        return cleaned_candidate

    lowered_candidate = cleaned_candidate.lower()

    for year in profile.explicit_years:
        if year not in cleaned_candidate:
            return ""

    additions: list[str] = []
    if profile.slot_phrases and not any(phrase in cleaned_candidate for phrase in profile.slot_phrases):
        additions.append(profile.slot_phrases[0])

    if profile.recency_terms and not any(term in cleaned_candidate for term in profile.recency_terms):
        additions.append(profile.recency_terms[0])

    for keyword in profile.keep_keywords:
        if keyword in cleaned_candidate:
            continue
        if keyword.lower() in lowered_candidate:
            continue
        additions.append(keyword)
        if len(additions) >= 2:
            break

    if additions:
        merged = f"{cleaned_candidate} {' '.join(additions)}".strip()
        return deduplicate_inline_terms(merged)
    return cleaned_candidate


def build_temporal_constraint_profile(query: str) -> TemporalConstraintProfile:
    stripped = query.strip()
    explicit_years = tuple(dict.fromkeys(match.group(1) for match in YEAR_PATTERN.finditer(stripped)))
    recency_terms = tuple(term for term in TEMPORAL_RECENCY_TERMS if term in stripped)
    slot_phrases = tuple(phrase for phrase in TEMPORAL_SLOT_PHRASES if phrase in stripped)
    keep_keywords = tuple(keyword for keyword in TEMPORAL_KEYWORDS if keyword in stripped)
    is_temporal = bool(explicit_years or recency_terms or slot_phrases or keep_keywords)
    return TemporalConstraintProfile(
        is_temporal=is_temporal,
        explicit_years=explicit_years,
        recency_terms=recency_terms,
        slot_phrases=slot_phrases,
        keep_keywords=keep_keywords,
    )


def deduplicate_inline_terms(text: str) -> str:
    parts = [item.strip() for item in re.split(r"\s+", text.strip()) if item.strip()]
    deduped: list[str] = []
    for part in parts:
        if part not in deduped:
            deduped.append(part)
    return " ".join(deduped)


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
