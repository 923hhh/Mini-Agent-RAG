from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.constants import IMAGE_QUERY_HINTS
from app.schemas.chat import ChatMessage, RetrievedReference
from app.services.llm_service import build_chat_model
from app.services.observability import append_jsonl_trace
from app.services.query_rewrite_service import format_history, sanitize_rewritten_query
from app.services.settings import AppSettings
from app.services.streaming_llm import stream_prompt_output


RAG_SYSTEM_PROMPT = """你是一个基于本地知识库的问答助手。
请严格依据提供的上下文回答用户问题，不要编造上下文中不存在的事实。
如果上下文不足以回答问题，请明确说明“根据当前检索到的内容，无法确定”并给出缺失信息。
不要把相似事实当成目标答案，不要补充常识性猜测。
回答尽量简洁、准确，并优先提炼要点。
如果问题包含多个并列子问题、多个角色、多个原因、多个措施或“哪些/分别/同时/以及”等要求，必须逐项覆盖，不得遗漏。
若上下文已经明确列出了清单、步骤、职责、原因或措施，回答时应尽量完整保留这些要点，而不是只做笼统概括。
优先使用分点或小标题作答，让每个子问题都有明确落点。
如果上下文包含“文本证据 / OCR 证据 / 视觉描述证据”，请区分哪些信息是直接证据，哪些只是推断。
不要把 OCR 识别文本与视觉描述混为同一类事实。
除非用户明确要求，不要在答案中写“根据上下文”“根据参考资料”“来源如下”等套话。"""

IMAGE_RAG_SYSTEM_PROMPT = """你是一个基于本地知识库的多模态问答助手。
当前问题更偏向图片、OCR 或视觉描述理解，请优先依据提供的图片相关证据回答。
请遵守以下规则：
1. OCR 文字属于直接文本证据，但可能有识别错误；引用时不要自动纠正成你主观认为更合理的内容。
2. 视觉描述属于图像理解结果，只能用于“看到的画面特征”判断，不能当作绝对事实扩展。
3. 如果 OCR 证据和视觉描述证据冲突，必须明确指出冲突，不要强行合并。
4. 如果图片证据不足，请明确说明“根据当前检索到的图片相关证据，无法确定”，不要拿普通正文内容替代图片结论。
5. 回答尽量简洁，先说可确认内容，再说可能推断。"""

CORRECTIVE_RAG_GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是 RAG 检索质量评估器。"
            "你需要判断当前检索证据是否足以回答用户问题。"
            "只允许输出 JSON，不要输出额外解释。"
            'JSON schema: {"grade":"relevant|partial|insufficient","reason":"<=60字","missing_aspects":"<=60字"}。',
        ),
        (
            "human",
            "对话历史：\n{history_text}\n\n用户问题：\n{query}\n\n当前证据摘要：\n{reference_summary}\n\n"
            "请评估当前证据覆盖度。",
        ),
    ]
)

CORRECTIVE_RAG_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是 RAG 的二次检索查询生成器。"
            "请基于原问题和当前证据缺口，输出一条更适合第二轮检索的短查询。"
            "必须保留型号、参数、专有名词和关键约束。"
            "不要回答问题，不要解释，只输出一行查询。",
        ),
        (
            "human",
            "对话历史：\n{history_text}\n\n用户问题：\n{query}\n\n"
            "当前分级：{grade}\n原因：{reason}\n缺失点：{missing_aspects}\n\n"
            "当前证据摘要：\n{reference_summary}\n\n"
            "请输出一条更适合二次检索的查询：",
        ),
    ]
)

RAG_COMPLETENESS_REVIEW_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是 RAG 答案完备性审校器。"
            "你只能依据给定证据检查当前答案是否漏答、缺少并列要点、遗漏角色职责或清单项。"
            "不要补充证据中不存在的事实。"
            "只允许输出 JSON，不要输出额外解释。"
            'JSON schema: {"status":"ok|revise","missing_points":["<=30字"],"revised_answer":"string"}。'
            "若当前答案已足够完整，status=ok，revised_answer 置空字符串。",
        ),
        (
            "human",
            "证据事实：\n{context}\n\n{coverage_requirements}"
            "用户问题：\n{query}\n\n当前答案草稿：\n{draft_answer}\n\n"
            "请检查是否存在漏答、列表不全、并列子问题未覆盖或角色遗漏。"
            "若需要修改，请直接给出修订后的最终答案。",
        ),
    ]
)

RAG_EVIDENCE_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是 RAG 证据抽取器。"
            "你只能依据给定证据抽取可直接用于回答用户问题的事实。"
            "不要补充常识，不要猜测，不要改写成答案。"
            "只允许输出 JSON。"
            'JSON schema: {"facts":["<=80字"],"unknown_points":["<=40字"],"checklist_coverage":[{"item":"string","covered":true|false}]}。',
        ),
        (
            "human",
            "用户问题：\n{query}\n\n{coverage_requirements}"
            "证据上下文：\n{context}\n\n"
            "请提取与回答直接相关的关键事实，并指出证据不足的点。",
        ),
    ]
)

RAG_ANSWER_FROM_EVIDENCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        (
            "human",
            "{memory_section}已抽取的证据事实如下：\n{evidence_facts}\n\n{coverage_requirements}"
            "用户问题：\n{query}\n\n"
            "请只依据上述证据事实作答。"
            "如果某个问题没有足够证据，请直接说明“根据当前检索到的内容，无法确定”。"
            "不要补充证据事实之外的新信息。",
        ),
    ]
)


@dataclass(frozen=True)
class RetrievalCoverageGrade:
    grade: str
    reason: str
    missing_aspects: str = ""


@dataclass(frozen=True)
class ContextBuildResult:
    text: str
    reference_count: int
    context_chars: int


@dataclass(frozen=True)
class EvidenceExtractionResult:
    facts_text: str
    fact_count: int
    unknown_count: int

def generate_rag_answer(
    settings: AppSettings,
    query: str,
    references: list[RetrievedReference],
    history: list[ChatMessage],
    agent_memory_context: str = "",
) -> str:
    prompt_kind = resolve_rag_prompt_kind(query, references)
    variables = build_rag_variables(
        query,
        references,
        history,
        agent_memory_context=agent_memory_context,
        compress_context=(prompt_kind == "default"),
    )
    context_meta: ContextBuildResult = variables.pop("_context_meta")
    llm = build_chat_model(settings, temperature=0.0)
    evidence_meta = EvidenceExtractionResult(
        facts_text=str(variables["context"]),
        fact_count=0,
        unknown_count=0,
    )
    if prompt_kind == "default":
        evidence_meta = extract_rag_evidence(
            settings=settings,
            query=query,
            context=str(variables["context"]),
            coverage_requirements=str(variables["coverage_requirements"]),
        )
        answer = generate_answer_from_evidence(
            llm=llm,
            history=history,
            memory_section=str(variables["memory_section"]),
            query=query,
            coverage_requirements=str(variables["coverage_requirements"]),
            evidence_facts=evidence_meta.facts_text,
        )
    else:
        prompt = build_rag_prompt(query, references)
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke(variables)
    answer = maybe_refine_rag_answer(
        settings=settings,
        query=query,
        references=references,
        context=evidence_meta.facts_text,
        coverage_requirements=str(variables["coverage_requirements"]),
        draft_answer=answer,
    )
    append_answer_trace(
        settings=settings,
        query=query,
        references=references,
        prompt_kind=prompt_kind,
        answer=answer,
        compressed_reference_count=context_meta.reference_count,
        compressed_context_chars=context_meta.context_chars,
        evidence_fact_count=evidence_meta.fact_count,
        evidence_unknown_count=evidence_meta.unknown_count,
        coverage_requirement_count=count_coverage_requirements(str(variables["coverage_requirements"])),
    )
    return answer


def stream_rag_answer(
    settings: AppSettings,
    query: str,
    references: list[RetrievedReference],
    history: list[ChatMessage],
    agent_memory_context: str = "",
) -> Iterator[str]:
    prompt_kind = resolve_rag_prompt_kind(query, references)
    prompt = build_rag_prompt(query, references)
    variables = build_rag_variables(query, references, history, agent_memory_context=agent_memory_context)
    context_meta: ContextBuildResult = variables.pop("_context_meta")
    answer_parts: list[str] = []
    for delta in stream_prompt_output(settings, prompt, variables, temperature=0.0):
        answer_parts.append(delta)
        yield delta
    append_answer_trace(
        settings=settings,
        query=query,
        references=references,
        prompt_kind=prompt_kind,
        answer="".join(answer_parts),
        compressed_reference_count=context_meta.reference_count,
        compressed_context_chars=context_meta.context_chars,
    )


def maybe_run_corrective_retrieval(
    *,
    settings: AppSettings,
    query: str,
    references: list[RetrievedReference],
    history: list[ChatMessage],
    top_k: int,
    score_threshold: float,
    retrieve: Callable[[str, int, float], list[RetrievedReference]],
    search_web: Callable[[str], list[RetrievedReference]] | None = None,
    source_type: str,
    target_name: str,
) -> list[RetrievedReference]:
    if not settings.kb.ENABLE_CORRECTIVE_RAG:
        return references

    initial_grade = grade_documents(
        settings=settings,
        query=query,
        references=references,
        history=history,
    )
    should_retry = not references or initial_grade.grade in {"partial", "insufficient"}
    if not should_retry:
        append_corrective_trace(
            settings=settings,
            query=query,
            source_type=source_type,
            target_name=target_name,
            initial_reference_count=len(references),
            initial_grade=initial_grade,
            triggered=False,
            follow_up_query="",
            follow_up_reference_count=0,
            final_references=references,
        )
        return references

    follow_up_query = generate_corrective_query(
        settings=settings,
        query=query,
        references=references,
        history=history,
        grade=initial_grade,
    )
    second_pass_top_k = min(
        20,
        max(top_k + 2, settings.kb.CORRECTIVE_RAG_SECOND_PASS_TOP_K),
    )
    second_pass_threshold = min(score_threshold, settings.kb.CORRECTIVE_RAG_SECOND_PASS_SCORE_THRESHOLD)
    final_limit = min(
        24,
        max(
            second_pass_top_k,
            top_k
            + (
                settings.kb.CORRECTIVE_WEB_SEARCH_TOP_K
                if settings.kb.ENABLE_CORRECTIVE_WEB_SEARCH and initial_grade.grade == "partial"
                else 0
            ),
        ),
    )
    follow_up_references: list[RetrievedReference] = []
    second_pass_error = ""
    try:
        follow_up_references = retrieve(
            follow_up_query,
            second_pass_top_k,
            second_pass_threshold,
        )
    except Exception as exc:
        second_pass_error = str(exc)

    final_references = merge_corrective_references(
        references,
        follow_up_references,
        limit=final_limit,
    )
    web_search_triggered = False
    web_search_query = ""
    web_reference_count = 0
    web_sources: list[str] = []
    web_error_message = ""
    if (
        search_web is not None
        and settings.kb.ENABLE_CORRECTIVE_WEB_SEARCH
        and initial_grade.grade == "partial"
    ):
        web_search_triggered = True
        web_search_query = follow_up_query or query.strip()
        try:
            web_references = search_web(web_search_query)
        except Exception as exc:
            web_error_message = str(exc)
        else:
            web_reference_count = len(web_references)
            web_sources = [item.source for item in web_references[:6]]
            final_references = merge_corrective_references(
                final_references,
                web_references,
                limit=final_limit,
            )

    error_messages = [message for message in (second_pass_error, web_error_message) if message]
    append_corrective_trace(
        settings=settings,
        query=query,
        source_type=source_type,
        target_name=target_name,
        initial_reference_count=len(references),
        initial_grade=initial_grade,
        triggered=True,
        follow_up_query=follow_up_query,
        follow_up_reference_count=len(follow_up_references),
        final_references=final_references,
        error_message="; ".join(error_messages),
        web_search_triggered=web_search_triggered,
        web_search_query=web_search_query,
        web_reference_count=web_reference_count,
        web_sources=web_sources,
        web_error_message=web_error_message,
    )
    return final_references


def build_rag_prompt(
    query: str,
    references: list[RetrievedReference],
) -> ChatPromptTemplate:
    system_prompt = select_rag_system_prompt(query, references)
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("history"),
            (
                "human",
                "{memory_section}参考上下文如下：\n{context}\n\n{coverage_requirements}"
                "用户问题：\n{query}\n\n"
                "请基于参考上下文回答问题。"
                "若上方提供长期记忆证据且与知识库证据冲突，须分别说明来源，勿强行合并。",
            ),
        ]
    )


def select_rag_system_prompt(
    query: str,
    references: list[RetrievedReference],
) -> str:
    if should_use_image_rag_prompt(query, references):
        return IMAGE_RAG_SYSTEM_PROMPT
    return RAG_SYSTEM_PROMPT


def resolve_rag_prompt_kind(
    query: str,
    references: list[RetrievedReference],
) -> str:
    return "image" if should_use_image_rag_prompt(query, references) else "default"


def should_use_image_rag_prompt(
    query: str,
    references: list[RetrievedReference],
) -> bool:
    lowered_query = query.strip().lower()
    query_hint_hit = any(hint in lowered_query for hint in IMAGE_QUERY_HINTS)

    image_evidence_count = 0
    text_evidence_count = 0
    for ref in references:
        group = resolve_reference_context_group(ref)
        if group in {"ocr", "vision"}:
            image_evidence_count += 1
        else:
            text_evidence_count += 1

    if query_hint_hit and image_evidence_count > 0:
        return True
    if image_evidence_count > 0 and text_evidence_count == 0:
        return True
    if image_evidence_count >= 2 and image_evidence_count >= text_evidence_count:
        return True
    if query_hint_hit and not references:
        return True
    return False


def build_rag_variables(
    query: str,
    references: list[RetrievedReference],
    history: list[ChatMessage],
    *,
    agent_memory_context: str = "",
    compress_context: bool = False,
) -> dict[str, object]:
    history_messages = convert_history(history)
    context_meta = build_context(references, compress=compress_context)
    memory_section = ""
    trimmed_memory = agent_memory_context.strip()
    if trimmed_memory:
        memory_section = (
            "【长期记忆证据】（来自历史会话抽取，可能与知识库不一致）\n"
            f"{trimmed_memory}\n\n"
        )
    return {
        "history": history_messages,
        "memory_section": memory_section,
        "context": context_meta.text if context_meta.text else "当前没有检索到可用上下文。",
        "coverage_requirements": build_coverage_requirements(query),
        "query": query,
        "_context_meta": context_meta,
    }


def build_context(
    references: list[RetrievedReference],
    *,
    compress: bool = False,
) -> ContextBuildResult:
    if not references:
        return ContextBuildResult(text="", reference_count=0, context_chars=0)

    ordered_references = compress_references_for_answer(references) if compress else references

    grouped_blocks = {
        "text": [],
        "ocr": [],
        "vision": [],
    }

    for index, ref in enumerate(ordered_references, start=1):
        grouped_blocks[resolve_reference_context_group(ref)].append(
            format_reference_block(index, ref, compressed=compress)
        )

    sections: list[str] = []
    section_order = (
        ("text", "文本证据"),
        ("ocr", "OCR 证据"),
        ("vision", "视觉描述证据"),
    )
    for key, title in section_order:
        blocks = grouped_blocks[key]
        if not blocks:
            continue
        sections.append(f"## {title}\n" + "\n\n".join(blocks))

    text = "\n\n".join(sections)
    return ContextBuildResult(
        text=text,
        reference_count=len(ordered_references),
        context_chars=len(text),
    )


def build_coverage_requirements(query: str) -> str:
    normalized = str(query or "").strip()
    if not normalized:
        return ""

    lines = split_query_into_requirements(normalized)
    if len(lines) <= 1:
        return ""

    rendered = "\n".join(
        f"{index}. {line}" for index, line in enumerate(lines, start=1)
    )
    return (
        "请确保回答至少覆盖以下子问题或要点：\n"
        f"{rendered}\n\n"
    )


def split_query_into_requirements(query: str) -> list[str]:
    cleaned = query.replace("问题：", " ").replace("请回答：", " ").strip()
    for token in ("\r", "\n", "\t"):
        cleaned = cleaned.replace(token, " ")
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return []

    candidate_parts = [cleaned]
    split_patterns = (
        "同时",
        "以及",
        "并且",
        "并说明",
        "并回答",
        "并分别",
        "分别说明",
        "分别指出",
        "；",
        ";",
        "？",
        "?",
    )
    for marker in split_patterns:
        next_parts: list[str] = []
        changed = False
        for part in candidate_parts:
            pieces = [piece.strip(" ，,。；;？?") for piece in part.split(marker)]
            pieces = [piece for piece in pieces if piece]
            if len(pieces) > 1:
                changed = True
                next_parts.extend(pieces)
            else:
                next_parts.append(part)
        candidate_parts = next_parts
        if changed:
            break

    deduped: list[str] = []
    for part in candidate_parts:
        normalized_part = part.strip(" ，,。；;？?")
        if not normalized_part:
            continue
        if normalized_part not in deduped:
            deduped.append(normalized_part)
    return deduped


def maybe_refine_rag_answer(
    *,
    settings: AppSettings,
    query: str,
    references: list[RetrievedReference],
    context: str,
    coverage_requirements: str,
    draft_answer: str,
) -> str:
    if not should_run_answer_completeness_review(
        query=query,
        references=references,
        coverage_requirements=coverage_requirements,
        draft_answer=draft_answer,
    ):
        return draft_answer

    model_name = settings.model.QUERY_REWRITE_MODEL.strip() or settings.model.DEFAULT_LLM_MODEL
    try:
        llm = build_chat_model(settings, model_name=model_name, temperature=0.0)
        chain = RAG_COMPLETENESS_REVIEW_PROMPT | llm | StrOutputParser()
        raw_output = chain.invoke(
            {
                "context": context,
                "coverage_requirements": coverage_requirements,
                "query": query.strip(),
                "draft_answer": draft_answer.strip(),
            }
        )
    except Exception:
        return draft_answer

    payload = extract_json_payload(raw_output)
    if not isinstance(payload, dict):
        return draft_answer

    status = str(payload.get("status", "")).strip().lower()
    revised_answer = str(payload.get("revised_answer", "")).strip()
    if status != "revise" or not revised_answer:
        return draft_answer
    return revised_answer


def extract_rag_evidence(
    *,
    settings: AppSettings,
    query: str,
    context: str,
    coverage_requirements: str,
) -> EvidenceExtractionResult:
    if not context.strip() or context.strip() == "当前没有检索到可用上下文。":
        return EvidenceExtractionResult(
            facts_text="暂无可用证据事实。",
            fact_count=0,
            unknown_count=1,
        )

    model_name = settings.model.QUERY_REWRITE_MODEL.strip() or settings.model.DEFAULT_LLM_MODEL
    try:
        llm = build_chat_model(settings, model_name=model_name, temperature=0.0)
        chain = RAG_EVIDENCE_EXTRACTION_PROMPT | llm | StrOutputParser()
        raw_output = chain.invoke(
            {
                "query": query.strip(),
                "coverage_requirements": coverage_requirements,
                "context": context,
            }
        )
    except Exception:
        return EvidenceExtractionResult(
            facts_text=context,
            fact_count=0,
            unknown_count=0,
        )

    payload = extract_json_payload(raw_output)
    if not isinstance(payload, dict):
        return EvidenceExtractionResult(
            facts_text=context,
            fact_count=0,
            unknown_count=0,
        )

    facts = [str(item).strip() for item in payload.get("facts", []) if str(item).strip()]
    unknown_points = [str(item).strip() for item in payload.get("unknown_points", []) if str(item).strip()]
    lines: list[str] = []
    if facts:
        lines.append("已确认事实：")
        lines.extend(f"{index}. {fact}" for index, fact in enumerate(facts, start=1))
    if unknown_points:
        lines.append("\n证据不足点：")
        lines.extend(f"- {item}" for item in unknown_points)
    facts_text = "\n".join(lines).strip() or context
    return EvidenceExtractionResult(
        facts_text=facts_text,
        fact_count=len(facts),
        unknown_count=len(unknown_points),
    )


def generate_answer_from_evidence(
    *,
    llm: object,
    history: list[ChatMessage],
    memory_section: str,
    query: str,
    coverage_requirements: str,
    evidence_facts: str,
) -> str:
    chain = RAG_ANSWER_FROM_EVIDENCE_PROMPT | llm | StrOutputParser()
    return chain.invoke(
        {
            "history": convert_history(history),
            "memory_section": memory_section,
            "query": query,
            "coverage_requirements": coverage_requirements,
            "evidence_facts": evidence_facts,
        }
    )


def should_run_answer_completeness_review(
    *,
    query: str,
    references: list[RetrievedReference],
    coverage_requirements: str,
    draft_answer: str,
) -> bool:
    if not references:
        return False
    if should_use_image_rag_prompt(query, references):
        return False
    if not draft_answer.strip():
        return False
    if coverage_requirements.strip():
        return True

    checklist_hints = ("哪些", "哪几", "分别", "同时", "以及", "角色", "原因", "措施", "步骤", "职责")
    return any(hint in query for hint in checklist_hints)


def compress_references_for_answer(
    references: list[RetrievedReference],
    *,
    max_blocks: int = 4,
    max_chars: int = 2200,
) -> list[RetrievedReference]:
    grouped: dict[str, list[RetrievedReference]] = {}
    for ref in references:
        sample_id = infer_reference_sample_id(ref)
        key = sample_id or f"source:{ref.source_path or ref.source}"
        grouped.setdefault(key, []).append(ref)

    selected: list[RetrievedReference] = []
    total_chars = 0
    for _, group in sorted(
        grouped.items(),
        key=lambda item: max(ref.relevance_score for ref in item[1]),
        reverse=True,
    ):
        for ref in dedupe_reference_group(group):
            preview_len = len(build_reference_evidence_text(ref))
            if selected and total_chars + preview_len > max_chars:
                break
            selected.append(ref)
            total_chars += preview_len
            if len(selected) >= max_blocks:
                return selected
        if len(selected) >= max_blocks or total_chars >= max_chars:
            break
    return selected or references[:max_blocks]


def dedupe_reference_group(group: list[RetrievedReference]) -> list[RetrievedReference]:
    best_by_fingerprint: dict[str, RetrievedReference] = {}
    for ref in sorted(group, key=lambda item: item.relevance_score, reverse=True):
        fingerprint = build_reference_fingerprint(ref)
        existing = best_by_fingerprint.get(fingerprint)
        if existing is None or is_reference_better(ref, existing):
            best_by_fingerprint[fingerprint] = ref
    return sorted(best_by_fingerprint.values(), key=lambda item: item.relevance_score, reverse=True)


def is_reference_better(candidate: RetrievedReference, existing: RetrievedReference) -> bool:
    if bool(candidate.evidence_summary) != bool(existing.evidence_summary):
        return bool(candidate.evidence_summary)
    candidate_len = len(build_reference_evidence_text(candidate))
    existing_len = len(build_reference_evidence_text(existing))
    if candidate_len != existing_len:
        return candidate_len < existing_len
    return candidate.relevance_score > existing.relevance_score


def build_reference_fingerprint(ref: RetrievedReference) -> str:
    base = ref.evidence_summary or ref.content_preview or ref.content or ""
    normalized = normalize_text_for_fingerprint(base)
    return normalized[:160] or ref.chunk_id


def build_reference_evidence_text(ref: RetrievedReference) -> str:
    if ref.evidence_summary:
        return ref.evidence_summary.strip()
    if ref.content_preview:
        return ref.content_preview.strip()
    return ref.content.strip()[:420]


def normalize_text_for_fingerprint(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip().lower()
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", "", normalized)
    return normalized


def infer_reference_sample_id(ref: RetrievedReference) -> str:
    for value in (ref.source_path, ref.chunk_id, ref.source):
        for part in str(value or "").replace("\\", "/").split("/"):
            if re.fullmatch(r"[0-9a-f]{24}", part, flags=re.IGNORECASE):
                return part
    return ""


def count_coverage_requirements(text: str) -> int:
    stripped = (text or "").strip()
    if not stripped:
        return 0
    return sum(1 for line in stripped.splitlines() if re.match(r"^\d+\.\s", line.strip()))


def grade_documents(
    *,
    settings: AppSettings,
    query: str,
    references: list[RetrievedReference],
    history: list[ChatMessage],
) -> RetrievalCoverageGrade:
    heuristic = heuristic_grade_documents(references)
    if not references:
        return heuristic

    parsed = invoke_corrective_grade_llm(
        settings=settings,
        query=query,
        references=references[: settings.kb.CORRECTIVE_RAG_MAX_REFERENCES_TO_GRADE],
        history=history,
    )
    if parsed is None:
        return heuristic
    return parsed


def heuristic_grade_documents(
    references: list[RetrievedReference],
) -> RetrievalCoverageGrade:
    if not references:
        return RetrievalCoverageGrade(
            grade="insufficient",
            reason="当前未检索到相关证据。",
            missing_aspects="缺少可回答问题的上下文。",
        )

    relevance_scores = [max(0.0, float(item.relevance_score)) for item in references]
    max_relevance = max(relevance_scores, default=0.0)
    average_relevance = sum(relevance_scores) / max(1, len(relevance_scores))
    if len(references) >= 2 and max_relevance >= 0.78 and average_relevance >= 0.62:
        return RetrievalCoverageGrade(
            grade="relevant",
            reason="当前证据覆盖度较高。",
            missing_aspects="",
        )
    if max_relevance >= 0.55 or average_relevance >= 0.42:
        return RetrievalCoverageGrade(
            grade="partial",
            reason="已有部分相关证据，但覆盖不完整。",
            missing_aspects="可能缺少关键条件或补充细节。",
        )
    return RetrievalCoverageGrade(
        grade="insufficient",
        reason="当前证据相关性偏弱。",
        missing_aspects="需要更聚焦的检索结果。",
    )


def invoke_corrective_grade_llm(
    *,
    settings: AppSettings,
    query: str,
    references: list[RetrievedReference],
    history: list[ChatMessage],
) -> RetrievalCoverageGrade | None:
    model_name = settings.model.QUERY_REWRITE_MODEL.strip() or settings.model.DEFAULT_LLM_MODEL
    try:
        llm = build_chat_model(settings, model_name=model_name, temperature=0.0)
        chain = CORRECTIVE_RAG_GRADE_PROMPT | llm | StrOutputParser()
        raw_output = chain.invoke(
            {
                "query": query.strip(),
                "history_text": format_history(history),
                "reference_summary": summarize_references_for_grading(references),
            }
        )
    except Exception:
        return None

    payload = extract_json_payload(raw_output)
    if not isinstance(payload, dict):
        return None
    grade = str(payload.get("grade", "")).strip().lower()
    if grade not in {"relevant", "partial", "insufficient"}:
        return None
    reason = str(payload.get("reason", "")).strip() or "模型未提供理由。"
    missing_aspects = str(payload.get("missing_aspects", "")).strip()
    return RetrievalCoverageGrade(
        grade=grade,
        reason=reason[:120],
        missing_aspects=missing_aspects[:120],
    )


def generate_corrective_query(
    *,
    settings: AppSettings,
    query: str,
    references: list[RetrievedReference],
    history: list[ChatMessage],
    grade: RetrievalCoverageGrade,
) -> str:
    model_name = settings.model.QUERY_REWRITE_MODEL.strip() or settings.model.DEFAULT_LLM_MODEL
    try:
        llm = build_chat_model(settings, model_name=model_name, temperature=0.0)
        chain = CORRECTIVE_RAG_QUERY_PROMPT | llm | StrOutputParser()
        raw_output = chain.invoke(
            {
                "query": query.strip(),
                "history_text": format_history(history),
                "grade": grade.grade,
                "reason": grade.reason,
                "missing_aspects": grade.missing_aspects or "未明确指出。",
                "reference_summary": summarize_references_for_grading(
                    references[: settings.kb.CORRECTIVE_RAG_MAX_REFERENCES_TO_GRADE]
                ),
            }
        )
    except Exception:
        return query.strip()

    cleaned = sanitize_rewritten_query(raw_output)
    if not cleaned:
        return query.strip()
    if len(cleaned) > max(128, len(query.strip()) * 2):
        return query.strip()
    return cleaned


def summarize_references_for_grading(
    references: list[RetrievedReference],
) -> str:
    if not references:
        return "无可用证据。"

    lines: list[str] = []
    for index, ref in enumerate(references, start=1):
        summary_parts = [f"[{index}] source={ref.source}"]
        if ref.section_title:
            summary_parts.append(f"section={ref.section_title}")
        summary_parts.append(f"relevance={ref.relevance_score:.3f}")
        preview = (ref.evidence_summary or ref.content_preview or ref.content).replace("\n", " ").strip()
        lines.append(" | ".join(summary_parts) + f" | preview={preview[:180]}")
    return "\n".join(lines)


def extract_json_payload(text: str) -> dict[str, object] | None:
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    if cleaned.startswith("```"):
        lines = [line for line in cleaned.splitlines() if not line.startswith("```")]
        cleaned = "\n".join(lines).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def merge_corrective_references(
    initial: list[RetrievedReference],
    follow_up: list[RetrievedReference],
    *,
    limit: int,
) -> list[RetrievedReference]:
    best_by_chunk_id: dict[str, RetrievedReference] = {}
    for item in [*initial, *follow_up]:
        existing = best_by_chunk_id.get(item.chunk_id)
        if existing is None or item.relevance_score > existing.relevance_score:
            best_by_chunk_id[item.chunk_id] = item
    ranked = sorted(
        best_by_chunk_id.values(),
        key=lambda item: item.relevance_score,
        reverse=True,
    )
    return ranked[:limit]


def resolve_reference_context_group(ref: RetrievedReference) -> str:
    source_modality = (ref.source_modality or "").strip().lower()
    if source_modality == "ocr":
        return "ocr"
    if source_modality in {"vision", "image"}:
        return "vision"
    if source_modality == "ocr+vision":
        return "vision"
    if ref.ocr_text and not ref.image_caption:
        return "ocr"
    if ref.image_caption:
        return "vision"
    return "text"


def format_reference_block(index: int, ref: RetrievedReference, *, compressed: bool = False) -> str:
    metadata_parts = [f"source={ref.source}"]
    if ref.page is not None:
        metadata_parts.append(f"page={ref.page}")
    if ref.section_title:
        metadata_parts.append(f"section={ref.section_title}")
    if ref.source_modality:
        metadata_parts.append(f"modality={ref.source_modality}")
    if ref.content_type:
        metadata_parts.append(f"type={ref.content_type}")
    metadata_parts.append(f"relevance={ref.relevance_score:.3f}")

    evidence_lines: list[str] = [f"[{index}] " + " | ".join(metadata_parts)]
    if ref.evidence_summary:
        evidence_lines.append(f"evidence_summary: {ref.evidence_summary}")
    if ref.ocr_text:
        evidence_lines.append(f"ocr_text: {ref.ocr_text[:240]}")
    if ref.image_caption:
        evidence_lines.append(f"image_caption: {ref.image_caption[:240]}")
    body = build_reference_evidence_text(ref) if compressed else ref.content
    evidence_lines.append(f"content:\n{body}")
    return "\n".join(evidence_lines)


def append_answer_trace(
    *,
    settings: AppSettings,
    query: str,
    references: list[RetrievedReference],
    prompt_kind: str,
    answer: str,
    compressed_reference_count: int | None = None,
    compressed_context_chars: int | None = None,
    evidence_fact_count: int | None = None,
    evidence_unknown_count: int | None = None,
    coverage_requirement_count: int | None = None,
) -> None:
    source_modalities = count_reference_attribute(references, "source_modality")
    evidence_types = count_reference_attribute(references, "evidence_type")
    context_groups = count_context_groups(references)
    has_text_context = context_groups.get("text", 0) > 0
    has_image_context = (
        context_groups.get("ocr", 0) > 0 or context_groups.get("vision", 0) > 0
    )
    append_jsonl_trace(
        settings,
        "answer_trace",
        {
            "event_type": "answer",
            "query": query,
            "prompt_kind": prompt_kind,
            "reference_count": len(references),
            "source_modalities": source_modalities,
            "evidence_types": evidence_types,
            "context_groups": context_groups,
            "has_text_evidence": has_text_context,
            "has_ocr_evidence": "ocr" in source_modalities or "ocr+vision" in source_modalities,
            "has_vision_evidence": "vision" in source_modalities or "ocr+vision" in source_modalities or "image" in source_modalities,
            "has_image_side_evidence": has_image_context,
            "has_joint_text_image_evidence": has_text_context and has_image_context,
            "has_multimodal_context": len([key for key, value in context_groups.items() if value > 0]) >= 2,
            "compressed_reference_count": compressed_reference_count,
            "compressed_context_chars": compressed_context_chars,
            "evidence_fact_count": evidence_fact_count,
            "evidence_unknown_count": evidence_unknown_count,
            "coverage_requirement_count": coverage_requirement_count,
            "answer_preview": answer[:240],
        },
    )


def append_corrective_trace(
    *,
    settings: AppSettings,
    query: str,
    source_type: str,
    target_name: str,
    initial_reference_count: int,
    initial_grade: RetrievalCoverageGrade,
    triggered: bool,
    follow_up_query: str,
    follow_up_reference_count: int,
    final_references: list[RetrievedReference],
    error_message: str = "",
    web_search_triggered: bool = False,
    web_search_query: str = "",
    web_reference_count: int = 0,
    web_sources: list[str] | None = None,
    web_error_message: str = "",
) -> None:
    append_jsonl_trace(
        settings,
        "corrective_rag_trace",
        {
            "event_type": "corrective_rag",
            "query": query,
            "source_type": source_type,
            "target_name": target_name,
            "initial_reference_count": initial_reference_count,
            "initial_grade": initial_grade.grade,
            "grade_reason": initial_grade.reason,
            "missing_aspects": initial_grade.missing_aspects,
            "triggered": triggered,
            "follow_up_query": follow_up_query,
            "follow_up_reference_count": follow_up_reference_count,
            "web_search_triggered": web_search_triggered,
            "web_search_query": web_search_query,
            "web_reference_count": web_reference_count,
            "web_sources": web_sources or [],
            "web_error_message": web_error_message,
            "final_reference_count": len(final_references),
            "final_sources": [item.source for item in final_references[:6]],
            "error_message": error_message,
        },
    )


def count_reference_attribute(
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


def count_context_groups(
    references: list[RetrievedReference],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ref in references:
        key = resolve_reference_context_group(ref)
        counts[key] = counts.get(key, 0) + 1
    return counts


def convert_history(history: list[ChatMessage]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for item in history:
        if item.role == "user":
            messages.append(HumanMessage(content=item.content))
        elif item.role == "assistant":
            messages.append(AIMessage(content=item.content))
        elif item.role == "system":
            messages.append(SystemMessage(content=item.content))
    return messages
