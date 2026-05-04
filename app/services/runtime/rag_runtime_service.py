"""RAG 运行时编排辅助服务。"""

from __future__ import annotations

import json
from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.constants import IMAGE_QUERY_HINTS
from app.schemas.chat import ChatMessage, RetrievedReference
from app.services.core.observability import append_jsonl_trace
from app.services.core.settings import AppSettings
from app.services.models.llm_service import build_chat_model
from app.services.retrieval.answer_guard_service import (
    build_answer_requirements,
    build_coverage_requirements,
    is_temporal_answer_query,
    split_query_into_requirements,
)
from app.services.retrieval.context_build_service import build_context
from app.services.retrieval.evidence_packing_service import (
    count_context_groups,
    count_reference_attribute,
    resolve_reference_context_group,
)
from app.services.retrieval.query_answer_policy_service import build_query_answer_policy
from app.services.retrieval.query_rewrite_service import format_history, sanitize_rewritten_query
from app.services.retrieval.reference_overview import build_reference_overview


@dataclass(frozen=True)
class RetrievalCoverageGrade:
    grade: str
    reason: str
    missing_aspects: str = ""


def build_rag_prompt(
    query: str,
    references: list[RetrievedReference],
    *,
    system_prompt: str,
    image_system_prompt: str,
) -> ChatPromptTemplate:
    selected_system_prompt = select_rag_system_prompt(
        query,
        references,
        system_prompt=system_prompt,
        image_system_prompt=image_system_prompt,
    )
    return ChatPromptTemplate.from_messages(
        [
            ("system", selected_system_prompt),
            MessagesPlaceholder("history"),
            (
                "human",
                "{memory_section}参考上下文如下：\n{context}\n\n{coverage_requirements}{answer_requirements}"
                "用户问题：\n{query}\n\n"
                "请基于参考上下文回答问题。"
                "若上方提供长期记忆证据且与知识库证据冲突，须分别说明来源，勿强行合并。",
            ),
        ]
    )


def select_rag_system_prompt(
    query: str,
    references: list[RetrievedReference],
    *,
    system_prompt: str,
    image_system_prompt: str,
) -> str:
    if should_use_image_rag_prompt(query, references):
        return image_system_prompt
    return system_prompt


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
    settings: AppSettings,
    query: str,
    references: list[RetrievedReference],
    history: list[ChatMessage],
    *,
    agent_memory_context: str = "",
) -> dict[str, object]:
    policy = build_query_answer_policy(query, references)
    history_messages = convert_history(history)
    coverage_requirements = build_coverage_requirements(
        query,
        references,
        settings=settings,
    )
    answer_requirements = build_answer_requirements(
        query,
        references,
        settings=settings,
    )
    context = build_context(
        query,
        references,
        policy=policy,
    )
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
        "context": context if context else "当前没有检索到可用上下文。",
        "coverage_requirements": coverage_requirements,
        "answer_requirements": answer_requirements,
        "query": query,
    }


def grade_documents(
    *,
    settings: AppSettings,
    query: str,
    references: list[RetrievedReference],
    history: list[ChatMessage],
    corrective_grade_prompt: ChatPromptTemplate,
) -> RetrievalCoverageGrade:
    heuristic = heuristic_grade_documents(references)
    if not references:
        return heuristic

    parsed = invoke_corrective_grade_llm(
        settings=settings,
        query=query,
        references=references[: settings.kb.CORRECTIVE_RAG_MAX_REFERENCES_TO_GRADE],
        history=history,
        corrective_grade_prompt=corrective_grade_prompt,
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
    average_relevance = sum(relevance_scores) / max(1, len(references))
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
    corrective_grade_prompt: ChatPromptTemplate,
) -> RetrievalCoverageGrade | None:
    model_name = settings.model.QUERY_REWRITE_MODEL.strip() or settings.model.DEFAULT_LLM_MODEL
    try:
        llm = build_chat_model(settings, model_name=model_name, temperature=0.0)
        chain = corrective_grade_prompt | llm | StrOutputParser()
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
    corrective_query_prompt: ChatPromptTemplate,
) -> str:
    model_name = settings.model.QUERY_REWRITE_MODEL.strip() or settings.model.DEFAULT_LLM_MODEL
    try:
        llm = build_chat_model(settings, model_name=model_name, temperature=0.0)
        chain = corrective_query_prompt | llm | StrOutputParser()
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


def append_answer_trace(
    *,
    settings: AppSettings,
    query: str,
    references: list[RetrievedReference],
    prompt_kind: str,
    answer: str,
) -> None:
    policy = build_query_answer_policy(query, references)
    reference_overview = build_reference_overview(references)
    source_modalities = count_reference_attribute(references, "source_modality")
    evidence_types = count_reference_attribute(references, "evidence_type")
    context_groups = count_context_groups(references)
    has_text_context = context_groups.get("text", 0) > 0
    has_timeseries_context = context_groups.get("timeseries", 0) > 0
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
            "ts_reference_count": reference_overview.timeseries_count,
            "source_modalities": source_modalities,
            "evidence_types": evidence_types,
            "context_groups": context_groups,
            "has_text_evidence": has_text_context,
            "has_ts_evidence": reference_overview.timeseries_count > 0,
            "has_timeseries_evidence": has_timeseries_context,
            "has_ocr_evidence": "ocr" in source_modalities or "ocr+vision" in source_modalities,
            "has_vision_evidence": "vision" in source_modalities or "ocr+vision" in source_modalities or "image" in source_modalities,
            "has_image_side_evidence": has_image_context,
            "has_joint_text_image_evidence": has_text_context and has_image_context,
            "has_text_ts_joint_coverage": reference_overview.has_text_ts_joint_coverage,
            "has_text_ts_joint_evidence": has_text_context and has_timeseries_context,
            "has_multimodal_context": len([key for key, value in context_groups.items() if value > 0]) >= 2,
            "direct_answer_query": policy.should_direct_answer,
            "temporal_answer_query": is_temporal_answer_query(query),
            "temporal_constraint_detected": is_temporal_answer_query(query),
            "coverage_requirement_count": policy.requirement_count,
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
