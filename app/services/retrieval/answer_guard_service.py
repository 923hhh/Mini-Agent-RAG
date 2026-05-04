"""RAG 答案审校与完备性保护服务。"""

from __future__ import annotations

import json
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.constants import IMAGE_QUERY_HINTS
from app.schemas.chat import RetrievedReference
from app.services.core.settings import AppSettings
from app.services.models.llm_service import build_chat_model
from app.services.retrieval.evidence_packing_service import resolve_reference_context_group


def is_timeseries_answer_guard_enabled(settings: AppSettings | None) -> bool:
    if settings is None:
        return True
    return bool(getattr(settings.kb, "ENABLE_TIMESERIES_RETRIEVAL_EXTENSION", True))


def build_coverage_requirements(
    query: str,
    references: list[RetrievedReference] | None = None,
    *,
    settings: AppSettings | None = None,
) -> str:
    normalized = extract_primary_question_text(query)
    if not normalized:
        return ""

    lines = split_query_into_requirements(normalized)
    if len(lines) <= 1:
        lines = infer_comparative_coverage_points(normalized)
    if is_timeseries_answer_guard_enabled(settings):
        timeseries_points = infer_timeseries_coverage_points(
            normalized,
            references or [],
        )
        if timeseries_points:
            for item in timeseries_points:
                if item not in lines:
                    lines.append(item)
    if is_procedural_query(normalized) and has_numbered_step_references(references or []):
        procedural_points = [
            "完整覆盖上下文中已经给出的操作步骤",
            "不要遗漏后续编号步骤、注意事项或取下部件清单",
        ]
        for item in procedural_points:
            if item not in lines:
                lines.append(item)
    if not lines:
        return ""
    return "【回答覆盖要求】\n- " + "\n- ".join(lines) + "\n\n"


def build_answer_requirements(
    query: str,
    references: list[RetrievedReference] | None = None,
    *,
    settings: AppSettings | None = None,
) -> str:
    normalized = extract_primary_question_text(query)
    if not normalized:
        return ""

    lines: list[str] = []
    if should_directly_answer_query(normalized):
        lines.append("第一句先直接给出结论，再补充必要说明。")
    if is_multi_doc_comparative_query(normalized):
        lines.append("若涉及多个对象，必须逐项对照说明，不得只给笼统总结。")
    if is_temporal_answer_query(normalized):
        lines.append("回答中必须保留具体时间点、年份或日期，不要弱化时间条件。")
    if is_procedural_query(normalized):
        lines.append("若上下文包含编号步骤，必须按原编号顺序完整列出，不得擅自截断或合并后续步骤。")
        lines.append("不要写“上下文未完整列出”“具体内容略”等保守性说明，除非证据确实缺失。")
        if has_numbered_step_references(references or []):
            lines.append("优先保留原文中的部件名、螺栓规格、数量和提示信息，不要只做笼统概括。")
    if (
        is_timeseries_answer_guard_enabled(settings)
        and has_timeseries_references(references or [])
        and has_text_references(references or [])
    ):
        lines.append("若同时存在时间序列证据与文本证据，回答需分别覆盖趋势观察与事件背景。")
    if not lines:
        return ""
    return "【回答形式要求】\n- " + "\n- ".join(lines) + "\n\n"


def split_query_into_requirements(query: str) -> list[str]:
    normalized = str(query or "").strip()
    if not normalized:
        return []
    separators = ("；", ";", "。", "\n")
    segments = [normalized]
    for separator in separators:
        next_segments: list[str] = []
        for segment in segments:
            next_segments.extend(part.strip(" ，,。；;？?") for part in segment.split(separator))
        segments = [part for part in next_segments if part]

    conjunction_markers = ("分别", "同时", "以及", "并且", "还要", "此外")
    requirements: list[str] = []
    for segment in segments:
        requirements.append(segment)
        for marker in conjunction_markers:
            if marker in segment:
                parts = [part.strip(" ，,。；;？?") for part in segment.split(marker)]
                for part in parts:
                    if part and part not in requirements:
                        requirements.append(part)
    deduped: list[str] = []
    for item in requirements:
        if item and item not in deduped:
            deduped.append(item)
    return deduped


def has_timeseries_references(references: list[RetrievedReference]) -> bool:
    return any(
        (ref.source_modality or "").strip().lower() == "timeseries"
        for ref in references
    )


def has_text_references(references: list[RetrievedReference]) -> bool:
    return any(
        resolve_reference_context_group(ref) == "text"
        and (ref.source_modality or "").strip().lower() != "timeseries"
        for ref in references
    )


def answer_mentions_timeseries_observation(answer: str) -> bool:
    normalized = str(answer or "").strip()
    if not normalized:
        return False
    observation_markers = (
        "趋势",
        "上升",
        "下降",
        "波动",
        "峰值",
        "谷值",
        "时间范围",
    )
    return any(marker in normalized for marker in observation_markers)


def answer_mentions_event_background(answer: str) -> bool:
    normalized = str(answer or "").strip()
    if not normalized:
        return False
    background_markers = (
        "背景",
        "原因",
        "事件",
        "由于",
        "导致",
        "文本证据",
        "冷空气",
    )
    return any(marker in normalized for marker in background_markers)


def extract_primary_question_text(query: str) -> str:
    cleaned = str(query or "").strip()
    if not cleaned:
        return ""

    question_markers = (
        "问题：",
        "问题:",
        "请回答：",
        "请回答:",
        "问：",
        "问:",
    )
    last_marker_index = -1
    last_marker_length = 0
    for marker in question_markers:
        marker_index = cleaned.rfind(marker)
        if marker_index > last_marker_index:
            last_marker_index = marker_index
            last_marker_length = len(marker)

    if last_marker_index >= 0:
        candidate = cleaned[last_marker_index + last_marker_length :].strip()
        if candidate:
            return candidate
    return cleaned


def infer_comparative_coverage_points(query: str) -> list[str]:
    normalized = str(query or "").strip(" ，,。；;？?")
    if not normalized or not is_multi_doc_comparative_query(normalized):
        return []

    points: list[str] = []
    if "相似" in normalized and any(marker in normalized for marker in ("之处", "点", "特征", "特点")):
        points.append("比较对象的相似之处或共同点")
    if "共同" in normalized and any(marker in normalized for marker in ("目标", "点", "特征", "特点")):
        points.append("比较对象的共同目标或共同点")
    if any(marker in normalized for marker in ("独特特点", "各自特点", "特点", "特征", "区别", "不同")):
        points.append("比较对象各自的独特特点或区别")

    deduped: list[str] = []
    for point in points:
        if point not in deduped:
            deduped.append(point)
    return deduped


def infer_timeseries_coverage_points(
    query: str,
    references: list[RetrievedReference],
) -> list[str]:
    if not has_timeseries_references(references):
        return []

    points: list[str] = ["时间范围与主要变化趋势"]
    normalized = str(query or "").strip()
    if any(marker in normalized for marker in ("异常", "峰值", "谷值", "拐点", "波动")):
        points.append("关键异常点或峰值谷值")
    if has_text_references(references) and any(
        marker in normalized for marker in ("背景", "原因", "事件", "为什么", "影响")
    ):
        points.append("对应事件背景或文本原因说明")

    deduped: list[str] = []
    for point in points:
        if point not in deduped:
            deduped.append(point)
    return deduped


def should_directly_answer_query(query: str) -> bool:
    normalized = extract_primary_question_text(query)
    if not normalized:
        return False
    if len(split_query_into_requirements(normalized)) > 1:
        return False

    direct_markers = (
        "谁",
        "哪位",
        "哪个",
        "是什么",
        "多少",
        "几",
        "哪一年",
        "哪年",
        "何时",
        "什么时候",
        "什么时间",
        "现任",
        "当前",
        "最新",
    )
    return any(marker in normalized for marker in direct_markers)


def is_procedural_query(query: str) -> bool:
    normalized = extract_primary_question_text(query)
    if not normalized:
        return False
    procedural_markers = (
        "步骤",
        "流程",
        "如何",
        "怎么",
        "怎样",
        "拆卸",
        "安装",
        "更换",
        "检查",
        "测量",
        "调整",
        "操作",
    )
    return any(marker in normalized for marker in procedural_markers)


def is_numeric_fact_query(query: str) -> bool:
    normalized = extract_primary_question_text(query)
    if not normalized:
        return False
    numeric_markers = (
        "多少",
        "几",
        "数值",
        "标准值",
        "范围",
        "扭矩",
        "压力",
        "间隙",
        "规格",
        "尺寸",
        "厚度",
        "温度",
        "电压",
        "电流",
    )
    return any(marker in normalized for marker in numeric_markers)


def is_symbol_explanation_query(query: str) -> bool:
    normalized = extract_primary_question_text(query)
    if not normalized:
        return False
    explanation_markers = (
        "分别表示什么",
        "表示什么",
        "分别代表什么",
        "代表什么",
        "含义是什么",
        "什么意思",
    )
    if any(marker in normalized for marker in explanation_markers):
        return True
    symbol_markers = ("IN", "EX", "L", "M", "A", "B", "C", "D")
    upper = normalized.upper()
    symbol_count = sum(1 for marker in symbol_markers if marker in upper)
    return symbol_count >= 2


def has_numbered_step_references(references: list[RetrievedReference]) -> bool:
    numbered_line_count = 0
    for ref in references:
        text = str(ref.content or "")
        for line in text.splitlines():
            normalized = line.strip()
            if re.match(r"^\d+[.、]\s*", normalized):
                numbered_line_count += 1
                if numbered_line_count >= 2:
                    return True
    return False


def is_multi_doc_comparative_query(query: str) -> bool:
    normalized = extract_primary_question_text(query)
    if not normalized:
        return False

    relation_markers = ("与", "和", "及", "以及", "对比", "比较")
    comparative_markers = (
        "共同",
        "区别",
        "不同",
        "相同",
        "相似",
        "异同",
        "差异",
        "特点",
        "特征",
        "分别",
        "各自",
        "对比",
        "比较",
    )
    entity_markers = ("专业", "学院", "方向", "项目")
    has_relation = any(marker in normalized for marker in relation_markers)
    has_comparative = any(marker in normalized for marker in comparative_markers)
    has_entity = any(marker in normalized for marker in entity_markers)
    return has_relation and has_comparative and has_entity


def is_temporal_answer_query(query: str) -> bool:
    normalized = extract_primary_question_text(query)
    if not normalized:
        return False
    temporal_markers = (
        "当前",
        "最新",
        "现任",
        "截至",
        "哪一年",
        "哪年",
        "何时",
        "什么时候",
        "什么时间",
        "日期",
        "时间",
    )
    if any(marker in normalized for marker in temporal_markers):
        return True
    if re.search(r"(?:19|20)\d{2}", normalized):
        return True
    if re.search(r"\d{1,2}\s*月\s*\d{1,2}\s*[日号]", normalized):
        return True
    return is_implicit_current_role_query(normalized)


def is_implicit_current_role_query(query: str) -> bool:
    normalized = str(query or "").strip()
    if not normalized:
        return False

    explicit_historical_markers = (
        "曾任",
        "历任",
        "前任",
        "上一任",
        "原",
        "时任",
        "当时",
        "哪一任",
        "哪位曾",
    )
    if any(marker in normalized for marker in explicit_historical_markers):
        return False

    who_markers = ("谁", "哪位")
    role_markers = (
        "校长",
        "院长",
        "主任",
        "书记",
        "局长",
        "部长",
        "市长",
        "县长",
        "董事长",
        "总经理",
        "ceo",
        "负责人",
        "馆长",
    )
    lowered = normalized.lower()
    has_who_marker = any(marker in normalized for marker in who_markers)
    has_role_marker = any(marker in lowered for marker in role_markers)
    return has_who_marker and has_role_marker


def maybe_refine_rag_answer(
    *,
    settings: AppSettings,
    completeness_review_prompt: ChatPromptTemplate,
    factual_review_prompt: ChatPromptTemplate,
    query: str,
    references: list[RetrievedReference],
    context: str,
    coverage_requirements: str,
    answer_requirements: str,
    draft_answer: str,
) -> str:
    refined_answer = draft_answer

    if should_run_answer_completeness_review(
        query=query,
        references=references,
        coverage_requirements=coverage_requirements,
        draft_answer=draft_answer,
        settings=settings,
    ):
        refined_answer = (
            invoke_answer_revision_review(
                settings=settings,
                prompt=completeness_review_prompt,
                context=context,
                coverage_requirements=coverage_requirements,
                answer_requirements=answer_requirements,
                query=query,
                draft_answer=refined_answer,
            )
            or refined_answer
        )

    if should_run_timeseries_joint_completeness_retry(
        settings=settings,
        query=query,
        references=references,
        answer=refined_answer,
    ):
        refined_answer = (
            invoke_answer_revision_review(
                settings=settings,
                prompt=completeness_review_prompt,
                context=context,
                coverage_requirements=coverage_requirements,
                answer_requirements=answer_requirements,
                query=query,
                draft_answer=refined_answer,
            )
            or refined_answer
        )

    if should_run_answer_factual_review(
        query=query,
        references=references,
        draft_answer=refined_answer,
        settings=settings,
    ):
        refined_answer = (
            invoke_answer_revision_review(
                settings=settings,
                prompt=factual_review_prompt,
                context=context,
                coverage_requirements=coverage_requirements,
                answer_requirements=answer_requirements,
                query=query,
                draft_answer=refined_answer,
            )
            or refined_answer
        )

    return cleanup_generated_answer(refined_answer)


def cleanup_generated_answer(answer: str) -> str:
    normalized = str(answer or "").strip()
    if not normalized:
        return ""

    cleaned = remove_known_truncated_tail_phrases(normalized)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def remove_known_truncated_tail_phrases(answer: str) -> str:
    cleaned = str(answer or "")
    truncated_patterns = (
        r"[，,、]?\s*多方合力共[。；;，,]?",
        r"[，,、]?\s*具体如下[:：]?[。；;，,]?",
        r"[，,、]?\s*主要有以下[:：]?[。；;，,]?",
        r"[，,、]?\s*包括以下[:：]?[。；;，,]?",
    )
    for pattern in truncated_patterns:
        cleaned = re.sub(pattern, "。", cleaned)
    cleaned = re.sub(r"仅提到[“\"]?[。；，、][”\"]?这一概括性表述", "仅提到概括性表述", cleaned)
    cleaned = re.sub(r"[“\"]?[。；，、][”\"]?这一概括性表述", "概括性表述", cleaned)
    cleaned = re.sub(r"。{2,}", "。", cleaned)
    cleaned = re.sub(r"[，,、]\s*。", "。", cleaned)
    return cleaned


def invoke_answer_revision_review(
    *,
    settings: AppSettings,
    prompt: ChatPromptTemplate,
    context: str,
    coverage_requirements: str,
    answer_requirements: str,
    query: str,
    draft_answer: str,
) -> str | None:
    model_name = settings.model.QUERY_REWRITE_MODEL.strip() or settings.model.DEFAULT_LLM_MODEL
    try:
        llm = build_chat_model(settings, model_name=model_name, temperature=0.0)
        chain = prompt | llm | StrOutputParser()
        raw_output = chain.invoke(
            {
                "context": context,
                "coverage_requirements": coverage_requirements,
                "answer_requirements": answer_requirements,
                "query": query.strip(),
                "draft_answer": draft_answer.strip(),
            }
        )
    except Exception:
        return None

    payload = extract_json_payload(raw_output)
    if not isinstance(payload, dict):
        return None

    status = str(payload.get("status", "")).strip().lower()
    revised_answer = str(payload.get("revised_answer", "")).strip()
    if status != "revise" or not revised_answer:
        return None
    return revised_answer


def should_run_answer_completeness_review(
    *,
    query: str,
    references: list[RetrievedReference],
    coverage_requirements: str,
    draft_answer: str,
    settings: AppSettings | None = None,
) -> bool:
    if not references:
        return False
    if should_use_image_rag_prompt(query, references):
        return False
    if not draft_answer.strip():
        return False
    if coverage_requirements.strip():
        return True

    if requires_timeseries_joint_coverage(query, references, settings=settings):
        return True

    checklist_hints = ("哪些", "哪几", "分别", "同时", "以及", "角色", "原因", "措施", "步骤", "职责")
    if any(hint in query for hint in checklist_hints):
        return True
    return is_procedural_query(query)


def should_run_answer_factual_review(
    *,
    query: str,
    references: list[RetrievedReference],
    draft_answer: str,
    settings: AppSettings | None = None,
) -> bool:
    if not references:
        return False
    if should_use_image_rag_prompt(query, references):
        return False
    if not draft_answer.strip():
        return False
    return True


def requires_timeseries_joint_coverage(
    query: str,
    references: list[RetrievedReference],
    *,
    settings: AppSettings | None = None,
) -> bool:
    if not is_timeseries_answer_guard_enabled(settings):
        return False
    if not has_timeseries_references(references):
        return False
    if not has_text_references(references):
        return False
    normalized = extract_primary_question_text(query)
    return any(marker in normalized for marker in ("背景", "原因", "事件", "为什么", "影响"))


def should_run_timeseries_joint_completeness_retry(
    *,
    settings: AppSettings | None = None,
    query: str,
    references: list[RetrievedReference],
    answer: str,
) -> bool:
    if not requires_timeseries_joint_coverage(query, references, settings=settings):
        return False
    return not (
        answer_mentions_timeseries_observation(answer)
        and answer_mentions_event_background(answer)
    )


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
