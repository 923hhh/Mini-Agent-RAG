from __future__ import annotations

from collections.abc import Iterator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.schemas.chat import ChatMessage, RetrievedReference
from app.services.llm_service import build_chat_model
from app.services.observability import append_jsonl_trace
from app.services.settings import AppSettings
from app.services.streaming_llm import stream_prompt_output


RAG_SYSTEM_PROMPT = """你是一个基于本地知识库的问答助手。
请优先依据提供的上下文回答用户问题，不要编造上下文中不存在的事实。
如果上下文不足以回答问题，请明确说明“根据当前检索到的内容，无法确定”并给出缺失信息。
回答尽量简洁、准确，并优先提炼要点。
如果上下文包含“文本证据 / OCR 证据 / 视觉描述证据”，请区分哪些信息是直接证据，哪些只是推断。
不要把 OCR 识别文本与视觉描述混为同一类事实。"""

IMAGE_RAG_SYSTEM_PROMPT = """你是一个基于本地知识库的多模态问答助手。
当前问题更偏向图片、OCR 或视觉描述理解，请优先依据提供的图片相关证据回答。
请遵守以下规则：
1. OCR 文字属于直接文本证据，但可能有识别错误；引用时不要自动纠正成你主观认为更合理的内容。
2. 视觉描述属于图像理解结果，只能用于“看到的画面特征”判断，不能当作绝对事实扩展。
3. 如果 OCR 证据和视觉描述证据冲突，必须明确指出冲突，不要强行合并。
4. 如果图片证据不足，请明确说明“根据当前检索到的图片相关证据，无法确定”，不要拿普通正文内容替代图片结论。
5. 回答尽量简洁，先说可确认内容，再说可能推断。"""

IMAGE_QUERY_HINTS = (
    "图",
    "图片",
    "图像",
    "照片",
    "截图",
    "画面",
    "看图",
    "图里",
    "图中",
    "这张图",
    "这幅图",
    "ocr",
    "识别",
    "图片文字",
    "图像描述",
    "视觉描述",
)


def generate_rag_answer(
    settings: AppSettings,
    query: str,
    references: list[RetrievedReference],
    history: list[ChatMessage],
) -> str:
    prompt_kind = resolve_rag_prompt_kind(query, references)
    prompt = build_rag_prompt(query, references)
    variables = build_rag_variables(query, references, history)
    llm = build_chat_model(settings)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke(variables)
    append_answer_trace(
        settings=settings,
        query=query,
        references=references,
        prompt_kind=prompt_kind,
        answer=answer,
    )
    return answer


def stream_rag_answer(
    settings: AppSettings,
    query: str,
    references: list[RetrievedReference],
    history: list[ChatMessage],
) -> Iterator[str]:
    prompt_kind = resolve_rag_prompt_kind(query, references)
    prompt = build_rag_prompt(query, references)
    variables = build_rag_variables(query, references, history)
    answer_parts: list[str] = []
    for delta in stream_prompt_output(settings, prompt, variables):
        answer_parts.append(delta)
        yield delta
    append_answer_trace(
        settings=settings,
        query=query,
        references=references,
        prompt_kind=prompt_kind,
        answer="".join(answer_parts),
    )


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
                "参考上下文如下：\n{context}\n\n用户问题：\n{query}\n\n请基于参考上下文回答问题。",
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
) -> dict[str, object]:
    history_messages = convert_history(history)
    context = build_context(references)
    return {
        "history": history_messages,
        "context": context if context else "当前没有检索到可用上下文。",
        "query": query,
    }


def build_context(references: list[RetrievedReference]) -> str:
    if not references:
        return ""

    grouped_blocks = {
        "text": [],
        "ocr": [],
        "vision": [],
    }

    for index, ref in enumerate(references, start=1):
        grouped_blocks[resolve_reference_context_group(ref)].append(
            format_reference_block(index, ref)
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

    return "\n\n".join(sections)


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


def format_reference_block(index: int, ref: RetrievedReference) -> str:
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
    evidence_lines.append(f"content:\n{ref.content}")
    return "\n".join(evidence_lines)


def append_answer_trace(
    *,
    settings: AppSettings,
    query: str,
    references: list[RetrievedReference],
    prompt_kind: str,
    answer: str,
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
            "answer_preview": answer[:240],
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
