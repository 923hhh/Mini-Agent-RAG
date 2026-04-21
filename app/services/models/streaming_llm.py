"""提供基于流式输出的 LLM 生成辅助函数。"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from app.services.models.llm_service import build_chat_model
from app.services.core.settings import AppSettings


def stream_prompt_output(
    settings: AppSettings,
    prompt: ChatPromptTemplate,
    variables: dict[str, Any],
    *,
    model_name: str | None = None,
    temperature: float | None = None,
) -> Iterator[str]:
    llm = build_chat_model(
        settings,
        model_name=model_name,
        temperature=temperature,
    )
    prompt_value = prompt.invoke(variables)
    for chunk in llm.stream(prompt_value):
        text = normalize_chunk_content(getattr(chunk, "content", ""))
        if text:
            yield text


def stream_messages_output(
    settings: AppSettings,
    messages: list[BaseMessage],
    *,
    model_name: str | None = None,
    temperature: float | None = None,
) -> Iterator[str]:
    llm = build_chat_model(
        settings,
        model_name=model_name,
        temperature=temperature,
    )
    for chunk in llm.stream(messages):
        text = normalize_chunk_content(getattr(chunk, "content", ""))
        if text:
            yield text


def normalize_chunk_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    if content is None:
        return ""
    return str(content)

