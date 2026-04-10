from __future__ import annotations

import os
from typing import Any

from app.services.network import build_httpx_client
from app.services.settings import AppSettings


def build_chat_model(
    settings: AppSettings,
    model_name: str | None = None,
    temperature: float | None = None,
) -> Any:
    provider = normalize_llm_provider(settings.model.LLM_PROVIDER)
    resolved_model_name = model_name or settings.model.DEFAULT_LLM_MODEL
    resolved_temperature = temperature if temperature is not None else settings.model.TEMPERATURE

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=resolved_model_name,
            base_url=settings.model.OLLAMA_BASE_URL,
            temperature=resolved_temperature,
        )

    if provider == "openai_compatible":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "当前 LLM_PROVIDER=openai_compatible，但未安装 `langchain-openai`。"
                "请先执行 `pip install -r requirements.txt`。"
            ) from exc

        base_url = resolve_openai_compatible_base_url(settings)
        api_key = resolve_openai_compatible_api_key(settings)
        if not base_url:
            raise ValueError(
                "当前 LLM_PROVIDER=openai_compatible，但未配置 OPENAI_COMPATIBLE_BASE_URL。"
            )
        if not api_key:
            raise ValueError(
                "当前 LLM_PROVIDER=openai_compatible，但未配置 OPENAI_COMPATIBLE_API_KEY。"
            )

        return ChatOpenAI(
            model=resolved_model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=resolved_temperature,
            timeout=settings.model.OPENAI_COMPATIBLE_TIMEOUT_SECONDS,
            max_retries=settings.model.OPENAI_COMPATIBLE_MAX_RETRIES,
            http_client=build_httpx_client(
                timeout=settings.model.OPENAI_COMPATIBLE_TIMEOUT_SECONDS,
            ),
        )

    raise ValueError(f"不支持的 LLM_PROVIDER: {settings.model.LLM_PROVIDER}")


def normalize_llm_provider(provider: str) -> str:
    normalized = provider.strip().lower().replace("-", "_")
    aliases = {
        "api": "openai_compatible",
        "openai": "openai_compatible",
        "openai_compatible": "openai_compatible",
        "ollama": "ollama",
    }
    return aliases.get(normalized, normalized)


def resolve_openai_compatible_base_url(settings: AppSettings) -> str:
    configured = settings.model.OPENAI_COMPATIBLE_BASE_URL.strip()
    if configured:
        return configured
    for env_name in ("OPENAI_COMPATIBLE_BASE_URL", "OPENAI_BASE_URL"):
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return ""


def resolve_openai_compatible_api_key(settings: AppSettings) -> str:
    configured = settings.model.OPENAI_COMPATIBLE_API_KEY.strip()
    if configured:
        return configured

    for env_name in (
        "OPENAI_COMPATIBLE_API_KEY",
        "OPENAI_API_KEY",
        "DASHSCOPE_API_KEY",
        "DEEPSEEK_API_KEY",
    ):
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return ""
