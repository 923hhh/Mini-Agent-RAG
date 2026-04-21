"""创建 embedding 模型并提供批量向量化能力。"""

from __future__ import annotations

from typing import Any

from app.services.core.settings import AppSettings
from app.services.models.llm_service import (
    normalize_llm_provider,
    resolve_openai_compatible_api_key,
    resolve_openai_compatible_base_url,
)


def build_embeddings(settings: AppSettings, model_name: str | None = None) -> Any:
    # EMBEDDING_PROVIDER 优先；留空则跟随 LLM_PROVIDER
    raw_provider = (settings.model.EMBEDDING_PROVIDER or settings.model.LLM_PROVIDER).strip()
    provider = normalize_llm_provider(raw_provider)
    resolved_model_name = model_name or settings.model.DEFAULT_EMBEDDING_MODEL

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=resolved_model_name,
            base_url=settings.model.OLLAMA_BASE_URL,
        )

    if provider == "openai_compatible":
        try:
            from langchain_openai import OpenAIEmbeddings
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

        return OpenAIEmbeddings(
            model=resolved_model_name,
            base_url=base_url,
            api_key=api_key,
            request_timeout=settings.model.OPENAI_COMPATIBLE_TIMEOUT_SECONDS,
            max_retries=settings.model.OPENAI_COMPATIBLE_MAX_RETRIES,
        )

    raise ValueError(f"不支持的 Embedding provider: {raw_provider}")


def embed_texts_batched(
    embeddings: Any,
    texts: list[str],
    batch_size: int,
) -> list[list[float]]:
    vectors: list[list[float]] = []
    if not texts:
        return vectors
    for start in range(0, len(texts), max(1, batch_size)):
        vectors.extend(embeddings.embed_documents(texts[start : start + max(1, batch_size)]))
    return vectors

