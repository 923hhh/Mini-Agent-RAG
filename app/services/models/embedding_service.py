"""创建 embedding 模型并提供批量向量化能力。"""

from __future__ import annotations

import os
from typing import Any

from app.services.core.settings import AppSettings
from app.services.models.llm_service import (
    normalize_llm_provider,
    resolve_openai_compatible_api_key,
    resolve_openai_compatible_base_url,
)


def _resolve_embedding_api_key(settings: AppSettings) -> str:
    configured = settings.model.EMBEDDING_API_KEY.strip()
    if configured:
        return configured
    for env_name in ("SILICONFLOW_API_KEY", "EMBEDDING_API_KEY"):
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return resolve_openai_compatible_api_key(settings)


def _resolve_embedding_base_url(settings: AppSettings) -> str:
    configured = settings.model.EMBEDDING_BASE_URL.strip()
    if configured:
        return configured
    return resolve_openai_compatible_base_url(settings)


QUERY_INSTRUCTION = "给定一个问题，检索相关的文档段落"

_INSTRUCT_MODELS = {"qwen3-embedding", "qwen3-embedding-8b"}


def _needs_instruction(model_name: str) -> bool:
    return any(tag in model_name.lower() for tag in _INSTRUCT_MODELS)


class _InstructEmbeddings:
    """Wraps an embedding model to prepend instruction prefix on queries."""

    def __init__(self, inner: Any, instruction: str):
        self._inner = inner
        self._instruction = instruction

    def embed_query(self, text: str) -> list[float]:
        prefixed = f"Instruct: {self._instruction}\nQuery: {text}"
        return self._inner.embed_query(prefixed)

    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_documents(texts)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def build_embeddings(settings: AppSettings, model_name: str | None = None) -> Any:
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

        base_url = _resolve_embedding_base_url(settings)
        api_key = _resolve_embedding_api_key(settings)
        if not base_url:
            raise ValueError(
                "当前 EMBEDDING_PROVIDER=openai_compatible，但未配置 EMBEDDING_BASE_URL 或 OPENAI_COMPATIBLE_BASE_URL。"
            )
        if not api_key:
            raise ValueError(
                "当前 EMBEDDING_PROVIDER=openai_compatible，但未配置 EMBEDDING_API_KEY 或 SILICONFLOW_API_KEY 环境变量。"
            )

        inner = OpenAIEmbeddings(
            model=resolved_model_name,
            base_url=base_url,
            api_key=api_key,
            request_timeout=settings.model.OPENAI_COMPATIBLE_TIMEOUT_SECONDS,
            max_retries=settings.model.OPENAI_COMPATIBLE_MAX_RETRIES,
        )

        if _needs_instruction(resolved_model_name):
            return _InstructEmbeddings(inner, QUERY_INSTRUCTION)
        return inner

    raise ValueError(f"不支持的 Embedding provider: {raw_provider}")


def embed_texts_batched(
    embeddings: Any,
    texts: list[str],
    batch_size: int,
    max_workers: int = 3,
    max_retries: int = 5,
) -> list[list[float]]:
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not texts:
        return []

    bs = max(1, batch_size)
    batches = [texts[i : i + bs] for i in range(0, len(texts), bs)]

    def _embed_with_retry(batch: list[str]) -> list[list[float]]:
        for attempt in range(max_retries):
            try:
                return embeddings.embed_documents(batch)
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep(min(2 ** attempt, 30))
        return []

    if max_workers <= 1 or len(batches) <= 2:
        results: list[list[list[float]]] = [_embed_with_retry(b) for b in batches]
    else:
        results = [None] * len(batches)  # type: ignore[list-item]
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_embed_with_retry, batch): idx
                for idx, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

    return [vec for batch_result in results for vec in batch_result]

