from __future__ import annotations

from langchain_ollama import OllamaEmbeddings

from app.services.settings import AppSettings


def build_embeddings(settings: AppSettings, model_name: str | None = None) -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=model_name or settings.model.DEFAULT_EMBEDDING_MODEL,
        base_url=settings.model.OLLAMA_BASE_URL,
    )


def embed_texts_batched(
    embeddings: OllamaEmbeddings,
    texts: list[str],
    batch_size: int,
) -> list[list[float]]:
    vectors: list[list[float]] = []
    if not texts:
        return vectors
    for start in range(0, len(texts), max(1, batch_size)):
        vectors.extend(embeddings.embed_documents(texts[start : start + max(1, batch_size)]))
    return vectors
