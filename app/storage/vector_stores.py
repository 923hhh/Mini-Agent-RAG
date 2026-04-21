"""封装向量库目录、加载与持久化逻辑。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.services.core.settings import AppSettings
from app.storage.filters import MetadataFilters, matches_metadata_filters, metadata_filters_to_chroma_where


@dataclass(frozen=True)
class VectorStoreEntry:
    chunk_id: str
    page_content: str
    metadata: dict[str, Any]
    embedding: list[float]


class BaseVectorStoreAdapter(ABC):
    def __init__(
        self,
        settings: AppSettings,
        vector_store_dir: Path,
        embeddings: Any,
        collection_name: str,
    ) -> None:
        self.settings = settings
        self.vector_store_dir = vector_store_dir
        self.embeddings = embeddings
        self.collection_name = collection_name

    @abstractmethod
    def exists(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def build(self, entries: list[VectorStoreEntry]) -> None:
        raise NotImplementedError

    @abstractmethod
    def append(self, entries: list[VectorStoreEntry]) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_all_documents(self) -> dict[str, Document]:
        raise NotImplementedError

    @abstractmethod
    def similarity_search_with_score(
        self,
        query: str,
        k: int,
        *,
        metadata_filters: MetadataFilters | None = None,
        fetch_k: int | None = None,
    ) -> list[tuple[Document, float]]:
        raise NotImplementedError


class FaissVectorStoreAdapter(BaseVectorStoreAdapter):
    def exists(self) -> bool:
        return (self.vector_store_dir / "index.faiss").exists() and (self.vector_store_dir / "index.pkl").exists()

    def build(self, entries: list[VectorStoreEntry]) -> None:
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        if not entries:
            raise ValueError("FAISS 向量库构建失败：没有可写入的向量条目。")
        vector_store = FAISS.from_embeddings(
            text_embeddings=[(item.page_content, item.embedding) for item in entries],
            embedding=self.embeddings,
            metadatas=[item.metadata for item in entries],
            ids=[item.chunk_id for item in entries],
        )
        vector_store.save_local(str(self.vector_store_dir))

    def append(self, entries: list[VectorStoreEntry]) -> None:
        if not entries:
            return
        vector_store = FAISS.load_local(
            str(self.vector_store_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        vector_store.add_embeddings(
            text_embeddings=[(item.page_content, item.embedding) for item in entries],
            metadatas=[item.metadata for item in entries],
            ids=[item.chunk_id for item in entries],
        )
        vector_store.save_local(str(self.vector_store_dir))

    def load_all_documents(self) -> dict[str, Document]:
        vector_store = FAISS.load_local(
            str(self.vector_store_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        documents: dict[str, Document] = {}
        for docstore_id in vector_store.index_to_docstore_id.values():
            document = vector_store.docstore.search(docstore_id)
            if not isinstance(document, Document):
                continue
            chunk_id = str(document.metadata.get("chunk_id") or docstore_id)
            documents[chunk_id] = document
        return documents

    def similarity_search_with_score(
        self,
        query: str,
        k: int,
        *,
        metadata_filters: MetadataFilters | None = None,
        fetch_k: int | None = None,
    ) -> list[tuple[Document, float]]:
        vector_store = FAISS.load_local(
            str(self.vector_store_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        hits = vector_store.similarity_search_with_score(query, k=fetch_k or k)
        if metadata_filters is None or not metadata_filters.filters:
            return [(document, float(score)) for document, score in hits[:k]]

        filtered: list[tuple[Document, float]] = []
        for document, score in hits:
            if matches_metadata_filters(document.metadata, metadata_filters):
                filtered.append((document, float(score)))
            if len(filtered) >= k:
                break
        return filtered


class ChromaVectorStoreAdapter(BaseVectorStoreAdapter):
    def __init__(
        self,
        settings: AppSettings,
        vector_store_dir: Path,
        embeddings: Any,
        collection_name: str,
    ) -> None:
        super().__init__(
            settings,
            vector_store_dir,
            embeddings,
            _sanitize_chroma_collection_name(collection_name),
        )
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as exc:
            raise RuntimeError(
                "当前环境未安装 chromadb，无法使用 Chroma 向量库。"
                "请执行 `pip install chromadb` 或 `pip install -r requirements.txt`。"
            ) from exc

        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.vector_store_dir),
            settings=ChromaSettings(
                persist_directory=str(self.vector_store_dir),
                anonymized_telemetry=False,
            ),
        )

    def exists(self) -> bool:
        try:
            return self._collection().count() > 0
        except Exception:
            return False

    def build(self, entries: list[VectorStoreEntry]) -> None:
        self._reset_collection()
        self.append(entries)

    def append(self, entries: list[VectorStoreEntry]) -> None:
        if not entries:
            return
        collection = self._collection()
        collection.upsert(
            ids=[item.chunk_id for item in entries],
            embeddings=[item.embedding for item in entries],
            documents=[item.page_content for item in entries],
            metadatas=[_transform_chroma_metadata(item.metadata) for item in entries],
        )

    def load_all_documents(self) -> dict[str, Document]:
        collection = self._collection()
        payload = collection.get(include=["documents", "metadatas"])
        documents: dict[str, Document] = {}
        ids = payload.get("ids", [])
        docs = payload.get("documents", [])
        metadatas = payload.get("metadatas", [])
        for chunk_id, page_content, metadata in zip(ids, docs, metadatas, strict=False):
            document_metadata = dict(metadata or {})
            document_metadata["chunk_id"] = chunk_id
            documents[str(chunk_id)] = Document(
                page_content=str(page_content or ""),
                metadata=document_metadata,
            )
        return documents

    def similarity_search_with_score(
        self,
        query: str,
        k: int,
        *,
        metadata_filters: MetadataFilters | None = None,
        fetch_k: int | None = None,
    ) -> list[tuple[Document, float]]:
        where, needs_post_filter = metadata_filters_to_chroma_where(metadata_filters)
        collection = self._collection()
        payload = collection.query(
            query_embeddings=[self.embeddings.embed_query(query)],
            n_results=fetch_k or k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        documents = payload.get("documents", [[]])[0]
        metadatas = payload.get("metadatas", [[]])[0]
        distances = payload.get("distances", [[]])[0]
        ids = payload.get("ids", [[]])[0]

        hits: list[tuple[Document, float]] = []
        for chunk_id, page_content, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
            strict=False,
        ):
            document_metadata = dict(metadata or {})
            document_metadata["chunk_id"] = chunk_id
            document = Document(
                page_content=str(page_content or ""),
                metadata=document_metadata,
            )
            if needs_post_filter and not matches_metadata_filters(document.metadata, metadata_filters):
                continue
            hits.append((document, float(distance)))
            if len(hits) >= k:
                break
        return hits

    def _collection(self):
        return self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _reset_collection(self) -> None:
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection()


def build_vector_store_adapter(
    settings: AppSettings,
    vector_store_dir: Path,
    embeddings: Any,
    *,
    collection_name: str,
    vector_store_type: str | None = None,
) -> BaseVectorStoreAdapter:
    resolved_type = (vector_store_type or settings.kb.DEFAULT_VS_TYPE).strip().lower()
    if resolved_type == "faiss":
        return FaissVectorStoreAdapter(settings, vector_store_dir, embeddings, collection_name)
    if resolved_type == "chroma":
        return ChromaVectorStoreAdapter(settings, vector_store_dir, embeddings, collection_name)
    raise ValueError(f"不支持的向量存储类型: {resolved_type}")


def vector_store_index_exists(vector_store_dir: Path, vector_store_type: str | None = None) -> bool:
    normalized = (vector_store_type or "").strip().lower()
    faiss_exists = (vector_store_dir / "index.faiss").exists() and (vector_store_dir / "index.pkl").exists()
    chroma_exists = (vector_store_dir / "chroma.sqlite3").exists()
    if normalized == "faiss":
        return faiss_exists
    if normalized == "chroma":
        return chroma_exists
    return faiss_exists or chroma_exists


def _transform_chroma_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    transformed: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            transformed[key] = value
    return transformed


def _sanitize_chroma_collection_name(name: str) -> str:
    normalized = "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in name).strip("-.")
    if 3 <= len(normalized) <= 63 and ".." not in normalized:
        return normalized
    return f"kb-{sha1(name.encode('utf-8')).hexdigest()[:16]}"

