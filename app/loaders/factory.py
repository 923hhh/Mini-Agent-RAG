"""按文件类型选择并创建对应的加载器。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.documents import Document

from .metadata import load_sidecar_file_metadata
from .registry import get_knowledge_registry

if TYPE_CHECKING:
    from app.services.core.settings import AppSettings


LOADER_EXCLUDED_FILENAMES = {
    ".rag_file_metadata.json",
}


class BaseKnowledge(ABC):
    supported_extensions: tuple[str, ...] = ()

    def __init__(
        self,
        path: Path,
        content_dir: Path,
        settings: "AppSettings | None" = None,
    ) -> None:
        self.path = path
        self.content_dir = content_dir
        self.settings = settings
        self.relative_path = path.relative_to(content_dir).as_posix()
        self.base_metadata = {
            "source": path.name,
            "source_path": str(path.resolve()),
            "relative_path": self.relative_path,
            "extension": path.suffix.lower(),
            "title": path.stem,
            "content_type": "document_text",
            "source_modality": "text",
            "original_file_type": path.suffix.lower().lstrip("."),
            "evidence_summary": path.stem,
        }
        self.base_metadata.update(load_sidecar_file_metadata(content_dir, self.relative_path))
        title = str(self.base_metadata.get("title", "")).strip()
        if title:
            self.base_metadata["evidence_summary"] = title

    @classmethod
    def supports(cls, path: Path) -> bool:
        return path.suffix.lower() in cls.supported_extensions

    @abstractmethod
    def load(self) -> list[Document]:
        raise NotImplementedError


class KnowledgeFactory:
    @classmethod
    def _registry(cls) -> list[type[BaseKnowledge]]:
        return get_knowledge_registry()

    @classmethod
    def create(
        cls,
        path: Path,
        content_dir: Path,
        settings: "AppSettings | None" = None,
    ) -> BaseKnowledge:
        for knowledge_cls in cls._registry():
            if knowledge_cls.supports(path):
                return knowledge_cls(path, content_dir, settings=settings)
        raise ValueError(f"暂不支持的文件类型: {path}")

    @classmethod
    def load(
        cls,
        path: Path,
        content_dir: Path,
        settings: "AppSettings | None" = None,
    ) -> list[Document]:
        return cls.create(path, content_dir, settings=settings).load()


def list_supported_files(content_dir: Path, supported_extensions: list[str]) -> list[Path]:
    normalized = {ext.lower() for ext in supported_extensions}
    files: list[Path] = []
    for path in content_dir.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in normalized
            and path.name not in LOADER_EXCLUDED_FILENAMES
        ):
            files.append(path)
    return sorted(files)


def load_documents(
    content_dir: Path,
    supported_extensions: list[str],
    settings: "AppSettings | None" = None,
) -> tuple[list[Document], list[Path]]:
    files = list_supported_files(content_dir, supported_extensions)
    documents: list[Document] = []

    for path in files:
        documents.extend(load_file(path, content_dir, settings=settings))

    return documents, files


def load_file(
    path: Path,
    content_dir: Path,
    settings: "AppSettings | None" = None,
) -> list[Document]:
    return KnowledgeFactory.load(path, content_dir, settings=settings)


__all__ = [
    "BaseKnowledge",
    "KnowledgeFactory",
    "list_supported_files",
    "load_documents",
    "load_file",
]

