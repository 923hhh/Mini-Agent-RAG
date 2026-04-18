from __future__ import annotations

from abc import ABC, abstractmethod
import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.documents import Document

if TYPE_CHECKING:
    from app.services.settings import AppSettings


FILE_METADATA_BLOCKED_KEYS = {
    "source",
    "source_path",
    "relative_path",
    "extension",
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
        from .image import ImageKnowledge
        from .office import DocxKnowledge, EpubKnowledge
        from .pdf import PdfKnowledge
        from .text import MarkdownKnowledge, TextKnowledge

        return [
            MarkdownKnowledge,
            TextKnowledge,
            PdfKnowledge,
            DocxKnowledge,
            EpubKnowledge,
            ImageKnowledge,
        ]

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
        if path.is_file() and path.suffix.lower() in normalized:
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


def load_sidecar_file_metadata(content_dir: Path, relative_path: str) -> dict[str, str]:
    metadata_map = load_sidecar_metadata_map(content_dir)
    raw_payload = metadata_map.get(relative_path, {})
    sanitized: dict[str, str] = {}
    for key, value in raw_payload.items():
        normalized_key = str(key).strip()
        if not normalized_key or normalized_key in FILE_METADATA_BLOCKED_KEYS:
            continue
        sanitized[normalized_key] = str(value).strip()
    return sanitized


def load_sidecar_metadata_map(content_dir: Path) -> dict[str, dict[str, str]]:
    metadata_path = content_dir / ".rag_file_metadata.json"
    if not metadata_path.exists():
        return {}

    stat = metadata_path.stat()
    return _load_sidecar_metadata_map_cached(
        str(metadata_path.resolve()),
        stat.st_mtime_ns,
        stat.st_size,
    )


@lru_cache(maxsize=16)
def _load_sidecar_metadata_map_cached(
    metadata_path_str: str,
    _mtime_ns: int,
    _size: int,
) -> dict[str, dict[str, str]]:
    metadata_path = Path(metadata_path_str)
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    raw_files = payload.get("files")
    if not isinstance(raw_files, dict):
        return {}

    normalized: dict[str, dict[str, str]] = {}
    for relative_path, raw_metadata in raw_files.items():
        if not isinstance(raw_metadata, dict):
            continue
        normalized[str(relative_path)] = {
            str(key): str(value)
            for key, value in raw_metadata.items()
            if value is not None
        }
    return normalized


__all__ = [
    "BaseKnowledge",
    "KnowledgeFactory",
    "list_supported_files",
    "load_documents",
    "load_file",
]
