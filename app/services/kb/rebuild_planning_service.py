"""知识库重建规划与 build manifest 服务。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.services.core.settings import AppSettings
from app.storage.vector_stores import vector_store_index_exists


class BuildManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relative_path: str
    source_path: str
    extension: str
    size_bytes: int
    modified_at: float
    sha256: str
    raw_documents: int
    chunks: int
    cache_file: str


class KnowledgeBaseBuildManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    knowledge_base_name: str
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    text_splitter_name: str = "ChineseRecursiveTextSplitter"
    vector_store_type: str = "faiss"
    built_at: datetime
    files_total: int
    raw_documents_total: int
    chunks_total: int
    files: list[BuildManifestEntry]


@dataclass
class FileSnapshot:
    path: Path
    relative_path: str
    size_bytes: int
    modified_at: float
    extension: str
    sha256: str | None = None


@dataclass
class FileBuildPlan:
    snapshot: FileSnapshot
    change_kind: Literal["reuse", "new", "modified"]
    manifest_entry: BuildManifestEntry | None = None


def plan_rebuild(
    *,
    settings: AppSettings,
    files: list[Path],
    content_dir: Path,
    existing_manifest: KnowledgeBaseBuildManifest | None,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
    metadata_path: Path,
    vector_store_dir: Path,
    chunk_cache_dir: Path,
    is_chunk_cache_available: Callable[[Path], bool],
    compute_file_sha256: Callable[[Path], str],
) -> tuple[list[FileBuildPlan], list[BuildManifestEntry], str]:
    previous_entries = {entry.relative_path: entry for entry in (existing_manifest.files if existing_manifest else [])}
    current_relative_paths = {path.relative_to(content_dir).as_posix() for path in files}
    deleted_entries = [
        entry for relative_path, entry in previous_entries.items() if relative_path not in current_relative_paths
    ]
    config_changed = (
        existing_manifest is None
        or existing_manifest.chunk_size != chunk_size
        or existing_manifest.chunk_overlap != chunk_overlap
        or existing_manifest.embedding_model != embedding_model
        or existing_manifest.text_splitter_name != settings.kb.TEXT_SPLITTER_NAME
        or existing_manifest.vector_store_type != settings.kb.DEFAULT_VS_TYPE
    )
    can_use_incremental = (
        settings.kb.ENABLE_INCREMENTAL_REBUILD
        and existing_manifest is not None
        and vector_store_index_exists(vector_store_dir, settings.kb.DEFAULT_VS_TYPE)
        and metadata_path.exists()
        and not config_changed
    )
    plans: list[FileBuildPlan] = []
    for path in files:
        relative_path = path.relative_to(content_dir).as_posix()
        stat = path.stat()
        snapshot = FileSnapshot(
            path=path,
            relative_path=relative_path,
            size_bytes=stat.st_size,
            modified_at=stat.st_mtime,
            extension=path.suffix.lower(),
        )
        previous_entry = previous_entries.get(relative_path)
        if not can_use_incremental or previous_entry is None:
            change_kind: Literal["new", "modified"] = "new" if previous_entry is None else "modified"
            plans.append(FileBuildPlan(snapshot=snapshot, change_kind=change_kind, manifest_entry=previous_entry))
            continue
        cache_path = chunk_cache_dir / previous_entry.cache_file
        if not is_chunk_cache_available(cache_path) or not settings.kb.ENABLE_FILE_HASH_CACHE:
            plans.append(FileBuildPlan(snapshot=snapshot, change_kind="modified", manifest_entry=previous_entry))
            continue
        if previous_entry.size_bytes == snapshot.size_bytes and abs(previous_entry.modified_at - snapshot.modified_at) < 1e-6:
            plans.append(
                FileBuildPlan(
                    snapshot=snapshot,
                    change_kind="reuse",
                    manifest_entry=previous_entry.model_copy(update={"source_path": str(path.resolve())}),
                )
            )
            continue
        snapshot.sha256 = compute_file_sha256(path)
        if snapshot.sha256 == previous_entry.sha256:
            plans.append(
                FileBuildPlan(
                    snapshot=snapshot,
                    change_kind="reuse",
                    manifest_entry=previous_entry.model_copy(
                        update={
                            "source_path": str(path.resolve()),
                            "size_bytes": snapshot.size_bytes,
                            "modified_at": snapshot.modified_at,
                        }
                    ),
                )
            )
            continue
        plans.append(FileBuildPlan(snapshot=snapshot, change_kind="modified", manifest_entry=previous_entry))
    if not can_use_incremental:
        return plans, deleted_entries, "full"
    rebuild_kinds = {plan.change_kind for plan in plans if plan.change_kind != "reuse"}
    if not rebuild_kinds and not deleted_entries:
        return plans, deleted_entries, "reuse"
    if deleted_entries or "modified" in rebuild_kinds or not settings.kb.ENABLE_APPEND_INDEX:
        return plans, deleted_entries, "full"
    return plans, deleted_entries, "append"


def build_manifest_from_plans(
    *,
    knowledge_base_name: str,
    plans: list[FileBuildPlan],
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
    text_splitter_name: str,
    vector_store_type: str,
    rebuilt_outputs: dict[str, tuple[BuildManifestEntry, object]] | None = None,
) -> KnowledgeBaseBuildManifest:
    entries: list[BuildManifestEntry] = []
    for plan in sorted(plans, key=lambda item: item.snapshot.relative_path):
        if rebuilt_outputs and plan.snapshot.relative_path in rebuilt_outputs:
            entries.append(rebuilt_outputs[plan.snapshot.relative_path][0])
        elif plan.manifest_entry is not None:
            entries.append(plan.manifest_entry)
    return KnowledgeBaseBuildManifest(
        knowledge_base_name=knowledge_base_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        text_splitter_name=text_splitter_name,
        vector_store_type=vector_store_type,
        built_at=datetime.now(UTC),
        files_total=len(entries),
        raw_documents_total=sum(entry.raw_documents for entry in entries),
        chunks_total=sum(entry.chunks for entry in entries),
        files=entries,
    )


def load_build_manifest(path: Path) -> KnowledgeBaseBuildManifest | None:
    if not path.exists():
        return None
    return KnowledgeBaseBuildManifest.model_validate_json(path.read_text(encoding="utf-8"))


def write_build_manifest(path: Path, manifest: KnowledgeBaseBuildManifest) -> None:
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
