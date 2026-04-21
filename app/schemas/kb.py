"""定义知识库管理相关的数据模型。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

class DocumentChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    doc_id: str
    source: str
    source_path: str
    extension: str
    chunk_index: int
    page: int | None = None
    page_end: int | None = None
    title: str | None = None
    section_title: str | None = None
    section_path: str | None = None
    section_index: int | None = None
    content_type: str | None = None
    source_modality: str | None = None
    original_file_type: str | None = None
    ocr_text: str | None = None
    ocr_language: str | None = None
    image_caption: str | None = None
    evidence_summary: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    content_length: int
    content_preview: str


class DocumentChunkRecord(DocumentChunk):
    model_config = ConfigDict(extra="forbid")


class RebuildKnowledgeBaseResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    knowledge_base_name: str
    content_dir: Path
    vector_store_dir: Path
    metadata_path: Path
    build_manifest_path: Path | None = None
    files_processed: int
    raw_documents: int
    chunks: int
    incremental_rebuild: bool = False
    index_mode: str = "full"
    files_total: int | None = None
    files_reused: int = 0
    files_rebuilt: int = 0
    files_deleted: int = 0
    chunks_reused: int = 0
    chunks_embedded: int = 0
    vector_store_type: str = "faiss"
    image_vlm_enabled_for_build: bool = False
    force_full_rebuild: bool = False
    stage_timings_seconds: dict[str, float] = Field(default_factory=dict)


class RebuildKnowledgeBaseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    knowledge_base_name: str = Field(min_length=1)
    chunk_size: int | None = Field(default=None, ge=1)
    chunk_overlap: int | None = Field(default=None, ge=0)
    embedding_model: str | None = None
    enable_image_vlm_for_build: bool = False
    force_full_rebuild: bool = False


class RebuildTaskAccepted(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_id: str
    knowledge_base_name: str
    status: str
    progress: float = 0.0
    progress_message: str | None = None
    created_at: datetime


class RebuildTaskStatus(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_id: str
    knowledge_base_name: str
    status: str
    progress: float = 0.0
    progress_message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error_message: str | None = None
    result: RebuildKnowledgeBaseResult | None = None


class KnowledgeBaseSummary(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    knowledge_base_name: str
    content_dir: Path
    vector_store_dir: Path
    files: list[str]
    file_count: int
    vector_store_type: str = "faiss"
    index_exists: bool
    metadata_exists: bool


class KnowledgeBaseUploadResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    scope: str
    knowledge_base_name: str | None = None
    knowledge_id: str | None = None
    content_dir: Path
    vector_store_dir: Path | None = None
    metadata_path: Path | None = None
    saved_files: list[str]
    overwritten_files: list[str] = Field(default_factory=list)
    skipped_files: list[str] = Field(default_factory=list)
    files_processed: int
    raw_documents: int | None = None
    chunks: int | None = None
    auto_rebuild: bool = False
    requires_rebuild: bool = False
    rebuild_result: RebuildKnowledgeBaseResult | None = None
    created_at: datetime | None = None
    last_accessed_at: datetime | None = None
    expires_at: datetime | None = None
    ttl_minutes: int | None = None
    touch_on_access: bool | None = None
    cleanup_policy: str | None = None


class TempKnowledgeManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    knowledge_id: str
    scope: str = "temp"
    created_at: datetime
    last_accessed_at: datetime
    expires_at: datetime
    ttl_minutes: int = Field(ge=1)
    touch_on_access: bool = True
    cleanup_policy: str
    cleanup_state: str = "active"
    deleted_at: datetime | None = None
    saved_files: list[str] = Field(default_factory=list)


class TempKnowledgeCleanupEntry(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    knowledge_id: str
    knowledge_dir: Path
    expired: bool
    removed: bool
    reason: str


class TempKnowledgeCleanupResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    cleanup_reason: str
    expired_only: bool
    requested_knowledge_id: str | None = None
    ttl_minutes: int
    scanned: int
    removed: int
    skipped: int
    entries: list[TempKnowledgeCleanupEntry]
