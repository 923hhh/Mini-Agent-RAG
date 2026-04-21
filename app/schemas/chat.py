"""定义聊天问答相关的请求与响应模型。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.storage.filters import MetadataFilters


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class RetrievedDoc(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    source: str
    source_path: str
    extension: str
    page: int | None = None
    page_end: int | None = None
    title: str | None = None
    section_title: str | None = None
    section_path: str | None = None
    section_index: int | None = None
    content_type: str | None = None
    source_modality: str | None = None
    evidence_type: str | None = None
    used_for_answer: bool = False
    original_file_type: str | None = None
    ocr_text: str | None = None
    image_caption: str | None = None
    evidence_summary: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    content: str
    content_preview: str
    raw_score: float
    relevance_score: float


class RetrievedReference(RetrievedDoc):
    model_config = ConfigDict(extra="forbid")


class ReferenceOverview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference_count: int = 0
    text_count: int = 0
    image_side_count: int = 0
    multimodal_count: int = 0
    has_joint_text_image_coverage: bool = False
    source_modality_counts: dict[str, int] = Field(default_factory=dict)
    evidence_type_counts: dict[str, int] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    source_type: Literal["local_kb", "temp_kb"] = "local_kb"
    knowledge_base_name: str = ""
    knowledge_id: str = ""
    top_k: int = Field(default=10, ge=1, le=20)
    score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    history: list[ChatMessage] = Field(default_factory=list)
    metadata_filters: MetadataFilters | None = None
    stream: bool = False


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    references: list[RetrievedReference]
    reference_overview: ReferenceOverview = Field(default_factory=ReferenceOverview)
    source_type: str
    knowledge_base_name: str
    used_context: bool
    stream: bool


class ToolCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_index: int = Field(ge=1)
    tool_name: str
    arguments: dict[str, Any]
    output: str
    status: Literal["success", "error", "skipped"]
    error_message: str | None = None


class AgentStepRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_index: int = Field(ge=1)
    kind: Literal["tool", "stop", "final"]
    status: Literal["success", "error", "stopped", "completed"]
    summary: str
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    output: str = ""


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, Any]


class MemoryOverview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    semantic_hits: int = 0
    episode_hits: int = 0
    turn_hits: int = 0
    used_memory: bool = False


class AgentChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    session_id: str | None = None
    knowledge_base_name: str = ""
    top_k: int = Field(default=10, ge=1, le=20)
    score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    history: list[ChatMessage] = Field(default_factory=list)
    metadata_filters: MetadataFilters | None = None
    allowed_tools: list[str] = Field(default_factory=list)
    max_steps: int = Field(default=4, ge=1, le=8)
    stream: bool = False

    @field_validator("session_id", mode="before")
    @classmethod
    def normalize_session_id(cls, value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value


class AgentChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    tool_calls: list[ToolCallRecord]
    steps: list[AgentStepRecord]
    references: list[RetrievedReference]
    reference_overview: ReferenceOverview = Field(default_factory=ReferenceOverview)
    knowledge_base_name: str
    used_tools: bool
    stream: bool
    session_id: str | None = None
    memory_overview: MemoryOverview | None = None
