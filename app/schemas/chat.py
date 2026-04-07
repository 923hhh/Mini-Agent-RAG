from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

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


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    source_type: Literal["local_kb", "temp_kb"] = "local_kb"
    knowledge_base_name: str = ""
    knowledge_id: str = ""
    top_k: int = Field(default=4, ge=1, le=20)
    score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    history: list[ChatMessage] = Field(default_factory=list)
    metadata_filters: MetadataFilters | None = None
    stream: bool = False


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    references: list[RetrievedReference]
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


class AgentChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    knowledge_base_name: str = ""
    top_k: int = Field(default=4, ge=1, le=20)
    score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    history: list[ChatMessage] = Field(default_factory=list)
    metadata_filters: MetadataFilters | None = None
    allowed_tools: list[str] = Field(default_factory=list)
    max_steps: int = Field(default=4, ge=1, le=8)
    stream: bool = False


class AgentChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    tool_calls: list[ToolCallRecord]
    steps: list[AgentStepRecord]
    references: list[RetrievedReference]
    knowledge_base_name: str
    used_tools: bool
    stream: bool
