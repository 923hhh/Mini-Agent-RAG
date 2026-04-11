from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class BasicSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    PROJECT_NAME: str = "mini-agent-rag2"
    PYTHON_VERSION: str = "3.11"
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000
    WEBUI_HOST: str = "127.0.0.1"
    WEBUI_PORT: int = 8501
    DATA_ROOT: str = "./data"
    KB_ROOT_PATH: str = "./data/knowledge_base"
    TEMP_ROOT_PATH: str = "./data/temp"
    LOG_PATH: str = "./data/logs"
    VECTOR_STORE_PATH: str = "./data/vector_store"
    ENABLE_AGENT_MEMORY: bool = False
    AGENT_MEMORY_ROOT: str = "./data/agent_memory"
    AGENT_MEMORY_EPISODE_MAX_TURNS: int = Field(default=8, ge=2, le=64)
    AGENT_MEMORY_SEMANTIC_TOP_K: int = Field(default=5, ge=1, le=20)
    AGENT_MEMORY_EPISODE_TOP_K: int = Field(default=3, ge=1, le=20)
    AGENT_MEMORY_ENABLE_TURN_EXPANSION: bool = False
    AGENT_MEMORY_CONTEXT_CHAR_BUDGET: int = Field(default=3200, ge=200, le=16000)


class KBSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    DEFAULT_VS_TYPE: str = "faiss"
    TEXT_SPLITTER_NAME: str = "ChineseRecursiveTextSplitter"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    VECTOR_SEARCH_TOP_K: int = 4
    SCORE_THRESHOLD: float = 0.5
    ENABLE_QUERY_REWRITE: bool = True
    ENABLE_MULTI_QUERY_RETRIEVAL: bool = True
    ENABLE_HYDE: bool = False
    ENABLE_HYBRID_RETRIEVAL: bool = True
    ENABLE_CORRECTIVE_RAG: bool = False
    ENABLE_CORRECTIVE_WEB_SEARCH: bool = False
    ENABLE_HEURISTIC_RERANK: bool = True
    ENABLE_MODEL_RERANK: bool = False
    ENABLE_INCREMENTAL_REBUILD: bool = True
    ENABLE_FILE_HASH_CACHE: bool = True
    ENABLE_CHUNK_CACHE: bool = True
    ENABLE_APPEND_INDEX: bool = True
    ENABLE_SMALL_TO_BIG_CONTEXT: bool = True
    ENABLE_MULTIMODAL_TRACE_LOG: bool = True
    HYBRID_DENSE_TOP_K: int = Field(default=30, ge=1, le=100)
    HYBRID_LEXICAL_TOP_K: int = Field(default=30, ge=1, le=100)
    HYBRID_RERANK_TOP_K: int = Field(default=5, ge=1, le=50)
    HYBRID_RRF_K: int = Field(default=60, ge=1, le=200)
    HYBRID_DENSE_SCORE_WEIGHT: float = Field(default=0.35, ge=0.0, le=2.0)
    HYBRID_LEXICAL_SCORE_WEIGHT: float = Field(default=0.25, ge=0.0, le=2.0)
    MULTI_QUERY_MAX_QUERIES: int = Field(default=3, ge=1, le=6)
    CORRECTIVE_RAG_SECOND_PASS_TOP_K: int = Field(default=6, ge=1, le=20)
    CORRECTIVE_RAG_SECOND_PASS_SCORE_THRESHOLD: float = Field(default=0.25, ge=0.0, le=1.0)
    CORRECTIVE_RAG_MAX_REFERENCES_TO_GRADE: int = Field(default=4, ge=1, le=8)
    CORRECTIVE_WEB_SEARCH_PROVIDER: str = "duckduckgo_html"
    CORRECTIVE_WEB_SEARCH_ENDPOINT: str = "https://html.duckduckgo.com/html/"
    CORRECTIVE_WEB_SEARCH_TOP_K: int = Field(default=3, ge=1, le=8)
    CORRECTIVE_WEB_SEARCH_TIMEOUT_SECONDS: int = Field(default=12, ge=1, le=60)
    CORRECTIVE_WEB_SEARCH_SNIPPET_MAX_CHARS: int = Field(default=240, ge=80, le=800)
    RERANK_CANDIDATES_TOP_N: int = Field(default=12, ge=1, le=50)
    RERANK_SCORE_THRESHOLD: float = Field(default=0.0, ge=0.0, le=1.0)
    RERANK_FALLBACK_TO_HEURISTIC: bool = True
    METADATA_FILTER_DENSE_FETCH_MULTIPLIER: int = Field(default=5, ge=1, le=20)
    TRACE_LOG_MAX_REFERENCES: int = Field(default=8, ge=1, le=20)
    EMBEDDING_BATCH_SIZE: int = Field(default=32, ge=1, le=512)
    DOC_PARSE_WORKERS: int = Field(default=4, ge=1, le=16)
    SMALL_TO_BIG_EXPAND_CHUNKS: int = Field(default=1, ge=0, le=5)
    IMAGE_OCR_ENABLED: bool = True
    IMAGE_OCR_BACKEND: str = "tesseract"
    IMAGE_OCR_INSTRUCTION_PAGE_BACKEND: str = "paddle"
    IMAGE_OCR_FAST_MODE: bool = True
    IMAGE_OCR_LANGUAGE: str = "chi_sim+eng"
    IMAGE_OCR_MAX_SIDE: int = Field(default=1600, ge=0, le=10000)
    IMAGE_OCR_EARLY_STOP_CHARS: int = Field(default=24, ge=0, le=5000)
    OCR_TESSERACT_CMD: str = ""
    PADDLE_OCR_LANGUAGE: str = "ch"
    PADDLE_OCR_USE_ANGLE_CLS: bool = True
    PADDLE_OCR_DET_LIMIT_SIDE_LEN: int = Field(default=1600, ge=0, le=10000)
    PADDLE_OCR_MIN_SCORE: float = Field(default=0.45, ge=0.0, le=1.0)
    OCR_MIN_CONFIDENCE: float = Field(default=60.0, ge=0.0, le=100.0)
    OCR_MIN_TEXT_LENGTH: int = Field(default=6, ge=0, le=200)
    OCR_MIN_MEANINGFUL_RATIO: float = Field(default=0.6, ge=0.0, le=1.0)
    TEMP_KB_TTL_MINUTES: int = Field(default=120, ge=1)
    TEMP_KB_CLEANUP_ON_STARTUP: bool = True
    TEMP_KB_TOUCH_ON_ACCESS: bool = True
    SUPPORTED_EXTENSIONS: list[str] = Field(
        default_factory=lambda: [
            ".txt",
            ".md",
            ".pdf",
            ".docx",
            ".epub",
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".webp",
        ]
    )


class ModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    LLM_PROVIDER: str = "ollama"
    EMBEDDING_PROVIDER: str = ""  # 留空则跟随 LLM_PROVIDER；可单独设为 "ollama" 或 "openai_compatible"
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    OPENAI_COMPATIBLE_BASE_URL: str = ""
    OPENAI_COMPATIBLE_API_KEY: str = ""
    OPENAI_COMPATIBLE_TIMEOUT_SECONDS: int = Field(default=120, ge=1, le=600)
    OPENAI_COMPATIBLE_MAX_RETRIES: int = Field(default=2, ge=0, le=10)
    DEFAULT_LLM_MODEL: str = "qwen2.5:7b"
    QUERY_REWRITE_MODEL: str = ""
    DEFAULT_EMBEDDING_MODEL: str = "bge-m3:latest"
    RERANK_MODEL: str = "BAAI/bge-reranker-base"
    RERANK_DEVICE: str = "cpu"
    AGENT_MODEL: str = "qwen2.5:7b"
    IMAGE_VLM_ENABLED: bool = False
    IMAGE_VLM_AUTO_CAPTION_ENABLED: bool = False
    IMAGE_VLM_PROVIDER: str = "openai_compatible"
    IMAGE_VLM_API_STYLE: str = "chat_completions"
    IMAGE_VLM_BASE_URL: str = ""
    IMAGE_VLM_API_KEY: str = ""
    IMAGE_VLM_MODEL: str = "deepseek-ai/deepseek-vl2"
    IMAGE_VLM_TIMEOUT_SECONDS: int = Field(default=120, ge=1, le=600)
    IMAGE_VLM_MAX_TOKENS: int = Field(default=768, ge=16, le=2048)
    IMAGE_VLM_MAX_SIDE: int = Field(default=1600, ge=0, le=10000)
    IMAGE_VLM_PROMPT: str = ""
    IMAGE_VLM_USE_OCR_CONTEXT: bool = True
    IMAGE_VLM_AUTO_TRIGGER_BY_OCR: bool = False
    IMAGE_VLM_SKIP_IF_OCR_CHARS_AT_LEAST: int = Field(default=20, ge=0, le=5000)
    IMAGE_VLM_ONLY_WHEN_OCR_EMPTY: bool = True
    IMAGE_VLM_REGION_CAPTION_ENABLED: bool = True
    IMAGE_VLM_REGION_MAX_REGIONS: int = Field(default=3, ge=0, le=6)
    IMAGE_VLM_REGION_MIN_SIDE: int = Field(default=320, ge=0, le=4000)
    TEMPERATURE: float = 0.3
    MAX_TOKENS: int = 2048
    ENABLE_STREAMING: bool = True


class AppSettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    project_root: Path
    config_root: Path
    basic: BasicSettings
    kb: KBSettings
    model: ModelSettings

    def resolve_path(self, path_like: str | Path) -> Path:
        path = Path(path_like)
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

    @property
    def data_root(self) -> Path:
        return self.resolve_path(self.basic.DATA_ROOT)

    @property
    def knowledge_base_root(self) -> Path:
        return self.resolve_path(self.basic.KB_ROOT_PATH)

    @property
    def temp_root(self) -> Path:
        return self.resolve_path(self.basic.TEMP_ROOT_PATH)

    @property
    def log_root(self) -> Path:
        return self.resolve_path(self.basic.LOG_PATH)

    @property
    def vector_store_root(self) -> Path:
        return self.resolve_path(self.basic.VECTOR_STORE_PATH)

    @property
    def agent_memory_root(self) -> Path:
        return self.resolve_path(self.basic.AGENT_MEMORY_ROOT)

    def agent_memory_session_dir(self, session_id: str) -> Path:
        return self.agent_memory_root / session_id

    def knowledge_base_dir(self, knowledge_base_name: str) -> Path:
        return self.knowledge_base_root / knowledge_base_name

    def knowledge_base_content_dir(self, knowledge_base_name: str) -> Path:
        return self.knowledge_base_dir(knowledge_base_name) / "content"

    def vector_store_dir(self, knowledge_base_name: str) -> Path:
        return self.vector_store_root / knowledge_base_name

    def vector_store_manifest_path(self, knowledge_base_name: str) -> Path:
        return self.vector_store_dir(knowledge_base_name) / "build_manifest.json"

    def vector_store_bm25_index_path(self, knowledge_base_name: str) -> Path:
        return self.vector_store_dir(knowledge_base_name) / "bm25_index.json"

    def vector_store_cache_dir(self, knowledge_base_name: str) -> Path:
        return self.vector_store_dir(knowledge_base_name) / "cache"

    def vector_store_chunk_cache_dir(self, knowledge_base_name: str) -> Path:
        return self.vector_store_cache_dir(knowledge_base_name) / "chunks"

    def temp_knowledge_dir(self, knowledge_id: str) -> Path:
        return self.temp_root / knowledge_id

    def temp_content_dir(self, knowledge_id: str) -> Path:
        return self.temp_knowledge_dir(knowledge_id) / "content"

    def temp_vector_store_dir(self, knowledge_id: str) -> Path:
        return self.temp_knowledge_dir(knowledge_id) / "vector_store"

    def resolve_ocr_tesseract_cmd(self) -> str:
        for env_key in ("OCR_TESSERACT_CMD", "TESSERACT_CMD"):
            raw_value = os.getenv(env_key, "").strip()
            if raw_value:
                return str(self.resolve_path(raw_value))

        configured = self.kb.OCR_TESSERACT_CMD.strip()
        if not configured:
            return ""
        return str(self.resolve_path(configured))


DEFAULT_CONFIG_MODELS: dict[str, type[BaseModel]] = {
    "basic_settings.yaml": BasicSettings,
    "kb_settings.yaml": KBSettings,
    "model_settings.yaml": ModelSettings,
}

ENV_FILE_CANDIDATES = (
    ".env",
    "configs/.env",
)

SENSITIVE_CONFIG_FIELDS: dict[str, set[str]] = {
    "model_settings.yaml": {
        "OPENAI_COMPATIBLE_API_KEY",
        "IMAGE_VLM_API_KEY",
    }
}


def default_config_data() -> dict[str, dict[str, Any]]:
    return {
        filename: model().model_dump()
        for filename, model in DEFAULT_CONFIG_MODELS.items()
    }


def dump_yaml(data: dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def read_yaml_file(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw or {}


def write_yaml_file(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_yaml(data), encoding="utf-8")


def load_project_env(project_root: Path) -> list[Path]:
    loaded_paths: list[Path] = []
    for relative_path in ENV_FILE_CANDIDATES:
        path = project_root / relative_path
        if not path.exists() or not path.is_file():
            continue
        apply_env_file(path)
        loaded_paths.append(path)
    return loaded_paths


def apply_env_file(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        normalized_key = key.strip()
        if not normalized_key:
            continue
        os.environ.setdefault(normalized_key, normalize_env_value(raw_value.strip()))


def normalize_env_value(value: str) -> str:
    if not value:
        return ""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    if " #" in value:
        return value.split(" #", 1)[0].rstrip()
    return value


def sanitize_config_data(
    model: type[BaseModel],
    data: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    allowed = set(model.model_fields)
    sanitized = {key: value for key, value in data.items() if key in allowed}
    unknown = sorted(key for key in data if key not in allowed)
    return sanitized, unknown


def sanitize_sensitive_config_values(
    filename: str,
    data: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    sensitive_fields = SENSITIVE_CONFIG_FIELDS.get(filename, set())
    if not sensitive_fields:
        return data, []

    sanitized = dict(data)
    ignored: list[str] = []
    for key in sorted(sensitive_fields):
        value = sanitized.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        sanitized[key] = ""
        ignored.append(key)
    return sanitized, ignored


def validate_sensitive_config_updates(
    filename: str,
    updates: dict[str, Any],
) -> None:
    sensitive_fields = SENSITIVE_CONFIG_FIELDS.get(filename, set())
    forbidden = sorted(
        key
        for key in updates
        if key in sensitive_fields
        and (
            not isinstance(updates.get(key), str)
            or bool(str(updates.get(key)).strip())
        )
    )
    if not forbidden:
        return
    raise ValueError(
        f"{filename} 中的敏感配置项只支持通过环境变量提供: {', '.join(forbidden)}"
    )


def save_config_values(
    project_root: Path,
    filename: str,
    updates: dict[str, Any],
) -> dict[str, Any]:
    model = DEFAULT_CONFIG_MODELS.get(filename)
    if model is None:
        raise ValueError(f"不支持保存的配置文件: {filename}")

    allowed = set(model.model_fields)
    unknown_update_keys = sorted(key for key in updates if key not in allowed)
    if unknown_update_keys:
        raise ValueError(
            f"{filename} 包含未识别的配置项: {', '.join(unknown_update_keys)}"
        )
    validate_sensitive_config_updates(filename, updates)

    config_path = project_root / "configs" / filename
    current_data = read_yaml_file(config_path) if config_path.exists() else {}
    if not isinstance(current_data, dict):
        raise ValueError(f"{filename} 配置格式无效，根节点必须是映射对象。")

    merged_data = dict(current_data)
    merged_data.update(updates)
    merged_data, _ = sanitize_sensitive_config_values(filename, merged_data)

    sanitized, _ = sanitize_config_data(model, merged_data)
    validated = model.model_validate(sanitized)
    validated_data = validated.model_dump()

    # 保留当前文件中其他未知字段，避免把已有扩展配置意外抹掉。
    preserved_unknown = {
        key: value for key, value in merged_data.items() if key not in allowed
    }
    final_data = {**validated_data, **preserved_unknown}
    write_yaml_file(config_path, final_data)
    clear_settings_cache()
    return final_data


def load_settings(project_root: Path) -> AppSettings:
    load_project_env(project_root.resolve())
    normalized_project_root = str(project_root.resolve())
    return _load_settings_cached(normalized_project_root)


def clear_settings_cache() -> None:
    _load_settings_cached.cache_clear()


@lru_cache(maxsize=8)
def _load_settings_cached(project_root_str: str) -> AppSettings:
    project_root = Path(project_root_str)
    config_root = project_root / "configs"

    try:
        basic_raw = read_yaml_file(config_root / "basic_settings.yaml")
        kb_raw = read_yaml_file(config_root / "kb_settings.yaml")
        model_raw = read_yaml_file(config_root / "model_settings.yaml")
        model_raw, model_sensitive = sanitize_sensitive_config_values(
            "model_settings.yaml",
            model_raw,
        )
        basic_data, basic_unknown = sanitize_config_data(
            BasicSettings,
            basic_raw,
        )
        kb_data, kb_unknown = sanitize_config_data(
            KBSettings,
            kb_raw,
        )
        model_data, model_unknown = sanitize_config_data(
            ModelSettings,
            model_raw,
        )
        basic = BasicSettings.model_validate(basic_data)
        kb = KBSettings.model_validate(kb_data)
        model = ModelSettings.model_validate(model_data)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"配置文件不存在: {exc.filename}") from exc
    except ValidationError as exc:
        raise ValueError(f"配置文件校验失败:\n{exc}") from exc

    ignored_fields = {
        "basic_settings.yaml": basic_unknown,
        "kb_settings.yaml": kb_unknown,
        "model_settings.yaml": model_unknown,
    }
    for filename, unknown in ignored_fields.items():
        if unknown:
            print(f"[settings] 忽略未识别配置项 {filename}: {', '.join(unknown)}")
    if model_sensitive:
        print(
            "[settings] 忽略 model_settings.yaml 中的敏感配置项: "
            f"{', '.join(model_sensitive)}；请改用环境变量提供。"
        )

    return AppSettings(
        project_root=project_root,
        config_root=config_root,
        basic=basic,
        kb=kb,
        model=model,
    )
