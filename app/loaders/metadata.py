"""读取文档 sidecar 元数据。"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


FILE_METADATA_BLOCKED_KEYS = {
    "source",
    "source_path",
    "relative_path",
    "extension",
}


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
    "load_sidecar_file_metadata",
    "load_sidecar_metadata_map",
]
