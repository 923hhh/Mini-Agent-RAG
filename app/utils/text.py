"""提供文本清洗、截断与格式化等通用工具。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from langchain_core.documents import Document


HEADER_KEYS = ("Header1", "Header2", "Header3")


def coerce_optional_text(value: object) -> str | None:
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    return None


def extract_header_metadata(metadata: Mapping[str, object]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for key in HEADER_KEYS:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            headers[key] = value.strip()
    return headers


def extract_document_headers(document: Document) -> dict[str, str]:
    return extract_header_metadata(document.metadata)


def deduplicate_strings(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result
