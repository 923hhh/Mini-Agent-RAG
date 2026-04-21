"""评测阶段处理参考文档与命中信息的工具函数。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


SAMPLE_ID_PATTERN = re.compile(r"^[0-9a-f]{24}$", re.IGNORECASE)


def infer_reference_sample_id(source_path: str) -> str:
    normalized_path = str(source_path or "").strip()
    if not normalized_path:
        return ""

    for part in reversed(Path(normalized_path).parts[:-1]):
        token = str(part).strip()
        if SAMPLE_ID_PATTERN.fullmatch(token):
            return token
    return ""


def build_top_reference_details(references) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for ref in references:
        details.append(
            {
                "source": getattr(ref, "source", ""),
                "source_path": getattr(ref, "source_path", ""),
                "sample_id": infer_reference_sample_id(getattr(ref, "source_path", "")),
                "source_modality": getattr(ref, "source_modality", ""),
                "raw_score": float(getattr(ref, "raw_score", 0.0) or 0.0),
                "relevance_score": float(getattr(ref, "relevance_score", 0.0) or 0.0),
            }
        )
    return details


def extract_reference_texts(references) -> list[str]:
    parts: list[str] = []
    for ref in references:
        for value in (
            getattr(ref, "content", ""),
            getattr(ref, "ocr_text", ""),
            getattr(ref, "image_caption", ""),
            getattr(ref, "evidence_summary", ""),
        ):
            text = str(value or "").strip()
            if text:
                parts.append(text)
    return parts


def extract_reference_contents(references) -> list[str]:
    parts: list[str] = []
    for ref in references:
        text = str(getattr(ref, "content", "") or "").strip()
        if text:
            parts.append(text)
    return parts


def build_reference_eval_text(references) -> str:
    return "\n".join(extract_reference_texts(references))
