from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.settings import AppSettings


def append_jsonl_trace(
    settings: AppSettings,
    trace_name: str,
    payload: dict[str, Any],
) -> None:
    if not settings.kb.ENABLE_MULTIMODAL_TRACE_LOG:
        return

    trace_path = _trace_path(settings, trace_name)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _trace_path(settings: AppSettings, trace_name: str) -> Path:
    normalized = trace_name.strip().replace(" ", "_")
    return settings.log_root / f"{normalized}.jsonl"
