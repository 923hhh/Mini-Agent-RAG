"""封装流式响应相关的 API 辅助逻辑。"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any


SSE_MEDIA_TYPE = "text/event-stream"


def sse_event(event_type: str, payload: dict[str, Any]) -> str:
    body = json.dumps({"type": event_type, **payload}, ensure_ascii=False)
    return f"event: {event_type}\ndata: {body}\n\n"


def iter_text_chunks(text: str, chunk_size: int = 16) -> Iterator[str]:
    if not text:
        return
    for index in range(0, len(text), chunk_size):
        yield text[index : index + chunk_size]
