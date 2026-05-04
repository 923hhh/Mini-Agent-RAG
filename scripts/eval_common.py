from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_jsonl(path: Path, *, encoding: str = "utf-8") -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding=encoding).splitlines()
        if line.strip()
    ]


def write_json_report(path: Path, payload: dict[str, Any]) -> Path:
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return resolved


__all__ = [
    "PROJECT_ROOT",
    "load_jsonl",
    "write_json_report",
]
