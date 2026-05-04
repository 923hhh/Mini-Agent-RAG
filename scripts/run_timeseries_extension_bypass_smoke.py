from __future__ import annotations

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starlette.datastructures import UploadFile

from app.retrievers.local_kb import search_temp_knowledge_base
from app.services.core.settings import load_settings
from app.services.kb.kb_ingestion_service import upload_temp_files
from app.services.runtime.temp_kb_service import cleanup_temp_knowledge_bases


DEFAULT_CASES_PATH = PROJECT_ROOT / "data" / "eval" / "timeseries" / "timeseries_minimal_cases_20260426.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "eval" / "timeseries" / "timeseries_extension_bypass_smoke.json"
ASSET_DIR = PROJECT_ROOT / "data" / "eval" / "timeseries" / "assets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 timeseries 扩展开关关闭状态 smoke。")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--case-count", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--keep-temp-kb", action="store_true")
    return parser.parse_args()


def discover_asset_files() -> tuple[Path, ...]:
    files = [
        path
        for path in sorted(ASSET_DIR.iterdir())
        if path.is_file() and path.suffix.lower() in {".json", ".txt", ".md"}
    ]
    if not files:
        raise FileNotFoundError(f"未在 {ASSET_DIR} 发现可上传的时间序列回归资产。")
    return tuple(files)


def load_cases(path: Path, limit: int) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        cases.append(json.loads(line))
        if len(cases) >= limit:
            break
    return cases


def build_upload_file(path: Path) -> UploadFile:
    return UploadFile(filename=path.name, file=BytesIO(path.read_bytes()))


def find_last_trace_record(path: Path, query: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    matched: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("event_type") != "retrieval":
            continue
        if str(payload.get("query", "")).strip() == query.strip():
            matched = payload
    return matched


def main() -> int:
    args = parse_args()
    base_settings = load_settings(PROJECT_ROOT)
    settings = base_settings.model_copy(
        update={
            "kb": base_settings.kb.model_copy(
                update={"ENABLE_TIMESERIES_RETRIEVAL_EXTENSION": False}
            )
        }
    )
    cases = load_cases(args.cases, args.case_count)
    upload_response = upload_temp_files(
        settings=settings,
        files=[build_upload_file(path) for path in discover_asset_files()],
    )
    knowledge_id = upload_response.knowledge_id

    try:
        results: list[dict[str, Any]] = []
        trace_path = settings.log_root / "retrieval_trace.jsonl"
        for case in cases:
            query = str(case["query"])
            print(f"[timeseries-bypass-smoke] running {case['case_id']}")
            references = search_temp_knowledge_base(
                settings=settings,
                knowledge_id=knowledge_id,
                query=query,
                top_k=args.top_k,
                score_threshold=args.score_threshold,
                history=[],
            )
            trace = find_last_trace_record(trace_path, query)
            passed = (
                bool(references)
                and trace.get("timeseries_extension_enabled") is False
                and trace.get("timeseries_extension_bypassed") is True
                and trace.get("timeseries_extension_bypass_reason") == "disabled_by_setting"
                and trace.get("timeseries_branch_used") is False
            )
            results.append(
                {
                    "case_id": case["case_id"],
                    "query": query,
                    "reference_count": len(references),
                    "top_sources": [ref.source for ref in references[:5]],
                    "trace_excerpt": {
                        "timeseries_extension_enabled": trace.get("timeseries_extension_enabled"),
                        "timeseries_extension_bypassed": trace.get("timeseries_extension_bypassed"),
                        "timeseries_extension_bypass_reason": trace.get("timeseries_extension_bypass_reason"),
                        "timeseries_branch_used": trace.get("timeseries_branch_used"),
                        "topk_modality_sequence": trace.get("topk_modality_sequence"),
                    },
                    "passed": passed,
                }
            )
    finally:
        if not args.keep_temp_kb:
            cleanup_temp_knowledge_bases(
                settings,
                knowledge_id=knowledge_id,
                expired_only=False,
                cleanup_reason="timeseries_extension_bypass_smoke",
            )

    summary = {
        "case_count": len(results),
        "passed": sum(1 for item in results if item["passed"]),
        "failed": sum(1 for item in results if not item["passed"]),
        "pass_rate": round(sum(1 for item in results if item["passed"]) / max(1, len(results)), 4),
        "timeseries_extension_enabled": False,
    }
    payload = {
        "mode": "timeseries_extension_bypass_smoke",
        "generated_at": "2026-05-03",
        "summary": summary,
        "results": results,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
