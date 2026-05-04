from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_LEGACY_PATH = PROJECT_ROOT / "data" / "eval" / "full_chain_small_regression_20260424.json"
DEFAULT_TIMESERIES_PATH = PROJECT_ROOT / "data" / "eval" / "timeseries_minimal_regression_20260426.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "eval" / "domain_small_regression_suite_20260426.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="聚合当前 domain 小回归与时间序列专项小回归。")
    parser.add_argument("--legacy", type=Path, default=DEFAULT_LEGACY_PATH)
    parser.add_argument("--timeseries", type=Path, default=DEFAULT_TIMESERIES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    legacy_payload = load_json(args.legacy)
    timeseries_payload = load_json(args.timeseries)

    suite_payload = {
        "generated_at": "2026-04-26",
        "mode": "domain_small_regression_suite",
        "sources": {
            "legacy_full_chain": str(args.legacy.relative_to(PROJECT_ROOT)),
            "timeseries_regression": str(args.timeseries.relative_to(PROJECT_ROOT)),
        },
        "summary": build_summary(legacy_payload, timeseries_payload),
        "groups": [
            build_legacy_group(legacy_payload),
            build_timeseries_group(timeseries_payload),
        ],
    }
    args.output.write_text(json.dumps(suite_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(suite_payload, ensure_ascii=False, indent=2))
    return 0


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_summary(
    legacy_payload: dict[str, Any],
    timeseries_payload: dict[str, Any],
) -> dict[str, Any]:
    legacy_case_count = int(legacy_payload.get("case_count", len(legacy_payload.get("results", []))))
    timeseries_case_count = int(timeseries_payload.get("case_count", len(timeseries_payload.get("results", []))))
    timeseries_summary = timeseries_payload.get("summary", {})
    return {
        "combined_case_count": legacy_case_count + timeseries_case_count,
        "legacy_case_count": legacy_case_count,
        "timeseries_case_count": timeseries_case_count,
        "timeseries_passed": int(timeseries_summary.get("passed", 0)),
        "timeseries_failed": int(timeseries_summary.get("failed", 0)),
        "timeseries_pass_rate": float(timeseries_summary.get("pass_rate", 0.0)),
        "timeseries_single_modality_ok_case_count": int(timeseries_summary.get("single_modality_ok_case_count", 0)),
        "timeseries_single_modality_ok_failed": int(timeseries_summary.get("single_modality_ok_failed", 0)),
        "timeseries_joint_required_case_count": int(timeseries_summary.get("joint_required_case_count", 0)),
        "timeseries_joint_required_failed": int(timeseries_summary.get("joint_required_failed", 0)),
        "timeseries_ranking_failure_count": int(timeseries_summary.get("ranking_failure_count", 0)),
        "timeseries_recall_failure_count": int(timeseries_summary.get("recall_failure_count", 0)),
        "timeseries_generation_failure_count": int(timeseries_summary.get("generation_failure_count", 0)),
        "timeseries_phase2_ready": bool(timeseries_summary.get("phase2_ready", False)),
        "timeseries_top1_timeseries_case_count": int(timeseries_summary.get("top1_timeseries_case_count", 0)),
        "timeseries_in_top2_case_count": int(timeseries_summary.get("timeseries_in_top2_case_count", 0)),
        "timeseries_text_in_top2_case_count": int(timeseries_summary.get("text_in_top2_case_count", 0)),
    }


def build_legacy_group(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results", [])
    return {
        "group_name": "legacy_full_chain_small_regression",
        "mode": payload.get("mode", ""),
        "case_count": int(payload.get("case_count", len(results))),
        "results": results,
    }


def build_timeseries_group(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results", [])
    return {
        "group_name": "timeseries_small_regression",
        "mode": payload.get("mode", ""),
        "case_count": int(payload.get("case_count", len(results))),
        "summary": payload.get("summary", {}),
        "results": results,
    }


if __name__ == "__main__":
    raise SystemExit(main())
