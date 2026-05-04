from __future__ import annotations

import argparse
import json
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starlette.datastructures import UploadFile

from app.chains.rag import generate_rag_answer
from app.retrievers.local_kb import search_temp_knowledge_base
from app.schemas.chat import ChatRequest
from app.services.core.settings import load_settings
from app.services.kb.kb_ingestion_service import upload_temp_files
from app.services.retrieval.reference_overview import build_reference_overview
from app.services.runtime.temp_kb_service import cleanup_temp_knowledge_bases


DEFAULT_CASES_PATH = PROJECT_ROOT / "data" / "eval" / "timeseries" / "timeseries_minimal_cases_20260426.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "eval" / "timeseries" / "timeseries_minimal_regression_20260426.json"
ASSET_DIR = PROJECT_ROOT / "data" / "eval" / "timeseries" / "assets"


def discover_asset_files() -> tuple[Path, ...]:
    files = [
        path
        for path in sorted(ASSET_DIR.iterdir())
        if path.is_file() and path.suffix.lower() in {".json", ".txt", ".md"}
    ]
    if not files:
        raise FileNotFoundError(f"未在 {ASSET_DIR} 发现可上传的时间序列回归资产。")
    return tuple(files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行时间序列最小真实检索+生成回归。")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--keep-temp-kb", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings(PROJECT_ROOT)
    cases = load_cases(args.cases)
    upload_response = upload_temp_files(
        settings=settings,
        files=[build_upload_file(path) for path in discover_asset_files()],
    )
    knowledge_id = upload_response.knowledge_id

    try:
        results = []
        for case in cases:
            print(f"[timeseries-regression] running {case['case_id']}")
            try:
                result = run_case(
                    settings=settings,
                    knowledge_id=knowledge_id,
                    case=case,
                    top_k=args.top_k,
                    score_threshold=args.score_threshold,
                )
            except Exception as exc:  # pragma: no cover - 回归脚本容错
                result = {
                    "case_id": case["case_id"],
                    "task": case["task"],
                    "query": case["query"],
                    "error": repr(exc),
                    "evaluation": {
                        "passed": False,
                        "error": True,
                    },
                }
            results.append(result)
    finally:
        if not args.keep_temp_kb:
            cleanup_temp_knowledge_bases(
                settings,
                knowledge_id=knowledge_id,
                expired_only=False,
                cleanup_reason="timeseries_minimal_regression",
            )

    summary = build_summary(results)
    payload = {
        "generated_at": "2026-04-26",
        "mode": "timeseries_minimal_regression",
        "knowledge_id": knowledge_id,
        "case_count": len(results),
        "summary": summary,
        "results": results,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        cases.append(json.loads(line))
    return cases


def build_upload_file(path: Path) -> UploadFile:
    return UploadFile(filename=path.name, file=BytesIO(path.read_bytes()))


def run_case(
    *,
    settings,
    knowledge_id: str,
    case: dict[str, Any],
    top_k: int,
    score_threshold: float,
) -> dict[str, Any]:
    request = ChatRequest(
        query=str(case["query"]),
        source_type="temp_kb",
        knowledge_id=knowledge_id,
        top_k=top_k,
        score_threshold=score_threshold,
    )
    references = retry_call(
        lambda: search_temp_knowledge_base(
            settings=settings,
            knowledge_id=knowledge_id,
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            history=request.history,
        )
    )
    answer = retry_call(
        lambda: generate_rag_answer(
            settings=settings,
            query=request.query,
            references=references,
            history=request.history,
        )
    )
    reference_overview = build_reference_overview(references)
    retrieval_trace = find_last_trace_record(
        settings.log_root / "retrieval_trace.jsonl",
        request.query,
        "retrieval",
    )
    answer_trace = find_last_trace_record(
        settings.log_root / "answer_trace.jsonl",
        request.query,
        "answer",
    )
    evaluation = evaluate_case(
        case=case,
        references=references,
        answer=answer,
        reference_overview=reference_overview.model_dump(),
        retrieval_trace=retrieval_trace,
        answer_trace=answer_trace,
    )
    top_modalities = [
        (getattr(ref, "source_modality", "") or "").strip() or "missing"
        for ref in references[:5]
    ]
    return {
        "case_id": case["case_id"],
        "task": case["task"],
        "evaluation_mode": str(case.get("evaluation_mode", "single_modality_ok")),
        "query": request.query,
        "require_joint_coverage": bool(case.get("require_joint_coverage", False)),
        "required_modalities": list(case.get("required_modalities", [])),
        "preferred_modalities": list(case.get("preferred_modalities", [])),
        "reference_count": len(references),
        "reference_overview": reference_overview.model_dump(),
        "top_sources": [ref.source for ref in references[:5]],
        "top_modalities": top_modalities,
        "top1_source_modality": top_modalities[0] if top_modalities else "missing",
        "answer": answer,
        "retrieval_trace_excerpt": {
            "ts_reference_count": retrieval_trace.get("ts_reference_count"),
            "has_ts_evidence": retrieval_trace.get("has_ts_evidence"),
            "has_text_ts_joint_coverage": retrieval_trace.get("has_text_ts_joint_coverage"),
            "temporal_constraint_detected": retrieval_trace.get("temporal_constraint_detected"),
            "timeseries_branch_used": retrieval_trace.get("timeseries_branch_used"),
            "joint_query_detected": retrieval_trace.get("joint_query_detected"),
            "joint_rerank_applied": retrieval_trace.get("joint_rerank_applied"),
            "topk_modality_sequence": retrieval_trace.get("topk_modality_sequence"),
            "topk_has_text_ts_joint_coverage": retrieval_trace.get("topk_has_text_ts_joint_coverage"),
        },
        "answer_trace_excerpt": {
            "ts_reference_count": answer_trace.get("ts_reference_count"),
            "has_ts_evidence": answer_trace.get("has_ts_evidence"),
            "has_text_ts_joint_coverage": answer_trace.get("has_text_ts_joint_coverage"),
            "temporal_constraint_detected": answer_trace.get("temporal_constraint_detected"),
        },
        "evaluation": evaluation,
    }


def find_last_trace_record(path: Path, query: str, event_type: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    matched: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        record = json.loads(line)
        if record.get("event_type") == event_type and record.get("query") == query:
            matched = record
    return matched


def retry_call(func, *, attempts: int = 3, delay_seconds: float = 1.0):
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:  # pragma: no cover - 回归脚本容错
            last_error = exc
            if attempt >= attempts:
                raise
            time.sleep(delay_seconds)
    raise last_error  # type: ignore[misc]


def evaluate_case(
    *,
    case: dict[str, Any],
    references,
    answer: str,
    reference_overview: dict[str, Any],
    retrieval_trace: dict[str, Any],
    answer_trace: dict[str, Any],
) -> dict[str, Any]:
    reference_modalities = {
        (getattr(ref, "source_modality", "") or "").strip()
        for ref in references
    }
    evaluation_mode = str(case.get("evaluation_mode", "single_modality_ok")).strip() or "single_modality_ok"
    required_modalities = set(case.get("required_modalities", []))
    missing_modalities = sorted(item for item in required_modalities if item not in reference_modalities)
    require_joint_coverage = bool(case.get("require_joint_coverage", False))
    require_temporal_constraint = bool(case.get("require_temporal_constraint", False))
    keyword_group_results = [
        keyword_group_hit(answer, group)
        for group in case.get("answer_keyword_groups", [])
    ]
    keyword_groups_passed = all(keyword_group_results) if keyword_group_results else True
    joint_coverage_passed = (
        bool(reference_overview.get("has_text_ts_joint_coverage", False))
        if require_joint_coverage
        else True
    )
    temporal_trace_passed = (
        bool(retrieval_trace.get("temporal_constraint_detected") or answer_trace.get("temporal_constraint_detected"))
        if require_temporal_constraint
        else True
    )
    trace_ts_passed = (
        bool(retrieval_trace.get("has_ts_evidence") and answer_trace.get("has_ts_evidence"))
        if "timeseries" in required_modalities
        else True
    )
    passed = (
        not missing_modalities
        and keyword_groups_passed
        and joint_coverage_passed
        and temporal_trace_passed
        and trace_ts_passed
    )
    failure_cause = classify_failure_cause(
        passed=passed,
        required_modalities=required_modalities,
        missing_modalities=missing_modalities,
        require_joint_coverage=require_joint_coverage,
        retrieval_trace=retrieval_trace,
        reference_overview=reference_overview,
    )
    return {
        "passed": passed,
        "evaluation_mode": evaluation_mode,
        "missing_modalities": missing_modalities,
        "keyword_group_results": keyword_group_results,
        "keyword_groups_passed": keyword_groups_passed,
        "joint_coverage_passed": joint_coverage_passed,
        "temporal_trace_passed": temporal_trace_passed,
        "trace_ts_passed": trace_ts_passed,
        "failure_cause": failure_cause,
    }


def classify_failure_cause(
    *,
    passed: bool,
    required_modalities: set[str],
    missing_modalities: list[str],
    require_joint_coverage: bool,
    retrieval_trace: dict[str, Any],
    reference_overview: dict[str, Any],
) -> str:
    if passed:
        return "passed"
    if missing_modalities:
        return "recall_failure"
    if require_joint_coverage and not bool(reference_overview.get("has_text_ts_joint_coverage", False)):
        return "recall_failure"

    topk_modalities = list(retrieval_trace.get("topk_modality_sequence") or [])
    top2_modalities = topk_modalities[:2]
    has_ts_evidence = bool(retrieval_trace.get("has_ts_evidence"))
    has_joint_coverage = bool(retrieval_trace.get("has_text_ts_joint_coverage"))
    if "timeseries" in required_modalities and not has_ts_evidence:
        return "recall_failure"
    if require_joint_coverage and has_joint_coverage and not ("timeseries" in top2_modalities and "text" in top2_modalities):
        return "ranking_failure"
    if bool(retrieval_trace.get("timeseries_branch_used")) and has_ts_evidence and has_joint_coverage:
        return "generation_failure"
    return "generation_failure"


def keyword_group_hit(answer: str, group: list[str]) -> bool:
    normalized = str(answer or "")
    return any(keyword in normalized for keyword in group)


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(1 for item in results if item["evaluation"]["passed"])
    joint_required_cases = [item for item in results if item.get("evaluation_mode") == "joint_required"]
    single_modality_ok_cases = [item for item in results if item.get("evaluation_mode") == "single_modality_ok"]
    joint_required_failed = sum(1 for item in joint_required_cases if not item["evaluation"]["passed"])
    single_modality_ok_failed = sum(1 for item in single_modality_ok_cases if not item["evaluation"]["passed"])
    ranking_failures = sum(1 for item in results if item["evaluation"].get("failure_cause") == "ranking_failure")
    recall_failures = sum(1 for item in results if item["evaluation"].get("failure_cause") == "recall_failure")
    generation_failures = sum(1 for item in results if item["evaluation"].get("failure_cause") == "generation_failure")
    joint_cases = [
        item
        for item in results
        if isinstance(item.get("reference_overview"), dict)
        and item["reference_overview"].get("has_text_ts_joint_coverage")
    ]
    top1_timeseries_cases = sum(
        1 for item in results if item.get("top1_source_modality") == "timeseries"
    )
    timeseries_in_top2_cases = sum(
        1 for item in results if "timeseries" in item.get("top_modalities", [])[:2]
    )
    text_in_top2_cases = sum(
        1 for item in joint_required_cases if "text" in item.get("top_modalities", [])[:2]
    )
    return {
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": round(passed / max(1, len(results)), 3),
        "single_modality_ok_case_count": len(single_modality_ok_cases),
        "single_modality_ok_failed": single_modality_ok_failed,
        "joint_required_case_count": len(joint_required_cases),
        "joint_required_failed": joint_required_failed,
        "ranking_failure_count": ranking_failures,
        "recall_failure_count": recall_failures,
        "generation_failure_count": generation_failures,
        "phase2_ready": bool(
            ranking_failures >= 3
            or (
                len(joint_required_cases) > 0
                and joint_required_failed / len(joint_required_cases) >= 0.15
                and ranking_failures > 0
            )
        ),
        "avg_ts_reference_count": round(
            sum(
                int(item.get("reference_overview", {}).get("timeseries_count", 0))
                for item in results
                if isinstance(item.get("reference_overview"), dict)
            )
            / max(1, len(results)),
            3,
        ),
        "joint_coverage_case_count": len(joint_cases),
        "top1_timeseries_case_count": top1_timeseries_cases,
        "timeseries_in_top2_case_count": timeseries_in_top2_cases,
        "text_in_top2_case_count": text_in_top2_cases,
    }


if __name__ == "__main__":
    raise SystemExit(main())
