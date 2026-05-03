from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.schemas.chat import RetrievedReference
from app.services.core.settings import load_settings
from app.services.retrieval.answer_guard_service import (
    build_answer_requirements,
    build_coverage_requirements,
    is_timeseries_answer_guard_enabled,
    requires_timeseries_joint_coverage,
)
from app.services.runtime.rag_runtime_service import build_rag_variables


OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "eval"
    / "timeseries"
    / "timeseries_answer_guard_bypass_smoke.json"
)


def build_test_references() -> list[RetrievedReference]:
    return [
        RetrievedReference(
            chunk_id="ts-1",
            source="air_quality_series_clean_u.json#beijing_pm25_clean_u",
            source_path="data/eval/timeseries_minimal_assets/air_quality_series_clean_u.json",
            extension=".json",
            title="beijing_pm25_clean_u",
            source_modality="timeseries",
            evidence_type="text",
            content='{"city":"北京","channel":"pm25"}',
            content_preview='{"city":"北京","channel":"pm25"}',
            raw_score=0.1,
            relevance_score=0.95,
        ),
        RetrievedReference(
            chunk_id="txt-1",
            source="event_background_clean_u.txt",
            source_path="data/eval/timeseries_minimal_assets/event_background_clean_u.txt",
            extension=".txt",
            title="event_background_clean_u",
            source_modality="text",
            evidence_type="text",
            content="1月3日污染缓解与扩散条件改善有关。",
            content_preview="1月3日污染缓解与扩散条件改善有关。",
            raw_score=0.2,
            relevance_score=0.88,
        ),
    ]


def main() -> int:
    base_settings = load_settings(PROJECT_ROOT)
    disabled_settings = base_settings.model_copy(
        update={
            "kb": base_settings.kb.model_copy(
                update={"ENABLE_TIMESERIES_RETRIEVAL_EXTENSION": False}
            )
        }
    )
    references = build_test_references()
    query = "请结合时间序列和文本背景说明北京pm25变化原因。"

    enabled_payload = {
        "guard_enabled": is_timeseries_answer_guard_enabled(base_settings),
        "coverage_requirements": build_coverage_requirements(
            query,
            references,
            settings=base_settings,
        ),
        "answer_requirements": build_answer_requirements(
            query,
            references,
            settings=base_settings,
        ),
        "requires_joint_coverage": requires_timeseries_joint_coverage(
            query,
            references,
            settings=base_settings,
        ),
        "rag_variables": build_rag_variables(
            base_settings,
            query,
            references,
            [],
            is_multi_doc_comparative=False,
            should_direct_answer=False,
            requirement_count=1,
        ),
    }
    disabled_payload = {
        "guard_enabled": is_timeseries_answer_guard_enabled(disabled_settings),
        "coverage_requirements": build_coverage_requirements(
            query,
            references,
            settings=disabled_settings,
        ),
        "answer_requirements": build_answer_requirements(
            query,
            references,
            settings=disabled_settings,
        ),
        "requires_joint_coverage": requires_timeseries_joint_coverage(
            query,
            references,
            settings=disabled_settings,
        ),
        "rag_variables": build_rag_variables(
            disabled_settings,
            query,
            references,
            [],
            is_multi_doc_comparative=False,
            should_direct_answer=False,
            requirement_count=1,
        ),
    }

    pass_checks = [
        enabled_payload["guard_enabled"] is True,
        disabled_payload["guard_enabled"] is False,
        enabled_payload["requires_joint_coverage"] is True,
        disabled_payload["requires_joint_coverage"] is False,
        "趋势观察与事件背景" in enabled_payload["answer_requirements"],
        "趋势观察与事件背景" not in disabled_payload["answer_requirements"],
        "对应事件背景或文本原因说明" in enabled_payload["coverage_requirements"],
        "对应事件背景或文本原因说明" not in disabled_payload["coverage_requirements"],
        "趋势观察与事件背景" in enabled_payload["rag_variables"]["answer_requirements"],
        "趋势观察与事件背景" not in disabled_payload["rag_variables"]["answer_requirements"],
    ]

    payload = {
        "mode": "timeseries_answer_guard_bypass_smoke",
        "generated_at": "2026-05-03",
        "summary": {
            "passed": all(pass_checks),
            "check_count": len(pass_checks),
            "passed_count": sum(1 for item in pass_checks if item),
        },
        "enabled": enabled_payload,
        "disabled": disabled_payload,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
