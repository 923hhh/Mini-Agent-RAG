from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    FactualCorrectness,
    Faithfulness,
    LLMContextRecall,
    ResponseRelevancy,
)

try:
    from scripts.eval_common import PROJECT_ROOT, load_jsonl, write_json_report
except ModuleNotFoundError:
    from eval_common import PROJECT_ROOT, load_jsonl, write_json_report
from app.chains.rag import generate_rag_answer
from app.retrievers.local_kb import search_local_knowledge_base
from app.schemas.chat import RetrievedReference
from app.services.core.settings import AppSettings, load_settings
from app.services.models.embedding_service import build_embeddings
from app.services.models.llm_service import build_chat_model, resolve_openai_compatible_api_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_ragas_eval",
        description="Run RAGAS evaluation against a local knowledge base using an existing testset.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="RAGAS testset jsonl path.",
    )
    parser.add_argument(
        "--knowledge-base-name",
        type=str,
        default="motor_manual_chunk_eval",
        help="Local knowledge base name used for retrieval and generation.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Top-k references retrieved for each query.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Score threshold passed to local retrieval.",
    )
    parser.add_argument(
        "--judge-llm-provider",
        choices=("auto", "ollama", "openai_compatible"),
        default="auto",
        help="Judge LLM provider for RAGAS metrics.",
    )
    parser.add_argument(
        "--judge-llm-model",
        type=str,
        default="",
        help="Optional override for judge LLM model.",
    )
    parser.add_argument(
        "--judge-embedding-provider",
        choices=("auto", "ollama", "openai_compatible"),
        default="auto",
        help="Judge embedding provider for response relevancy.",
    )
    parser.add_argument(
        "--judge-embedding-model",
        type=str,
        default="",
        help="Optional override for judge embedding model.",
    )
    parser.add_argument(
        "--ragas-batch-size",
        type=int,
        default=4,
        help="Batch size for ragas.evaluate.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional output report path. Defaults near dataset.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root. Defaults to repository root.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    settings = load_settings(project_root)
    dataset_path = args.dataset_path.resolve()
    if not dataset_path.exists():
        print(f"测试集不存在: {dataset_path}", file=sys.stderr)
        return 1

    cases = load_testset_cases(dataset_path)
    if not cases:
        print("测试集为空，无法评测。", file=sys.stderr)
        return 1

    runtime_samples: list[SingleTurnSample] = []
    detail_rows: list[dict[str, Any]] = []
    error_count = 0

    for index, case in enumerate(cases, start=1):
        query = str(case.get("user_input", "")).strip()
        reference = str(case.get("reference", "")).strip()
        reference_contexts = [
            str(item).strip()
            for item in (case.get("reference_contexts") or [])
            if str(item).strip()
        ]
        if not query or not reference:
            error_count += 1
            detail_rows.append(
                {
                    "case_index": index,
                    "query": query,
                    "error": "missing_query_or_reference",
                }
            )
            continue

        try:
            references = search_local_knowledge_base(
                settings=settings,
                knowledge_base_name=args.knowledge_base_name,
                query=query,
                top_k=args.top_k,
                score_threshold=args.score_threshold,
            )
            answer = generate_rag_answer(
                settings=settings,
                query=query,
                references=references,
                history=[],
            )
        except Exception as exc:
            error_count += 1
            detail_rows.append(
                {
                    "case_index": index,
                    "query": query,
                    "error": str(exc),
                }
            )
            continue

        retrieved_contexts = [build_retrieved_context_text(item) for item in references]
        runtime_samples.append(
            SingleTurnSample(
                user_input=query,
                retrieved_contexts=retrieved_contexts,
                response=answer,
                reference=reference,
                reference_contexts=reference_contexts,
            )
        )
        detail_rows.append(
            {
                "case_index": index,
                "query": query,
                "response": answer,
                "reference": reference,
                "retrieved_context_count": len(retrieved_contexts),
                "top_sections": [ref.section_title for ref in references[:3]],
            }
        )

    if not runtime_samples:
        print("没有成功生成任何可评测样本。", file=sys.stderr)
        return 1

    judge_settings = build_judge_settings(
        settings,
        llm_provider=args.judge_llm_provider,
        llm_model=args.judge_llm_model.strip(),
        embedding_provider=args.judge_embedding_provider,
        embedding_model=args.judge_embedding_model.strip(),
    )
    judge_llm = build_chat_model(
        judge_settings,
        model_name=judge_settings.model.QUERY_REWRITE_MODEL.strip()
        or judge_settings.model.DEFAULT_LLM_MODEL,
        temperature=0.0,
    )
    judge_embeddings = build_embeddings(
        judge_settings,
        model_name=judge_settings.model.DEFAULT_EMBEDDING_MODEL,
    )

    metrics = [
        LLMContextRecall(name="llm_context_recall"),
        Faithfulness(),
        FactualCorrectness(),
        ResponseRelevancy(name="response_relevancy"),
    ]

    dataset = EvaluationDataset(samples=runtime_samples)
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        batch_size=args.ragas_batch_size,
        raise_exceptions=False,
        show_progress=True,
    )

    report = build_report(
        dataset_path=dataset_path,
        knowledge_base_name=args.knowledge_base_name,
        cases=cases,
        evaluated_samples=runtime_samples,
        error_count=error_count,
        judge_settings=judge_settings,
        metrics=metrics,
        detail_rows=detail_rows,
        result=result,
        ragas_batch_size=args.ragas_batch_size,
    )
    output_path = resolve_output_path(args.output_path, dataset_path)
    write_json_report(output_path, report)
    print(f"[ragas-eval] report => {output_path}")
    return 0


def load_testset_cases(dataset_path: Path) -> list[dict[str, Any]]:
    return load_jsonl(dataset_path, encoding="utf-8")


def build_retrieved_context_text(reference: RetrievedReference) -> str:
    section = f"{reference.section_title}\n" if reference.section_title else ""
    return f"{section}{reference.content}".strip()


def build_judge_settings(
    settings: AppSettings,
    *,
    llm_provider: str,
    llm_model: str,
    embedding_provider: str,
    embedding_model: str,
) -> AppSettings:
    effective_llm_provider = resolve_effective_llm_provider(settings, llm_provider)
    effective_embedding_provider = resolve_effective_embedding_provider(
        effective_llm_provider,
        embedding_provider,
    )
    model_updates: dict[str, object] = {
        "LLM_PROVIDER": effective_llm_provider,
        "EMBEDDING_PROVIDER": effective_embedding_provider,
    }
    if llm_model:
        model_updates["DEFAULT_LLM_MODEL"] = llm_model
        model_updates["QUERY_REWRITE_MODEL"] = llm_model
    elif not settings.model.QUERY_REWRITE_MODEL.strip():
        model_updates["QUERY_REWRITE_MODEL"] = settings.model.DEFAULT_LLM_MODEL
    if embedding_model:
        model_updates["DEFAULT_EMBEDDING_MODEL"] = embedding_model
    return settings.model_copy(
        update={"model": settings.model.model_copy(update=model_updates)}
    )


def resolve_effective_llm_provider(settings: AppSettings, requested_provider: str) -> str:
    if requested_provider in {"ollama", "openai_compatible"}:
        return requested_provider
    if settings.model.LLM_PROVIDER == "openai_compatible":
        if resolve_openai_compatible_api_key(settings).strip():
            return "openai_compatible"
        return "ollama"
    return settings.model.LLM_PROVIDER


def resolve_effective_embedding_provider(
    effective_llm_provider: str,
    requested_provider: str,
) -> str:
    if requested_provider in {"ollama", "openai_compatible"}:
        return requested_provider
    return effective_llm_provider


def resolve_output_path(output_path: Path | None, dataset_path: Path) -> Path:
    if output_path is not None:
        return output_path.resolve()
    return dataset_path.with_name(dataset_path.stem.replace("_testset_", "_eval_") + ".json")


def build_report(
    *,
    dataset_path: Path,
    knowledge_base_name: str,
    cases: list[dict[str, Any]],
    evaluated_samples: list[SingleTurnSample],
    error_count: int,
    judge_settings: AppSettings,
    metrics: list[Any],
    detail_rows: list[dict[str, Any]],
    result: Any,
    ragas_batch_size: int,
) -> dict[str, Any]:
    result_df = result.to_pandas()
    records = result_df.to_dict(orient="records")
    metric_names = [metric.name for metric in metrics]
    metric_summary: dict[str, float | None] = {}
    for metric_name in metric_names:
        values: list[float] = []
        for record in records:
            value = record.get(metric_name)
            if isinstance(value, (int, float)) and not math.isnan(float(value)):
                values.append(float(value))
        metric_summary[metric_name] = sum(values) / len(values) if values else None

    detail_map = {
        row["case_index"]: row
        for row in detail_rows
        if isinstance(row.get("case_index"), int)
    }
    details: list[dict[str, Any]] = []
    record_index = 0
    for case_index in range(1, len(cases) + 1):
        base_row = dict(detail_map.get(case_index, {"case_index": case_index}))
        if "error" in base_row:
            details.append(base_row)
            continue
        if record_index >= len(records):
            details.append(base_row)
            continue
        metric_row = records[record_index]
        record_index += 1
        for metric_name in metric_names:
            base_row[metric_name] = metric_row.get(metric_name)
        details.append(base_row)

    return {
        "benchmark": "manual_pdf_ragas",
        "dataset_path": str(dataset_path),
        "knowledge_base_name": knowledge_base_name,
        "case_total": len(cases),
        "evaluated_cases": len(evaluated_samples),
        "error_count": error_count,
        "judge_model": judge_settings.model.QUERY_REWRITE_MODEL
        or judge_settings.model.DEFAULT_LLM_MODEL,
        "judge_embedding_model": judge_settings.model.DEFAULT_EMBEDDING_MODEL,
        "metrics": metric_names,
        "language": "chinese",
        "ragas_batch_size": ragas_batch_size,
        "metric_summary": metric_summary,
        "details": details,
    }


if __name__ == "__main__":
    raise SystemExit(main())
