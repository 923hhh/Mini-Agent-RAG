from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
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
from app.schemas.chat import ChatMessage, RetrievedReference
from app.services.core.settings import AppSettings, load_settings
from app.services.models.embedding_service import build_embeddings
from app.services.models.llm_service import build_chat_model, resolve_openai_compatible_api_key


CRUD_100_RETRIEVAL_PATH = PROJECT_ROOT / "data" / "eval" / "crud" / "crud_rag_3qa_full_retrieval_cases_100.jsonl"
CRUD_ANSWER_PATH = PROJECT_ROOT / "data" / "eval" / "crud" / "crud_rag_3qa_full_crud_rag_3qa_train.jsonl"
DOMAIN_100_PATH = PROJECT_ROOT / "data" / "eval" / "domain" / "domainrag_small_batch_100_domainrag_small_batch.jsonl"
DEFAULT_OLLAMA_JUDGE_LLM_MODEL = "qwen2.5:7b"
DEFAULT_OLLAMA_JUDGE_EMBEDDING_MODEL = "bge-m3:latest"


def log_status(message: str) -> None:
    print(f"[generation-eval] {message}", flush=True)


def render_progress(completed: int, total: int, *, width: int = 24) -> str:
    total = max(total, 1)
    completed = min(max(completed, 0), total)
    ratio = completed / total
    filled = min(width, int(ratio * width))
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {completed}/{total} ({ratio * 100:5.1f}%)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_generation_quality_eval_100",
        description="Run 100-sample generation quality evaluation with parallel retrieval and generation.",
    )
    parser.add_argument(
        "--dataset",
        choices=("crud100", "domain100"),
        required=True,
        help="Which 100-sample benchmark to evaluate.",
    )
    parser.add_argument(
        "--knowledge-base-name",
        type=str,
        default="",
        help="Override knowledge base name. Defaults are derived from the benchmark dataset.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k references retrieved for each query.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.35,
        help="Score threshold passed to local retrieval.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads used for parallel retrieval and generation.",
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
        help="Optional output report path.",
    )
    return parser.parse_args()


def build_history_messages(history_qa: list[Any] | None) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    for item in history_qa or []:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        if question:
            messages.append(ChatMessage(role="user", content=question))
        answers = item.get("answers")
        answer_text = flatten_answer_field(answers)
        if answer_text:
            messages.append(ChatMessage(role="assistant", content=answer_text))
    return messages


def flatten_answer_field(raw_answer: Any) -> str:
    if isinstance(raw_answer, str):
        return raw_answer.strip()
    if isinstance(raw_answer, list):
        parts: list[str] = []
        for item in raw_answer:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    parts.append(text)
            elif isinstance(item, list):
                nested = [str(x).strip() for x in item if str(x).strip()]
                if nested:
                    parts.append("、".join(nested))
        if parts:
            return parts[0]
    return ""


def build_reference_contexts_from_positive_references(positive_references: list[dict[str, Any]]) -> list[str]:
    contexts: list[str] = []
    for item in positive_references:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        contents = str(item.get("contents", "")).strip()
        if not contents:
            continue
        contexts.append(f"{title}\n{contents}".strip() if title else contents)
    return contexts


def build_retrieved_context_text(reference: RetrievedReference) -> str:
    section = f"{reference.section_title}\n" if reference.section_title else ""
    return f"{section}{reference.content}".strip()


def load_domain100_cases() -> tuple[list[dict[str, Any]], str]:
    rows = load_jsonl(DOMAIN_100_PATH, encoding="utf-8")
    cases: list[dict[str, Any]] = []
    default_kb = ""
    for row in rows:
        if not default_kb:
            default_kb = str(row.get("knowledge_base_name", "")).strip()
        positive_raw = row.get("positive_references")
        if isinstance(positive_raw, dict):
            positive_references = [positive_raw]
        elif isinstance(positive_raw, list):
            positive_references = [item for item in positive_raw if isinstance(item, dict)]
        else:
            positive_single = row.get("positive_reference")
            if isinstance(positive_single, dict):
                positive_references = [positive_single]
            elif isinstance(positive_single, list):
                positive_references = [item for item in positive_single if isinstance(item, dict)]
            else:
                positive_references = []

        question = str(row.get("question", "")).strip()
        if not question:
            continue
        history = build_history_messages(row.get("history_qa"))
        query = question
        if str(row.get("domainrag_task", "")).strip() == "time-sensitive_qa":
            year = str(row.get("date", "")).strip()
            if year:
                query = f"{year}年 {question}"

        reference = flatten_answer_field(row.get("answers") or row.get("answer"))
        if not reference:
            continue
        cases.append(
            {
                "case_id": str(row.get("case_id", "")),
                "query": question,
                "retrieval_query": query,
                "reference": reference,
                "reference_contexts": build_reference_contexts_from_positive_references(positive_references),
                "history": history,
                "metadata": {
                    "task": str(row.get("domainrag_task", "")),
                    "source_dataset": "domain100",
                },
            }
        )
    return cases, default_kb


def load_crud100_cases() -> tuple[list[dict[str, Any]], str]:
    retrieval_rows = load_jsonl(CRUD_100_RETRIEVAL_PATH, encoding="utf-8")
    answer_rows = load_jsonl(CRUD_ANSWER_PATH, encoding="utf-8")
    answer_by_case_id = {
        str(row.get("case_id", "")).strip(): row
        for row in answer_rows
        if str(row.get("case_id", "")).strip()
    }

    cases: list[dict[str, Any]] = []
    default_kb = ""
    for row in retrieval_rows:
        if not default_kb:
            default_kb = str(row.get("knowledge_base_name", "")).strip()
        case_id = str(row.get("case_id", "")).strip()
        answer_row = answer_by_case_id.get(case_id)
        if not answer_row:
            continue
        query = str(row.get("query", "")).strip()
        reference = str(answer_row.get("answer", "")).strip()
        if not query or not reference:
            continue
        expected_references = [
            item for item in (row.get("expected_references") or [])
            if isinstance(item, dict)
        ]
        reference_contexts = [
            str(item.get("content", "")).strip()
            for item in expected_references
            if str(item.get("content", "")).strip()
        ]
        cases.append(
            {
                "case_id": case_id,
                "query": query,
                "retrieval_query": query,
                "reference": reference,
                "reference_contexts": reference_contexts,
                "history": [],
                "metadata": {
                    "task": str(row.get("category", "")),
                    "source_dataset": "crud100",
                },
            }
        )
    return cases, default_kb


def load_benchmark_cases(dataset_name: str) -> tuple[list[dict[str, Any]], str, str]:
    if dataset_name == "domain100":
        cases, default_kb = load_domain100_cases()
        return cases, default_kb, str(DOMAIN_100_PATH)
    if dataset_name == "crud100":
        cases, default_kb = load_crud100_cases()
        return cases, default_kb, str(CRUD_100_RETRIEVAL_PATH)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def evaluate_generation_case(
    *,
    settings: AppSettings,
    knowledge_base_name: str,
    top_k: int,
    score_threshold: float,
    case: dict[str, Any],
) -> dict[str, Any]:
    references = search_local_knowledge_base(
        settings=settings,
        knowledge_base_name=knowledge_base_name,
        query=case["retrieval_query"],
        top_k=top_k,
        score_threshold=score_threshold,
        history=case["history"],
    )
    answer = generate_rag_answer(
        settings=settings,
        query=case["query"],
        references=references,
        history=case["history"],
    )
    retrieved_contexts = [build_retrieved_context_text(item) for item in references]
    return {
        "case_id": case["case_id"],
        "query": case["query"],
        "reference": case["reference"],
        "reference_contexts": case["reference_contexts"],
        "retrieved_contexts": retrieved_contexts,
        "response": answer,
        "top_sections": [ref.section_title for ref in references[:3]],
        "reference_count": len(references),
        "metadata": case["metadata"],
    }


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


def looks_like_ollama_model_name(model_name: str) -> bool:
    normalized = model_name.strip()
    return ":" in normalized if normalized else False


def select_ollama_fallback_model(candidate: str, default: str) -> str:
    normalized = candidate.strip()
    if looks_like_ollama_model_name(normalized):
        return normalized
    return default


def is_openai_compatible_404_error(exc: Exception) -> bool:
    message = str(exc)
    lowered = message.lower()
    return "404" in lowered or "notfounderror" in lowered or "not found" in lowered


def build_runtime_judges(judge_settings: AppSettings) -> tuple[Any, Any]:
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
    return judge_llm, judge_embeddings


def build_ollama_fallback_judge_settings(
    settings: AppSettings,
    args: argparse.Namespace,
    current_judge_settings: AppSettings,
) -> AppSettings:
    requested_llm_model = args.judge_llm_model.strip()
    requested_embedding_model = args.judge_embedding_model.strip()
    fallback_llm_model = select_ollama_fallback_model(
        requested_llm_model
        or current_judge_settings.model.QUERY_REWRITE_MODEL
        or current_judge_settings.model.DEFAULT_LLM_MODEL
        or settings.model.AGENT_MODEL,
        DEFAULT_OLLAMA_JUDGE_LLM_MODEL,
    )
    fallback_embedding_model = select_ollama_fallback_model(
        requested_embedding_model
        or current_judge_settings.model.DEFAULT_EMBEDDING_MODEL
        or settings.model.DEFAULT_EMBEDDING_MODEL,
        DEFAULT_OLLAMA_JUDGE_EMBEDDING_MODEL,
    )
    return build_judge_settings(
        settings,
        llm_provider="ollama",
        llm_model=fallback_llm_model,
        embedding_provider="ollama",
        embedding_model=fallback_embedding_model,
    )


def resolve_output_path(output_path: Path | None, dataset_name: str) -> Path:
    if output_path is not None:
        return output_path.resolve()
    return (PROJECT_ROOT / "data" / "eval" / f"{dataset_name}_generation_quality_100.json").resolve()


def build_report(
    *,
    dataset_name: str,
    dataset_path: str,
    knowledge_base_name: str,
    evaluated_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
    judge_settings: AppSettings,
    metrics: list[Any],
    result: Any,
    workers: int,
    top_k: int,
    score_threshold: float,
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

    details: list[dict[str, Any]] = []
    evaluated_iter = iter(records)
    for row in evaluated_rows:
        detail = dict(row)
        metric_row = next(evaluated_iter, {})
        for metric_name in metric_names:
            detail[metric_name] = metric_row.get(metric_name)
        details.append(detail)
    details.extend(error_rows)

    display_summary = {
        "Context Recall": metric_summary.get("llm_context_recall"),
        "Faithfulness": metric_summary.get("faithfulness"),
        "Factual Correctness": metric_summary.get("factual_correctness"),
        "Answer Relevance": metric_summary.get("response_relevancy"),
    }

    return {
        "benchmark": f"{dataset_name}_generation_quality_100",
        "dataset": dataset_name,
        "dataset_path": dataset_path,
        "knowledge_base_name": knowledge_base_name,
        "case_total": len(evaluated_rows) + len(error_rows),
        "evaluated_cases": len(evaluated_rows),
        "error_count": len(error_rows),
        "judge_model": judge_settings.model.QUERY_REWRITE_MODEL
        or judge_settings.model.DEFAULT_LLM_MODEL,
        "judge_embedding_model": judge_settings.model.DEFAULT_EMBEDDING_MODEL,
        "metrics": metric_names,
        "display_metric_summary": display_summary,
        "metric_summary": metric_summary,
        "workers": workers,
        "top_k": top_k,
        "score_threshold": score_threshold,
        "ragas_batch_size": ragas_batch_size,
        "details": details,
    }


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        print("--workers must be >= 1.", file=sys.stderr)
        return 1

    log_status("loading settings")
    settings = load_settings(PROJECT_ROOT)
    log_status(f"loading benchmark cases for {args.dataset}")
    cases, default_kb, dataset_path = load_benchmark_cases(args.dataset)
    if not cases:
        print("No benchmark cases loaded.", file=sys.stderr)
        return 1

    knowledge_base_name = args.knowledge_base_name.strip() or default_kb
    if not knowledge_base_name:
        print("knowledge_base_name is empty.", file=sys.stderr)
        return 1

    log_status(
        f"loaded {len(cases)} cases; kb={knowledge_base_name}; workers={args.workers}; "
        f"top_k={args.top_k}; score_threshold={args.score_threshold}"
    )
    log_status("running parallel retrieval and generation")

    indexed_results: list[dict[str, Any] | None] = [None] * len(cases)
    error_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_index = {
            executor.submit(
                evaluate_generation_case,
                settings=settings,
                knowledge_base_name=knowledge_base_name,
                top_k=args.top_k,
                score_threshold=args.score_threshold,
                case=case,
            ): index
            for index, case in enumerate(cases)
        }
        completed = 0
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            case = cases[index]
            try:
                indexed_results[index] = future.result()
            except Exception as exc:
                error_rows.append(
                    {
                        "case_id": case["case_id"],
                        "query": case["query"],
                        "error": str(exc),
                        "metadata": case["metadata"],
                    }
                )
            completed += 1
            progress = render_progress(completed, len(cases))
            print(
                f"\r[generation-eval] {progress} last={case['case_id']}",
                end="" if completed < len(cases) else "\n",
                flush=True,
            )

    evaluated_rows = [item for item in indexed_results if item is not None]
    if not evaluated_rows:
        print("No successful cases were generated for evaluation.", file=sys.stderr)
        return 1

    log_status("building ragas dataset")
    runtime_samples = [
        SingleTurnSample(
            user_input=row["query"],
            retrieved_contexts=row["retrieved_contexts"],
            response=row["response"],
            reference=row["reference"],
            reference_contexts=row["reference_contexts"],
        )
        for row in evaluated_rows
    ]

    log_status("building judge models")
    judge_settings = build_judge_settings(
        settings,
        llm_provider=args.judge_llm_provider,
        llm_model=args.judge_llm_model.strip(),
        embedding_provider=args.judge_embedding_provider,
        embedding_model=args.judge_embedding_model.strip(),
    )
    judge_llm, judge_embeddings = build_runtime_judges(judge_settings)

    metrics = [
        LLMContextRecall(name="llm_context_recall"),
        Faithfulness(),
        FactualCorrectness(),
        ResponseRelevancy(name="response_relevancy"),
    ]

    log_status("running ragas scoring")
    dataset = EvaluationDataset(samples=runtime_samples)
    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=judge_llm,
            embeddings=judge_embeddings,
            batch_size=args.ragas_batch_size,
            raise_exceptions=False,
            show_progress=True,
        )
    except Exception as exc:
        if judge_settings.model.LLM_PROVIDER != "openai_compatible" or not is_openai_compatible_404_error(exc):
            raise
        log_status(
            "openai_compatible judge returned 404 during ragas scoring; "
            "falling back to ollama judge and retrying once"
        )
        judge_settings = build_ollama_fallback_judge_settings(settings, args, judge_settings)
        log_status(
            f"fallback judge llm={judge_settings.model.QUERY_REWRITE_MODEL or judge_settings.model.DEFAULT_LLM_MODEL}; "
            f"embedding={judge_settings.model.DEFAULT_EMBEDDING_MODEL}"
        )
        judge_llm, judge_embeddings = build_runtime_judges(judge_settings)
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
        dataset_name=args.dataset,
        dataset_path=dataset_path,
        knowledge_base_name=knowledge_base_name,
        evaluated_rows=evaluated_rows,
        error_rows=error_rows,
        judge_settings=judge_settings,
        metrics=metrics,
        result=result,
        workers=args.workers,
        top_k=args.top_k,
        score_threshold=args.score_threshold,
        ragas_batch_size=args.ragas_batch_size,
    )
    output_path = resolve_output_path(args.output_path, args.dataset)
    log_status(f"writing report to {output_path}")
    write_json_report(output_path, report)
    print(json.dumps(report["display_metric_summary"], ensure_ascii=False, indent=2))
    print(f"[done] output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
