from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.chains.rag import generate_rag_answer
from app.retrievers.local_kb import search_local_knowledge_base
from app.schemas.chat import ChatMessage
from app.services.crud_eval_cases import (
    SUPPORTED_TASKS,
    build_cases,
    load_crud_rag_items,
    normalize_tasks,
    resolve_data_file,
)
from app.services.embedding_service import build_embeddings
from app.services.eval_reference_utils import (
    build_top_reference_details,
    extract_reference_contents,
)
from app.services.llm_service import build_chat_model
from app.services.reference_overview import build_reference_overview
from app.services.settings import AppSettings, load_settings


DEFAULT_LANGUAGE = "chinese"
DEFAULT_METRICS = (
    "llm_context_recall",
    "faithfulness",
    "factual_correctness",
)
OPTIONAL_METRICS = ("response_relevancy",)
SUPPORTED_METRICS = DEFAULT_METRICS + OPTIONAL_METRICS
RAGAS_CACHE_ROOT = PROJECT_ROOT / "data" / "cache" / "ragas"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the current local knowledge-base RAG with real RAGAS metrics on CRUD_RAG-style tasks."
    )
    parser.add_argument(
        "--case-file",
        default="",
        help="Generic JSONL case file. Supports CRUD exported eval files and DomainRAG JSONL files.",
    )
    parser.add_argument(
        "--knowledge-base-name",
        required=True,
        help="Knowledge base name inside the current project.",
    )
    parser.add_argument(
        "--crud-rag-root",
        default="",
        help="Path to the local CRUD_RAG repository root. Used to auto-locate split_merged.json.",
    )
    parser.add_argument(
        "--data-file",
        default="",
        help="Direct path to CRUD_RAG task data (JSON or JSONL). Ignored when --case-file is provided.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[],
        help="Tasks to evaluate. Supported: quest_answer summary",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Retrieval top_k.")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Retrieval score threshold.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate at most N samples after filtering. 0 means all.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="How many successful samples to score per RAGAS batch.",
    )
    parser.add_argument(
        "--show-cases",
        action="store_true",
        help="Print one JSON line per case result.",
    )
    parser.add_argument(
        "--judge-model",
        default="",
        help="Optional evaluator LLM override. Defaults to the current DEFAULT_LLM_MODEL.",
    )
    parser.add_argument(
        "--judge-embedding-model",
        default="",
        help="Optional evaluator embedding model override. Only used when response_relevancy is enabled.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[],
        help="Explicit RAGAS metric names. Supported: llm_context_recall faithfulness factual_correctness response_relevancy",
    )
    parser.add_argument(
        "--include-response-relevancy",
        action="store_true",
        help="Append response_relevancy to the selected metric set.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to save the final JSON summary. Defaults to data/eval/<kb>_ragas_report.json.",
    )
    args = parser.parse_args()

    selected_tasks = normalize_tasks(args.tasks) if args.tasks else []
    metric_names = normalize_metric_names(
        args.metrics,
        include_response_relevancy=args.include_response_relevancy,
    )
    cases, data_path = load_eval_cases(
        case_file=args.case_file,
        data_file=args.data_file,
        crud_rag_root=args.crud_rag_root,
        selected_tasks=selected_tasks,
        limit=args.limit,
    )

    settings = load_settings(PROJECT_ROOT)
    summary = evaluate_cases_with_ragas(
        settings=settings,
        knowledge_base_name=args.knowledge_base_name,
        cases=cases,
        top_k=args.top_k,
        score_threshold=args.score_threshold,
        show_cases=args.show_cases,
        batch_size=args.batch_size,
        dataset_path=data_path,
        metric_names=metric_names,
        judge_model_name=args.judge_model.strip(),
        judge_embedding_model_name=args.judge_embedding_model.strip(),
    )

    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    output_path = resolve_output_path(args.output, args.knowledge_base_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    return 0


def normalize_metric_names(
    values: list[str],
    *,
    include_response_relevancy: bool,
) -> list[str]:
    raw_names = list(values) if values else list(DEFAULT_METRICS)
    normalized: list[str] = []
    supported = set(SUPPORTED_METRICS)
    for value in raw_names:
        metric_name = str(value).strip().lower().replace("-", "_")
        if metric_name not in supported:
            raise ValueError(
                f"不支持的 RAGAS 指标: {value}。支持: {', '.join(SUPPORTED_METRICS)}"
            )
        if metric_name not in normalized:
            normalized.append(metric_name)

    if include_response_relevancy and "response_relevancy" not in normalized:
        normalized.append("response_relevancy")
    return normalized


def resolve_output_path(output: str, knowledge_base_name: str) -> Path:
    if output:
        path = Path(output)
        if not path.is_absolute():
            return (PROJECT_ROOT / path).resolve()
        return path
    return PROJECT_ROOT / "data" / "eval" / f"{knowledge_base_name}_ragas_report.json"


def load_eval_cases(
    *,
    case_file: str,
    data_file: str,
    crud_rag_root: str,
    selected_tasks: list[str],
    limit: int,
) -> tuple[list[dict[str, Any]], Path]:
    if case_file.strip():
        path = Path(case_file)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        cases = load_generic_case_file(path, selected_tasks=selected_tasks, limit=limit)
        return cases, path

    if not selected_tasks:
        selected_tasks = list(SUPPORTED_TASKS)
    data_path = resolve_data_file(data_file, crud_rag_root)
    raw_items = load_crud_rag_items(data_path)
    cases = build_cases(raw_items, selected_tasks)
    if limit > 0:
        cases = cases[:limit]
    return cases, data_path


def load_generic_case_file(
    path: Path,
    *,
    selected_tasks: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"评测文件不存在: {path}")

    task_filter = set(selected_tasks) if selected_tasks else None
    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        raw_case = json.loads(stripped)
        case = normalize_generic_case(raw_case, source_path=path, line_number=line_number)
        if case is None:
            continue
        if task_filter is not None and case["task"] not in task_filter:
            continue
        cases.append(case)
        if limit > 0 and len(cases) >= limit:
            break
    return cases


def normalize_generic_case(
    raw_case: dict[str, Any],
    *,
    source_path: Path,
    line_number: int,
) -> dict[str, Any] | None:
    if is_domainrag_case(raw_case):
        return normalize_domainrag_case(raw_case, source_path=source_path, line_number=line_number)
    return normalize_crud_export_case(raw_case, fallback_index=line_number)


def is_domainrag_case(raw_case: dict[str, Any]) -> bool:
    return "question" in raw_case and (
        "positive_reference" in raw_case or "positive_references" in raw_case
    )


def normalize_domainrag_case(
    raw_case: dict[str, Any],
    *,
    source_path: Path,
    line_number: int,
) -> dict[str, Any] | None:
    question = str(raw_case.get("question", "")).strip()
    # DomainRAG tasks are not fully uniform: most use `answers`, while
    # multi-doc style samples often provide a single `answer` string.
    gold_answer = extract_domainrag_gold_answer(raw_case.get("answers"))
    if not gold_answer:
        gold_answer = extract_domainrag_gold_answer(raw_case.get("answer"))
    if not question or not gold_answer:
        return None

    case_id = str(raw_case.get("case_id", "")).strip()
    if not case_id:
        raw_id = raw_case.get("id")
        case_id = str(raw_id).strip() if raw_id is not None else ""
    if not case_id:
        case_id = f"{source_path.stem}-{line_number}"

    task_name = str(raw_case.get("domainrag_task", "")).strip() or source_path.parent.name or source_path.stem
    history = build_domainrag_history(raw_case.get("history_qa"))
    return {
        "case_id": case_id,
        "task": task_name,
        "source_task": task_name,
        "event": "",
        "question": question,
        "gold_answer": gold_answer,
        "retrieval_query": question,
        "generation_query": question,
        "history": history,
    }


def normalize_crud_export_case(raw_case: dict[str, Any], *, fallback_index: int) -> dict[str, Any] | None:
    task = str(raw_case.get("task", "")).strip().lower()
    if task == "event_summary":
        task = "summary"
    source_task = str(raw_case.get("source_task", raw_case.get("task", task))).strip()
    if task not in SUPPORTED_TASKS:
        return None

    case_id = str(raw_case.get("case_id", "")).strip() or f"{task}-{fallback_index:06d}"
    event = str(raw_case.get("event", "")).strip()

    if task == "summary":
        gold_answer = coerce_text(raw_case.get("summary") or raw_case.get("answer"))
        if not event or not gold_answer:
            return None
        retrieval_query = event
        generation_query = f"请基于知识库检索内容，对以下事件写一段简洁摘要：\n{event}"
        question = ""
    else:
        question = str(raw_case.get("question", raw_case.get("questions", ""))).strip()
        gold_answer = coerce_text(raw_case.get("answer"))
        if not question or not gold_answer:
            return None
        retrieval_query = f"{event}\n{question}".strip() if event else question
        generation_query = f"事件背景：{event}\n\n问题：{question}" if event else question

    return {
        "case_id": case_id,
        "task": task,
        "source_task": source_task or task,
        "event": event,
        "question": question,
        "gold_answer": gold_answer,
        "retrieval_query": retrieval_query,
        "generation_query": generation_query,
        "history": [],
    }


def extract_domainrag_gold_answer(value: Any) -> str:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, list):
                for nested in item:
                    text = str(nested).strip()
                    if text:
                        return text
                continue
            text = str(item).strip()
            if text:
                return text
        return ""
    if value is None:
        return ""
    return str(value).strip()


def build_domainrag_history(value: Any) -> list[ChatMessage]:
    history: list[ChatMessage] = []
    if not isinstance(value, list):
        return history

    for item in value:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = extract_domainrag_gold_answer(item.get("answers"))
        if question:
            history.append(ChatMessage(role="user", content=question))
        if answer:
            history.append(ChatMessage(role="assistant", content=answer))
    return history


def coerce_text(value: Any) -> str:
    if isinstance(value, list):
        for item in value:
            text = str(item).strip()
            if text:
                return text
        return ""
    if value is None:
        return ""
    return str(value).strip()


def evaluate_cases_with_ragas(
    *,
    settings: AppSettings,
    knowledge_base_name: str,
    cases: list[dict[str, str]],
    top_k: int,
    score_threshold: float,
    show_cases: bool,
    batch_size: int,
    dataset_path: Path,
    metric_names: list[str],
    judge_model_name: str,
    judge_embedding_model_name: str,
) -> dict[str, Any]:
    ragas_modules = load_ragas_modules()
    judge_model_resolved = judge_model_name or settings.model.DEFAULT_LLM_MODEL
    judge_embedding_resolved = ""
    judge_llm = ragas_modules["LangchainLLMWrapper"](
        build_chat_model(settings, model_name=judge_model_resolved, temperature=0.0)
    )

    needs_embeddings = "response_relevancy" in metric_names
    judge_embeddings = None
    if needs_embeddings:
        judge_embedding_resolved = (
            judge_embedding_model_name or settings.model.DEFAULT_EMBEDDING_MODEL
        )
        judge_embeddings = ragas_modules["LangchainEmbeddingsWrapper"](
            build_embeddings(settings, model_name=judge_embedding_resolved)
        )

    metrics = build_ragas_metrics(metric_names, ragas_modules)
    ensure_metric_prompts(
        metrics=metrics,
        language=DEFAULT_LANGUAGE,
        judge_llm=judge_llm,
    )

    details: list[dict[str, Any]] = []
    scorable_cases: list[dict[str, Any]] = []

    for case in cases:
        detail = build_case_detail(
            case=case,
            settings=settings,
            knowledge_base_name=knowledge_base_name,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        detail["ragas_metrics"] = zero_metric_scores(metric_names)
        details.append(detail)
        if detail["status"] == "ok":
            scorable_cases.append(
                {
                    "case": case,
                    "detail": detail,
                }
            )

    score_case_batches(
        ragas_modules=ragas_modules,
        metric_names=metric_names,
        metrics=metrics,
        judge_llm=judge_llm,
        judge_embeddings=judge_embeddings,
        scorable_cases=scorable_cases,
        batch_size=max(1, batch_size),
    )

    metric_accumulator = MetricAccumulator(metric_names)
    error_count = 0
    for detail in details:
        metric_accumulator.update(detail["ragas_metrics"])
        if detail["status"] != "ok":
            error_count += 1
        if show_cases:
            print(json.dumps(detail, ensure_ascii=False))

    summary = {
        "benchmark": "CRUD_RAG_ragas",
        "dataset_path": str(dataset_path),
        "knowledge_base_name": knowledge_base_name,
        "case_total": len(cases),
        "evaluated_cases": len(cases),
        "error_count": error_count,
        "judge_model": judge_model_resolved,
        "judge_embedding_model": judge_embedding_resolved,
        "metrics": metric_names,
        "language": DEFAULT_LANGUAGE,
        "ragas_batch_size": max(1, batch_size),
        "prompt_cache_dir": str((RAGAS_CACHE_ROOT / DEFAULT_LANGUAGE).resolve()),
        "metric_summary": metric_accumulator.summary(),
        "details": details if show_cases else [],
    }
    return summary


def build_case_detail(
    *,
    case: dict[str, str],
    settings: AppSettings,
    knowledge_base_name: str,
    top_k: int,
    score_threshold: float,
) -> dict[str, Any]:
    history = list(case.get("history", []))
    base_detail: dict[str, Any] = {
        "case_id": case["case_id"],
        "task": case["task"],
        "source_task": case.get("source_task", case["task"]),
        "query": case["generation_query"],
        "retrieval_query": case["retrieval_query"],
        "gold_answer": case["gold_answer"],
        "history_turns": len(history),
        "reference_count": 0,
        "top_sources": [],
        "top_references": [],
        "retrieved_contexts": [],
        "answer": "",
        "status": "ok",
        "error": "",
    }

    try:
        references = search_local_knowledge_base(
            settings=settings,
            knowledge_base_name=knowledge_base_name,
            query=case["retrieval_query"],
            top_k=top_k,
            score_threshold=score_threshold,
            history=history,
        )
    except Exception as exc:
        base_detail["status"] = "retrieval_error"
        base_detail["error"] = str(exc)
        return base_detail

    base_detail["reference_count"] = len(references)
    base_detail["reference_overview"] = build_reference_overview(references).model_dump()
    base_detail["top_sources"] = [ref.source for ref in references]
    base_detail["top_references"] = build_top_reference_details(references)
    base_detail["retrieved_contexts"] = extract_reference_contents(references)

    try:
        answer = generate_rag_answer(
            settings=settings,
            query=case["generation_query"],
            references=references,
            history=history,
        )
    except Exception as exc:
        base_detail["status"] = "generation_error"
        base_detail["error"] = str(exc)
        return base_detail

    base_detail["answer"] = answer
    return base_detail


def score_case_batches(
    *,
    ragas_modules: dict[str, Any],
    metric_names: list[str],
    metrics: list[Any],
    judge_llm: Any,
    judge_embeddings: Any,
    scorable_cases: list[dict[str, Any]],
    batch_size: int,
) -> None:
    for batch in iter_chunks(scorable_cases, batch_size):
        try:
            batch_scores = evaluate_case_batch_with_ragas(
                ragas_modules=ragas_modules,
                metric_names=metric_names,
                metrics=metrics,
                judge_llm=judge_llm,
                judge_embeddings=judge_embeddings,
                batch=batch,
                batch_size=batch_size,
            )
        except Exception as exc:
            batch_error = str(exc)
            for entry in batch:
                detail = entry["detail"]
                try:
                    detail["ragas_metrics"] = evaluate_single_case_with_ragas(
                        ragas_modules=ragas_modules,
                        metric_names=metric_names,
                        metrics=metrics,
                        judge_llm=judge_llm,
                        judge_embeddings=judge_embeddings,
                        case=entry["case"],
                        answer=str(detail["answer"]),
                        retrieved_contexts=list(detail["retrieved_contexts"]),
                    )
                except Exception as inner_exc:
                    detail["status"] = "ragas_error"
                    detail["error"] = f"{batch_error}; fallback={inner_exc}"
            continue

        for entry, metric_scores in zip(batch, batch_scores, strict=True):
            entry["detail"]["ragas_metrics"] = metric_scores


def evaluate_case_batch_with_ragas(
    *,
    ragas_modules: dict[str, Any],
    metric_names: list[str],
    metrics: list[Any],
    judge_llm: Any,
    judge_embeddings: Any,
    batch: list[dict[str, Any]],
    batch_size: int,
) -> list[dict[str, float]]:
    samples = [
        ragas_modules["SingleTurnSample"](
            user_input=entry["case"]["generation_query"],
            retrieved_contexts=list(entry["detail"]["retrieved_contexts"]),
            response=str(entry["detail"]["answer"]),
            reference=entry["case"]["gold_answer"],
        )
        for entry in batch
    ]
    dataset = ragas_modules["EvaluationDataset"](samples=samples)
    result = ragas_modules["evaluate"](
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        raise_exceptions=False,
        show_progress=False,
        batch_size=max(1, batch_size),
    )
    rows = list(getattr(result, "scores", []) or [])
    if len(rows) != len(batch):
        raise RuntimeError(
            f"RAGAS 批量返回条数异常: expected={len(batch)} actual={len(rows)}"
        )
    return [coerce_metric_scores(row, metric_names) for row in rows]


def evaluate_single_case_with_ragas(
    *,
    ragas_modules: dict[str, Any],
    metric_names: list[str],
    metrics: list[Any],
    judge_llm: Any,
    judge_embeddings: Any,
    case: dict[str, str],
    answer: str,
    retrieved_contexts: list[str],
) -> dict[str, float]:
    sample = ragas_modules["SingleTurnSample"](
        user_input=case["generation_query"],
        retrieved_contexts=retrieved_contexts,
        response=answer,
        reference=case["gold_answer"],
    )
    dataset = ragas_modules["EvaluationDataset"](samples=[sample])
    result = ragas_modules["evaluate"](
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        raise_exceptions=False,
        show_progress=False,
    )
    if not getattr(result, "scores", None):
        raise RuntimeError("RAGAS 未返回单条评分结果。")

    return coerce_metric_scores(result.scores[0], metric_names)


def coerce_metric_scores(row: dict[str, Any], metric_names: list[str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    invalid_metrics: list[str] = []
    for metric_name in metric_names:
        raw_value = resolve_metric_value(row, metric_name)
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            invalid_metrics.append(metric_name)
            continue
        if not math.isfinite(numeric_value):
            invalid_metrics.append(metric_name)
            continue
        scores[metric_name] = numeric_value

    if invalid_metrics:
        raise RuntimeError(f"RAGAS 指标无有效数值: {', '.join(invalid_metrics)}")
    return scores


def iter_chunks(items: list[Any], chunk_size: int) -> list[list[Any]]:
    return [
        items[index : index + chunk_size]
        for index in range(0, len(items), max(1, chunk_size))
    ]


def resolve_metric_value(row: dict[str, Any], metric_name: str) -> Any:
    if metric_name in row:
        return row[metric_name]

    prefixed_matches = [
        value
        for key, value in row.items()
        if key == metric_name or key.startswith(f"{metric_name}(")
    ]
    if prefixed_matches:
        return prefixed_matches[0]
    return None


def load_ragas_modules() -> dict[str, Any]:
    try:
        from ragas import evaluate
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        from ragas.embeddings.base import LangchainEmbeddingsWrapper
        from ragas.llms.base import LangchainLLMWrapper
        from ragas.metrics import (
            Faithfulness,
            FactualCorrectness,
            LLMContextRecall,
            ResponseRelevancy,
        )
    except ImportError as exc:
        raise RuntimeError(
            "未安装 ragas 相关依赖。请先执行 `python -m pip install -r requirements.txt`。"
        ) from exc

    return {
        "evaluate": evaluate,
        "EvaluationDataset": EvaluationDataset,
        "SingleTurnSample": SingleTurnSample,
        "LangchainEmbeddingsWrapper": LangchainEmbeddingsWrapper,
        "LangchainLLMWrapper": LangchainLLMWrapper,
        "Faithfulness": Faithfulness,
        "FactualCorrectness": FactualCorrectness,
        "LLMContextRecall": LLMContextRecall,
        "ResponseRelevancy": ResponseRelevancy,
    }


def build_ragas_metrics(metric_names: list[str], ragas_modules: dict[str, Any]) -> list[Any]:
    metrics: list[Any] = []
    for metric_name in metric_names:
        if metric_name == "llm_context_recall":
            metrics.append(ragas_modules["LLMContextRecall"](name="llm_context_recall"))
            continue
        if metric_name == "faithfulness":
            metrics.append(ragas_modules["Faithfulness"](name="faithfulness"))
            continue
        if metric_name == "factual_correctness":
            metrics.append(
                ragas_modules["FactualCorrectness"](
                    name="factual_correctness",
                    language=DEFAULT_LANGUAGE,
                )
            )
            continue
        if metric_name == "response_relevancy":
            metrics.append(ragas_modules["ResponseRelevancy"](name="response_relevancy"))
            continue
        raise ValueError(f"不支持的指标: {metric_name}")
    return metrics


def ensure_metric_prompts(
    *,
    metrics: list[Any],
    language: str,
    judge_llm: Any,
) -> None:
    language_cache_dir = RAGAS_CACHE_ROOT / language
    language_cache_dir.mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        metric_cache_dir = language_cache_dir / metric.name
        metric_cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            loaded_prompts = metric.load_prompts(str(metric_cache_dir), language=language)
        except Exception:
            adapted_prompts = asyncio.run(
                metric.adapt_prompts(language=language, llm=judge_llm, adapt_instruction=False)
            )
            metric.set_prompts(**adapted_prompts)
            metric.save_prompts(str(metric_cache_dir))
            continue
        metric.set_prompts(**loaded_prompts)


def zero_metric_scores(metric_names: list[str]) -> dict[str, float]:
    return {metric_name: 0.0 for metric_name in metric_names}


class MetricAccumulator:
    def __init__(self, metric_names: list[str]) -> None:
        self.metric_names = list(metric_names)
        self.values: dict[str, list[float]] = {metric_name: [] for metric_name in metric_names}

    def update(self, metric_values: dict[str, float]) -> None:
        for metric_name in self.metric_names:
            self.values[metric_name].append(float(metric_values.get(metric_name, 0.0)))

    def summary(self) -> dict[str, float]:
        rendered: dict[str, float] = {}
        for metric_name in self.metric_names:
            values = self.values.get(metric_name, [])
            if not values:
                rendered[metric_name] = 0.0
                continue
            rendered[metric_name] = statistics.fmean(values)
        return rendered


if __name__ == "__main__":
    raise SystemExit(main())
