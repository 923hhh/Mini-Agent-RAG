from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.chains.rag import generate_rag_answer
from app.retrievers.local_kb import search_local_knowledge_base
from app.services.crud_eval_cases import (
    SUPPORTED_TASKS,
    build_cases,
    load_crud_rag_items,
    normalize_tasks,
    resolve_data_file,
)
from app.services.eval_reference_utils import (
    build_reference_eval_text,
    build_top_reference_details,
    infer_reference_sample_id,
)
from app.services.reference_overview import build_reference_overview
from app.services.settings import load_settings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the current local knowledge-base RAG with CRUD_RAG-style tasks."
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
        help="Direct path to CRUD_RAG task data (JSON or JSONL). Overrides --crud-rag-root.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(SUPPORTED_TASKS),
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
        "--skip-generation",
        action="store_true",
        help="Only evaluate retrieval-side proxy metrics without calling the generation chain.",
    )
    parser.add_argument(
        "--show-cases",
        action="store_true",
        help="Print one JSON line per case result.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to save the final JSON summary.",
    )
    args = parser.parse_args()

    selected_tasks = normalize_tasks(args.tasks)
    data_path = resolve_data_file(args.data_file, args.crud_rag_root)
    raw_items = load_crud_rag_items(data_path)
    cases = build_cases(raw_items, selected_tasks)
    if args.limit > 0:
        cases = cases[: args.limit]

    settings = load_settings(PROJECT_ROOT)
    summary = evaluate_cases(
        settings=settings,
        knowledge_base_name=args.knowledge_base_name,
        cases=cases,
        top_k=args.top_k,
        score_threshold=args.score_threshold,
        skip_generation=args.skip_generation,
        show_cases=args.show_cases,
        dataset_path=data_path,
    )

    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    return 0


def evaluate_cases(
    *,
    settings,
    knowledge_base_name: str,
    cases: list[dict[str, str]],
    top_k: int,
    score_threshold: float,
    skip_generation: bool,
    show_cases: bool,
    dataset_path: Path,
) -> dict[str, Any]:
    retrieval_overall = MetricAccumulator()
    generation_overall = MetricAccumulator()
    task_groups: dict[str, dict[str, MetricAccumulator]] = defaultdict(
        lambda: {
            "retrieval": MetricAccumulator(),
            "generation": MetricAccumulator(),
        }
    )
    details: list[dict[str, Any]] = []

    for case in cases:
        try:
            references = search_local_knowledge_base(
                settings=settings,
                knowledge_base_name=knowledge_base_name,
                query=case["retrieval_query"],
                top_k=top_k,
                score_threshold=score_threshold,
            )
        except Exception as exc:
            detail = {
                "case_id": case["case_id"],
                "task": case["task"],
                "query": case["generation_query"],
                "status": "retrieval_error",
                "error": str(exc),
            }
            details.append(detail)
            if show_cases:
                print(json.dumps(detail, ensure_ascii=False))
            continue

        reference_overview = build_reference_overview(references).model_dump()
        context_text = build_reference_eval_text(references)
        gold_answer = case["gold_answer"]
        retrieval_metrics = compute_retrieval_proxy_metrics(
            task=case["task"],
            references=references,
            context_text=context_text,
            gold_answer=gold_answer,
        )
        retrieval_overall.update(retrieval_metrics)
        task_groups[case["task"]]["retrieval"].update(retrieval_metrics)

        detail: dict[str, Any] = {
            "case_id": case["case_id"],
            "task": case["task"],
            "source_task": case.get("source_task", case["task"]),
            "query": case["generation_query"],
            "retrieval_query": case["retrieval_query"],
            "gold_answer": gold_answer,
            "reference_count": len(references),
            "reference_overview": reference_overview,
            "retrieval_metrics": retrieval_metrics,
            "top_sources": [ref.source for ref in references],
            "top_source_paths": [ref.source_path for ref in references],
            "top_source_sample_ids": [infer_reference_sample_id(ref.source_path) for ref in references],
            "top_references": build_top_reference_details(references),
            "top_modalities": [ref.source_modality for ref in references],
            "status": "ok",
        }

        if not skip_generation:
            try:
                answer = generate_rag_answer(
                    settings=settings,
                    query=case["generation_query"],
                    references=references,
                    history=[],
                )
                generation_metrics = compute_generation_metrics(answer=answer, gold_answer=gold_answer)
                generation_overall.update(generation_metrics)
                task_groups[case["task"]]["generation"].update(generation_metrics)
                detail["answer"] = answer
                detail["generation_metrics"] = generation_metrics
            except Exception as exc:
                detail["status"] = "generation_error"
                detail["generation_error"] = str(exc)
        details.append(detail)
        if show_cases:
            print(json.dumps(detail, ensure_ascii=False))

    summary = {
        "benchmark": "CRUD_RAG_adapted",
        "dataset_path": str(dataset_path),
        "knowledge_base_name": knowledge_base_name,
        "task_filter": sorted(task_groups.keys()),
        "case_total": len(cases),
        "metric_note": (
            "CRUD_RAG 原始数据不提供 gold 文档 ID。本脚本的 retrieval 指标为代理指标，"
            "主要衡量检索结果非空率、gold 答案在检索上下文中的覆盖情况，以及上下文与 gold 答案的字符级重合度。"
        ),
        "retrieval_metrics": retrieval_overall.summary(),
        "generation_metrics": generation_overall.summary() if not skip_generation else {},
        "task_breakdown": {
            task: {
                "retrieval_metrics": values["retrieval"].summary(),
                "generation_metrics": values["generation"].summary() if not skip_generation else {},
            }
            for task, values in sorted(task_groups.items())
        },
        "details": details if show_cases else [],
    }
    return summary
def compute_retrieval_proxy_metrics(
    *,
    task: str,
    references,
    context_text: str,
    gold_answer: str,
) -> dict[str, float]:
    normalized_context = normalize_text(context_text)
    normalized_gold = normalize_text(gold_answer)
    has_references = 1.0 if references else 0.0
    answer_in_context = 0.0
    if task == "quest_answer" and normalized_gold and normalized_gold in normalized_context:
        answer_in_context = 1.0

    return {
        "retrieval_non_empty": has_references,
        "reference_count": float(len(references)),
        "context_answer_substring_hit": answer_in_context,
        "context_char_f1": char_f1(context_text, gold_answer),
        "context_rouge_l_f1": rouge_l_f1(context_text, gold_answer),
    }


def compute_generation_metrics(*, answer: str, gold_answer: str) -> dict[str, float]:
    return {
        "answer_exact_match": exact_match(answer, gold_answer),
        "answer_char_f1": char_f1(answer, gold_answer),
        "answer_rouge_l_f1": rouge_l_f1(answer, gold_answer),
        "answer_bleu_1": sentence_bleu(answer, gold_answer, max_n=1),
        "answer_bleu_2": sentence_bleu(answer, gold_answer, max_n=2),
        "answer_bleu_4": sentence_bleu(answer, gold_answer, max_n=4),
    }


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    collapsed = "".join(normalized.split())
    return collapsed


def char_tokens(text: str) -> list[str]:
    return list(normalize_text(text))


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(gold) and normalize_text(gold) else 0.0


def char_f1(pred: str, gold: str) -> float:
    pred_tokens = char_tokens(pred)
    gold_tokens = char_tokens(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    overlap = sum(min(pred_counter[token], gold_counter[token]) for token in pred_counter)
    if overlap <= 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rouge_l_f1(pred: str, gold: str) -> float:
    pred_tokens = char_tokens(pred)
    gold_tokens = char_tokens(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0

    lcs = lcs_length(pred_tokens, gold_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0

    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for index, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def sentence_bleu(pred: str, gold: str, *, max_n: int) -> float:
    pred_tokens = char_tokens(pred)
    gold_tokens = char_tokens(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0

    precisions: list[float] = []
    for n in range(1, max_n + 1):
        pred_ngrams = Counter(tuple(pred_tokens[i : i + n]) for i in range(max(len(pred_tokens) - n + 1, 0)))
        gold_ngrams = Counter(tuple(gold_tokens[i : i + n]) for i in range(max(len(gold_tokens) - n + 1, 0)))
        total = sum(pred_ngrams.values())
        if total == 0:
            precisions.append(0.0)
            continue
        matched = sum(min(count, gold_ngrams[gram]) for gram, count in pred_ngrams.items())
        precisions.append((matched + 1.0) / (total + 1.0))

    if min(precisions) <= 0:
        return 0.0

    log_precision = sum(math.log(value) for value in precisions) / max_n
    bp = 1.0
    if len(pred_tokens) < len(gold_tokens):
        bp = math.exp(1 - (len(gold_tokens) / max(len(pred_tokens), 1)))
    return bp * math.exp(log_precision)

class MetricAccumulator:
    def __init__(self) -> None:
        self.values: dict[str, list[float]] = defaultdict(list)

    def update(self, metric_values: dict[str, float]) -> None:
        for key, value in metric_values.items():
            self.values[key].append(float(value))

    def summary(self) -> dict[str, float]:
        if not self.values:
            return {}
        rendered: dict[str, float] = {}
        for key, values in sorted(self.values.items()):
            if not values:
                continue
            rendered[key] = statistics.fmean(values)
        rendered["evaluated_cases"] = float(max((len(v) for v in self.values.values()), default=0))
        return rendered


if __name__ == "__main__":
    raise SystemExit(main())
