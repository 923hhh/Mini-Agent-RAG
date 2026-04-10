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
from app.services.reference_overview import build_reference_overview
from app.services.settings import load_settings


SUPPORTED_TASKS = ("quest_answer", "summary")


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


def normalize_tasks(values: list[str]) -> list[str]:
    supported = set(SUPPORTED_TASKS)
    normalized: list[str] = []
    for value in values:
        task = str(value).strip().lower()
        if task not in supported:
            raise ValueError(f"不支持的任务类型: {value}。支持: {', '.join(SUPPORTED_TASKS)}")
        if task not in normalized:
            normalized.append(task)
    if not normalized:
        raise ValueError("至少需要一个任务类型。")
    return normalized


def resolve_data_file(data_file: str, crud_rag_root: str) -> Path:
    if data_file:
        path = Path(data_file)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"CRUD_RAG 数据文件不存在: {path}")
        return path

    if not crud_rag_root:
        raise ValueError("请提供 --data-file 或 --crud-rag-root。")

    root = Path(crud_rag_root)
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"CRUD_RAG 仓库目录不存在: {root}")

    candidates = [
        root / "split_merged.json",
        root / "data" / "split_merged.json",
        root / "dataset" / "split_merged.json",
        root / "datasets" / "split_merged.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    found = sorted(root.rglob("split_merged.json"))
    if found:
        return found[0]
    raise FileNotFoundError(f"在 {root} 下未找到 split_merged.json。")


def load_crud_rag_items(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        items: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                items.append(obj)
        return items

    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    return normalize_json_payload_to_items(raw)


def normalize_json_payload_to_items(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]

    if not isinstance(raw, dict):
        raise ValueError("CRUD_RAG 数据格式无效，期望 JSON 数组、JSONL 或映射对象。")

    items: list[dict[str, Any]] = []
    if "data" in raw and isinstance(raw["data"], list):
        items.extend(item for item in raw["data"] if isinstance(item, dict))
    if "items" in raw and isinstance(raw["items"], list):
        items.extend(item for item in raw["items"] if isinstance(item, dict))

    for task_name in SUPPORTED_TASKS:
        task_value = raw.get(task_name)
        if isinstance(task_value, list):
            for item in task_value:
                if isinstance(item, dict):
                    merged = dict(item)
                    merged.setdefault("task", task_name)
                    items.append(merged)

    if items:
        return items

    raise ValueError("无法从 JSON 中解析出 CRUD_RAG 样本列表。")


def build_cases(
    items: list[dict[str, Any]],
    selected_tasks: list[str],
) -> list[dict[str, str]]:
    task_set = set(selected_tasks)
    cases: list[dict[str, str]] = []
    for index, item in enumerate(items, start=1):
        task = infer_task_name(item)
        if task not in task_set:
            continue

        case = build_case(item, task, fallback_index=index)
        if case is None:
            continue
        cases.append(case)
    return cases


def infer_task_name(item: dict[str, Any]) -> str | None:
    explicit_keys = ("task", "task_name", "type", "category")
    for key in explicit_keys:
        value = str(item.get(key, "")).strip().lower()
        if value in SUPPORTED_TASKS:
            return value

    has_question = bool(first_non_empty(item, ("question", "questions", "query_question", "ask")))
    has_answer = bool(first_non_empty(item, ("answer", "answers", "gold_answer", "target_answer")))
    has_summary = bool(first_non_empty(item, ("summary", "gold_summary", "target_summary")))
    has_event = bool(first_non_empty(item, ("event", "title", "context", "instruction")))

    if has_question and has_answer:
        return "quest_answer"
    if has_summary and has_event:
        return "summary"
    return None


def build_case(
    item: dict[str, Any],
    task: str,
    *,
    fallback_index: int,
) -> dict[str, str] | None:
    case_id = (
        first_non_empty(item, ("case_id", "id", "uid", "sample_id"))
        or f"{task}-{fallback_index:06d}"
    )
    event = first_non_empty(item, ("event", "title", "context", "instruction"))

    if task == "summary":
        gold_answer = first_non_empty(item, ("summary", "gold_summary", "target_summary"))
        if not event or not gold_answer:
            return None
        retrieval_query = event
        generation_query = f"请基于知识库检索内容，对以下事件写一段简洁摘要：\n{event}"
        return {
            "case_id": case_id,
            "task": task,
            "event": event,
            "question": "",
            "gold_answer": gold_answer,
            "retrieval_query": retrieval_query,
            "generation_query": generation_query,
        }

    question = first_non_empty(item, ("question", "questions", "query_question", "ask"))
    gold_answer = first_non_empty(item, ("answer", "answers", "gold_answer", "target_answer"))
    if not question or not gold_answer:
        return None
    retrieval_query = f"{event}\n{question}".strip() if event else question
    if event:
        generation_query = f"事件背景：{event}\n\n问题：{question}"
    else:
        generation_query = question
    return {
        "case_id": case_id,
        "task": task,
        "event": event,
        "question": question,
        "gold_answer": gold_answer,
        "retrieval_query": retrieval_query,
        "generation_query": generation_query,
    }


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
            "query": case["generation_query"],
            "retrieval_query": case["retrieval_query"],
            "gold_answer": gold_answer,
            "reference_count": len(references),
            "reference_overview": reference_overview,
            "retrieval_metrics": retrieval_metrics,
            "top_sources": [ref.source for ref in references],
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


def build_reference_eval_text(references) -> str:
    parts: list[str] = []
    for ref in references:
        for value in (
            getattr(ref, "content", ""),
            getattr(ref, "ocr_text", ""),
            getattr(ref, "image_caption", ""),
            getattr(ref, "evidence_summary", ""),
        ):
            text = str(value or "").strip()
            if text:
                parts.append(text)
    return "\n".join(parts)


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


def first_non_empty(item: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = item.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            for nested in value:
                text = str(nested).strip()
                if text:
                    return text
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


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
