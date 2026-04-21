"""构造 CRUD 评测样本与对应参考信息。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SUPPORTED_TASKS = ("quest_answer", "summary")
TASK_ALIASES = {
    "event_summary": "summary",
    "questanswer_1doc": "quest_answer",
    "questanswer_2docs": "quest_answer",
    "questanswer_3docs": "quest_answer",
}


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

    supported_keys = list(SUPPORTED_TASKS) + list(TASK_ALIASES)
    for task_name in supported_keys:
        task_value = raw.get(task_name)
        if isinstance(task_value, list):
            for item in task_value:
                if isinstance(item, dict):
                    merged = dict(item)
                    merged.setdefault("task", TASK_ALIASES.get(task_name, task_name))
                    merged.setdefault("source_task", task_name)
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
        value = TASK_ALIASES.get(value, value)
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
        first_non_empty(item, ("case_id", "id", "ID", "uid", "sample_id"))
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
            "source_task": str(item.get("source_task", item.get("task", ""))).strip(),
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
        "source_task": str(item.get("source_task", item.get("task", ""))).strip(),
        "event": event,
        "question": question,
        "gold_answer": gold_answer,
        "retrieval_query": retrieval_query,
        "generation_query": generation_query,
    }


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
