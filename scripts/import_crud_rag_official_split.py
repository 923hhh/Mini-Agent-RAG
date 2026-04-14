from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.kb_ingestion_service import (
    ensure_knowledge_base_layout,
    rebuild_knowledge_base,
    render_rebuild_summary,
)
from app.services.settings import load_settings


SUPPORTED_SOURCE_TASKS = (
    "questanswer_1doc",
    "questanswer_2docs",
    "questanswer_3docs",
    "event_summary",
)
TASK_ALIASES = {
    "questanswer_1doc": "quest_answer",
    "questanswer_2docs": "quest_answer",
    "questanswer_3docs": "quest_answer",
    "event_summary": "summary",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="import_crud_rag_official_split",
        description="Export the official local CRUD_RAG split JSON into a dedicated knowledge base and evaluation file.",
    )
    parser.add_argument("--kb-name", required=True, help="Target knowledge base name.")
    parser.add_argument(
        "--data-file",
        required=True,
        type=Path,
        help="Path to CRUD_RAG official split JSON, usually data/crud_split/split_merged.json.",
    )
    parser.add_argument(
        "--source-tasks",
        nargs="+",
        default=list(SUPPORTED_SOURCE_TASKS),
        help="Subset of official source tasks to import.",
    )
    parser.add_argument(
        "--limit-per-task",
        type=int,
        default=0,
        help="Import at most N samples per source task. 0 means all.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite existing exported files.",
    )
    parser.add_argument(
        "--auto-rebuild",
        action="store_true",
        help="Rebuild the target knowledge base after export.",
    )
    parser.add_argument(
        "--force-full-rebuild",
        action="store_true",
        help="Force a full rebuild when --auto-rebuild is enabled.",
    )
    parser.add_argument("--chunk-size", type=int, default=None, help="Optional rebuild chunk size override.")
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Optional rebuild chunk overlap override.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Optional rebuild embedding model override.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root. Defaults to the current repository root.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    settings = load_settings(project_root)
    data_file = args.data_file.resolve()
    if not data_file.exists():
        raise SystemExit(f"数据文件不存在: {data_file}")

    selected_source_tasks = normalize_source_tasks(args.source_tasks)
    payload = json.loads(data_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("官方 split JSON 格式无效，期望顶层为对象。")

    content_dir, _ = ensure_knowledge_base_layout(settings, args.kb_name)
    export_root = content_dir / "crud_rag_official_split"
    export_root.mkdir(parents=True, exist_ok=True)

    file_metadata_map: dict[str, dict[str, str]] = {}
    eval_cases: list[dict[str, str]] = []
    imported_samples = 0
    saved_files = 0
    overwritten_files = 0
    skipped_files = 0
    task_sample_counts: dict[str, int] = {}

    for source_task in selected_source_tasks:
        rows = payload.get(source_task)
        if not isinstance(rows, list):
            continue

        imported_for_task = 0
        for row in rows:
            if args.limit_per_task > 0 and imported_for_task >= args.limit_per_task:
                break
            if not isinstance(row, dict):
                continue

            sample_id = coerce_text(row.get("ID")) or coerce_text(row.get("id"))
            if not sample_id:
                sample_id = f"{source_task}_{imported_samples + 1:06d}"
            event = coerce_text(row.get("event"))
            sample_dir = export_root / source_task / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            exported_docs = build_documents_for_row(source_task=source_task, row=row, event=event)
            if not exported_docs:
                continue

            event_id = build_event_id(event)
            for file_name, text in exported_docs:
                target_path = sample_dir / file_name
                status = write_text_file(
                    path=target_path,
                    text=text,
                    overwrite_existing=args.overwrite_existing,
                )
                if status == "saved":
                    saved_files += 1
                elif status == "overwritten":
                    overwritten_files += 1
                else:
                    skipped_files += 1

                relative_path = target_path.relative_to(content_dir).as_posix()
                file_metadata_map[relative_path] = {
                    "dataset_name": data_file.name,
                    "doc_type": infer_doc_type(file_name),
                    "sample_id": sample_id,
                    "source_task": source_task,
                    "event_id": event_id,
                    "group_retrieval_strategy": "same_sample_id",
                    "doc_index": str(extract_doc_index(file_name)),
                }

            eval_case = build_eval_case(source_task=source_task, row=row, sample_id=sample_id, event=event)
            if eval_case is not None:
                eval_cases.append(eval_case)

            imported_samples += 1
            imported_for_task += 1

        task_sample_counts[source_task] = imported_for_task

    eval_dir = project_root / "data" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_path = eval_dir / f"{args.kb_name}_official_split.jsonl"
    write_jsonl(eval_path, eval_cases)

    metadata_map_path = content_dir / ".rag_file_metadata.json"
    metadata_payload = {
        "source": "CRUD_RAG_official_split",
        "data_file": str(data_file),
        "source_tasks": selected_source_tasks,
        "files": file_metadata_map,
    }
    metadata_map_path.write_text(
        json.dumps(metadata_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest_path = settings.knowledge_base_dir(args.kb_name) / "crud_rag_official_split_import_manifest.json"
    manifest_payload = {
        "source": "local_official_split",
        "data_file": str(data_file),
        "source_tasks": selected_source_tasks,
        "task_sample_counts": task_sample_counts,
        "sample_count": imported_samples,
        "saved_files": saved_files,
        "overwritten_files": overwritten_files,
        "skipped_files": skipped_files,
        "eval_case_count": len(eval_cases),
        "eval_case_path": str(eval_path),
        "export_root": str(export_root),
        "metadata_map_path": str(metadata_map_path),
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        render_import_summary(
            kb_name=args.kb_name,
            data_file=data_file,
            export_root=export_root,
            manifest_path=manifest_path,
            eval_path=eval_path,
            imported_samples=imported_samples,
            task_sample_counts=task_sample_counts,
            saved_files=saved_files,
            overwritten_files=overwritten_files,
            skipped_files=skipped_files,
            eval_case_count=len(eval_cases),
        )
    )

    if args.auto_rebuild:
        rebuild_result = rebuild_knowledge_base(
            settings=settings,
            knowledge_base_name=args.kb_name,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embedding_model=args.embedding_model,
            force_full_rebuild=args.force_full_rebuild,
            progress_callback=render_rebuild_progress,
        )
        print()
        print(render_rebuild_summary(rebuild_result))

    return 0


def normalize_source_tasks(values: list[str]) -> list[str]:
    supported = set(SUPPORTED_SOURCE_TASKS)
    normalized: list[str] = []
    for value in values:
        task = str(value).strip().lower()
        if task not in supported:
            raise ValueError(f"不支持的官方任务: {value}。支持: {', '.join(SUPPORTED_SOURCE_TASKS)}")
        if task not in normalized:
            normalized.append(task)
    if not normalized:
        raise ValueError("至少需要一个 source_task。")
    return normalized


def build_documents_for_row(
    *,
    source_task: str,
    row: dict[str, Any],
    event: str,
) -> list[tuple[str, str]]:
    if source_task == "event_summary":
        text = coerce_text(row.get("text"))
        rendered = build_named_document(event=event, label="原文", text=text)
        return [("source.txt", rendered)] if rendered else []

    documents: list[tuple[str, str]] = []
    for index in (1, 2, 3):
        key = f"news{index}"
        text = coerce_text(row.get(key))
        rendered = build_named_document(event=event, label=f"新闻{index}", text=text)
        if rendered:
            documents.append((f"news{index}.txt", rendered))
    return documents


def build_eval_case(
    *,
    source_task: str,
    row: dict[str, Any],
    sample_id: str,
    event: str,
) -> dict[str, str] | None:
    task = TASK_ALIASES[source_task]
    if task == "summary":
        gold_answer = coerce_text(row.get("summary"))
        if not event or not gold_answer:
            return None
        return {
            "case_id": sample_id,
            "task": task,
            "source_task": source_task,
            "event": event,
            "summary": gold_answer,
        }

    question = coerce_text(row.get("question")) or coerce_text(row.get("questions"))
    answer = coerce_text(row.get("answer")) or coerce_text(row.get("answers"))
    if not question or not answer:
        return None
    return {
        "case_id": sample_id,
        "task": task,
        "source_task": source_task,
        "event": event,
        "question": question,
        "answer": answer,
    }


def build_named_document(*, event: str, label: str, text: str) -> str:
    if not text:
        return ""
    parts = []
    if event:
        parts.append(f"事件：{event}")
    parts.append(f"{label}：")
    parts.append(text)
    return "\n\n".join(parts).strip()


def coerce_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(item).strip() for item in value if str(item).strip()).strip()
    return str(value).strip()


def infer_doc_type(file_name: str) -> str:
    lowered = file_name.lower()
    if lowered.startswith("news"):
        return "news"
    if lowered == "source.txt":
        return "source"
    return "text"


def extract_doc_index(file_name: str) -> int:
    lowered = file_name.lower()
    for number in (1, 2, 3):
        if f"news{number}" in lowered:
            return number
    return 0


def build_event_id(event: str) -> str:
    normalized = event.strip()
    if not normalized:
        return ""
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def write_text_file(*, path: Path, text: str, overwrite_existing: bool) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    existed_before = path.exists()
    if existed_before and not overwrite_existing:
        return "skipped"
    path.write_text(text, encoding="utf-8")
    return "overwritten" if existed_before else "saved"


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def render_import_summary(
    *,
    kb_name: str,
    data_file: Path,
    export_root: Path,
    manifest_path: Path,
    eval_path: Path,
    imported_samples: int,
    task_sample_counts: dict[str, int],
    saved_files: int,
    overwritten_files: int,
    skipped_files: int,
    eval_case_count: int,
) -> str:
    lines = [
        "CRUD_RAG 官方 split 导入完成",
        f"知识库名称: {kb_name}",
        f"数据文件: {data_file}",
        f"导出目录: {export_root}",
        f"导入清单文件: {manifest_path}",
        f"评测样例文件: {eval_path}",
        f"导入样本数: {imported_samples}",
        f"保存文件数: {saved_files}",
        f"覆盖文件数: {overwritten_files}",
        f"跳过文件数: {skipped_files}",
        f"评测样例数: {eval_case_count}",
    ]
    if task_sample_counts:
        lines.append("任务分布:")
        for task_name, count in sorted(task_sample_counts.items()):
            lines.append(f"  - {task_name}: {count}")
    return "\n".join(lines)


def render_rebuild_progress(progress: float, message: str) -> None:
    percent = max(0.0, min(100.0, progress * 100))
    print(f"{percent:6.2f}% {message}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
