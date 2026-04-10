from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.kb_ingestion_service import (
    ensure_knowledge_base_layout,
    rebuild_knowledge_base,
    render_rebuild_summary,
)
from app.services.settings import load_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="import_crud_rag_3qa",
        description="Download Hugging Face CRUD_RAG_3QA, export it into a local knowledge base, and generate evaluation cases.",
    )
    parser.add_argument("--kb-name", required=True, help="Target knowledge base name.")
    parser.add_argument(
        "--dataset-name",
        default="AndrewTsai0406/CRUD_RAG_3QA",
        help="Hugging Face dataset name.",
    )
    parser.add_argument("--split", default="train", help="Dataset split.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Import at most N rows. 0 means all rows in the split.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite existing exported news files.",
    )
    parser.add_argument(
        "--include-thoughts-doc",
        action="store_true",
        help="Also export the annotator thoughts field into the knowledge base. Disabled by default because it can leak reasoning hints.",
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
    rows = load_dataset_rows(dataset_name=args.dataset_name, split=args.split, limit=args.limit)
    if not rows:
        raise SystemExit("未读取到任何 CRUD_RAG_3QA 样本。")

    content_dir, _ = ensure_knowledge_base_layout(settings, args.kb_name)
    export_root = content_dir / "crud_rag_3qa" / args.split
    export_root.mkdir(parents=True, exist_ok=True)
    file_metadata_map: dict[str, dict[str, str]] = {}

    saved_files = 0
    overwritten_files = 0
    skipped_files = 0
    eval_cases: list[dict[str, str]] = []

    for row in rows:
        sample_id = str(row.get("id", "")).strip() or f"sample_{len(eval_cases) + 1:06d}"
        sample_dir = export_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        event = str(row.get("event", "")).strip()
        event_id = build_event_id(event)
        question = coerce_text(row.get("questions"))
        answer = coerce_text(row.get("answers"))
        news_docs = [
            ("news1.txt", build_news_document(event=event, label="新闻1", text=coerce_text(row.get("news1")))),
            ("news2.txt", build_news_document(event=event, label="新闻2", text=coerce_text(row.get("news2")))),
            ("news3.txt", build_news_document(event=event, label="新闻3", text=coerce_text(row.get("news3")))),
        ]
        thoughts = coerce_text(row.get("thoughts"))
        if args.include_thoughts_doc and thoughts:
            news_docs.append(("thoughts.txt", build_thoughts_document(event=event, thoughts=thoughts)))

        for file_name, text in news_docs:
            if not text.strip():
                continue
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
            news_index = extract_news_index(file_name)
            file_metadata_map[relative_path] = {
                "dataset_name": args.dataset_name,
                "doc_type": "news",
                "sample_id": sample_id,
                "event_id": event_id,
                "group_retrieval_strategy": "same_sample_id",
                "chunk_size_override": "700",
                "news_index": str(news_index),
            }

        if question and answer:
            eval_cases.append(
                {
                    "case_id": sample_id,
                    "task": "quest_answer",
                    "event": event,
                    "question": question,
                    "answer": answer,
                }
            )

    eval_dir = project_root / "data" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_path = eval_dir / f"{args.kb_name}_crud_rag_3qa_{args.split}.jsonl"
    write_jsonl(eval_path, eval_cases)

    metadata_map_path = content_dir / ".rag_file_metadata.json"
    metadata_payload = {
        "source": "CRUD_RAG_3QA",
        "dataset_name": args.dataset_name,
        "split": args.split,
        "files": file_metadata_map,
    }
    metadata_map_path.write_text(
        json.dumps(metadata_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest_path = settings.knowledge_base_dir(args.kb_name) / "crud_rag_3qa_import_manifest.json"
    manifest_payload = {
        "source": "huggingface",
        "dataset_name": args.dataset_name,
        "split": args.split,
        "row_count": len(rows),
        "include_thoughts_doc": args.include_thoughts_doc,
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
            dataset_name=args.dataset_name,
            split=args.split,
            export_root=export_root,
            manifest_path=manifest_path,
            eval_path=eval_path,
            row_count=len(rows),
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
        )
        print()
        print(render_rebuild_summary(rebuild_result))

    return 0


def load_dataset_rows(*, dataset_name: str, split: str, limit: int) -> list[dict[str, object]]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "读取 Hugging Face 数据集需要安装 `datasets`。请先执行 `pip install datasets`。"
        ) from exc

    dataset = load_dataset(dataset_name, split=split)
    rows: list[dict[str, object]] = []
    for index, row in enumerate(dataset):
        rows.append(dict(row))
        if limit > 0 and index + 1 >= limit:
            break
    return rows


def coerce_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(item).strip() for item in value if str(item).strip()).strip()
    return str(value).strip()


def build_news_document(*, event: str, label: str, text: str) -> str:
    if not text:
        return ""
    parts = []
    if event:
        parts.append(f"事件：{event}")
    parts.append(f"{label}：")
    parts.append(text)
    return "\n\n".join(parts).strip()


def build_thoughts_document(*, event: str, thoughts: str) -> str:
    parts = []
    if event:
        parts.append(f"事件：{event}")
    parts.append("新闻整合思路：")
    parts.append(thoughts)
    return "\n\n".join(parts).strip()


def build_event_id(event: str) -> str:
    normalized = event.strip()
    if not normalized:
        return ""
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def extract_news_index(file_name: str) -> int:
    lowered = file_name.lower()
    for number in (1, 2, 3):
        if f"news{number}" in lowered:
            return number
    return 0


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
    dataset_name: str,
    split: str,
    export_root: Path,
    manifest_path: Path,
    eval_path: Path,
    row_count: int,
    saved_files: int,
    overwritten_files: int,
    skipped_files: int,
    eval_case_count: int,
) -> str:
    lines = [
        "CRUD_RAG_3QA 导入完成",
        f"知识库名称: {kb_name}",
        f"数据集: {dataset_name}",
        f"数据切分: {split}",
        f"导出目录: {export_root}",
        f"导入清单文件: {manifest_path}",
        f"评测样例文件: {eval_path}",
        f"样本数: {row_count}",
        f"新增文件数: {saved_files}",
        f"覆盖文件数: {overwritten_files}",
        f"跳过文件数: {skipped_files}",
        f"评测样例数: {eval_case_count}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
