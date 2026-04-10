from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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


JSON_TEXT_KEYS = (
    "content",
    "text",
    "document",
    "doc",
    "body",
    "article",
    "passage",
    "news",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="import_crud_rag_kb",
        description="Import the CRUD_RAG retrieval corpus into the current local knowledge base.",
    )
    parser.add_argument("--kb-name", required=True, help="Target knowledge base name.")
    parser.add_argument(
        "--crud-rag-root",
        required=True,
        type=Path,
        help="Path to the local CRUD_RAG repository root.",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=None,
        help="Optional explicit path to the retrieval document directory. Defaults to CRUD_RAG/data/80000_docs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Import at most N source files. 0 means all.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite existing files in the target knowledge base content directory.",
    )
    parser.add_argument(
        "--auto-rebuild",
        action="store_true",
        help="Rebuild the knowledge base after import completes.",
    )
    parser.add_argument(
        "--force-full-rebuild",
        action="store_true",
        help="Force a full rebuild after import. Only works together with --auto-rebuild.",
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
    crud_rag_root = args.crud_rag_root.resolve()
    settings = load_settings(project_root)

    docs_dir = resolve_docs_dir(crud_rag_root, args.docs_dir)
    source_files = discover_source_files(docs_dir, limit=args.limit)
    if not source_files:
        raise SystemExit(f"未在 {docs_dir} 下发现可导入文件。")

    content_dir, _ = ensure_knowledge_base_layout(settings, args.kb_name)
    summary = import_source_files(
        source_files=source_files,
        docs_dir=docs_dir,
        content_dir=content_dir,
        supported_extensions={ext.lower() for ext in settings.kb.SUPPORTED_EXTENSIONS},
        overwrite_existing=args.overwrite_existing,
    )
    manifest_path = write_import_manifest(
        settings=settings,
        knowledge_base_name=args.kb_name,
        crud_rag_root=crud_rag_root,
        docs_dir=docs_dir,
        summary=summary,
    )

    print(render_import_summary(args.kb_name, docs_dir, content_dir, manifest_path, summary))

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


def resolve_docs_dir(crud_rag_root: Path, docs_dir: Path | None) -> Path:
    if docs_dir is not None:
        resolved = docs_dir.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"指定的 docs 目录不存在: {resolved}")
        return resolved

    candidates = [
        crud_rag_root / "data" / "80000_docs",
        crud_rag_root / "80000_docs",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"未在 {crud_rag_root} 下找到 80000_docs。请用 --docs-dir 显式指定语料目录。"
    )


def discover_source_files(docs_dir: Path, *, limit: int) -> list[Path]:
    files = [path for path in docs_dir.rglob("*") if path.is_file()]
    files.sort(key=lambda item: item.relative_to(docs_dir).as_posix().lower())
    if limit > 0:
        return files[:limit]
    return files


def import_source_files(
    *,
    source_files: list[Path],
    docs_dir: Path,
    content_dir: Path,
    supported_extensions: set[str],
    overwrite_existing: bool,
) -> dict[str, Any]:
    copied_files: list[str] = []
    converted_files: list[str] = []
    skipped_files: list[str] = []
    overwritten_files: list[str] = []
    unsupported_files: list[str] = []
    exported_records = 0

    for source_path in source_files:
        relative_path = source_path.relative_to(docs_dir)
        suffix = source_path.suffix.lower()

        if suffix in supported_extensions:
            target_path = content_dir / relative_path
            status = copy_file(
                source_path=source_path,
                target_path=target_path,
                overwrite_existing=overwrite_existing,
            )
            register_file_status(
                status=status,
                target_path=target_path.relative_to(content_dir).as_posix(),
                copied_files=copied_files,
                overwritten_files=overwritten_files,
                skipped_files=skipped_files,
            )
            continue

        if suffix in {".json", ".jsonl"}:
            record_count, status_items = export_json_container_file(
                source_path=source_path,
                relative_path=relative_path,
                content_dir=content_dir,
                overwrite_existing=overwrite_existing,
            )
            exported_records += record_count
            if record_count > 0:
                converted_files.append(relative_path.as_posix())
            for status, path_text in status_items:
                if status == "saved":
                    copied_files.append(path_text)
                elif status == "overwritten":
                    overwritten_files.append(path_text)
                else:
                    skipped_files.append(path_text)
            continue

        if suffix == "":
            normalized_target = (content_dir / relative_path).with_suffix(".txt")
            status = copy_file(
                source_path=source_path,
                target_path=normalized_target,
                overwrite_existing=overwrite_existing,
            )
            register_file_status(
                status=status,
                target_path=normalized_target.relative_to(content_dir).as_posix(),
                copied_files=copied_files,
                overwritten_files=overwritten_files,
                skipped_files=skipped_files,
            )
            continue

        unsupported_files.append(relative_path.as_posix())

    return {
        "source_file_count": len(source_files),
        "saved_count": len(copied_files),
        "overwritten_count": len(overwritten_files),
        "skipped_count": len(skipped_files),
        "unsupported_count": len(unsupported_files),
        "converted_container_count": len(converted_files),
        "exported_record_count": exported_records,
        "saved_files": copied_files,
        "overwritten_files": overwritten_files,
        "skipped_files": skipped_files,
        "unsupported_files": unsupported_files,
        "converted_files": converted_files,
    }


def copy_file(*, source_path: Path, target_path: Path, overwrite_existing: bool) -> str:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and not overwrite_existing:
        return "skipped"
    if target_path.exists():
        shutil.copy2(source_path, target_path)
        return "overwritten"
    shutil.copy2(source_path, target_path)
    return "saved"


def register_file_status(
    *,
    status: str,
    target_path: str,
    copied_files: list[str],
    overwritten_files: list[str],
    skipped_files: list[str],
) -> None:
    if status == "saved":
        copied_files.append(target_path)
    elif status == "overwritten":
        overwritten_files.append(target_path)
    else:
        skipped_files.append(target_path)


def export_json_container_file(
    *,
    source_path: Path,
    relative_path: Path,
    content_dir: Path,
    overwrite_existing: bool,
) -> tuple[int, list[tuple[str, str]]]:
    items = load_json_container(source_path)
    if not items:
        return 0, []

    output_dir = content_dir / relative_path.parent / relative_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    record_count = 0
    status_items: list[tuple[str, str]] = []
    for index, item in enumerate(items, start=1):
        text = extract_text_from_container_item(item)
        if not text:
            continue
        file_name = build_container_record_name(item, index=index)
        target_path = output_dir / file_name
        status = write_text_file(
            target_path=target_path,
            text=text,
            overwrite_existing=overwrite_existing,
        )
        record_count += 1
        status_items.append((status, target_path.relative_to(content_dir).as_posix()))
    return record_count, status_items


def load_json_container(path: Path) -> list[Any]:
    if path.suffix.lower() == ".jsonl":
        items: list[Any] = []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            items.append(json.loads(stripped))
        return items

    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        if isinstance(raw.get("data"), list):
            return list(raw["data"])
        if isinstance(raw.get("items"), list):
            return list(raw["items"])
        return [raw]
    return []


def extract_text_from_container_item(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if not isinstance(item, dict):
        return ""

    for key in JSON_TEXT_KEYS:
        value = item.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text

    parts: list[str] = []
    title = str(item.get("title", "")).strip()
    if title:
        parts.append(title)
    for key, value in item.items():
        if key in {"id", "uid", "doc_id", "title"}:
            continue
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                parts.append(f"{key}: {stripped}")
    return "\n\n".join(parts).strip()


def build_container_record_name(item: Any, *, index: int) -> str:
    if isinstance(item, dict):
        for key in ("id", "uid", "doc_id", "title"):
            value = str(item.get(key, "")).strip()
            if value:
                slug = slugify_filename(value)
                if slug:
                    return f"{slug}.txt"
    return f"{index:06d}.txt"


def slugify_filename(value: str) -> str:
    safe_chars: list[str] = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            safe_chars.append(char)
        elif char in {" ", "\t"}:
            safe_chars.append("_")
    slug = "".join(safe_chars).strip("._")
    if len(slug) > 80:
        slug = slug[:80].rstrip("._")
    if slug:
        return slug
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()[:12]
    return f"doc_{digest}"


def write_text_file(*, target_path: Path, text: str, overwrite_existing: bool) -> str:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    existed_before = target_path.exists()
    if existed_before and not overwrite_existing:
        return "skipped"
    target_path.write_text(text, encoding="utf-8")
    return "overwritten" if existed_before else "saved"


def write_import_manifest(
    *,
    settings,
    knowledge_base_name: str,
    crud_rag_root: Path,
    docs_dir: Path,
    summary: dict[str, Any],
) -> Path:
    manifest_path = settings.knowledge_base_dir(knowledge_base_name) / "crud_rag_import_manifest.json"
    payload = {
        "source": "CRUD_RAG",
        "crud_rag_root": str(crud_rag_root),
        "docs_dir": str(docs_dir),
        **summary,
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def render_import_summary(
    knowledge_base_name: str,
    docs_dir: Path,
    content_dir: Path,
    manifest_path: Path,
    summary: dict[str, Any],
) -> str:
    lines = [
        "CRUD_RAG 语料导入完成",
        f"知识库名称: {knowledge_base_name}",
        f"源语料目录: {docs_dir}",
        f"目标内容目录: {content_dir}",
        f"导入清单文件: {manifest_path}",
        f"扫描源文件数: {summary['source_file_count']}",
        f"新增文件数: {summary['saved_count']}",
        f"覆盖文件数: {summary['overwritten_count']}",
        f"跳过文件数: {summary['skipped_count']}",
        f"容器文件展开数: {summary['converted_container_count']}",
        f"容器导出记录数: {summary['exported_record_count']}",
        f"不支持文件数: {summary['unsupported_count']}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
