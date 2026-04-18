from __future__ import annotations

import argparse
import hashlib
import json
import re
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


DEFAULT_TASKS = (
    "extractive_qa",
    "conversation_qa",
    "multi-doc_qa",
    "time-sensitive_qa",
    "structured_qa",
)

TASK_CONFIGS: dict[str, dict[str, Any]] = {
    "extractive_qa": {
        "raw": Path("BCM/labeled_data/extractive_qa/basic_qa.jsonl"),
        "retrieved": [
            Path("BCM/labeled_data/extractive_qa/basic_qa_retrieved_dense.jsonl"),
            Path("BCM/labeled_data/extractive_qa/basic_qa_retrieved_bm25.jsonl"),
        ],
    },
    "conversation_qa": {
        "raw": Path("BCM/labeled_data/conversation_qa/conversation_qa.jsonl"),
        "retrieved": [
            Path("BCM/labeled_data/conversation_qa/conversation_qa_retrieved_dense.jsonl"),
            Path("BCM/labeled_data/conversation_qa/conversation_qa_retrieved_bm25.jsonl"),
        ],
    },
    "multi-doc_qa": {
        "raw": Path("BCM/labeled_data/multi-doc_qa/multidoc_qa.jsonl"),
        "retrieved": [
            Path("BCM/labeled_data/multi-doc_qa/multidoc_qa_retrieved_dense.jsonl"),
            Path("BCM/labeled_data/multi-doc_qa/multidoc_qa_retrieved_bm25.jsonl"),
        ],
    },
    "time-sensitive_qa": {
        "raw": Path("BCM/labeled_data/time-sensitive_qa/time_sensitive.jsonl"),
        "retrieved": [
            Path("BCM/labeled_data/time-sensitive_qa/time_sensitive_retrieved_dense.jsonl"),
            Path("BCM/labeled_data/time-sensitive_qa/time_sensitive_retrieved_bm25.jsonl"),
        ],
    },
    "structured_qa": {
        "raw": Path("BCM/labeled_data/structured_qa/structured_qa_twopositive.jsonl"),
        "retrieved": [],
    },
    "faithful_qa": {
        "raw": Path("BCM/labeled_data/faithful_qa/faithful_qa.jsonl"),
        "retrieved": [],
    },
    "noisy_qa": {
        "raw": Path("BCM/labeled_data/noisy_qa/noisy_qa_ver3.jsonl"),
        "retrieved": [],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="import_domainrag_kb",
        description=(
            "Import DomainRAG into the current local knowledge base format. "
            "If the official corpus directory is unavailable, the script can bootstrap "
            "a proxy corpus from labeled positive references and retrieved passages."
        ),
    )
    parser.add_argument("--kb-name", required=True, help="Target knowledge base name.")
    parser.add_argument(
        "--domainrag-root",
        required=True,
        type=Path,
        help="Path to the local DomainRAG repository root.",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help="Optional explicit path to the official DomainRAG corpus directory.",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Task name to include. Can be repeated. Defaults to a retrieval-focused subset.",
    )
    parser.add_argument(
        "--max-cases-per-task",
        type=int,
        default=0,
        help="Only export the first N cases from each selected task. 0 means all available cases.",
    )
    parser.add_argument(
        "--include-retrieved-psgs",
        action="store_true",
        help="When bootstrapping from labeled data, also import retrieved passages as distractor documents.",
    )
    parser.add_argument(
        "--retrieved-psgs-per-case",
        type=int,
        default=3,
        help="How many retrieved passages to keep from each retrieved JSONL file when bootstrapping.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite existing exported files in the target knowledge base content directory.",
    )
    parser.add_argument(
        "--auto-rebuild",
        action="store_true",
        help="Rebuild the knowledge base after export completes.",
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
    domainrag_root = args.domainrag_root.resolve()
    settings = load_settings(project_root)
    content_dir, _ = ensure_knowledge_base_layout(settings, args.kb_name)

    selected_tasks = resolve_tasks(args.task)
    task_payloads = load_task_payloads(
        domainrag_root=domainrag_root,
        task_names=selected_tasks,
        max_cases_per_task=args.max_cases_per_task,
    )
    if not task_payloads:
        raise SystemExit("未读取到任何 DomainRAG 样本。")

    corpus_dir = resolve_corpus_dir(domainrag_root, args.corpus_dir)
    if corpus_dir is not None:
        export_summary = export_official_corpus(
            corpus_dir=corpus_dir,
            content_dir=content_dir,
            overwrite_existing=args.overwrite_existing,
        )
        source_mode = "official_corpus"
        source_root = corpus_dir
    else:
        export_summary = export_proxy_corpus(
            task_payloads=task_payloads,
            content_dir=content_dir,
            overwrite_existing=args.overwrite_existing,
            include_retrieved_psgs=args.include_retrieved_psgs,
            retrieved_psgs_per_case=args.retrieved_psgs_per_case,
        )
        source_mode = "proxy_from_labeled_data"
        source_root = domainrag_root / "BCM" / "labeled_data"

    eval_path = write_eval_subset(
        project_root=project_root,
        kb_name=args.kb_name,
        task_payloads=task_payloads,
    )
    metadata_path = write_metadata_map(
        content_dir=content_dir,
        source_mode=source_mode,
        export_summary=export_summary,
    )
    manifest_path = write_import_manifest(
        settings=settings,
        knowledge_base_name=args.kb_name,
        domainrag_root=domainrag_root,
        source_root=source_root,
        source_mode=source_mode,
        selected_tasks=selected_tasks,
        export_summary=export_summary,
        eval_path=eval_path,
        metadata_path=metadata_path,
        max_cases_per_task=args.max_cases_per_task,
        include_retrieved_psgs=args.include_retrieved_psgs,
        retrieved_psgs_per_case=args.retrieved_psgs_per_case,
    )

    print(
        render_import_summary(
            kb_name=args.kb_name,
            source_mode=source_mode,
            source_root=source_root,
            manifest_path=manifest_path,
            eval_path=eval_path,
            metadata_path=metadata_path,
            selected_tasks=selected_tasks,
            export_summary=export_summary,
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


def resolve_tasks(raw_tasks: list[str]) -> list[str]:
    if not raw_tasks:
        return list(DEFAULT_TASKS)

    selected: list[str] = []
    for raw_task in raw_tasks:
        task = str(raw_task).strip()
        if not task:
            continue
        if task not in TASK_CONFIGS:
            supported = ", ".join(sorted(TASK_CONFIGS))
            raise ValueError(f"不支持的 task: {task}。可选值: {supported}")
        if task not in selected:
            selected.append(task)
    if not selected:
        raise ValueError("至少需要一个有效 task。")
    return selected


def resolve_corpus_dir(domainrag_root: Path, corpus_dir: Path | None) -> Path | None:
    candidates: list[Path] = []
    if corpus_dir is not None:
        candidates.append(corpus_dir.resolve())
    candidates.append((domainrag_root / "corpus").resolve())

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def load_task_payloads(
    *,
    domainrag_root: Path,
    task_names: list[str],
    max_cases_per_task: int,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for task_name in task_names:
        task_config = TASK_CONFIGS[task_name]
        raw_path = (domainrag_root / task_config["raw"]).resolve()
        raw_cases = load_jsonl(raw_path, limit=max_cases_per_task)
        if not raw_cases:
            continue

        retrieved_cases: list[dict[str, list[dict[str, Any]]]] = []
        for retrieved_relative in task_config["retrieved"]:
            retrieved_path = (domainrag_root / retrieved_relative).resolve()
            if not retrieved_path.exists():
                continue
            retrieved_rows = load_jsonl(retrieved_path, limit=len(raw_cases))
            retrieved_cases.append(
                {
                    "path": str(retrieved_path),
                    "rows": retrieved_rows,
                }
            )

        payloads.append(
            {
                "task_name": task_name,
                "raw_path": raw_path,
                "raw_cases": raw_cases,
                "retrieved_cases": retrieved_cases,
            }
        )
    return payloads


def load_jsonl(path: Path, *, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        rows.append(json.loads(stripped))
        if limit > 0 and len(rows) >= limit:
            break
    return rows


def export_official_corpus(
    *,
    corpus_dir: Path,
    content_dir: Path,
    overwrite_existing: bool,
) -> dict[str, Any]:
    export_root = content_dir / "domainrag_corpus"
    json_files = sorted(path for path in corpus_dir.rglob("*.json") if path.is_file())
    if not json_files:
        raise FileNotFoundError(f"未在 {corpus_dir} 下发现 DomainRAG corpus json 文件。")

    saved_files = 0
    overwritten_files = 0
    skipped_files = 0
    file_records: list[dict[str, str]] = []

    for source_path in json_files:
        doc = json.loads(source_path.read_text(encoding="utf-8"))
        relative_path = build_official_doc_relative_path(doc)
        target_path = export_root / relative_path
        text = render_official_doc_text(doc)
        status = write_text_file(
            path=target_path,
            text=text,
            overwrite_existing=overwrite_existing,
        )
        if status == "saved":
            saved_files += 1
        elif status == "overwritten":
            overwritten_files += 1
        else:
            skipped_files += 1
        file_records.append(
            {
                "relative_path": target_path.relative_to(content_dir).as_posix(),
                "source": str(source_path),
                "doc_id": str(doc.get("id", "")).strip(),
                "title": str(doc.get("title", "")).strip(),
                "url": str(doc.get("url", "")).strip(),
            }
        )

    return {
        "document_count": len(json_files),
        "saved_count": saved_files,
        "overwritten_count": overwritten_files,
        "skipped_count": skipped_files,
        "task_case_counts": {},
        "file_records": file_records,
    }


def export_proxy_corpus(
    *,
    task_payloads: list[dict[str, Any]],
    content_dir: Path,
    overwrite_existing: bool,
    include_retrieved_psgs: bool,
    retrieved_psgs_per_case: int,
) -> dict[str, Any]:
    export_root = content_dir / "domainrag_proxy"
    saved_files = 0
    overwritten_files = 0
    skipped_files = 0
    file_records: list[dict[str, str]] = []
    task_case_counts: dict[str, int] = {}
    seen_keys: set[str] = set()

    for payload in task_payloads:
        task_name = str(payload["task_name"])
        raw_cases = list(payload["raw_cases"])
        task_case_counts[task_name] = len(raw_cases)

        for case_index, raw_case in enumerate(raw_cases, start=1):
            case_id = resolve_case_id(raw_case, case_index=case_index)
            references = extract_case_references(raw_case)
            if include_retrieved_psgs:
                references.extend(
                    extract_retrieved_references(
                        retrieved_payloads=payload["retrieved_cases"],
                        case_index=case_index - 1,
                        retrieved_psgs_per_case=retrieved_psgs_per_case,
                    )
                )

            for reference_index, reference in enumerate(references, start=1):
                normalized = normalize_reference(reference)
                key = build_reference_key(normalized)
                if not key or key in seen_keys:
                    continue
                seen_keys.add(key)

                relative_path = build_proxy_relative_path(
                    task_name=task_name,
                    case_id=case_id,
                    reference=normalized,
                    reference_index=reference_index,
                )
                target_path = export_root / relative_path
                text = render_reference_text(normalized)
                status = write_text_file(
                    path=target_path,
                    text=text,
                    overwrite_existing=overwrite_existing,
                )
                if status == "saved":
                    saved_files += 1
                elif status == "overwritten":
                    overwritten_files += 1
                else:
                    skipped_files += 1
                file_records.append(
                    {
                        "relative_path": target_path.relative_to(content_dir).as_posix(),
                        "task_name": task_name,
                        "case_id": case_id,
                        "reference_id": normalized.get("id", ""),
                        "title": normalized.get("title", ""),
                        "url": normalized.get("url", ""),
                        "date": normalized.get("date", ""),
                    }
                )

    return {
        "document_count": len(file_records),
        "saved_count": saved_files,
        "overwritten_count": overwritten_files,
        "skipped_count": skipped_files,
        "task_case_counts": task_case_counts,
        "file_records": file_records,
    }


def extract_case_references(raw_case: dict[str, Any]) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    references.extend(coerce_reference_list(raw_case.get("positive_reference")))
    references.extend(coerce_reference_list(raw_case.get("positive_references")))

    history_qa = raw_case.get("history_qa")
    if isinstance(history_qa, list):
        for item in history_qa:
            if not isinstance(item, dict):
                continue
            references.extend(coerce_reference_list(item.get("positive_reference")))
    return references


def extract_retrieved_references(
    *,
    retrieved_payloads: list[dict[str, Any]],
    case_index: int,
    retrieved_psgs_per_case: int,
) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    limit = max(0, retrieved_psgs_per_case)
    if limit == 0:
        return references

    for payload in retrieved_payloads:
        rows = list(payload["rows"])
        if case_index >= len(rows):
            continue
        row = rows[case_index]
        retrieved_psgs = row.get("retrieved_psgs")
        if not isinstance(retrieved_psgs, list):
            continue
        for item in retrieved_psgs[:limit]:
            if isinstance(item, dict):
                references.append(item)
    return references


def coerce_reference_list(value: object) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return [value]
    return []


def normalize_reference(reference: dict[str, Any]) -> dict[str, str]:
    title = str(reference.get("title", "")).strip()
    url = str(reference.get("url", "")).strip()
    date = str(reference.get("date", "")).strip()
    reference_id = str(reference.get("id", "")).strip()
    psg_id = str(reference.get("psg_id", "")).strip()
    contents = str(reference.get("contents", reference.get("content", ""))).strip()
    return {
        "title": title,
        "url": url,
        "date": date,
        "id": reference_id,
        "psg_id": psg_id,
        "contents": contents,
    }


def build_reference_key(reference: dict[str, str]) -> str:
    parts = [
        reference.get("id", "").strip(),
        reference.get("psg_id", "").strip(),
        normalize_text(reference.get("url", "")),
        normalize_text(reference.get("title", "")),
        normalize_text(reference.get("contents", "")),
    ]
    merged = "\n".join(part for part in parts if part)
    if not merged:
        return ""
    return hashlib.sha1(merged.encode("utf-8")).hexdigest()


def render_reference_text(reference: dict[str, str]) -> str:
    parts = []
    if reference.get("title"):
        parts.append(f"标题：{reference['title']}")
    if reference.get("url"):
        parts.append(f"来源链接：{reference['url']}")
    if reference.get("date"):
        parts.append(f"日期：{reference['date']}")
    if reference.get("id"):
        parts.append(f"文档ID：{reference['id']}")
    if reference.get("psg_id"):
        parts.append(f"段落ID：{reference['psg_id']}")
    parts.append("正文：")
    parts.append(reference.get("contents", "").strip())
    return "\n\n".join(part for part in parts if part.strip()).strip() + "\n"


def build_proxy_relative_path(
    *,
    task_name: str,
    case_id: str,
    reference: dict[str, str],
    reference_index: int,
) -> Path:
    prefix = slugify(reference.get("title", "")) or "untitled"
    reference_id = reference.get("id", "").strip() or "noid"
    passage_id = reference.get("psg_id", "").strip() or f"ref{reference_index}"
    digest = build_reference_key(reference)[:12]
    file_name = f"{prefix}__doc-{reference_id}__psg-{passage_id}__{digest}.txt"
    return Path(task_name) / sanitize_segment(case_id) / file_name


def build_official_doc_relative_path(doc: dict[str, Any]) -> Path:
    title = slugify(str(doc.get("title", "")).strip()) or "untitled"
    doc_id = sanitize_segment(str(doc.get("id", "")).strip() or "noid")
    digest = hashlib.sha1(json.dumps(doc, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    file_name = f"{title}__doc-{doc_id}__{digest}.txt"
    return Path(file_name)


def render_official_doc_text(doc: dict[str, Any]) -> str:
    parts = []
    title = str(doc.get("title", "")).strip()
    url = str(doc.get("url", "")).strip()
    doc_id = str(doc.get("id", "")).strip()
    date = str(doc.get("date", "")).strip()
    if title:
        parts.append(f"标题：{title}")
    if url:
        parts.append(f"来源链接：{url}")
    if date:
        parts.append(f"日期：{date}")
    if doc_id:
        parts.append(f"文档ID：{doc_id}")

    passages = doc.get("passages")
    if isinstance(passages, list):
        rendered_passages = []
        for index, passage in enumerate(passages, start=1):
            text = str(passage).strip()
            if not text:
                continue
            rendered_passages.append(f"段落 {index}：\n{text}")
        if rendered_passages:
            parts.append("\n\n".join(rendered_passages))
    else:
        content = str(doc.get("contents", "")).strip()
        if content:
            parts.append(content)

    return "\n\n".join(part for part in parts if part.strip()).strip() + "\n"


def write_text_file(*, path: Path, text: str, overwrite_existing: bool) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite_existing:
        return "skipped"
    status = "overwritten" if path.exists() else "saved"
    path.write_text(text, encoding="utf-8")
    return status


def write_eval_subset(
    *,
    project_root: Path,
    kb_name: str,
    task_payloads: list[dict[str, Any]],
) -> Path:
    eval_dir = project_root / "data" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_path = eval_dir / f"{kb_name}_domainrag_small_batch.jsonl"

    with eval_path.open("w", encoding="utf-8") as handle:
        for payload in task_payloads:
            task_name = str(payload["task_name"])
            for index, raw_case in enumerate(payload["raw_cases"], start=1):
                enriched = dict(raw_case)
                enriched["knowledge_base_name"] = kb_name
                enriched["domainrag_task"] = task_name
                enriched.setdefault("case_id", f"{task_name}-{index}")
                handle.write(json.dumps(enriched, ensure_ascii=False) + "\n")
    return eval_path


def write_metadata_map(
    *,
    content_dir: Path,
    source_mode: str,
    export_summary: dict[str, Any],
) -> Path:
    metadata_path = content_dir / ".rag_file_metadata.json"
    files_payload: dict[str, dict[str, str]] = {}
    for record in export_summary.get("file_records", []):
        relative_path = str(record.get("relative_path", "")).strip()
        if not relative_path:
            continue
        files_payload[relative_path] = {
            "source": "DomainRAG",
            "source_mode": source_mode,
            "domainrag_task": str(record.get("task_name", "")).strip(),
            "case_id": str(record.get("case_id", "")).strip(),
            "reference_id": str(record.get("reference_id", "")).strip(),
            "url": str(record.get("url", "")).strip(),
            "title": str(record.get("title", "")).strip(),
            "date": str(record.get("date", "")).strip(),
        }

    payload = {
        "source": "DomainRAG",
        "source_mode": source_mode,
        "files": files_payload,
    }
    metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata_path


def write_import_manifest(
    *,
    settings: Any,
    knowledge_base_name: str,
    domainrag_root: Path,
    source_root: Path,
    source_mode: str,
    selected_tasks: list[str],
    export_summary: dict[str, Any],
    eval_path: Path,
    metadata_path: Path,
    max_cases_per_task: int,
    include_retrieved_psgs: bool,
    retrieved_psgs_per_case: int,
) -> Path:
    manifest_path = settings.knowledge_base_dir(knowledge_base_name) / "domainrag_import_manifest.json"
    manifest_payload = {
        "source": "DomainRAG",
        "domainrag_root": str(domainrag_root),
        "source_root": str(source_root),
        "source_mode": source_mode,
        "selected_tasks": selected_tasks,
        "max_cases_per_task": max_cases_per_task,
        "include_retrieved_psgs": include_retrieved_psgs,
        "retrieved_psgs_per_case": retrieved_psgs_per_case,
        "document_count": export_summary.get("document_count", 0),
        "saved_count": export_summary.get("saved_count", 0),
        "overwritten_count": export_summary.get("overwritten_count", 0),
        "skipped_count": export_summary.get("skipped_count", 0),
        "task_case_counts": export_summary.get("task_case_counts", {}),
        "eval_path": str(eval_path),
        "metadata_path": str(metadata_path),
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def render_import_summary(
    *,
    kb_name: str,
    source_mode: str,
    source_root: Path,
    manifest_path: Path,
    eval_path: Path,
    metadata_path: Path,
    selected_tasks: list[str],
    export_summary: dict[str, Any],
) -> str:
    lines = [
        "DomainRAG 导入完成",
        f"知识库名称: {kb_name}",
        f"导入模式: {source_mode}",
        f"来源目录: {source_root}",
        f"任务列表: {', '.join(selected_tasks)}",
        f"导出文档数: {export_summary.get('document_count', 0)}",
        f"新写入文件: {export_summary.get('saved_count', 0)}",
        f"覆盖文件: {export_summary.get('overwritten_count', 0)}",
        f"跳过文件: {export_summary.get('skipped_count', 0)}",
        f"评测子集: {eval_path}",
        f"元数据映射: {metadata_path}",
        f"导入清单: {manifest_path}",
    ]
    task_case_counts = export_summary.get("task_case_counts", {})
    if task_case_counts:
        lines.append("任务样本数:")
        for task_name in sorted(task_case_counts):
            lines.append(f"  - {task_name}: {task_case_counts[task_name]}")
    return "\n".join(lines)


def resolve_case_id(raw_case: dict[str, Any], *, case_index: int) -> str:
    raw_id = raw_case.get("case_id", raw_case.get("id", ""))
    case_id = str(raw_id).strip()
    if case_id:
        return sanitize_segment(case_id)
    return f"case_{case_index:04d}"


def normalize_text(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def sanitize_segment(value: str) -> str:
    cleaned = re.sub(r"[<>:\"/\\\\|?*]+", "_", value.strip())
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = cleaned.strip("._")
    return cleaned[:120] or "item"


def slugify(value: str) -> str:
    cleaned = normalize_text(value)
    if not cleaned:
        return ""
    cleaned = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "_", cleaned)
    cleaned = cleaned.strip("_")
    return cleaned[:80]


if __name__ == "__main__":
    raise SystemExit(main())
