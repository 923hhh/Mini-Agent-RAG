from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.settings import load_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_crud_retrieval_cases",
        description=(
            "Convert exported CRUD_RAG eval JSONL into retrieval-eval JSONL with "
            "expected_references pointing to the imported knowledge-base files."
        ),
    )
    parser.add_argument(
        "--input-file",
        required=True,
        type=Path,
        help="Source CRUD eval JSONL file.",
    )
    parser.add_argument(
        "--knowledge-base-name",
        required=True,
        help="Knowledge base name that contains the imported CRUD documents.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="CRUD split directory name under the imported knowledge base.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only export the first N cases. 0 means all.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Default retrieval top_k to attach to each case.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output JSONL path. Defaults to data/eval/<kb>_retrieval_cases_<limit or all>.jsonl",
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

    input_path = args.input_file.resolve() if args.input_file.is_absolute() else (project_root / args.input_file).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"输入评测文件不存在: {input_path}")

    content_dir = settings.knowledge_base_content_dir(args.knowledge_base_name)
    if not content_dir.exists():
        raise FileNotFoundError(f"知识库内容目录不存在: {content_dir}")

    output_path = resolve_output_path(
        output=args.output,
        project_root=project_root,
        knowledge_base_name=args.knowledge_base_name,
        limit=args.limit,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(input_path, limit=args.limit)
    cases = build_retrieval_cases(
        rows=rows,
        knowledge_base_name=args.knowledge_base_name,
        content_dir=content_dir,
        split=args.split,
        top_k=args.top_k,
    )
    write_jsonl(output_path, cases)

    summary = {
        "input_file": str(input_path),
        "knowledge_base_name": args.knowledge_base_name,
        "output_file": str(output_path),
        "case_count": len(cases),
        "top_k": args.top_k,
        "split": args.split,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def resolve_output_path(
    *,
    output: str,
    project_root: Path,
    knowledge_base_name: str,
    limit: int,
) -> Path:
    if output.strip():
        raw_path = Path(output)
        return raw_path if raw_path.is_absolute() else (project_root / raw_path).resolve()
    suffix = str(limit) if limit > 0 else "all"
    return project_root / "data" / "eval" / f"{knowledge_base_name}_retrieval_cases_{suffix}.jsonl"


def load_jsonl(path: Path, *, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            continue
        rows.append(payload)
        if limit > 0 and len(rows) >= limit:
            break
    return rows


def build_retrieval_cases(
    *,
    rows: list[dict[str, Any]],
    knowledge_base_name: str,
    content_dir: Path,
    split: str,
    top_k: int,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        case_id = str(row.get("case_id", "")).strip() or f"crud-{index:06d}"
        task = str(row.get("task", "")).strip() or "quest_answer"
        event = str(row.get("event", "")).strip()
        question = str(row.get("question", "")).strip()
        query = build_query(event=event, question=question, task=task)
        if not query:
            continue

        sample_dir = content_dir / "crud_rag_3qa" / split / case_id
        expected_references = build_expected_references(sample_dir)
        if not expected_references:
            continue

        cases.append(
            {
                "case_id": case_id,
                "category": f"crud_{task}",
                "knowledge_base_name": knowledge_base_name,
                "query": query,
                "top_k": top_k,
                "expected_references": expected_references,
            }
        )
    return cases


def build_query(*, event: str, question: str, task: str) -> str:
    normalized_task = task.strip().lower()
    if normalized_task == "summary":
        return event
    return f"{event}\n{question}".strip() if event else question


def build_expected_references(sample_dir: Path) -> list[dict[str, str]]:
    expected_references: list[dict[str, str]] = []
    for news_index in range(1, 4):
        file_path = sample_dir / f"news{news_index}.txt"
        if not file_path.exists():
            continue
        content = file_path.read_text(encoding="utf-8").strip()
        expected_references.append(
            {
                "source": file_path.name,
                "source_path": str(file_path.resolve()),
                "title": file_path.stem,
                "reference_id": file_path.stem,
                "content": content,
            }
        )
    return expected_references


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
