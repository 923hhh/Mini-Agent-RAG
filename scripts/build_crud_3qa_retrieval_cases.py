"""Build CRUD 3QA retrieval cases with expected_references.

This script converts the raw CRUD 3QA QA cases into retrieval-eval cases by
attaching the three source news files as `expected_references`.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "eval" / "crud_rag_3qa_full_crud_rag_3qa_train.jsonl"
DEFAULT_KB_ROOT = (
    PROJECT_ROOT
    / "data"
    / "knowledge_base"
    / "crud_rag_3qa_full"
    / "content"
    / "crud_rag_3qa"
    / "train"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "eval" / "crud_rag_3qa_full_retrieval_cases_1000.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Raw CRUD 3QA QA jsonl file.",
    )
    parser.add_argument(
        "--kb-root",
        type=Path,
        default=DEFAULT_KB_ROOT,
        help="Root directory containing case_id/news1.txt~news3.txt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output retrieval cases jsonl path.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=1000,
        help="How many cases to sample. <=0 means all cases.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260428,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k to store in each retrieval case.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def build_query(row: dict) -> str:
    event = str(row.get("event", "") or "").strip()
    question = str(row.get("question", "") or "").strip()
    if event and question:
        return f"{event}\n{question}"
    return question or event


def build_expected_references(kb_root: Path, case_id: str) -> list[dict]:
    expected_references: list[dict] = []
    for name in ("news1", "news2", "news3"):
        file_name = f"{name}.txt"
        source_path = kb_root / case_id / file_name
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source file for case {case_id}: {source_path}")
        expected_references.append(
            {
                "source": file_name,
                "source_path": str(source_path.resolve()),
                "title": name,
                "reference_id": name,
                "content": source_path.read_text(encoding="utf-8"),
            }
        )
    return expected_references


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.input.resolve())
    if not rows:
        raise RuntimeError("No CRUD 3QA rows loaded.")

    selected = list(rows)
    if args.sample_count > 0:
        if args.sample_count > len(rows):
            raise ValueError(
                f"sample-count={args.sample_count} exceeds total rows={len(rows)}"
            )
        rng = random.Random(args.seed)
        selected = rng.sample(rows, args.sample_count)

    payload_lines: list[str] = []
    kb_root = args.kb_root.resolve()
    for row in selected:
        case_id = str(row.get("case_id", "") or "").strip()
        if not case_id:
            raise ValueError("Encountered row without case_id.")
        item = {
            "case_id": case_id,
            "category": "crud_quest_answer",
            "knowledge_base_name": "crud_rag_3qa_full",
            "query": build_query(row),
            "top_k": args.top_k,
            "expected_references": build_expected_references(kb_root, case_id),
        }
        payload_lines.append(json.dumps(item, ensure_ascii=False))

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(payload_lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "input": str(args.input.resolve()),
                "output": str(output_path),
                "sample_count": len(selected),
                "seed": args.seed,
                "kb_root": str(kb_root),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
