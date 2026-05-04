from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CRUD_EVAL_DIR = PROJECT_ROOT / "data" / "eval" / "crud"
DEFAULT_CASES_DIR = DEFAULT_CRUD_EVAL_DIR / "shards" / "cases"
DEFAULT_CASES_PATTERN = "crud_rag_3qa_full_retrieval_cases_3188_part_*.jsonl"
PART_PATTERN = re.compile(r"_part_(\d+)\.jsonl$", re.IGNORECASE)


@dataclass(frozen=True)
class EvalShard:
    part_no: int
    cases_file: Path
    output_file: Path
    log_file: Path


@dataclass
class RunningShard:
    shard: EvalShard
    process: subprocess.Popen[bytes]
    started_at: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CRUD local KB evaluation shards with fixed-width auto backfill parallelism."
    )
    parser.add_argument(
        "--knowledge-base-name",
        type=str,
        default="crud_rag_3qa_full",
        help="Knowledge base name passed to run_crud_local_kb_eval.py.",
    )
    parser.add_argument(
        "--cases-dir",
        type=Path,
        default=DEFAULT_CASES_DIR,
        help="Directory containing shard jsonl files.",
    )
    parser.add_argument(
        "--cases-pattern",
        type=str,
        default=DEFAULT_CASES_PATTERN,
        help="Glob pattern for shard case files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_CRUD_EVAL_DIR / "shards" / "evals",
        help="Directory for shard eval json outputs.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_CRUD_EVAL_DIR / "shards" / "logs",
        help="Directory for per-shard stdout/stderr logs.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of shard processes to keep running concurrently.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k passed to eval script.")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.35,
        help="Score threshold passed to eval script.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip shards whose output json already exists. Recommended for long runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the shard plan without launching processes.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python executable used to launch child shard jobs.",
    )
    return parser.parse_args()


def extract_part_no(path: Path) -> int:
    match = PART_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"无法从文件名识别分片编号: {path.name}")
    return int(match.group(1))


def build_output_file(output_dir: Path, cases_file: Path) -> Path:
    name = cases_file.name.replace("_retrieval_cases_", "_local_eval_").replace(".jsonl", ".json")
    return output_dir / name


def build_log_file(log_dir: Path, cases_file: Path) -> Path:
    return log_dir / cases_file.name.replace(".jsonl", ".log")


def discover_shards(
    *,
    cases_dir: Path,
    cases_pattern: str,
    output_dir: Path,
    log_dir: Path,
) -> list[EvalShard]:
    shards: list[EvalShard] = []
    for path in sorted(cases_dir.glob(cases_pattern), key=extract_part_no):
        part_no = extract_part_no(path)
        shards.append(
            EvalShard(
                part_no=part_no,
                cases_file=path.resolve(),
                output_file=build_output_file(output_dir.resolve(), path),
                log_file=build_log_file(log_dir.resolve(), path),
            )
        )
    return shards


def launch_shard(
    *,
    shard: EvalShard,
    python_executable: Path,
    knowledge_base_name: str,
    top_k: int,
    score_threshold: float,
) -> RunningShard:
    shard.output_file.parent.mkdir(parents=True, exist_ok=True)
    shard.log_file.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(python_executable),
        str(PROJECT_ROOT / "scripts" / "run_crud_local_kb_eval.py"),
        "--knowledge-base-name",
        knowledge_base_name,
        "--cases-file",
        str(shard.cases_file),
        "--output",
        str(shard.output_file),
        "--top-k",
        str(top_k),
        "--score-threshold",
        str(score_threshold),
    ]
    log_handle = shard.log_file.open("wb")
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    return RunningShard(shard=shard, process=process, started_at=time.time())


def main() -> int:
    args = parse_args()
    if args.max_workers <= 0:
        raise ValueError("--max-workers 必须 >= 1")

    shards = discover_shards(
        cases_dir=args.cases_dir.resolve(),
        cases_pattern=args.cases_pattern,
        output_dir=args.output_dir.resolve(),
        log_dir=args.log_dir.resolve(),
    )
    if not shards:
        raise RuntimeError("没有找到任何分片文件，请检查 --cases-dir 和 --cases-pattern。")

    pending: list[EvalShard] = []
    skipped = 0
    for shard in shards:
        if args.resume and shard.output_file.exists():
            skipped += 1
            continue
        pending.append(shard)

    print(f"[plan] discovered={len(shards)} pending={len(pending)} skipped={skipped}")
    if args.dry_run:
        for shard in pending:
            print(f"[dry-run] part={shard.part_no:02d} cases={shard.cases_file.name} -> {shard.output_file.name}")
        return 0

    running: list[RunningShard] = []
    completed = 0
    failed = 0
    queue = list(pending)

    while queue or running:
        while queue and len(running) < args.max_workers:
            shard = queue.pop(0)
            job = launch_shard(
                shard=shard,
                python_executable=args.python.resolve(),
                knowledge_base_name=args.knowledge_base_name,
                top_k=args.top_k,
                score_threshold=args.score_threshold,
            )
            running.append(job)
            print(
                f"[start] part={shard.part_no:02d} "
                f"running={len(running)}/{args.max_workers} "
                f"log={shard.log_file.name}"
            )

        if not running:
            continue

        time.sleep(2.0)
        still_running: list[RunningShard] = []
        for job in running:
            return_code = job.process.poll()
            if return_code is None:
                still_running.append(job)
                continue
            elapsed = round(time.time() - job.started_at, 1)
            if return_code == 0:
                completed += 1
                print(
                    f"[done] part={job.shard.part_no:02d} "
                    f"elapsed={elapsed}s "
                    f"output={job.shard.output_file.name}"
                )
            else:
                failed += 1
                print(
                    f"[fail] part={job.shard.part_no:02d} "
                    f"exit={return_code} "
                    f"elapsed={elapsed}s "
                    f"log={job.shard.log_file.name}"
                )
        running = still_running

    print(
        f"[summary] total={len(shards)} skipped={skipped} completed={completed} failed={failed}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
