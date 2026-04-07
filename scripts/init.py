from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.init_service import initialize_project, render_init_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="init",
        description="Initialize local configs and runtime directories for the RAG agent project.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root to initialize. Defaults to the current repository root.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()

    try:
        result = initialize_project(project_root)
    except Exception as exc:
        print(f"初始化失败: {exc}", file=sys.stderr)
        return 1

    print(render_init_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
