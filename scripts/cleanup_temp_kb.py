from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.core.settings import load_settings
from app.services.runtime.temp_kb_service import cleanup_temp_knowledge_bases, render_temp_cleanup_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cleanup_temp_kb",
        description="Clean up expired or selected temporary knowledge bases.",
    )
    parser.add_argument(
        "--knowledge-id",
        type=str,
        default=None,
        help="Clean a specific temporary knowledge base by id.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Clean all temporary knowledge bases instead of only expired ones.",
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
    try:
        settings = load_settings(args.project_root.resolve())
        result = cleanup_temp_knowledge_bases(
            settings,
            knowledge_id=args.knowledge_id,
            expired_only=not args.all,
            cleanup_reason="manual_cli",
        )
    except Exception as exc:
        print(f"临时知识库清理失败: {exc}", file=sys.stderr)
        return 1

    print(render_temp_cleanup_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

