from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.core.settings import load_settings
from app.services.runtime.temp_kb_service import maybe_run_startup_cleanup, render_temp_cleanup_summary


def parse_args(settings) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="start_api",
        description="Start the FastAPI server for the local RAG agent project.",
    )
    parser.add_argument("--host", default=settings.basic.API_HOST)
    parser.add_argument("--port", type=int, default=settings.basic.API_PORT)
    parser.add_argument("--reload", action="store_true")
    return parser.parse_args()


def main() -> int:
    settings = load_settings(PROJECT_ROOT)
    args = parse_args(settings)
    try:
        cleanup_result = maybe_run_startup_cleanup(settings, startup_name="api")
    except Exception as exc:
        print(f"启动前临时知识库清理失败，将继续启动 API: {exc}", file=sys.stderr)
    else:
        if cleanup_result is not None and cleanup_result.scanned:
            print(render_temp_cleanup_summary(cleanup_result))
    uvicorn.run("app.api.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

