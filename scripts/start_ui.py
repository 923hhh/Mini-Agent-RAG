from __future__ import annotations

import argparse
import sys
from pathlib import Path

from streamlit.web import cli as stcli


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.settings import load_settings
from app.services.temp_kb_service import maybe_run_startup_cleanup, render_temp_cleanup_summary


def parse_args(settings) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="start_ui",
        description="Start the Streamlit UI for the local RAG agent project.",
    )
    parser.add_argument("--host", default=settings.basic.WEBUI_HOST)
    parser.add_argument("--port", type=int, default=settings.basic.WEBUI_PORT)
    return parser.parse_args()


def main() -> int:
    settings = load_settings(PROJECT_ROOT)
    args = parse_args(settings)
    try:
        cleanup_result = maybe_run_startup_cleanup(settings, startup_name="ui")
    except Exception as exc:
        print(f"启动前临时知识库清理失败，将继续启动 UI: {exc}", file=sys.stderr)
    else:
        if cleanup_result is not None and cleanup_result.scanned:
            print(render_temp_cleanup_summary(cleanup_result))
    app_path = PROJECT_ROOT / "app" / "ui" / "app.py"
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
        "--server.headless",
        "true",
    ]
    return stcli.main()


if __name__ == "__main__":
    raise SystemExit(main())
