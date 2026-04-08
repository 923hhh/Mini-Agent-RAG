from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.kb_ingestion_service import rebuild_knowledge_base, render_rebuild_summary
from app.services.settings import load_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rebuild_kb",
        description="Rebuild a local knowledge base into a persistent vector store.",
    )
    parser.add_argument("--kb-name", required=True, help="Knowledge base name.")
    parser.add_argument("--chunk-size", type=int, default=None, help="Override chunk size.")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="Override chunk overlap.")
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Override embedding model name configured in model_settings.yaml.",
    )
    parser.add_argument(
        "--enable-image-vlm-for-build",
        action="store_true",
        help="Enable image VLM captioning for this rebuild only, and force a full rebuild.",
    )
    parser.add_argument(
        "--force-full-rebuild",
        action="store_true",
        help="Force a full rebuild for this run and ignore incremental reuse.",
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

    try:
        settings = load_settings(project_root)
        result = rebuild_knowledge_base(
            settings=settings,
            knowledge_base_name=args.kb_name,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embedding_model=args.embedding_model,
            enable_image_vlm_for_build=args.enable_image_vlm_for_build,
            force_full_rebuild=args.force_full_rebuild,
        )
    except Exception as exc:
        print(f"知识库重建失败: {exc}", file=sys.stderr)
        return 1

    print(render_rebuild_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
