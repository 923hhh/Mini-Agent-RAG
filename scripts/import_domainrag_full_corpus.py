"""Import full DomainRAG corpus into a local Mini-Agent-RAG2 knowledge base.

This script converts the official DomainRAG corpus json files into plain text
files plus sidecar metadata understood by the current project, then optionally
triggers a full knowledge-base rebuild.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.core.settings import load_settings
from app.services.kb.kb_ingestion_service import ensure_knowledge_base_layout, rebuild_knowledge_base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domainrag-root",
        type=Path,
        required=True,
        help="Local DomainRAG-main root.",
    )
    parser.add_argument(
        "--corpus-json-root",
        type=Path,
        default=None,
        help="Directory with official corpus json files. "
        "Default: <domainrag-root>/corpus/rdzs/json_output",
    )
    parser.add_argument(
        "--kb-name",
        default="domainrag_full_corpus",
        help="Target local knowledge base name.",
    )
    parser.add_argument(
        "--clear-content",
        action="store_true",
        help="Delete existing knowledge base content before import.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Trigger a full rebuild after import.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1500,
        help="Chunk size override used when --rebuild is set.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=80,
        help="Chunk overlap override used when --rebuild is set.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Embedding model override used when --rebuild is set.",
    )
    return parser.parse_args()


def resolve_corpus_root(domainrag_root: Path, explicit_root: Path | None) -> Path:
    if explicit_root is not None:
        return explicit_root
    return domainrag_root / "corpus" / "rdzs" / "json_output"


def slugify(value: str, *, max_len: int = 80) -> str:
    text = re.sub(r"\s+", "_", str(value or "").strip())
    text = re.sub(r'[\\/:*?"<>|\0]+', "_", text)
    text = text.strip("._")
    if not text:
        text = "untitled"
    return text[:max_len]


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def build_passage_text(doc: dict[str, Any], passage: str, psg_id: int) -> str:
    title = normalize_text(doc.get("title", ""))
    url = normalize_text(doc.get("url", ""))
    date = normalize_text(doc.get("date", ""))
    body = normalize_text(passage)
    lines = [
        f"标题：{title}",
        f"URL：{url}",
        f"日期：{date}",
        f"文档ID：{doc.get('id', '')}",
        f"段落ID：{psg_id}",
        "",
        body,
    ]
    return "\n".join(lines).strip() + "\n"


def render_rebuild_progress(progress: float, message: str) -> None:
    percent = max(0.0, min(100.0, progress * 100))
    print(f"{percent:6.2f}% {message}", flush=True)


def main() -> int:
    args = parse_args()
    domainrag_root = args.domainrag_root.resolve()
    corpus_json_root = resolve_corpus_root(domainrag_root, args.corpus_json_root).resolve()

    if not corpus_json_root.exists():
        print(
            f"未找到 DomainRAG 全量语料目录: {corpus_json_root}\n"
            "请先把官方 corpus 解压到位，再重新运行。",
            file=sys.stderr,
        )
        return 1

    json_paths = sorted(corpus_json_root.glob("*.json"))
    if not json_paths:
        print(f"语料目录下没有 json 文件: {corpus_json_root}", file=sys.stderr)
        return 1

    settings = load_settings(PROJECT_ROOT)
    content_dir, _vector_store_dir = ensure_knowledge_base_layout(settings, args.kb_name)

    if args.clear_content and content_dir.exists():
        for child in content_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    export_root = content_dir / "domainrag_full_corpus"
    export_root.mkdir(parents=True, exist_ok=True)

    metadata_map: dict[str, dict[str, str]] = {}
    doc_count = 0
    passage_count = 0

    for json_path in tqdm(json_paths, desc="Import DomainRAG docs", unit="doc"):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        doc_id = str(data.get("id", "")).strip() or json_path.stem
        title = normalize_text(data.get("title", "")) or doc_id
        safe_dir = slugify(f"{title}__doc-{doc_id}", max_len=120)
        target_dir = export_root / safe_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        passages = data.get("passages", [])
        if not isinstance(passages, list):
            continue

        doc_count += 1
        for psg_id, passage in enumerate(passages):
            if not normalize_text(passage):
                continue
            filename = f"{slugify(title, max_len=80)}__doc-{doc_id}__psg-{psg_id}.txt"
            target_path = target_dir / filename
            target_path.write_text(build_passage_text(data, str(passage), psg_id), encoding="utf-8")

            relative_path = target_path.relative_to(content_dir).as_posix()
            metadata_map[relative_path] = {
                "title": title,
                "url": normalize_text(data.get("url", "")),
                "date": normalize_text(data.get("date", "")),
                "doc_id": doc_id,
                "psg_id": str(psg_id),
                "content_type": "document_text",
                "source_modality": "text",
                "original_file_type": "domainrag_json_passage",
                "evidence_summary": title,
            }
            passage_count += 1

    metadata_payload = {"files": metadata_map}
    metadata_path = content_dir / ".rag_file_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    import_manifest = {
        "source": "DomainRAG",
        "source_mode": "full_corpus_json_output",
        "domainrag_root": str(domainrag_root),
        "corpus_json_root": str(corpus_json_root),
        "kb_name": args.kb_name,
        "document_count": doc_count,
        "passage_count": passage_count,
        "export_root": str(export_root),
        "metadata_path": str(metadata_path),
    }
    manifest_path = settings.knowledge_base_root / args.kb_name / "domainrag_full_import_manifest.json"
    manifest_path.write_text(json.dumps(import_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] imported docs: {doc_count}")
    print(f"[done] imported passages: {passage_count}")
    print(f"[done] content dir: {content_dir}")
    print(f"[done] import manifest: {manifest_path}")

    if not args.rebuild:
        print("[done] import finished. rebuild not requested.")
        return 0

    result = rebuild_knowledge_base(
        settings=settings,
        knowledge_base_name=args.kb_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
        force_full_rebuild=True,
        progress_callback=render_rebuild_progress,
    )
    print(
        json.dumps(
            {
                "knowledge_base_name": result.knowledge_base_name,
                "files_processed": result.files_processed,
                "raw_documents": result.raw_documents,
                "chunks": result.chunks,
                "metadata_path": str(result.metadata_path),
                "vector_store_dir": str(result.vector_store_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
