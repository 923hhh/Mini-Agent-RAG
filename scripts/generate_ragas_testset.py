from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from ragas.testset import TestsetGenerator

try:
    from scripts.eval_common import PROJECT_ROOT, write_json_report
except ModuleNotFoundError:
    from eval_common import PROJECT_ROOT, write_json_report
from app.chains.text_splitter import split_documents
from app.loaders.factory import load_file
from app.services.core.settings import AppSettings, load_settings
from app.services.models.embedding_service import build_embeddings
from app.services.models.llm_service import (
    build_chat_model,
    resolve_openai_compatible_api_key,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="generate_ragas_testset",
        description="Use RAGAS to generate a synthetic RAG testset for a local document.",
    )
    parser.add_argument(
        "--pdf-path",
        type=Path,
        default=PROJECT_ROOT / "摩托车发动机维修手册.pdf",
        help="Target PDF path. Defaults to the motorcycle engine manual in repo root.",
    )
    parser.add_argument(
        "--testset-size",
        type=int,
        default=12,
        help="Number of synthetic samples to generate.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Optional chunk size override used before handing docs to RAGAS.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Optional chunk overlap override used before handing docs to RAGAS.",
    )
    parser.add_argument(
        "--llm-provider",
        choices=("auto", "ollama", "openai_compatible"),
        default="auto",
        help="LLM provider used by RAGAS generator. auto falls back to ollama when API key is missing.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="",
        help="Optional override for generator LLM model.",
    )
    parser.add_argument(
        "--embedding-provider",
        choices=("auto", "ollama", "openai_compatible"),
        default="auto",
        help="Embedding provider used by RAGAS generator. auto follows the effective LLM provider.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="",
        help="Optional override for generator embedding model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "manual",
        help="Output directory for generated jsonl/csv/report files.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root. Defaults to repository root.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    settings = load_settings(project_root)

    pdf_path = args.pdf_path.resolve()
    if not pdf_path.exists() or not pdf_path.is_file():
        print(f"目标 PDF 不存在: {pdf_path}", file=sys.stderr)
        return 1

    effective_settings = build_generation_settings(
        settings,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model.strip(),
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model.strip(),
    )

    chunk_size = args.chunk_size or settings.kb.CHUNK_SIZE
    chunk_overlap = settings.kb.CHUNK_OVERLAP if args.chunk_overlap is None else args.chunk_overlap
    source_documents = load_file(pdf_path, pdf_path.parent, settings=effective_settings)
    chunked_documents = split_documents(
        source_documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        splitter_name=effective_settings.kb.TEXT_SPLITTER_NAME,
    )
    if not chunked_documents:
        print("未能从目标 PDF 中解析出可用于生成测试集的文档块。", file=sys.stderr)
        return 1

    llm = build_chat_model(
        effective_settings,
        model_name=effective_settings.model.QUERY_REWRITE_MODEL.strip()
        or effective_settings.model.DEFAULT_LLM_MODEL,
        temperature=0.0,
    )
    embeddings = build_embeddings(
        effective_settings,
        model_name=effective_settings.model.DEFAULT_EMBEDDING_MODEL,
    )
    generator = TestsetGenerator.from_langchain(llm=llm, embedding_model=embeddings)

    print(
        f"[ragas] source_docs={len(source_documents)} chunks={len(chunked_documents)} "
        f"testset_size={args.testset_size}",
        flush=True,
    )
    testset = generator.generate_with_langchain_docs(
        chunked_documents,
        testset_size=args.testset_size,
        with_debugging_logs=False,
        raise_exceptions=True,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = slugify_stem(pdf_path.stem)
    base_name = f"{stem}_ragas_testset_{args.testset_size}"
    jsonl_path = output_dir / f"{base_name}.jsonl"
    csv_path = output_dir / f"{base_name}.csv"
    summary_path = output_dir / f"{base_name}_summary.json"

    testset.to_jsonl(str(jsonl_path))
    testset.to_csv(str(csv_path))
    summary = {
        "pdf_path": str(pdf_path),
        "source_document_count": len(source_documents),
        "chunk_count": len(chunked_documents),
        "testset_size": args.testset_size,
        "llm_provider": effective_settings.model.LLM_PROVIDER,
        "llm_model": effective_settings.model.QUERY_REWRITE_MODEL
        or effective_settings.model.DEFAULT_LLM_MODEL,
        "embedding_provider": effective_settings.model.EMBEDDING_PROVIDER
        or effective_settings.model.LLM_PROVIDER,
        "embedding_model": effective_settings.model.DEFAULT_EMBEDDING_MODEL,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "jsonl_path": str(jsonl_path),
        "csv_path": str(csv_path),
    }
    write_json_report(summary_path, summary)

    print(f"[ragas] jsonl => {jsonl_path}")
    print(f"[ragas] csv   => {csv_path}")
    print(f"[ragas] meta  => {summary_path}")
    return 0


def build_generation_settings(
    settings: AppSettings,
    *,
    llm_provider: str,
    llm_model: str,
    embedding_provider: str,
    embedding_model: str,
) -> AppSettings:
    effective_llm_provider = resolve_effective_llm_provider(settings, llm_provider)
    effective_embedding_provider = resolve_effective_embedding_provider(
        effective_llm_provider,
        embedding_provider,
    )

    model_updates: dict[str, object] = {
        "LLM_PROVIDER": effective_llm_provider,
        "EMBEDDING_PROVIDER": effective_embedding_provider,
    }
    if llm_model:
        model_updates["DEFAULT_LLM_MODEL"] = llm_model
        model_updates["QUERY_REWRITE_MODEL"] = llm_model
    elif not settings.model.QUERY_REWRITE_MODEL.strip():
        model_updates["QUERY_REWRITE_MODEL"] = settings.model.DEFAULT_LLM_MODEL
    if embedding_model:
        model_updates["DEFAULT_EMBEDDING_MODEL"] = embedding_model

    return settings.model_copy(
        update={
            "model": settings.model.model_copy(update=model_updates),
        }
    )


def resolve_effective_llm_provider(settings: AppSettings, requested_provider: str) -> str:
    if requested_provider in {"ollama", "openai_compatible"}:
        return requested_provider
    if settings.model.LLM_PROVIDER == "openai_compatible":
        if resolve_openai_compatible_api_key(settings).strip():
            return "openai_compatible"
        return "ollama"
    return settings.model.LLM_PROVIDER


def resolve_effective_embedding_provider(
    effective_llm_provider: str,
    requested_provider: str,
) -> str:
    if requested_provider in {"ollama", "openai_compatible"}:
        return requested_provider
    return effective_llm_provider


def slugify_stem(text: str) -> str:
    normalized = re.sub(r"\s+", "_", text.strip())
    normalized = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "document"


if __name__ == "__main__":
    raise SystemExit(main())
