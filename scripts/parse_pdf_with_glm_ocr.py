"""Parse scanned PDF with GLM-OCR page-by-page, export markdown, and optionally rebuild a KB."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib import error, request

try:
    import fitz
except ModuleNotFoundError:
    try:
        import pymupdf as fitz
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "未安装 PyMuPDF。请在当前运行脚本的 Python 环境中执行：pip install PyMuPDF"
        ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.core.settings import load_project_env, load_settings
from app.services.kb import ensure_knowledge_base_layout, rebuild_knowledge_base


DEFAULT_ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/layout_parsing"
DEFAULT_API_KEY_ENVS = ("BIGMODEL_API_KEY", "ZHIPUAI_API_KEY", "ZHIPU_API_KEY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-pdf",
        type=Path,
        required=True,
        help="Scanned PDF path.",
    )
    parser.add_argument(
        "--kb-name",
        required=True,
        help="Target local knowledge base name. Recommend using a new *_ocr KB.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Markdown output directory. Default: data/knowledge_base/<kb>/content/ocr_markdown",
    )
    parser.add_argument(
        "--model",
        default="glm-ocr",
        help="BigModel layout parsing model name.",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="BigModel layout parsing endpoint.",
    )
    parser.add_argument(
        "--api-key-env",
        default="BIGMODEL_API_KEY",
        help="Primary environment variable used to read API key.",
    )
    parser.add_argument(
        "--pages-per-batch",
        type=int,
        default=100,
        help="How many OCRed pages to merge into one markdown file.",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="1-based start page.",
    )
    parser.add_argument(
        "--end-page",
        type=int,
        default=0,
        help="1-based end page. 0 means till the last page.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of concurrent page OCR requests.",
    )
    parser.add_argument(
        "--render-scale",
        type=float,
        default=2.0,
        help="PDF page render scale before OCR.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=300,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--clear-content",
        action="store_true",
        help="Clear the target KB content directory before writing OCR markdown.",
    )
    parser.add_argument(
        "--skip-rebuild",
        action="store_true",
        help="Only export OCR markdown and skip KB rebuild.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Optional KB rebuild chunk size override.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Optional KB rebuild chunk overlap override.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Optional KB rebuild embedding model override.",
    )
    return parser.parse_args()


def log_status(message: str) -> None:
    print(f"[glm-ocr] {message}", flush=True)


def render_progress(completed: int, total: int, *, width: int = 24) -> str:
    total = max(total, 1)
    completed = min(max(completed, 0), total)
    ratio = completed / total
    filled = min(width, int(ratio * width))
    return f"[{'#' * filled}{'-' * (width - filled)}] {completed}/{total} ({ratio * 100:5.1f}%)"


def render_rebuild_progress(progress: float, message: str) -> None:
    percent = max(0.0, min(100.0, progress * 100))
    print(f"{percent:6.2f}% {message}", flush=True)


def sanitize_filename(value: str, *, max_len: int = 120) -> str:
    invalid = '\\/:*?"<>|'
    text = "".join("_" if char in invalid else char for char in str(value or "").strip())
    text = "_".join(part for part in text.split() if part)
    text = text.strip("._")
    return (text or "document")[:max_len]


def resolve_api_key(primary_env: str) -> str:
    for env_name in (primary_env, *DEFAULT_API_KEY_ENVS):
        key = env_name.strip()
        if not key:
            continue
        value = str(os.environ.get(key, "")).strip()
        if value:
            return value
    return ""


def resolve_page_range(page_count: int, start_page: int, end_page: int) -> tuple[int, int]:
    start = max(1, start_page)
    end = page_count if end_page <= 0 else min(page_count, end_page)
    if start > end:
        raise ValueError(f"非法页码范围: start_page={start_page}, end_page={end_page}, total={page_count}")
    return start, end


def build_page_batches(start_page: int, end_page: int, pages_per_batch: int) -> list[tuple[int, int]]:
    if pages_per_batch <= 0:
        raise ValueError("pages_per_batch 必须大于 0。")
    batches: list[tuple[int, int]] = []
    current = start_page
    while current <= end_page:
        batch_end = min(end_page, current + pages_per_batch - 1)
        batches.append((current, batch_end))
        current = batch_end + 1
    return batches


def call_glm_ocr(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    file_data_uri: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "file": file_data_uri,
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            raw = response.read()
    except error.HTTPError as exc:
        raw = exc.read()
        detail = raw.decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"请求失败: {exc}") from exc

    try:
        data = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("接口返回不是合法 JSON。") from exc

    api_error = data.get("error")
    if isinstance(api_error, dict):
        code = str(api_error.get("code", "")).strip()
        message = str(api_error.get("message", "")).strip()
        raise RuntimeError(f"{code or 'api_error'}: {message or json.dumps(api_error, ensure_ascii=False)}")

    return data


def render_page_png_data_uri(
    *,
    source_pdf: Path,
    page_number: int,
    render_scale: float,
) -> str:
    max_image_bytes = 10 * 1024 * 1024
    candidate_scales = [render_scale, 1.75, 1.5, 1.25, 1.0]
    source_doc = fitz.open(source_pdf)
    try:
        page = source_doc.load_page(page_number - 1)
        for scale in candidate_scales:
            if scale <= 0:
                continue
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            png_bytes = pix.tobytes("png")
            if len(png_bytes) <= max_image_bytes:
                encoded = base64.b64encode(png_bytes).decode("ascii")
                return f"data:image/png;base64,{encoded}"
    finally:
        source_doc.close()
    raise RuntimeError(f"第 {page_number} 页渲染后的 PNG 超过 10MB，无法提交给 GLM-OCR。")


def process_page(
    *,
    source_pdf: Path,
    page_number: int,
    raw_dir: Path,
    endpoint: str,
    api_key: str,
    model: str,
    timeout_seconds: int,
    render_scale: float,
    page_digits: int,
) -> dict[str, Any]:
    file_data_uri = render_page_png_data_uri(
        source_pdf=source_pdf,
        page_number=page_number,
        render_scale=render_scale,
    )
    response = call_glm_ocr(
        endpoint=endpoint,
        api_key=api_key,
        model=model,
        file_data_uri=file_data_uri,
        timeout_seconds=timeout_seconds,
    )
    markdown = str(response.get("md_results", "")).strip()
    if not markdown:
        raise RuntimeError("接口返回成功，但 md_results 为空。")
    raw_path = raw_dir / f"page_{page_number:0{page_digits}d}.json"
    raw_path.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "page_number": page_number,
        "markdown": markdown,
        "raw_path": str(raw_path),
        "task_id": str(response.get("id", "")).strip(),
        "request_id": str(response.get("request_id", "")).strip(),
    }


def clear_directory_children(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def list_conflicting_pdf_files(content_dir: Path) -> list[Path]:
    return sorted(path for path in content_dir.rglob("*.pdf") if path.is_file())


def write_markdown_batches(
    *,
    output_dir: Path,
    source_pdf: Path,
    page_results: list[dict[str, Any]],
    page_batches: list[tuple[int, int]],
    page_digits: int,
) -> list[dict[str, Any]]:
    stem = sanitize_filename(source_pdf.stem)
    result_by_page = {int(item["page_number"]): item for item in page_results}
    written_batches: list[dict[str, Any]] = []
    for start_page, end_page in page_batches:
        pieces: list[str] = []
        for page_number in range(start_page, end_page + 1):
            item = result_by_page.get(page_number)
            if not item:
                continue
            pieces.append(f"<!-- page: {page_number} -->\n{item['markdown']}".strip())
        merged_markdown = "\n\n".join(piece for piece in pieces if piece).strip()
        if not merged_markdown:
            continue
        suffix = f"pages_{start_page:0{page_digits}d}_{end_page:0{page_digits}d}"
        markdown_path = output_dir / f"{stem}__{suffix}.md"
        markdown_path.write_text(merged_markdown + "\n", encoding="utf-8")
        written_batches.append(
            {
                "start_page": start_page,
                "end_page": end_page,
                "markdown_path": str(markdown_path),
                "page_count": end_page - start_page + 1,
            }
        )
    return written_batches


def run_rebuild(
    *,
    kb_name: str,
    chunk_size: int | None,
    chunk_overlap: int | None,
    embedding_model: str | None,
) -> dict[str, Any]:
    settings = load_settings(PROJECT_ROOT)
    result = rebuild_knowledge_base(
        settings=settings,
        knowledge_base_name=kb_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        force_full_rebuild=True,
        progress_callback=render_rebuild_progress,
    )
    return {
        "knowledge_base_name": result.knowledge_base_name,
        "files_processed": result.files_processed,
        "raw_documents": result.raw_documents,
        "chunks": result.chunks,
        "metadata_path": str(result.metadata_path),
        "vector_store_dir": str(result.vector_store_dir),
    }


def main() -> int:
    args = parse_args()
    input_pdf = args.input_pdf.resolve()
    if not input_pdf.exists():
        print(f"未找到 PDF 文件: {input_pdf}", file=sys.stderr)
        return 1
    if input_pdf.suffix.lower() != ".pdf":
        print(f"输入文件不是 PDF: {input_pdf}", file=sys.stderr)
        return 1

    load_project_env(PROJECT_ROOT)
    api_key = resolve_api_key(args.api_key_env)
    if not api_key:
        print(
            "未找到 BigModel API Key。请先设置环境变量，例如：\n"
            "  PowerShell: $env:BIGMODEL_API_KEY=\"你的key\"",
            file=sys.stderr,
        )
        return 1

    settings = load_settings(PROJECT_ROOT)
    content_dir, _vector_store_dir = ensure_knowledge_base_layout(settings, args.kb_name)
    output_dir = args.output_dir.resolve() if args.output_dir else (content_dir / "ocr_markdown").resolve()
    raw_dir = (settings.knowledge_base_root / args.kb_name / "ocr_raw").resolve()
    manifest_path = (settings.knowledge_base_root / args.kb_name / "glm_ocr_manifest.json").resolve()

    if args.clear_content:
        clear_directory_children(content_dir)
        clear_directory_children(raw_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(input_pdf)
    try:
        page_count = doc.page_count
    finally:
        doc.close()

    start_page, end_page = resolve_page_range(page_count, args.start_page, args.end_page)
    batches = build_page_batches(start_page, end_page, min(args.pages_per_batch, 100))
    page_digits = max(3, len(str(page_count)))
    page_numbers = list(range(start_page, end_page + 1))

    log_status(
        f"input={input_pdf} total_pages={page_count} selected={start_page}-{end_page} "
        f"pages={len(page_numbers)} batches={len(batches)} workers={max(1, args.workers)} output_dir={output_dir}"
    )

    page_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {
            executor.submit(
                process_page,
                source_pdf=input_pdf,
                page_number=page_number,
                raw_dir=raw_dir,
                endpoint=args.endpoint,
                api_key=api_key,
                model=args.model,
                timeout_seconds=args.timeout_seconds,
                render_scale=args.render_scale,
                page_digits=page_digits,
            ): page_number
            for page_number in page_numbers
        }
        completed = 0
        total = len(future_map)
        for future in as_completed(future_map):
            page_number = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                print(file=sys.stderr)
                print(
                    f"[glm-ocr] OCR 失败: page={page_number}, error={exc}",
                    file=sys.stderr,
                )
                return 1
            page_results.append(result)
            completed += 1
            progress = render_progress(completed, total)
            sys.stdout.write(f"\r[glm-ocr] {progress} last=page {page_number}")
            sys.stdout.flush()
        print()

    page_results.sort(key=lambda item: int(item["page_number"]))
    batch_results = write_markdown_batches(
        output_dir=output_dir,
        source_pdf=input_pdf,
        page_results=page_results,
        page_batches=batches,
        page_digits=page_digits,
    )
    manifest = {
        "source_pdf": str(input_pdf),
        "kb_name": args.kb_name,
        "output_dir": str(output_dir),
        "raw_dir": str(raw_dir),
        "endpoint": args.endpoint,
        "model": args.model,
        "mode": "page_images",
        "total_pages": page_count,
        "selected_pages": {"start_page": start_page, "end_page": end_page},
        "pages_per_batch": min(args.pages_per_batch, 100),
        "ocr_page_count": len(page_results),
        "batch_count": len(batch_results),
        "batches": batch_results,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    log_status(f"markdown batches saved: {len(batch_results)}")
    log_status(f"manifest saved: {manifest_path}")

    if args.skip_rebuild:
        log_status("OCR markdown export finished. rebuild skipped.")
        return 0

    conflicting_pdfs = list_conflicting_pdf_files(content_dir)
    if conflicting_pdfs:
        preview = "\n".join(f"  - {path}" for path in conflicting_pdfs[:5])
        print(
            "目标知识库 content 目录中仍包含 PDF，自动 rebuild 可能再次走到原始扫描 PDF。\n"
            "请改用新的 kb_name，或加上 --clear-content 重新导入。\n"
            f"发现的 PDF:\n{preview}",
            file=sys.stderr,
        )
        return 1

    rebuild_result = run_rebuild(
        kb_name=args.kb_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
    )
    print(json.dumps(rebuild_result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
