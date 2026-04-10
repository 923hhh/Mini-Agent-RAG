from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_TESSERACT_RELATIVE_PATH = Path("./data/tools/Tesseract-OCR/tesseract.exe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="preprocess_pdf_ocr",
        description="OCR preprocess a scanned PDF into a text or markdown file before knowledge-base ingestion.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Input PDF path.")
    parser.add_argument("--output", type=Path, required=True, help="Output .txt or .md path.")
    parser.add_argument(
        "--language",
        default="chi_sim+eng",
        help="Tesseract language pack, for example chi_sim+eng or eng.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Render DPI for OCR.")
    parser.add_argument(
        "--prefer-text-layer",
        action="store_true",
        help="If a page already contains text, reuse the text layer instead of forcing OCR.",
    )
    parser.add_argument(
        "--tesseract-cmd",
        default="",
        help="Optional full path to tesseract executable.",
    )
    parser.add_argument(
        "--page-separator",
        default="\n\n---\n\n",
        help="Separator inserted between pages in the output file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        preprocess_pdf_with_ocr(
            input_path=args.input.resolve(),
            output_path=args.output.resolve(),
            language=args.language,
            dpi=args.dpi,
            prefer_text_layer=args.prefer_text_layer,
            tesseract_cmd=args.tesseract_cmd.strip(),
            page_separator=args.page_separator,
        )
    except Exception as exc:
        print(f"OCR 预处理失败: {exc}", file=sys.stderr)
        return 1

    print(f"OCR 预处理完成: {args.output.resolve()}")
    return 0


def preprocess_pdf_with_ocr(
    *,
    input_path: Path,
    output_path: Path,
    language: str,
    dpi: int,
    prefer_text_layer: bool,
    tesseract_cmd: str,
    page_separator: str,
) -> None:
    if input_path.suffix.lower() != ".pdf":
        raise ValueError(f"输入文件不是 PDF: {input_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    if output_path.suffix.lower() not in {".txt", ".md"}:
        raise ValueError("输出文件必须是 .txt 或 .md。")

    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "OCR 预处理需要安装 `PyMuPDF`。请先执行 `pip install -r requirements.txt`。"
        ) from exc

    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError(
            "OCR 预处理需要安装 `pytesseract`。请先执行 `pip install -r requirements.txt`。"
        ) from exc

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "OCR 预处理需要安装 `Pillow`。请先执行 `pip install -r requirements.txt`。"
        ) from exc

    resolved_tesseract_cmd = resolve_tesseract_cmd(tesseract_cmd)
    if resolved_tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = resolved_tesseract_cmd
    elif not shutil.which("tesseract"):
        raise RuntimeError(
            "未检测到 `tesseract` 可执行文件。请先安装 Tesseract OCR，"
            "或使用 `--tesseract-cmd <path>` / 环境变量 `OCR_TESSERACT_CMD` / `TESSERACT_CMD` 指定其完整路径。"
        )

    document = fitz.open(str(input_path))
    scale = max(dpi, 72) / 72.0
    matrix = fitz.Matrix(scale, scale)
    page_blocks: list[str] = []

    for page_index in range(document.page_count):
        page = document.load_page(page_index)
        page_text = page.get_text("text").strip() if prefer_text_layer else ""
        if page_text:
            extracted = page_text
        else:
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            extracted = pytesseract.image_to_string(image, lang=language).strip()

        title = f"Page {page_index + 1}"
        if output_path.suffix.lower() == ".md":
            page_blocks.append(f"## {title}\n\n{extracted}")
        else:
            page_blocks.append(f"[{title}]\n{extracted}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page_separator.join(page_blocks).strip() + "\n", encoding="utf-8")


def resolve_tesseract_cmd(cli_value: str) -> str:
    for raw_value in (
        cli_value.strip(),
        os.getenv("OCR_TESSERACT_CMD", "").strip(),
        os.getenv("TESSERACT_CMD", "").strip(),
    ):
        if not raw_value:
            continue
        candidate = Path(raw_value)
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        return str(candidate)

    fallback = (PROJECT_ROOT / DEFAULT_TESSERACT_RELATIVE_PATH).resolve()
    if fallback.exists():
        return str(fallback)
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
