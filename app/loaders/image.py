"""加载图片文件并提取 OCR 与视觉描述内容。"""

from __future__ import annotations

import os
import re
import shutil
import unicodedata
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document

from app.loaders.factory import BaseKnowledge, KnowledgeFactory, list_supported_files, load_documents, load_file
from app.loaders.office import DocxKnowledge, EpubKnowledge
from app.loaders.pdf import PdfKnowledge, PdfOutlineSection
from app.loaders.text import MarkdownKnowledge, TextKnowledge
from app.services.core.observability import append_jsonl_trace

if TYPE_CHECKING:
    from app.services.core.settings import AppSettings


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
INSTRUCTION_STEP_PATTERN = re.compile(r"^\s*(\d+)\s*(?:[\.、．)\]]\s*)?(.+)$")
INSTRUCTION_LIST_ITEM_PATTERN = re.compile(r"^\s*[oO○●•·\-]\s*(.+)$")
INSTRUCTION_NOTE_PATTERN = re.compile(r"^\s*(提示|注意|说明|备注)[:：]?\s*(.+)$")
INSTRUCTION_TITLE_PATTERN = re.compile(r"^\s*\d+(?:\.\d+)+\s*\S+")
INSTRUCTION_KEYWORDS = (
    "安装",
    "拆卸",
    "拆下",
    "装配",
    "维修",
    "保养",
    "气门",
    "气缸",
    "螺栓",
    "螺母",
    "扭矩",
    "垫片",
    "弹簧",
    "导向",
    "件号",
)


@dataclass(frozen=True)
class ImageRegionSpec:
    key: str
    title: str
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class InstructionPageParseResult:
    title: str
    steps: tuple[str, ...]
    note_lines: tuple[str, ...]
    parts_list: tuple[str, ...]


@dataclass(frozen=True)
class FigureBandSpec:
    key: str
    title: str
    box: tuple[int, int, int, int]


class ImageKnowledge(BaseKnowledge):
    supported_extensions = tuple(sorted(SUPPORTED_IMAGE_EXTENSIONS))

    def load(self) -> list[Document]:
        return _load_image(
            self.path,
            self.base_metadata,
            self.relative_path,
            settings=self.settings,
        )


__all__ = [
    "BaseKnowledge",
    "KnowledgeFactory",
    "list_supported_files",
    "load_documents",
    "load_file",
    "TextKnowledge",
    "MarkdownKnowledge",
    "PdfKnowledge",
    "PdfOutlineSection",
    "DocxKnowledge",
    "EpubKnowledge",
    "ImageKnowledge",
    "ImageRegionSpec",
    "InstructionPageParseResult",
    "FigureBandSpec",
]


def _load_image(
    path: Path,
    base_metadata: dict[str, str],
    relative_path: str,
    *,
    settings: "AppSettings | None" = None,
) -> list[Document]:
    ocr_enabled = True if settings is None else settings.kb.IMAGE_OCR_ENABLED
    image_title = base_metadata.get("title", path.stem)
    ocr_text = ""
    caption_text = ""
    raw_caption_text = ""
    region_caption_texts: list[str] = []
    region_documents: list[Document] = []
    instruction_documents: list[Document] = []
    ocr_error = ""
    caption_error = ""
    image_caption_reason = "disabled"
    ocr_char_count = 0
    caption_filtered = False
    caption_region_fn: Any | None = None
    format_caption_fn: Any | None = None
    ocr_backend_used = ""
    ocr_language_used = ""

    if ocr_enabled:
        language = "chi_sim+eng" if settings is None else settings.kb.IMAGE_OCR_LANGUAGE
        if settings is None:
            tesseract_cmd = ""
        else:
            tesseract_cmd = settings.resolve_ocr_tesseract_cmd()
        try:
            ocr_text, ocr_backend_used, ocr_language_used = _extract_image_text(
                path,
                backend=(
                    "tesseract" if settings is None else settings.kb.IMAGE_OCR_BACKEND
                ),
                instruction_page_backend=(
                    "" if settings is None else settings.kb.IMAGE_OCR_INSTRUCTION_PAGE_BACKEND
                ),
                language=language,
                tesseract_cmd=tesseract_cmd,
                paddle_language=(
                    "ch" if settings is None else settings.kb.PADDLE_OCR_LANGUAGE
                ),
                paddle_use_angle_cls=(
                    True if settings is None else settings.kb.PADDLE_OCR_USE_ANGLE_CLS
                ),
                paddle_det_limit_side_len=(
                    1600 if settings is None else settings.kb.PADDLE_OCR_DET_LIMIT_SIDE_LEN
                ),
                paddle_min_score=(
                    0.45 if settings is None else settings.kb.PADDLE_OCR_MIN_SCORE
                ),
                fast_mode=(
                    True if settings is None else settings.kb.IMAGE_OCR_FAST_MODE
                ),
                max_side=(
                    1600 if settings is None else settings.kb.IMAGE_OCR_MAX_SIDE
                ),
                early_stop_chars=(
                    24 if settings is None else settings.kb.IMAGE_OCR_EARLY_STOP_CHARS
                ),
                min_confidence=(
                    60.0 if settings is None else settings.kb.OCR_MIN_CONFIDENCE
                ),
                min_text_length=(
                    6 if settings is None else settings.kb.OCR_MIN_TEXT_LENGTH
                ),
                min_meaningful_ratio=(
                    0.6 if settings is None else settings.kb.OCR_MIN_MEANINGFUL_RATIO
                ),
            )
        except Exception as exc:
            ocr_error = str(exc)

    should_caption, image_caption_reason, ocr_char_count = _should_generate_image_caption(
        settings=settings,
        ocr_text=ocr_text,
    )
    if should_caption:
        try:
            from app.loaders.vlm import (
                caption_image_bytes_structured,
                caption_image_structured,
                format_structured_image_caption,
            )
            caption_region_fn = caption_image_bytes_structured
            format_caption_fn = format_structured_image_caption

            structured_caption = caption_image_structured(settings, path, ocr_text=ocr_text)
            raw_caption_text = structured_caption.raw_text.strip()
            formatted_caption = format_structured_image_caption(structured_caption)
            if _is_low_quality_image_caption(
                formatted_caption,
                raw_text=raw_caption_text,
            ):
                caption_text = ""
                caption_error = "low_quality_caption_filtered"
                caption_filtered = True
            else:
                caption_text = formatted_caption

            region_caption_texts, region_documents = _build_image_region_documents(
                path=path,
                image_title=image_title,
                base_metadata=base_metadata,
                relative_path=relative_path,
                settings=settings,
                caption_region_fn=caption_region_fn,
                format_caption_fn=format_caption_fn,
            )
        except Exception as exc:
            caption_error = str(exc)

    if _looks_like_instruction_page(ocr_text):
        instruction_documents = _build_instruction_page_documents(
            path=path,
            image_title=image_title,
            base_metadata=base_metadata,
            relative_path=relative_path,
            settings=settings,
            ocr_text=ocr_text,
            should_caption=should_caption,
            caption_region_fn=caption_region_fn,
            format_caption_fn=format_caption_fn,
        )

    normalized = _build_image_knowledge_text(
        path=path,
        ocr_text=ocr_text,
        caption_text=caption_text,
        region_caption_texts=region_caption_texts,
        ocr_enabled=ocr_enabled,
        ocr_error=ocr_error,
        caption_enabled=should_caption,
        caption_error=caption_error,
    )
    resolved_modality = _resolve_image_source_modality(
        ocr_text=ocr_text,
        has_visual_caption=bool(caption_text or region_caption_texts),
    )
    _append_image_caption_trace(
        settings=settings,
        path=path,
        relative_path=relative_path,
        source_modality=resolved_modality,
        ocr_text=ocr_text,
        ocr_error=ocr_error,
        ocr_char_count=ocr_char_count,
        should_caption=should_caption,
        image_caption_reason=image_caption_reason,
        raw_caption_text=raw_caption_text,
        caption_text=caption_text,
        caption_error=caption_error,
        caption_filtered=caption_filtered,
        caption_scope="full_image",
    )

    root_document = Document(
        page_content=normalized,
        metadata={
            **base_metadata,
            "doc_id": f"{relative_path}#image-main",
            "section_title": image_title,
            "section_path": image_title,
            "section_index": 0,
            "content_type": "image_evidence",
            "source_modality": resolved_modality,
            "ocr_text": ocr_text or None,
            "image_caption": caption_text or None,
            "evidence_summary": _build_image_evidence_summary(
                image_title=image_title,
                ocr_text=ocr_text,
                caption_text=caption_text,
                region_caption_texts=region_caption_texts,
            ),
            "image_ocr_enabled": ocr_enabled,
            "image_caption_enabled": should_caption,
            "image_caption_reason": image_caption_reason,
            "ocr_char_count": ocr_char_count,
            "ocr_filtered_char_count": len(re.sub(r"\s+", "", ocr_text)),
            "ocr_language": ocr_language_used if ocr_enabled else "",
            "image_caption_model": (
                settings.model.IMAGE_VLM_MODEL if settings is not None and should_caption else ""
            ),
            "ocr_error": ocr_error,
            "image_caption_error": caption_error,
        },
    )

    return [
        root_document,
        *region_documents,
        *instruction_documents,
    ]


def _extract_image_text(
    path: Path,
    *,
    backend: str,
    instruction_page_backend: str,
    language: str,
    tesseract_cmd: str,
    paddle_language: str,
    paddle_use_angle_cls: bool,
    paddle_det_limit_side_len: int,
    paddle_min_score: float,
    fast_mode: bool,
    max_side: int,
    early_stop_chars: int,
    min_confidence: float,
    min_text_length: int,
    min_meaningful_ratio: float,
) -> tuple[str, str, str]:
    primary_backend = _normalize_ocr_backend_name(backend)
    secondary_backend = _normalize_ocr_backend_name(instruction_page_backend)
    errors: list[str] = []

    def run_backend(backend_name: str) -> tuple[str, str]:
        if backend_name == "paddle":
            return (
                _extract_image_text_paddle(
                    path,
                    language=paddle_language,
                    use_angle_cls=paddle_use_angle_cls,
                    det_limit_side_len=paddle_det_limit_side_len,
                    min_score=paddle_min_score,
                    fast_mode=fast_mode,
                    max_side=max_side,
                    early_stop_chars=early_stop_chars,
                    min_text_length=min_text_length,
                    min_meaningful_ratio=min_meaningful_ratio,
                ),
                paddle_language,
            )
        return (
            _extract_image_text_tesseract(
                path,
                language=language,
                tesseract_cmd=tesseract_cmd,
                fast_mode=fast_mode,
                max_side=max_side,
                early_stop_chars=early_stop_chars,
                min_confidence=min_confidence,
                min_text_length=min_text_length,
                min_meaningful_ratio=min_meaningful_ratio,
            ),
            language,
        )

    primary_text = ""
    primary_language = ""
    try:
        primary_text, primary_language = run_backend(primary_backend)
    except Exception as exc:
        errors.append(f"{primary_backend}: {exc}")

    selected_text = primary_text.strip()
    selected_backend = primary_backend
    selected_language = primary_language

    should_try_secondary = _should_try_instruction_page_ocr_backend(
        primary_text=selected_text,
        primary_backend=primary_backend,
        secondary_backend=secondary_backend,
        path=path,
        max_side=max_side,
    )
    if should_try_secondary:
        try:
            secondary_text, secondary_language = run_backend(secondary_backend)
        except Exception as exc:
            errors.append(f"{secondary_backend}: {exc}")
        else:
            chosen_text, chosen_backend, chosen_language = _choose_better_ocr_result(
                primary_text=selected_text,
                primary_backend=primary_backend,
                primary_language=selected_language,
                secondary_text=secondary_text,
                secondary_backend=secondary_backend,
                secondary_language=secondary_language,
            )
            selected_text = chosen_text
            selected_backend = chosen_backend
            selected_language = chosen_language

    if selected_text.strip():
        return selected_text.strip(), selected_backend, selected_language
    if errors:
        raise RuntimeError(" | ".join(errors))
    return "", selected_backend, selected_language


def _extract_image_text_tesseract(
    path: Path,
    *,
    language: str,
    tesseract_cmd: str,
    fast_mode: bool,
    max_side: int,
    early_stop_chars: int,
    min_confidence: float,
    min_text_length: int,
    min_meaningful_ratio: float,
) -> str:
    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError(
            "图片 OCR 需要安装 `pytesseract`。"
            "请在当前虚拟环境中执行 `pip install pytesseract` 或 `pip install -r requirements.txt`。"
        ) from exc

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "图片 OCR 需要安装 `Pillow`。"
            "请在当前虚拟环境中执行 `pip install Pillow` 或 `pip install -r requirements.txt`。"
        ) from exc

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        tessdata_dir = Path(tesseract_cmd).resolve().parent / "tessdata"
        if tessdata_dir.exists():
            os.environ["TESSDATA_PREFIX"] = str(tessdata_dir)
    elif not shutil.which("tesseract"):
        raise RuntimeError(
            "图片 OCR 未检测到 `tesseract` 可执行文件。"
            "请先安装 Tesseract OCR，或通过环境变量 `OCR_TESSERACT_CMD` / `TESSERACT_CMD`"
            " 或在 kb_settings.yaml 中配置 OCR_TESSERACT_CMD。"
        )

    collected_lines: list[str] = []
    seen_lines: set[str] = set()
    with _open_image_without_bomb_warning(path) as original_image:
        prepared_image = _resize_image_for_ocr(original_image, max_side=max_side)
        for candidate_image in _generate_ocr_candidate_images(
            prepared_image,
            fast_mode=fast_mode,
        ):
            data = pytesseract.image_to_data(
                candidate_image,
                lang=language,
                output_type=pytesseract.Output.DICT,
            )
            for line_text in _extract_accepted_ocr_lines(
                data,
                min_confidence=min_confidence,
                min_text_length=min_text_length,
                min_meaningful_ratio=min_meaningful_ratio,
            ):
                normalized = line_text.strip()
                if not normalized or normalized in seen_lines:
                    continue
                seen_lines.add(normalized)
                collected_lines.append(normalized)
            if _should_stop_ocr_early(collected_lines, early_stop_chars=early_stop_chars):
                break

    return "\n".join(collected_lines).strip()


def _extract_image_text_paddle(
    path: Path,
    *,
    language: str,
    use_angle_cls: bool,
    det_limit_side_len: int,
    min_score: float,
    fast_mode: bool,
    max_side: int,
    early_stop_chars: int,
    min_text_length: int,
    min_meaningful_ratio: float,
) -> str:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "PaddleOCR 需要安装 `numpy`。"
            "请在当前虚拟环境中执行 `pip install numpy` 或 `pip install -r requirements.txt`。"
        ) from exc

    try:
        ocr_engine = _get_paddle_ocr_engine(
            language=language,
            use_angle_cls=use_angle_cls,
            det_limit_side_len=det_limit_side_len,
        )
    except ImportError as exc:
        raise RuntimeError(
            "PaddleOCR 需要安装 `paddleocr`。"
            "请在当前虚拟环境中执行 `pip install paddleocr`，并按平台安装 `paddlepaddle`。"
        ) from exc

    collected_lines: list[str] = []
    seen_lines: set[str] = set()
    with _open_image_without_bomb_warning(path) as original_image:
        prepared_image = _resize_image_for_ocr(original_image, max_side=max_side)
        for candidate_image in _generate_paddle_ocr_candidate_images(
            prepared_image,
            fast_mode=fast_mode,
        ):
            result = ocr_engine.ocr(np.asarray(candidate_image), cls=use_angle_cls)
            for line_text in _extract_accepted_paddle_lines(
                result,
                min_score=min_score,
                min_text_length=min_text_length,
                min_meaningful_ratio=min_meaningful_ratio,
            ):
                normalized = line_text.strip()
                if not normalized or normalized in seen_lines:
                    continue
                seen_lines.add(normalized)
                collected_lines.append(normalized)
            if _should_stop_ocr_early(collected_lines, early_stop_chars=early_stop_chars):
                break
    return "\n".join(collected_lines).strip()


@lru_cache(maxsize=4)
def _get_paddle_ocr_engine(
    *,
    language: str,
    use_angle_cls: bool,
    det_limit_side_len: int,
):
    from paddleocr import PaddleOCR

    return PaddleOCR(
        use_angle_cls=use_angle_cls,
        lang=language,
        show_log=False,
        det_limit_side_len=det_limit_side_len,
    )


def _normalize_ocr_backend_name(value: str) -> str:
    normalized = (value or "").strip().lower().replace("-", "_")
    if normalized in {"", "auto"}:
        return "tesseract"
    if normalized in {"paddle", "paddleocr"}:
        return "paddle"
    return "tesseract"


def _should_try_instruction_page_ocr_backend(
    *,
    primary_text: str,
    primary_backend: str,
    secondary_backend: str,
    path: Path,
    max_side: int,
) -> bool:
    if secondary_backend == primary_backend:
        return False
    if not secondary_backend:
        return False
    if _looks_like_instruction_page(primary_text):
        return True
    if _instruction_keyword_hits(primary_text) >= 2 and _instruction_signal_hits(primary_text) >= 1:
        return True
    if not primary_text.strip() and _looks_like_instruction_page_layout(path, max_side=max_side):
        return True
    compact_length = len(re.sub(r"\s+", "", primary_text or ""))
    if compact_length < 40 and _instruction_keyword_hits(primary_text) >= 1:
        return True
    return False


def _choose_better_ocr_result(
    *,
    primary_text: str,
    primary_backend: str,
    primary_language: str,
    secondary_text: str,
    secondary_backend: str,
    secondary_language: str,
) -> tuple[str, str, str]:
    normalized_primary = (primary_text or "").strip()
    normalized_secondary = (secondary_text or "").strip()
    if not normalized_secondary:
        return normalized_primary, primary_backend, primary_language
    if not normalized_primary:
        return normalized_secondary, secondary_backend, secondary_language

    primary_score = _score_instruction_ocr_text(normalized_primary)
    secondary_score = _score_instruction_ocr_text(normalized_secondary)
    if secondary_score > primary_score + 2:
        return normalized_secondary, secondary_backend, secondary_language

    primary_length = len(re.sub(r"\s+", "", normalized_primary))
    secondary_length = len(re.sub(r"\s+", "", normalized_secondary))
    if secondary_score >= primary_score and secondary_length > primary_length + 12:
        return normalized_secondary, secondary_backend, secondary_language
    return normalized_primary, primary_backend, primary_language


def _score_instruction_ocr_text(text: str) -> float:
    lines = _normalize_instruction_lines(text)
    step_hits = sum(1 for line in lines if _extract_instruction_step(line) is not None)
    list_hits = sum(1 for line in lines if INSTRUCTION_LIST_ITEM_PATTERN.match(line))
    note_hits = sum(1 for line in lines if INSTRUCTION_NOTE_PATTERN.match(line))
    keyword_hits = _instruction_keyword_hits(text)
    signal_hits = _instruction_signal_hits(text)
    compact_length = len(re.sub(r"\s+", "", text or ""))
    return (
        step_hits * 6
        + list_hits * 2
        + note_hits * 2
        + keyword_hits * 2
        + signal_hits * 3
        + min(compact_length, 120) / 20.0
    )


def _instruction_keyword_hits(text: str) -> int:
    return sum(1 for keyword in INSTRUCTION_KEYWORDS if keyword in (text or ""))


def _instruction_signal_hits(text: str) -> int:
    normalized = (text or "").lower()
    signals = ("件号", "n·m", "nm", "提示", "注意", "步骤")
    return sum(1 for signal in signals if signal in normalized)


def _looks_like_instruction_page_layout(path: Path, *, max_side: int) -> bool:
    try:
        from PIL import ImageOps
    except ImportError:
        return False

    with _open_image_without_bomb_warning(path) as original_image:
        prepared = _resize_image_for_ocr(original_image, max_side=min(max_side, 960))
        grayscale = ImageOps.autocontrast(prepared.convert("L"))
        analysis = _resize_image_for_analysis(
            grayscale,
            width=min(420, grayscale.width),
            height=max(1, int(grayscale.height * min(420, grayscale.width) / max(1, grayscale.width))),
        )
        pixels = analysis.load()
        total_pixels = max(1, analysis.width * analysis.height)
        white_pixels = 0
        dark_pixels = 0
        row_ratios: list[float] = []
        for y in range(analysis.height):
            row_dark = 0
            for x in range(analysis.width):
                value = pixels[x, y]
                if value > 245:
                    white_pixels += 1
                if value < 210:
                    dark_pixels += 1
                    row_dark += 1
            row_ratios.append(row_dark / max(1, analysis.width))

    white_ratio = white_pixels / total_pixels
    dark_ratio = dark_pixels / total_pixels
    bands = _group_dense_row_bands(
        row_ratios,
        threshold=0.025,
        max_gap=max(2, analysis.height // 90),
        min_rows=max(3, analysis.height // 70),
    )
    return white_ratio >= 0.55 and 0.01 <= dark_ratio <= 0.35 and len(bands) >= 2


def _generate_paddle_ocr_candidate_images(
    image,
    *,
    fast_mode: bool,
) -> list[Any]:
    try:
        from PIL import ImageOps
    except ImportError:
        return [image.copy()]

    base = image.convert("RGB")
    variants: list[Any] = [base, ImageOps.autocontrast(ImageOps.grayscale(base))]
    if not fast_mode:
        for left, top, right, bottom in _build_ocr_crop_boxes(base.width, base.height, fast_mode=False):
            variants.append(base.crop((left, top, right, bottom)))
    return variants


def _extract_accepted_paddle_lines(
    result: Any,
    *,
    min_score: float,
    min_text_length: int,
    min_meaningful_ratio: float,
) -> list[str]:
    items = _normalize_paddle_ocr_items(result)
    accepted_lines: list[str] = []
    for _, _, text, score in items:
        if score is not None and score < min_score:
            continue
        normalized = _normalize_ocr_token(text)
        if not normalized:
            continue
        if _should_keep_ocr_line(
            normalized,
            tokens=[normalized],
            min_text_length=min_text_length,
            min_meaningful_ratio=min_meaningful_ratio,
        ):
            accepted_lines.append(normalized)
    return accepted_lines


def _normalize_paddle_ocr_items(result: Any) -> list[tuple[float, float, str, float | None]]:
    if result is None:
        return []

    raw_items = result
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
        raw_items = result[0]

    normalized_items: list[tuple[float, float, str, float | None]] = []
    if not isinstance(raw_items, list):
        return normalized_items

    for item in raw_items:
        parsed = _parse_paddle_ocr_item(item)
        if parsed is None:
            continue
        normalized_items.append(parsed)

    normalized_items.sort(key=lambda item: (item[0], item[1]))
    return normalized_items


def _parse_paddle_ocr_item(item: Any) -> tuple[float, float, str, float | None] | None:
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        return None

    box = item[0]
    payload = item[1]
    if not isinstance(payload, (list, tuple)) or not payload:
        return None

    text = str(payload[0]).strip()
    if not text:
        return None

    score: float | None = None
    if len(payload) > 1:
        try:
            score = float(payload[1])
        except (TypeError, ValueError):
            score = None

    center_y, center_x = _paddle_box_center(box)
    return center_y, center_x, text, score


def _paddle_box_center(box: Any) -> tuple[float, float]:
    if not isinstance(box, (list, tuple)) or not box:
        return 0.0, 0.0

    xs: list[float] = []
    ys: list[float] = []
    for point in box:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            xs.append(float(point[0]))
            ys.append(float(point[1]))
        except (TypeError, ValueError):
            continue
    if not xs or not ys:
        return 0.0, 0.0
    return sum(ys) / len(ys), sum(xs) / len(xs)


def _extract_accepted_ocr_lines(
    data: dict[str, Any],
    *,
    min_confidence: float,
    min_text_length: int,
    min_meaningful_ratio: float,
) -> list[str]:
    text_items = data.get("text") or []
    conf_items = data.get("conf") or []
    block_items = data.get("block_num") or []
    par_items = data.get("par_num") or []
    line_items = data.get("line_num") or []

    line_tokens: dict[tuple[int, int, int], list[str]] = {}
    for index, raw_text in enumerate(text_items):
        token = _normalize_ocr_token(raw_text)
        if not token:
            continue

        raw_confidence = conf_items[index] if index < len(conf_items) else None
        confidence = _parse_ocr_confidence(raw_confidence)
        token_min_confidence = _effective_ocr_confidence_threshold(
            token,
            base_min_confidence=min_confidence,
        )
        if confidence is None or confidence < token_min_confidence:
            continue
        if not _is_meaningful_ocr_token(token):
            continue

        key = (
            int(block_items[index] if index < len(block_items) else 1 or 1),
            int(par_items[index] if index < len(par_items) else 1 or 1),
            int(line_items[index] if index < len(line_items) else 1 or 1),
        )
        line_tokens.setdefault(key, []).append(token)

    accepted_lines: list[str] = []
    for tokens in line_tokens.values():
        line_text = _join_ocr_tokens(tokens)
        if _should_keep_ocr_line(
            line_text,
            tokens=tokens,
            min_text_length=min_text_length,
            min_meaningful_ratio=min_meaningful_ratio,
        ):
            accepted_lines.append(line_text)
    return accepted_lines


def _generate_ocr_candidate_images(
    image,
    *,
    fast_mode: bool,
) -> list[Any]:
    try:
        from PIL import Image, ImageFilter, ImageOps
    except ImportError:
        return [image.copy()]

    base = image.convert("RGB")
    variants: list[Any] = [base]

    grayscale = ImageOps.grayscale(base)
    variants.append(ImageOps.autocontrast(grayscale))

    if hasattr(Image, "Resampling"):
        resample = Image.Resampling.LANCZOS
    else:
        resample = Image.LANCZOS
    enlarged = grayscale.resize(
        (max(1, grayscale.width * 2), max(1, grayscale.height * 2)),
        resample=resample,
    )
    variants.append(ImageOps.autocontrast(enlarged))

    if not fast_mode:
        sharpened = ImageOps.autocontrast(grayscale.filter(ImageFilter.SHARPEN))
        variants.append(sharpened)

        thresholded = ImageOps.autocontrast(grayscale).point(
            lambda value: 255 if value > 160 else 0
        )
        variants.append(thresholded)

    crop_boxes = _build_ocr_crop_boxes(base.width, base.height, fast_mode=fast_mode)
    if crop_boxes:
        for left, top, right, bottom in crop_boxes:
            cropped = base.crop((left, top, right, bottom))
            variants.append(cropped)
            cropped_gray = ImageOps.autocontrast(ImageOps.grayscale(cropped))
            variants.append(cropped_gray)

    return variants


def _build_ocr_crop_boxes(
    width: int,
    height: int,
    *,
    fast_mode: bool,
) -> list[tuple[int, int, int, int]]:
    if width < 240 or height < 120:
        return []

    boxes: list[tuple[int, int, int, int]] = []
    top_cut = max(1, int(height * 0.4))
    bottom_start = max(0, int(height * 0.6))
    middle_top = max(0, int(height * 0.2))
    middle_bottom = min(height, int(height * 0.8))

    boxes.append((0, 0, width, top_cut))
    boxes.append((0, middle_top, width, middle_bottom))
    if not fast_mode:
        boxes.append((0, bottom_start, width, height))

    if width >= 320 and not fast_mode:
        left_cut = max(1, int(width * 0.55))
        right_start = max(0, int(width * 0.45))
        boxes.append((0, 0, left_cut, height))
        boxes.append((right_start, 0, width, height))

    deduped: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for box in boxes:
        if box in seen:
            continue
        seen.add(box)
        deduped.append(box)
    return deduped


def _resize_image_for_ocr(
    image,
    *,
    max_side: int,
):
    try:
        from PIL import Image
    except ImportError:
        return image.copy()

    base = image.convert("RGB")
    if max_side <= 0:
        return base

    longest_side = max(base.width, base.height)
    if longest_side <= max_side:
        return base

    scale = max_side / max(1, longest_side)
    if hasattr(Image, "Resampling"):
        resample = Image.Resampling.LANCZOS
    else:
        resample = Image.LANCZOS
    return base.resize(
        (
            max(1, int(base.width * scale)),
            max(1, int(base.height * scale)),
        ),
        resample=resample,
    )


def _should_stop_ocr_early(
    lines: list[str],
    *,
    early_stop_chars: int,
) -> bool:
    if early_stop_chars <= 0 or not lines:
        return False
    compact = re.sub(r"\s+", "", "\n".join(lines))
    return len(compact) >= early_stop_chars


def _should_generate_image_caption(
    *,
    settings: "AppSettings | None",
    ocr_text: str,
) -> tuple[bool, str, int]:
    if settings is None:
        return False, "settings_missing", 0
    if not settings.model.IMAGE_VLM_ENABLED:
        return False, "vlm_disabled", 0

    normalized_ocr_text = _normalize_ocr_text_for_caption(ocr_text)
    ocr_char_count = len(normalized_ocr_text)
    if not settings.model.IMAGE_VLM_AUTO_CAPTION_ENABLED:
        return False, "auto_caption_disabled", ocr_char_count

    if settings.model.IMAGE_VLM_AUTO_TRIGGER_BY_OCR:
        threshold = settings.model.IMAGE_VLM_SKIP_IF_OCR_CHARS_AT_LEAST
        if ocr_char_count >= threshold:
            return False, "ocr_rich_skip_vlm", ocr_char_count
        return True, "ocr_sparse_use_vlm", ocr_char_count

    if settings.model.IMAGE_VLM_ONLY_WHEN_OCR_EMPTY:
        if ocr_char_count == 0:
            return True, "ocr_empty_use_vlm", ocr_char_count
        return False, "ocr_non_empty_skip_vlm", ocr_char_count

    return True, "always_use_vlm", ocr_char_count


def _normalize_ocr_text_for_caption(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _normalize_ocr_token(text: str) -> str:
    return re.sub(r"\s+", "", text or "").strip()


def _parse_ocr_confidence(raw_value: Any) -> float | None:
    try:
        value = float(str(raw_value).strip())
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None
    return value


def _join_ocr_tokens(tokens: list[str]) -> str:
    if not tokens:
        return ""

    parts: list[str] = [tokens[0]]
    for token in tokens[1:]:
        previous_last = parts[-1][-1]
        current_first = token[0]
        if _should_insert_space_between(previous_last, current_first):
            parts.append(f" {token}")
        else:
            parts.append(token)
    return "".join(parts).strip()


def _should_insert_space_between(left_char: str, right_char: str) -> bool:
    return _is_ascii_word_char(left_char) and _is_ascii_word_char(right_char)


def _is_ascii_word_char(char: str) -> bool:
    return char.isascii() and char.isalnum()


def _is_meaningful_ocr_token(token: str) -> bool:
    compact = re.sub(r"\s+", "", token)
    if not compact:
        return False
    meaningful_count = sum(1 for char in compact if _is_meaningful_ocr_char(char))
    return meaningful_count >= max(1, len(compact) // 2)


def _should_keep_ocr_line(
    text: str,
    *,
    tokens: list[str],
    min_text_length: int,
    min_meaningful_ratio: float,
) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return False

    contains_cjk = any(_is_cjk_char(char) for char in compact)
    contains_ascii_letters = any(char.isascii() and char.isalpha() for char in compact)

    effective_min_length = _effective_min_ocr_line_length(
        contains_cjk=contains_cjk,
        contains_ascii_letters=contains_ascii_letters,
        base_min_text_length=min_text_length,
    )
    if len(compact) < effective_min_length:
        return False

    meaningful_count = sum(1 for char in compact if _is_meaningful_ocr_char(char))
    effective_min_ratio = _effective_min_meaningful_ratio(
        contains_cjk=contains_cjk,
        contains_ascii_letters=contains_ascii_letters,
        base_min_meaningful_ratio=min_meaningful_ratio,
    )
    if meaningful_count / len(compact) < effective_min_ratio:
        return False
    if _looks_like_short_mixed_script_noise(tokens):
        return False
    if _looks_like_mixed_script_noise_line(compact):
        return False
    if _looks_like_latin_short_token_noise(tokens):
        return False
    if _looks_like_repeated_symbol_noise(compact):
        return False
    return True


def _looks_like_short_mixed_script_noise(tokens: list[str]) -> bool:
    if len(tokens) < 3:
        return False
    if not all(len(token) <= 3 for token in tokens):
        return False

    has_cjk = any(any(_is_cjk_char(char) for char in token) for token in tokens)
    has_latin = any(any(char.isascii() and char.isalpha() for char in token) for token in tokens)
    return has_cjk and has_latin


def _looks_like_mixed_script_noise_line(text: str) -> bool:
    has_cjk = any(_is_cjk_char(char) for char in text)
    has_latin = any(char.isascii() and char.isalpha() for char in text)
    if not (has_cjk and has_latin):
        return False
    return len(text) < 12


def _looks_like_latin_short_token_noise(tokens: list[str]) -> bool:
    latin_tokens = [
        token for token in tokens if any(char.isascii() and char.isalpha() for char in token)
    ]
    if not latin_tokens:
        return False
    if len(latin_tokens) >= 3 and all(len(token) <= 3 for token in latin_tokens):
        return True
    average_length = sum(len(token) for token in latin_tokens) / len(latin_tokens)
    return average_length < 2.2


def _looks_like_repeated_symbol_noise(text: str) -> bool:
    if not text:
        return True
    symbol_count = sum(1 for char in text if not _is_meaningful_ocr_char(char))
    return symbol_count > 0 and symbol_count / len(text) > 0.35


def _effective_ocr_confidence_threshold(
    token: str,
    *,
    base_min_confidence: float,
) -> float:
    compact = re.sub(r"\s+", "", token)
    if not compact:
        return base_min_confidence

    has_cjk = any(_is_cjk_char(char) for char in compact)
    has_latin = any(char.isascii() and char.isalpha() for char in compact)
    if has_cjk:
        if len(compact) >= 2:
            return max(40.0, base_min_confidence - 20.0)
        return max(50.0, base_min_confidence - 10.0)
    if has_latin and len(compact) <= 2:
        return min(95.0, base_min_confidence + 15.0)
    if has_latin and len(compact) <= 4:
        return min(90.0, base_min_confidence + 5.0)
    return base_min_confidence


def _effective_min_ocr_line_length(
    *,
    contains_cjk: bool,
    contains_ascii_letters: bool,
    base_min_text_length: int,
) -> int:
    if contains_cjk:
        return max(4, base_min_text_length - 2)
    if contains_ascii_letters:
        return max(8, base_min_text_length + 2)
    return base_min_text_length


def _effective_min_meaningful_ratio(
    *,
    contains_cjk: bool,
    contains_ascii_letters: bool,
    base_min_meaningful_ratio: float,
) -> float:
    if contains_cjk:
        return min(base_min_meaningful_ratio, 0.5)
    if contains_ascii_letters:
        return max(base_min_meaningful_ratio, 0.75)
    return base_min_meaningful_ratio


def _is_meaningful_ocr_char(char: str) -> bool:
    if _is_cjk_char(char):
        return True
    if char.isdigit():
        return True
    if char.isascii() and char.isalpha():
        return True

    category = unicodedata.category(char)
    return category.startswith("L") or category == "Nd"


def _is_cjk_char(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x4E00 <= codepoint <= 0x9FFF
        or 0x3400 <= codepoint <= 0x4DBF
        or 0xF900 <= codepoint <= 0xFAFF
    )


def _build_image_region_documents(
    *,
    path: Path,
    image_title: str,
    base_metadata: dict[str, str],
    relative_path: str,
    settings: "AppSettings | None",
    caption_region_fn: Any,
    format_caption_fn: Any,
) -> tuple[list[str], list[Document]]:
    if settings is None or not settings.model.IMAGE_VLM_REGION_CAPTION_ENABLED:
        return [], []

    try:
        from PIL import Image
    except ImportError:
        return [], []

    region_texts: list[str] = []
    region_documents: list[Document] = []
    with _open_image_without_bomb_warning(path) as original_image:
        base_image = _resize_image_for_region_caption(
            original_image,
            settings=settings,
        )
        region_specs = _build_image_region_specs(
            width=base_image.width,
            height=base_image.height,
            max_regions=settings.model.IMAGE_VLM_REGION_MAX_REGIONS,
            min_side=settings.model.IMAGE_VLM_REGION_MIN_SIDE,
        )
        for offset, region_spec in enumerate(region_specs, start=1):
            region_bytes = _render_image_region_bytes(base_image, region_spec.box)
            caption_error = ""
            caption_filtered = False
            raw_caption_text = ""
            caption_text = ""
            try:
                structured_caption = caption_region_fn(
                    settings,
                    image_bytes=region_bytes,
                    mime_type="image/png",
                    ocr_text="",
                    region_label=region_spec.title,
                )
                raw_caption_text = structured_caption.raw_text.strip()
                formatted_caption = format_caption_fn(
                    structured_caption,
                    include_region_label=True,
                )
                if _is_low_quality_image_caption(
                    formatted_caption,
                    raw_text=raw_caption_text,
                ):
                    caption_filtered = True
                    caption_error = "low_quality_caption_filtered"
                else:
                    caption_text = formatted_caption
            except Exception as exc:
                caption_error = str(exc)

            _append_image_caption_trace(
                settings=settings,
                path=path,
                relative_path=relative_path,
                source_modality="vision",
                ocr_text="",
                ocr_error="",
                ocr_char_count=0,
                should_caption=True,
                image_caption_reason="region_caption",
                raw_caption_text=raw_caption_text,
                caption_text=caption_text,
                caption_error=caption_error,
                caption_filtered=caption_filtered,
                caption_scope="region",
                region_label=region_spec.title,
            )

            if not caption_text:
                continue

            region_texts.append(caption_text)
            region_documents.append(
                Document(
                    page_content=_build_region_image_knowledge_text(
                        path=path,
                        region_title=region_spec.title,
                        caption_text=caption_text,
                    ),
                    metadata={
                        **base_metadata,
                        "doc_id": f"{relative_path}#region-{region_spec.key}",
                        "section_title": f"{image_title} {region_spec.title}",
                        "section_path": f"{image_title} > {region_spec.title}",
                        "section_index": offset,
                        "content_type": "image_region_evidence",
                        "source_modality": "vision",
                        "ocr_text": None,
                        "image_caption": caption_text,
                        "evidence_summary": _build_image_evidence_summary(
                            image_title=f"{image_title} {region_spec.title}",
                            ocr_text="",
                            caption_text=caption_text,
                            region_caption_texts=[],
                        ),
                    },
                )
            )

    return region_texts, region_documents


def _build_image_region_specs(
    *,
    width: int,
    height: int,
    max_regions: int,
    min_side: int,
) -> list[ImageRegionSpec]:
    if max_regions <= 0 or min(width, height) < max(1, min_side):
        return []

    specs = [
        ImageRegionSpec(
            key="top",
            title="上半区",
            box=(0, 0, width, max(1, int(height * 0.45))),
        ),
        ImageRegionSpec(
            key="middle",
            title="中部区域",
            box=(0, max(0, int(height * 0.2)), width, min(height, int(height * 0.8))),
        ),
        ImageRegionSpec(
            key="bottom",
            title="下半区",
            box=(0, max(0, int(height * 0.55)), width, height),
        ),
    ]
    return specs[:max_regions]


def _render_image_region_bytes(
    image,
    box: tuple[int, int, int, int],
) -> bytes:
    cropped = image.crop(box)
    buffer = BytesIO()
    cropped.save(buffer, format="PNG")
    return buffer.getvalue()


@contextmanager
def _open_image_without_bomb_warning(path: Path):
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "图片处理需要安装 `Pillow`。"
            "请在当前虚拟环境中执行 `pip install Pillow` 或 `pip install -r requirements.txt`。"
        ) from exc

    with warnings.catch_warnings():
        bomb_warning = getattr(Image, "DecompressionBombWarning", None)
        if bomb_warning is not None:
            warnings.simplefilter("ignore", bomb_warning)
        with Image.open(path) as original_image:
            yield original_image


def _resize_image_for_region_caption(
    image,
    *,
    settings: "AppSettings | None",
):
    max_side = _resolve_safe_region_max_side(settings)
    return _resize_image_for_ocr(image, max_side=max_side)


def _resolve_safe_region_max_side(settings: "AppSettings | None") -> int:
    if settings is None:
        return 1600

    candidates = [
        settings.kb.IMAGE_OCR_MAX_SIDE,
        settings.model.IMAGE_VLM_MAX_SIDE,
    ]
    valid_candidates = [value for value in candidates if value > 0]
    if not valid_candidates:
        return 0
    return max(valid_candidates)


def _looks_like_instruction_page(ocr_text: str) -> bool:
    lines = _normalize_instruction_lines(ocr_text)
    if len(lines) < 3:
        return False

    step_count = sum(
        1
        for line in lines
        if _extract_instruction_step(line) is not None
    )
    list_count = sum(1 for line in lines if INSTRUCTION_LIST_ITEM_PATTERN.match(line))
    note_count = sum(1 for line in lines if INSTRUCTION_NOTE_PATTERN.match(line))
    keyword_hits = sum(1 for keyword in INSTRUCTION_KEYWORDS if keyword in ocr_text)

    if step_count >= 2 and keyword_hits >= 2:
        return True
    if step_count >= 1 and keyword_hits >= 4:
        return True
    if step_count >= 1 and keyword_hits >= 3 and any("件号" in line or "n·m" in line.lower() or "nm" in line.lower() for line in lines):
        return True
    if step_count >= 1 and list_count >= 2 and note_count >= 1:
        return True
    return False


def _normalize_instruction_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in (text or "").splitlines():
        normalized = re.sub(r"\s+", " ", raw_line).strip()
        if not normalized:
            continue
        lines.append(normalized)
    return lines


def _build_instruction_page_documents(
    *,
    path: Path,
    image_title: str,
    base_metadata: dict[str, str],
    relative_path: str,
    settings: "AppSettings | None",
    ocr_text: str,
    should_caption: bool,
    caption_region_fn: Any | None,
    format_caption_fn: Any | None,
) -> list[Document]:
    parsed = _parse_instruction_page_content(ocr_text, fallback_title=image_title)
    if not parsed.steps and not parsed.parts_list:
        return []

    documents: list[Document] = []
    next_section_index = 100

    instruction_text = _build_instruction_text_content(parsed)
    if instruction_text:
        documents.append(
            Document(
                page_content=instruction_text,
                metadata={
                    **base_metadata,
                    "doc_id": f"{relative_path}#instruction-steps",
                    "section_title": f"{parsed.title} 操作步骤",
                    "section_path": f"{image_title} > 操作步骤",
                    "section_index": next_section_index,
                    "content_type": "instruction_text_evidence",
                    "source_modality": "ocr",
                    "ocr_text": instruction_text,
                    "image_caption": None,
                    "evidence_summary": _build_instruction_evidence_summary(
                        title=parsed.title,
                        items=parsed.steps or parsed.note_lines,
                        prefix="步骤",
                    ),
                },
            )
        )
        next_section_index += 1

    parts_text = _build_instruction_parts_content(parsed)
    if parts_text:
        documents.append(
            Document(
                page_content=parts_text,
                metadata={
                    **base_metadata,
                    "doc_id": f"{relative_path}#instruction-parts",
                    "section_title": f"{parsed.title} 零件清单",
                    "section_path": f"{image_title} > 零件清单",
                    "section_index": next_section_index,
                    "content_type": "instruction_parts_evidence",
                    "source_modality": "ocr",
                    "ocr_text": parts_text,
                    "image_caption": None,
                    "evidence_summary": _build_instruction_evidence_summary(
                        title=parsed.title,
                        items=parsed.parts_list,
                        prefix="零件",
                    ),
                },
            )
        )
        next_section_index += 1

    if should_caption and caption_region_fn is not None and format_caption_fn is not None:
        documents.extend(
            _build_instruction_figure_documents(
                path=path,
                image_title=image_title,
                page_title=parsed.title,
                base_metadata=base_metadata,
                relative_path=relative_path,
                settings=settings,
                start_section_index=next_section_index,
                caption_region_fn=caption_region_fn,
                format_caption_fn=format_caption_fn,
            )
        )

    return documents


def _parse_instruction_page_content(
    ocr_text: str,
    *,
    fallback_title: str,
) -> InstructionPageParseResult:
    lines = _normalize_instruction_lines(ocr_text)
    title = _extract_instruction_title(lines, fallback_title=fallback_title)

    steps: list[str] = []
    note_lines: list[str] = []
    parts_list: list[str] = []
    current_step_parts: list[str] = []
    list_mode = False

    def flush_step() -> None:
        nonlocal current_step_parts
        if not current_step_parts:
            return
        steps.append(" ".join(part.strip() for part in current_step_parts if part.strip()).strip())
        current_step_parts = []

    for line in lines:
        if line == title:
            continue

        note_match = INSTRUCTION_NOTE_PATTERN.match(line)
        if note_match:
            flush_step()
            list_mode = False
            prefix = note_match.group(1).strip()
            content = note_match.group(2).strip()
            note_lines.append(f"{prefix}: {content}" if content else prefix)
            continue

        step_match = _extract_instruction_step(line)
        if step_match is not None:
            flush_step()
            list_mode = False
            step_index, step_body = step_match
            current_step_parts = [f"{step_index}. {step_body}"]
            continue

        if _looks_like_parts_intro_line(line):
            flush_step()
            list_mode = True
            continue

        list_item_match = INSTRUCTION_LIST_ITEM_PATTERN.match(line)
        if list_item_match:
            item = list_item_match.group(1).strip()
            if not item:
                continue
            if list_mode or not current_step_parts:
                parts_list.append(item)
            else:
                current_step_parts.append(f"- {item}")
            continue

        if current_step_parts:
            current_step_parts.append(line)
            continue

        if list_mode:
            parts_list.append(line)

    flush_step()
    return InstructionPageParseResult(
        title=title,
        steps=tuple(_dedupe_preserve_order(steps)),
        note_lines=tuple(_dedupe_preserve_order(note_lines)),
        parts_list=tuple(_dedupe_preserve_order(parts_list)),
    )


def _extract_instruction_title(lines: list[str], *, fallback_title: str) -> str:
    for line in lines[:4]:
        if INSTRUCTION_TITLE_PATTERN.match(line):
            return line
    for line in lines[:4]:
        if len(line) <= 24 and any(keyword in line for keyword in INSTRUCTION_KEYWORDS):
            return line
    return fallback_title


def _extract_instruction_step(line: str) -> tuple[str, str] | None:
    match = INSTRUCTION_STEP_PATTERN.match(line)
    if match is None:
        return None
    step_index = match.group(1).strip()
    step_body = match.group(2).strip()
    if not step_body:
        return None
    if not _looks_like_instruction_step_body(step_body):
        return None
    return step_index, step_body


def _looks_like_instruction_step_body(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return False
    if any(keyword in compact for keyword in INSTRUCTION_KEYWORDS):
        return True
    has_cjk = any(_is_cjk_char(char) for char in compact)
    if has_cjk and len(compact) >= 4:
        return True
    return False


def _looks_like_parts_intro_line(line: str) -> bool:
    normalized = line.strip()
    if not normalized:
        return False
    intro_keywords = ("依次拆下", "零件如下", "包括以下", "所需零件", "零件清单", "拆下以下")
    if any(keyword in normalized for keyword in intro_keywords):
        return True
    return normalized.endswith(("：", ":")) and "零件" in normalized


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _build_instruction_text_content(parsed: InstructionPageParseResult) -> str:
    sections: list[str] = [f"说明书页面: {parsed.title}"]
    if parsed.steps:
        sections.append(
            "[操作步骤]\n"
            + "\n".join(f"{index}. {step}" for index, step in enumerate(parsed.steps, start=1))
        )
    if parsed.note_lines:
        sections.append("[提示说明]\n" + "\n".join(parsed.note_lines))
    return "\n\n".join(section for section in sections if section.strip()).strip()


def _build_instruction_parts_content(parsed: InstructionPageParseResult) -> str:
    if not parsed.parts_list:
        return ""
    return "\n\n".join(
        [
            f"说明书页面: {parsed.title}",
            "[零件清单]\n" + "\n".join(f"- {item}" for item in parsed.parts_list),
        ]
    ).strip()


def _build_instruction_evidence_summary(
    *,
    title: str,
    items: tuple[str, ...],
    prefix: str,
) -> str:
    compact = "；".join(item.strip() for item in items[:3] if item.strip())
    if compact:
        compact = compact[:120]
    return " | ".join(part for part in (title, f"{prefix}: {compact}" if compact else "") if part)


def _build_instruction_figure_documents(
    *,
    path: Path,
    image_title: str,
    page_title: str,
    base_metadata: dict[str, str],
    relative_path: str,
    settings: "AppSettings | None",
    start_section_index: int,
    caption_region_fn: Any,
    format_caption_fn: Any,
) -> list[Document]:
    if settings is None:
        return []

    try:
        from PIL import Image
    except ImportError:
        return []

    documents: list[Document] = []
    section_index = start_section_index
    with _open_image_without_bomb_warning(path) as original_image:
        base_image = _resize_image_for_region_caption(original_image, settings=settings)
        figure_specs = _build_instruction_figure_specs(base_image)
        for figure_spec in figure_specs:
            figure_bytes = _render_image_region_bytes(base_image, figure_spec.box)
            figure_doc = _build_instruction_visual_document(
                path=path,
                image_title=image_title,
                page_title=page_title,
                base_metadata=base_metadata,
                relative_path=relative_path,
                settings=settings,
                section_index=section_index,
                section_title=figure_spec.title,
                box=figure_spec.box,
                image_bytes=figure_bytes,
                content_type="instruction_figure_evidence",
                caption_region_fn=caption_region_fn,
                format_caption_fn=format_caption_fn,
                task_type="instruction_figure",
                caption_scope="instruction_figure",
            )
            if figure_doc is not None:
                documents.append(figure_doc)
                section_index += 1

            arrow_box = _find_red_arrow_focus_box(base_image, figure_spec.box)
            if arrow_box is None:
                continue
            arrow_bytes = _render_image_region_bytes(base_image, arrow_box)
            arrow_title = f"{figure_spec.title} 箭头局部"
            arrow_doc = _build_instruction_visual_document(
                path=path,
                image_title=image_title,
                page_title=page_title,
                base_metadata=base_metadata,
                relative_path=relative_path,
                settings=settings,
                section_index=section_index,
                section_title=arrow_title,
                box=arrow_box,
                image_bytes=arrow_bytes,
                content_type="instruction_arrow_evidence",
                caption_region_fn=caption_region_fn,
                format_caption_fn=format_caption_fn,
                task_type="arrow_focus",
                caption_scope="arrow_focus",
            )
            if arrow_doc is not None:
                documents.append(arrow_doc)
                section_index += 1

    return documents


def _build_instruction_visual_document(
    *,
    path: Path,
    image_title: str,
    page_title: str,
    base_metadata: dict[str, str],
    relative_path: str,
    settings: "AppSettings",
    section_index: int,
    section_title: str,
    box: tuple[int, int, int, int],
    image_bytes: bytes,
    content_type: str,
    caption_region_fn: Any,
    format_caption_fn: Any,
    task_type: str,
    caption_scope: str,
) -> Document | None:
    caption_text = ""
    raw_caption_text = ""
    caption_error = ""
    caption_filtered = False
    try:
        structured_caption = caption_region_fn(
            settings,
            image_bytes=image_bytes,
            mime_type="image/png",
            ocr_text="",
            region_label=section_title,
            task_type=task_type,
        )
        raw_caption_text = structured_caption.raw_text.strip()
        formatted_caption = format_caption_fn(
            structured_caption,
            include_region_label=True,
        )
        if _is_low_quality_image_caption(
            formatted_caption,
            raw_text=raw_caption_text,
        ):
            caption_filtered = True
            caption_error = "low_quality_caption_filtered"
        else:
            caption_text = formatted_caption
    except Exception as exc:
        caption_error = str(exc)

    _append_image_caption_trace(
        settings=settings,
        path=path,
        relative_path=relative_path,
        source_modality="vision",
        ocr_text="",
        ocr_error="",
        ocr_char_count=0,
        should_caption=True,
        image_caption_reason=task_type,
        raw_caption_text=raw_caption_text,
        caption_text=caption_text,
        caption_error=caption_error,
        caption_filtered=caption_filtered,
        caption_scope=caption_scope,
        region_label=section_title,
    )
    if not caption_text:
        return None

    return Document(
        page_content=_build_instruction_visual_text(
            path=path,
            page_title=page_title,
            section_title=section_title,
            caption_text=caption_text,
        ),
        metadata={
            **base_metadata,
            "doc_id": f"{relative_path}#{_slugify_instruction_section(section_title)}",
            "section_title": f"{page_title} {section_title}",
            "section_path": f"{image_title} > {page_title} > {section_title}",
            "section_index": section_index,
            "content_type": content_type,
            "source_modality": "vision",
            "ocr_text": None,
            "image_caption": caption_text,
            "evidence_summary": _build_image_evidence_summary(
                image_title=f"{page_title} {section_title}",
                ocr_text="",
                caption_text=caption_text,
                region_caption_texts=[],
            ),
            "region_box": ",".join(str(value) for value in box),
        },
    )


def _build_instruction_visual_text(
    *,
    path: Path,
    page_title: str,
    section_title: str,
    caption_text: str,
) -> str:
    return "\n\n".join(
        [
            f"图片文件: {path.name}",
            f"说明书页面: {page_title}",
            f"分析区域: {section_title}",
            f"[说明书配图描述]\n{caption_text}",
        ]
    ).strip()


def _slugify_instruction_section(title: str) -> str:
    normalized = re.sub(r"\s+", "-", title.strip().lower())
    normalized = re.sub(r"[^0-9a-z\u4e00-\u9fff\-]+", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    return normalized or "instruction-figure"


def _build_instruction_figure_specs(image) -> list[FigureBandSpec]:
    analysis_width = min(400, max(200, image.width))
    if image.width <= 0 or image.height <= 0:
        return []

    scale = analysis_width / max(1, image.width)
    analysis_height = max(1, int(image.height * scale))
    if analysis_height < 80:
        return []

    analysis_image = _resize_image_for_analysis(image, width=analysis_width, height=analysis_height)
    grayscale = analysis_image.convert("L")
    pixels = grayscale.load()
    row_ratios: list[float] = []
    for y in range(grayscale.height):
        dense_count = 0
        for x in range(grayscale.width):
            if pixels[x, y] < 245:
                dense_count += 1
        row_ratios.append(dense_count / max(1, grayscale.width))

    bands = _group_dense_row_bands(
        row_ratios,
        threshold=0.03,
        max_gap=max(3, analysis_height // 70),
        min_rows=max(16, analysis_height // 35),
    )

    figure_specs: list[FigureBandSpec] = []
    scale_back = image.width / max(1, analysis_width)
    for index, (start_row, end_row) in enumerate(bands, start=1):
        mean_ratio = sum(row_ratios[start_row : end_row + 1]) / max(1, end_row - start_row + 1)
        top = max(0, int(start_row * scale_back) - 18)
        bottom = min(image.height, int((end_row + 1) * scale_back) + 18)
        height = bottom - top
        if height < max(120, int(image.height * 0.12)):
            continue
        if mean_ratio < 0.08:
            continue

        box = _trim_white_margin_box(image, (0, top, image.width, bottom), padding=18)
        if box is None:
            continue
        if box[3] - box[1] < max(120, int(image.height * 0.1)):
            continue
        figure_specs.append(
            FigureBandSpec(
                key=f"figure-{index}",
                title=f"配图{len(figure_specs) + 1}",
                box=box,
            )
        )
        if len(figure_specs) >= 3:
            break

    return figure_specs


def _resize_image_for_analysis(image, *, width: int, height: int):
    try:
        from PIL import Image
    except ImportError:
        return image

    if hasattr(Image, "Resampling"):
        resample = Image.Resampling.BILINEAR
    else:
        resample = Image.BILINEAR
    return image.resize((width, height), resample=resample)


def _group_dense_row_bands(
    row_ratios: list[float],
    *,
    threshold: float,
    max_gap: int,
    min_rows: int,
) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start: int | None = None
    last_dense: int | None = None
    for index, ratio in enumerate(row_ratios):
        if ratio >= threshold:
            if start is None:
                start = index
            last_dense = index
            continue
        if start is None or last_dense is None:
            continue
        if index - last_dense <= max_gap:
            continue
        if last_dense - start + 1 >= min_rows:
            bands.append((start, last_dense))
        start = None
        last_dense = None
    if start is not None and last_dense is not None and last_dense - start + 1 >= min_rows:
        bands.append((start, last_dense))
    return bands


def _trim_white_margin_box(
    image,
    box: tuple[int, int, int, int],
    *,
    padding: int,
) -> tuple[int, int, int, int] | None:
    cropped = image.crop(box).convert("L")
    pixels = cropped.load()
    left = cropped.width
    top = cropped.height
    right = -1
    bottom = -1
    for y in range(cropped.height):
        for x in range(cropped.width):
            if pixels[x, y] >= 245:
                continue
            if x < left:
                left = x
            if y < top:
                top = y
            if x > right:
                right = x
            if y > bottom:
                bottom = y
    if right < left or bottom < top:
        return None

    absolute_left = max(box[0], box[0] + left - padding)
    absolute_top = max(box[1], box[1] + top - padding)
    absolute_right = min(box[2], box[0] + right + 1 + padding)
    absolute_bottom = min(box[3], box[1] + bottom + 1 + padding)
    return (absolute_left, absolute_top, absolute_right, absolute_bottom)


def _find_red_arrow_focus_box(
    image,
    figure_box: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    cropped = image.crop(figure_box).convert("RGB")
    pixels = cropped.load()
    left = cropped.width
    top = cropped.height
    right = -1
    bottom = -1
    red_count = 0
    for y in range(cropped.height):
        for x in range(cropped.width):
            red, green, blue = pixels[x, y]
            if red < 150 or red - green < 60 or red - blue < 60:
                continue
            if green > 140 or blue > 140:
                continue
            red_count += 1
            if x < left:
                left = x
            if y < top:
                top = y
            if x > right:
                right = x
            if y > bottom:
                bottom = y
    if red_count < 20 or right < left or bottom < top:
        return None

    expand_x = max(80, (right - left + 1) * 3)
    expand_y = max(70, (bottom - top + 1) * 3)
    absolute_box = (
        max(figure_box[0], figure_box[0] + left - expand_x),
        max(figure_box[1], figure_box[1] + top - expand_y),
        min(figure_box[2], figure_box[0] + right + 1 + expand_x),
        min(figure_box[3], figure_box[1] + bottom + 1 + expand_y),
    )
    return _trim_white_margin_box(image, absolute_box, padding=12) or absolute_box


def _build_region_image_knowledge_text(
    *,
    path: Path,
    region_title: str,
    caption_text: str,
) -> str:
    return "\n\n".join(
        [
            f"图片文件: {path.name}",
            f"分析区域: {region_title}",
            f"[区域视觉描述]\n{caption_text}",
        ]
    )


def _build_image_knowledge_text(
    *,
    path: Path,
    ocr_text: str,
    caption_text: str,
    region_caption_texts: list[str],
    ocr_enabled: bool,
    ocr_error: str,
    caption_enabled: bool,
    caption_error: str,
) -> str:
    sections: list[str] = [f"图片文件: {path.name}"]

    if ocr_text:
        sections.append(f"[图片文字 OCR]\n{ocr_text}")
    elif ocr_enabled and ocr_error:
        sections.append(f"[图片文字 OCR 失败]\n{ocr_error}")
    elif ocr_enabled:
        sections.append("[图片文字 OCR]\n未识别到明显文字。")
    else:
        sections.append("[图片文字 OCR]\n已关闭。")

    if caption_text:
        sections.append(f"[图片结构化描述]\n{caption_text}")
    elif caption_enabled and caption_error:
        sections.append(f"[图片内容描述生成失败]\n{caption_error}")
    elif caption_enabled:
        sections.append("[图片内容描述]\n未生成有效描述。")

    if region_caption_texts:
        sections.append(
            "[图片区域描述]\n"
            + "\n\n".join(
                f"[区域 {index}] {text}"
                for index, text in enumerate(region_caption_texts, start=1)
            )
        )

    return "\n\n".join(section.strip() for section in sections if section.strip())


def _resolve_image_source_modality(
    *,
    ocr_text: str,
    has_visual_caption: bool,
) -> str:
    if ocr_text and has_visual_caption:
        return "ocr+vision"
    if ocr_text:
        return "ocr"
    if has_visual_caption:
        return "vision"
    return "image"


def _build_image_evidence_summary(
    *,
    image_title: str,
    ocr_text: str,
    caption_text: str,
    region_caption_texts: list[str],
) -> str:
    parts = [image_title]
    if ocr_text:
        compact_ocr = re.sub(r"\s+", " ", ocr_text).strip()
        parts.append(f"OCR: {compact_ocr[:80]}")
    if caption_text:
        compact_caption = re.sub(r"\s+", " ", caption_text).strip()
        parts.append(f"Vision: {compact_caption[:80]}")
    elif region_caption_texts:
        compact_region = re.sub(r"\s+", " ", region_caption_texts[0]).strip()
        parts.append(f"VisionRegion: {compact_region[:80]}")
    return " | ".join(part for part in parts if part).strip()


def _is_low_quality_image_caption(
    text: str,
    *,
    raw_text: str = "",
) -> bool:
    normalized = _normalize_caption_quality_text(text, raw_text=raw_text)
    if not normalized:
        return True

    blocked_patterns = (
        "无法为您提供相关服务",
        "无法为你提供相关服务",
        "我无法为您提供相关服务",
        "我无法为你提供相关服务",
        "不符合公序良俗和相关规范要求",
        "不符合相关规范要求",
        "违反公序良俗和相关规范",
        "请你提供合规健康的内容",
        "请提供合规健康的内容",
        "请提供合规内容",
        "请你提供合规的正常内容",
        "请提供合规的正常内容",
        "内容包含低俗违规信息",
        "内容包含低俗淫秽信息",
        "涉及违规内容",
        "无法满足该请求",
        "请尝试提供其他话题",
        "我才能为你进行相关处理",
    )
    if any(pattern in normalized for pattern in blocked_patterns):
        return True

    if len(normalized) < 8:
        return True
    return False


def _normalize_caption_quality_text(
    text: str,
    *,
    raw_text: str = "",
) -> str:
    parts = [
        re.sub(r"\s+", "", part or "").lower()
        for part in (text, raw_text)
        if part and part.strip()
    ]
    return " ".join(parts).strip()


def _append_image_caption_trace(
    *,
    settings: "AppSettings | None",
    path: Path,
    relative_path: str,
    source_modality: str,
    ocr_text: str,
    ocr_error: str,
    ocr_char_count: int,
    should_caption: bool,
    image_caption_reason: str,
    raw_caption_text: str,
    caption_text: str,
    caption_error: str,
    caption_filtered: bool,
    caption_scope: str,
    region_label: str = "",
) -> None:
    if settings is None:
        return

    append_jsonl_trace(
        settings,
        "image_caption_trace",
        {
            "source": path.name,
            "relative_path": relative_path,
            "source_path": str(path.resolve()),
            "source_modality": source_modality,
            "caption_scope": caption_scope,
            "region_label": region_label,
            "ocr_char_count": ocr_char_count,
            "ocr_text_preview": _truncate_trace_text(ocr_text, limit=240),
            "ocr_error": ocr_error,
            "caption_attempted": should_caption,
            "caption_reason": image_caption_reason,
            "caption_filtered": caption_filtered,
            "raw_caption_text": _truncate_trace_text(raw_caption_text, limit=400),
            "kept_caption_text": _truncate_trace_text(caption_text, limit=400),
            "caption_error": caption_error,
        },
    )


def _truncate_trace_text(text: str, *, limit: int) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit]}..."

