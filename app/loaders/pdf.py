"""加载 PDF 文档并提取文本与页面级内容。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from langchain_core.documents import Document
from pypdf import PdfReader

from .factory import BaseKnowledge


@dataclass
class PdfOutlineSection:
    title: str
    path: str
    level: int
    page_number: int
    has_children: bool = False


class PdfKnowledge(BaseKnowledge):
    supported_extensions = (".pdf",)

    def load(self) -> list[Document]:
        return _load_pdf(self.path, self.base_metadata, self.relative_path)


def _load_pdf(path: Path, base_metadata: dict[str, str], relative_path: str) -> list[Document]:
    reader = PdfReader(str(path))
    outlined_documents = _load_pdf_outline_sections(reader, base_metadata, relative_path)
    if outlined_documents:
        return outlined_documents

    documents: list[Document] = []

    for index, page in enumerate(reader.pages, start=1):
        text = _clean_pdf_page_text(page.extract_text() or "", base_metadata)
        if not text.strip() or _looks_like_pdf_toc_page(text):
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    **base_metadata,
                    "doc_id": f"{relative_path}#page-{index}",
                    "page": index,
                    "page_end": index,
                    "section_title": f"{base_metadata.get('title', path.stem)} 第 {index} 页",
                    "section_path": f"{base_metadata.get('title', path.stem)} > 第 {index} 页",
                    "section_index": index - 1,
                },
            )
        )

    return documents


def _load_pdf_outline_sections(
    reader: PdfReader,
    base_metadata: dict[str, str],
    relative_path: str,
) -> list[Document]:
    outline_root = getattr(reader, "outline", None)
    if not outline_root:
        return []

    sections = _flatten_pdf_outline(reader, outline_root)
    if not sections:
        return []

    total_pages = len(reader.pages)
    documents: list[Document] = []
    for index, section in enumerate(sections):
        next_section = sections[index + 1] if index + 1 < len(sections) else None
        next_section_page = _find_next_distinct_section_page(sections, index)
        if section.has_children and next_section_page == section.page_number:
            continue

        start_page = max(1, min(section.page_number, total_pages))
        end_page = total_pages
        if next_section_page is not None and next_section_page > start_page:
            end_page = min(total_pages, next_section_page - 1)
        elif next_section_page is not None and next_section_page <= start_page:
            end_page = start_page

        page_texts: list[str] = []
        for page_number in range(start_page, end_page + 1):
            text = _clean_pdf_page_text(
                reader.pages[page_number - 1].extract_text() or "",
                base_metadata,
            )
            normalized = text.strip()
            if normalized:
                page_texts.append(normalized)
        if not page_texts:
            continue

        combined_text = "\n\n".join(page_texts)
        combined_text = _slice_outline_section_text(
            combined_text,
            current_title=section.title,
            next_title=(
                next_section.title
                if next_section is not None and next_section.page_number == section.page_number
                else None
            ),
        )
        if _looks_like_pdf_toc_page(combined_text):
            continue
        if not combined_text.strip():
            continue

        documents.append(
            Document(
                page_content=combined_text,
                metadata={
                    **base_metadata,
                    "doc_id": f"{relative_path}#section-{len(documents):04d}",
                    "page": start_page,
                    "page_end": end_page,
                    "title": section.title,
                    "section_title": section.title,
                    "section_path": section.path,
                    "section_index": len(documents),
                },
            )
        )

    return _compact_outline_documents(documents)


def _flatten_pdf_outline(
    reader: PdfReader,
    items: list[Any],
    *,
    level: int = 1,
    parent_path: list[str] | None = None,
) -> list[PdfOutlineSection]:
    base_path = list(parent_path or [])
    flattened: list[PdfOutlineSection] = []
    last_section: PdfOutlineSection | None = None

    for item in items:
        if isinstance(item, list):
            child_parent = base_path
            if last_section is not None:
                last_section.has_children = True
                child_parent = last_section.path.split(" > ")
            flattened.extend(
                _flatten_pdf_outline(
                    reader,
                    item,
                    level=level + 1,
                    parent_path=child_parent,
                )
            )
            continue

        title = _extract_pdf_outline_title(item)
        if not title:
            continue
        try:
            page_number = int(reader.get_destination_page_number(item)) + 1
        except Exception:
            continue

        path_parts = [part for part in (*base_path, title) if part]
        section = PdfOutlineSection(
            title=title,
            path=" > ".join(path_parts),
            level=level,
            page_number=page_number,
        )
        flattened.append(section)
        last_section = section

    return flattened


def _extract_pdf_outline_title(item: Any) -> str:
    for attr_name in ("title", "/Title"):
        value = getattr(item, attr_name, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(item, dict):
            dict_value = item.get(attr_name)
            if isinstance(dict_value, str) and dict_value.strip():
                return dict_value.strip()
    return ""


def _find_next_distinct_section_page(
    sections: list[PdfOutlineSection],
    current_index: int,
) -> int | None:
    current_page = sections[current_index].page_number
    for section in sections[current_index + 1 :]:
        if section.page_number != current_page:
            return section.page_number
    return None


PDF_PAGE_NUMBER_PATTERN = re.compile(r"^\s*No\.\s*\d+\s*/\s*\d+\s*$", re.IGNORECASE)
PDF_SHORT_OUTLINE_DOC_LENGTH = 80


def _clean_pdf_page_text(text: str, base_metadata: dict[str, str]) -> str:
    title = str(base_metadata.get("title", "")).strip()
    lines = [line.strip() for line in text.splitlines()]
    cleaned: list[str] = []
    for line in lines:
        if not line:
            if cleaned and cleaned[-1]:
                cleaned.append("")
            continue
        if PDF_PAGE_NUMBER_PATTERN.match(line):
            continue
        if title and line == title:
            continue
        cleaned.append(line)
    while cleaned and not cleaned[-1]:
        cleaned.pop()
    return "\n".join(cleaned).strip()


def _looks_like_pdf_toc_page(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 8:
        return False

    numbered = 0
    for line in lines:
        if re.match(r"^(?:[一二三四五六七八九十]+、|\d+\.\d+)", line):
            numbered += 1
    return numbered >= max(6, int(len(lines) * 0.5))


def _slice_outline_section_text(
    text: str,
    *,
    current_title: str,
    next_title: str | None,
) -> str:
    start_offset = _find_section_heading_offset(text, current_title)
    if start_offset is not None:
        text = text[start_offset:].lstrip()

    if next_title:
        end_offset = _find_section_heading_offset(text, next_title)
        if end_offset is not None and end_offset > 0:
            text = text[:end_offset].rstrip()
    return text.strip()


def _find_section_heading_offset(text: str, heading: str) -> int | None:
    normalized_heading = _normalize_heading_text(heading)
    if not normalized_heading:
        return None

    exact_offset = text.find(heading)
    if exact_offset >= 0:
        return exact_offset

    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        normalized_line = _normalize_heading_text(stripped)
        if normalized_line and (
            normalized_line == normalized_heading
            or normalized_line.startswith(normalized_heading)
        ):
            inner_offset = line.find(stripped)
            return offset + max(inner_offset, 0)
        offset += len(line)
    return None


def _normalize_heading_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip()


def _compact_outline_documents(documents: list[Document]) -> list[Document]:
    if len(documents) <= 1:
        return documents

    pending = list(documents)
    compacted: list[Document] = []
    index = 0
    while index < len(pending):
        current = pending[index]
        next_doc = pending[index + 1] if index + 1 < len(pending) else None
        previous = compacted[-1] if compacted else None

        if _should_drop_outline_document(current, previous, next_doc):
            index += 1
            continue

        if next_doc is not None and _should_merge_outline_forward(current, next_doc):
            pending[index + 1] = _merge_outline_documents(current, next_doc)
            index += 1
            continue

        if previous is not None and _should_merge_outline_backward(previous, current):
            compacted[-1] = _merge_outline_documents(previous, current)
            index += 1
            continue

        compacted.append(current)
        index += 1

    return compacted


def _should_drop_outline_document(
    current: Document,
    previous: Document | None,
    next_doc: Document | None,
) -> bool:
    if not _is_heading_only_outline_document(current):
        return False
    if not current.page_content.strip():
        return True
    if next_doc is not None and (
        _is_descendant_outline_document(next_doc, current)
        or _shares_outline_parent(current, next_doc)
    ):
        return True
    if previous is not None and (
        _is_descendant_outline_document(current, previous)
        or _shares_outline_parent(previous, current)
    ):
        return True
    return True


def _should_merge_outline_forward(current: Document, next_doc: Document) -> bool:
    if not _is_short_outline_document(current):
        return False
    return _shares_outline_parent(current, next_doc)


def _should_merge_outline_backward(previous: Document, current: Document) -> bool:
    if not _is_short_outline_document(current):
        return False
    return _shares_outline_parent(previous, current)


def _merge_outline_documents(first: Document, second: Document) -> Document:
    first_text = first.page_content.strip()
    second_text = second.page_content.strip()
    if first_text and second_text:
        merged_text = f"{first_text}\n\n{second_text}"
    else:
        merged_text = first_text or second_text
    metadata = dict(second.metadata)
    metadata["page"] = min(
        _coerce_outline_page(first.metadata.get("page")),
        _coerce_outline_page(second.metadata.get("page")),
    )
    metadata["page_end"] = max(
        _coerce_outline_page(first.metadata.get("page_end")),
        _coerce_outline_page(second.metadata.get("page_end")),
    )
    return Document(page_content=merged_text, metadata=metadata)


def _is_short_outline_document(document: Document) -> bool:
    content = document.page_content.strip()
    if not content:
        return True
    if _is_heading_only_outline_document(document):
        return True
    return len(content) < PDF_SHORT_OUTLINE_DOC_LENGTH


def _is_heading_only_outline_document(document: Document) -> bool:
    content = document.page_content.strip()
    title = str(document.metadata.get("section_title", "")).strip()
    if not content:
        return True
    if title and content == title:
        return True
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines or not title:
        return len(content) < 20
    return len(lines) <= 2 and lines[0] == title and len(content) < 40


def _outline_path_parts(document: Document) -> tuple[str, ...]:
    path = str(document.metadata.get("section_path", "")).strip()
    if not path:
        return ()
    return tuple(part.strip() for part in path.split(" > ") if part.strip())


def _is_descendant_outline_document(document: Document, ancestor: Document) -> bool:
    path = _outline_path_parts(document)
    ancestor_path = _outline_path_parts(ancestor)
    if len(path) <= len(ancestor_path) or not ancestor_path:
        return False
    return path[: len(ancestor_path)] == ancestor_path


def _shares_outline_parent(left: Document, right: Document) -> bool:
    left_path = _outline_path_parts(left)
    right_path = _outline_path_parts(right)
    if len(left_path) < 2 or len(right_path) < 2:
        return False
    return left_path[:-1] == right_path[:-1]


def _coerce_outline_page(value: object) -> int:
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except Exception:
        return 0


__all__ = ["PdfOutlineSection", "PdfKnowledge"]
