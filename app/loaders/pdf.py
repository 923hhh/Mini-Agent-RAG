from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
        text = page.extract_text() or ""
        if not text.strip():
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
            text = reader.pages[page_number - 1].extract_text() or ""
            normalized = text.strip()
            if normalized:
                page_texts.append(normalized)
        if not page_texts:
            continue

        documents.append(
            Document(
                page_content="\n\n".join(page_texts),
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

    return documents


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


__all__ = ["PdfOutlineSection", "PdfKnowledge"]
