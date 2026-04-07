from __future__ import annotations

from dataclasses import dataclass
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


CHINESE_SEPARATORS = [
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    "：",
    "，",
    " ",
    "",
]


MARKDOWN_HEADERS = [
    ("###", "Header3"),
    ("##", "Header2"),
    ("#", "Header1"),
]


def build_text_splitter(
    chunk_size: int,
    chunk_overlap: int,
    splitter_name: str = "ChineseRecursiveTextSplitter",
):
    normalized_name = splitter_name.strip() or "ChineseRecursiveTextSplitter"
    if normalized_name == "MarkdownHeaderTextSplitter":
        return MarkdownHeaderTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    if normalized_name == "ChineseRecursiveTextSplitter":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=CHINESE_SEPARATORS,
            length_function=len,
            is_separator_regex=False,
            keep_separator=True,
        )
    raise ValueError(f"不支持的 TEXT_SPLITTER_NAME: {normalized_name}")


def split_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
    splitter_name: str = "ChineseRecursiveTextSplitter",
) -> list[Document]:
    splitter = build_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        splitter_name=splitter_name,
    )
    return splitter.split_documents(documents)


@dataclass(frozen=True)
class MarkdownSection:
    content: str
    metadata: dict[str, str]


class MarkdownHeaderTextSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=CHINESE_SEPARATORS,
            length_function=len,
            is_separator_regex=False,
            keep_separator=True,
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        chunks: list[Document] = []
        for document in documents:
            sections = self._extract_sections(document.page_content)
            if not sections:
                chunks.extend(self._fallback_splitter.split_documents([document]))
                continue

            for section in sections:
                metadata = dict(document.metadata)
                metadata.update(section.metadata)
                self._fill_section_metadata(metadata)
                section_document = Document(
                    page_content=section.content,
                    metadata=metadata,
                )
                chunks.extend(self._fallback_splitter.split_documents([section_document]))
        return chunks

    def _extract_sections(self, text: str) -> list[MarkdownSection]:
        lines = text.splitlines()
        current_lines: list[str] = []
        header_stack: dict[int, str] = {}
        current_metadata: dict[str, str] = {}
        sections: list[MarkdownSection] = []
        in_code_block = False

        def flush() -> None:
            nonlocal current_lines
            content = "\n".join(line.rstrip() for line in current_lines).strip()
            if content:
                sections.append(
                    MarkdownSection(
                        content=content,
                        metadata=dict(current_metadata),
                    )
                )
            current_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                current_lines.append(line)
                continue

            if not in_code_block:
                matched = _match_markdown_header(stripped)
                if matched is not None:
                    level, header_name, title = matched
                    flush()
                    header_stack = {
                        key: value
                        for key, value in header_stack.items()
                        if key < level
                    }
                    header_stack[level] = title
                    current_metadata = {
                        name: header_stack[level_value]
                        for level_value, name in ((1, "Header1"), (2, "Header2"), (3, "Header3"))
                        if level_value in header_stack
                    }
                    continue

            if stripped or current_lines:
                current_lines.append(line)
        flush()
        return sections

    def _fill_section_metadata(self, metadata: dict[str, object]) -> None:
        headers = [
            str(metadata[key]).strip()
            for key in ("Header1", "Header2", "Header3")
            if isinstance(metadata.get(key), str) and str(metadata.get(key)).strip()
        ]
        if not headers:
            return
        if not metadata.get("section_title"):
            metadata["section_title"] = headers[-1]
        if not metadata.get("section_path"):
            metadata["section_path"] = " > ".join(headers)


def _match_markdown_header(line: str) -> tuple[int, str, str] | None:
    if not line:
        return None
    for marker, name in MARKDOWN_HEADERS:
        if line.startswith(marker) and (len(line) == len(marker) or line[len(marker)] == " "):
            title = re.sub(r"\s+", " ", line[len(marker) :].strip())
            if not title:
                return None
            return marker.count("#"), name, title
    return None
