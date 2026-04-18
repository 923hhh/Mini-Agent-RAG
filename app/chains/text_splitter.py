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
    if not documents:
        return []

    normalized_name = splitter_name.strip() or "ChineseRecursiveTextSplitter"
    splitter_cache: dict[str, object] = {}
    chunks: list[Document] = []
    for document in documents:
        resolved_name = resolve_document_splitter_name(document, normalized_name)
        splitter = splitter_cache.get(resolved_name)
        if splitter is None:
            splitter = build_text_splitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                splitter_name=resolved_name,
            )
            splitter_cache[resolved_name] = splitter
        chunks.extend(splitter.split_documents([document]))
    return chunks


def resolve_document_splitter_name(
    document: Document,
    default_splitter_name: str,
) -> str:
    if default_splitter_name == "MarkdownHeaderTextSplitter":
        return default_splitter_name
    extension = str(document.metadata.get("extension", "")).strip().lower()
    if extension == ".md":
        return "MarkdownHeaderTextSplitter"
    return default_splitter_name


@dataclass(frozen=True)
class MarkdownSection:
    content: str
    metadata: dict[str, str]


class MarkdownHeaderTextSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_section_length = max(120, min(360, chunk_overlap + 120))
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
            sections = self._merge_small_sections(sections)

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
                if _is_markdown_horizontal_rule(stripped):
                    continue
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

    def _merge_small_sections(self, sections: list[MarkdownSection]) -> list[MarkdownSection]:
        if len(sections) <= 1:
            return sections

        merged: list[MarkdownSection] = []
        pending = list(sections)
        index = 0
        while index < len(pending):
            current = pending[index]
            if len(current.content.strip()) >= self.min_section_length:
                merged.append(current)
                index += 1
                continue

            previous = merged[-1] if merged else None
            following = pending[index + 1] if index + 1 < len(pending) else None
            current_rendered = self._render_section_for_merge(current)

            if following is not None and self._should_merge_forward(current, previous, following):
                pending[index + 1] = MarkdownSection(
                    content=self._join_section_contents(current_rendered, following.content),
                    metadata=dict(following.metadata),
                )
                index += 1
                continue

            if previous is not None:
                merged[-1] = MarkdownSection(
                    content=self._join_section_contents(previous.content, current_rendered),
                    metadata=dict(previous.metadata),
                )
                index += 1
                continue

            merged.append(current)
            index += 1
        return merged

    def _should_merge_forward(
        self,
        current: MarkdownSection,
        previous: MarkdownSection | None,
        following: MarkdownSection,
    ) -> bool:
        if previous is None:
            return True
        current_path = _section_header_path(current.metadata)
        prev_path = _section_header_path(previous.metadata)
        next_path = _section_header_path(following.metadata)
        previous_score = _shared_prefix_length(current_path, prev_path)
        next_score = _shared_prefix_length(current_path, next_path)
        if next_score != previous_score:
            return next_score > previous_score
        return len(following.content) >= len(previous.content)

    def _render_section_for_merge(self, section: MarkdownSection) -> str:
        title = _section_title_from_metadata(section.metadata)
        content = section.content.strip()
        if not title:
            return content
        if content.startswith(title):
            return content
        if not content:
            return title
        return f"{title}\n{content}"

    def _join_section_contents(self, first: str, second: str) -> str:
        first_cleaned = first.strip()
        second_cleaned = second.strip()
        if not first_cleaned:
            return second_cleaned
        if not second_cleaned:
            return first_cleaned
        return f"{first_cleaned}\n\n{second_cleaned}"

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


def _is_markdown_horizontal_rule(line: str) -> bool:
    if not line:
        return False
    normalized = line.replace(" ", "")
    if len(normalized) < 3:
        return False
    return normalized in {"---", "***", "___"}


def _section_header_path(metadata: dict[str, str]) -> tuple[str, ...]:
    return tuple(
        str(metadata[key]).strip()
        for key in ("Header1", "Header2", "Header3")
        if str(metadata.get(key, "")).strip()
    )


def _section_title_from_metadata(metadata: dict[str, str]) -> str:
    path = _section_header_path(metadata)
    if not path:
        return ""
    return path[-1]


def _shared_prefix_length(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    size = min(len(left), len(right))
    count = 0
    for index in range(size):
        if left[index] != right[index]:
            break
        count += 1
    return count
