from __future__ import annotations

from langchain_core.documents import Document

from .factory import BaseKnowledge


class TextKnowledge(BaseKnowledge):
    supported_extensions = (".txt",)

    def load(self) -> list[Document]:
        text = self.path.read_text(encoding="utf-8", errors="ignore")
        return [
            Document(
                page_content=text,
                metadata={
                    **self.base_metadata,
                    "doc_id": self.relative_path,
                },
            )
        ]


class MarkdownKnowledge(BaseKnowledge):
    supported_extensions = (".md",)

    def load(self) -> list[Document]:
        text = self.path.read_text(encoding="utf-8", errors="ignore")
        return [
            Document(
                page_content=text,
                metadata={
                    **self.base_metadata,
                    "doc_id": self.relative_path,
                },
            )
        ]


__all__ = ["TextKnowledge", "MarkdownKnowledge"]
