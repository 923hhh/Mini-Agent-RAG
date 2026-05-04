"""维护知识库加载器注册表。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .factory import BaseKnowledge


def get_knowledge_registry() -> list[type["BaseKnowledge"]]:
    from .image import ImageKnowledge
    from .office import DocxKnowledge, EpubKnowledge
    from .pdf import PdfKnowledge
    from .text import MarkdownKnowledge, TextKnowledge
    from .timeseries import TimeSeriesKnowledge

    return [
        TimeSeriesKnowledge,
        MarkdownKnowledge,
        TextKnowledge,
        PdfKnowledge,
        DocxKnowledge,
        EpubKnowledge,
        ImageKnowledge,
    ]


__all__ = ["get_knowledge_registry"]
