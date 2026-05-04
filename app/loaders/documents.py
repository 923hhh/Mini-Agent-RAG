"""定义统一的文档对象与加载结果结构。"""

from __future__ import annotations

from .factory import BaseKnowledge, KnowledgeFactory, list_supported_files, load_documents, load_file
from .image import FigureBandSpec, ImageKnowledge, ImageRegionSpec, InstructionPageParseResult
from .office import DocxKnowledge, EpubKnowledge
from .pdf import PdfKnowledge, PdfOutlineSection
from .text import MarkdownKnowledge, TextKnowledge
from .timeseries import TimeSeriesKnowledge

__all__ = [
    "BaseKnowledge",
    "KnowledgeFactory",
    "list_supported_files",
    "load_documents",
    "load_file",
    "TextKnowledge",
    "MarkdownKnowledge",
    "TimeSeriesKnowledge",
    "PdfKnowledge",
    "PdfOutlineSection",
    "DocxKnowledge",
    "EpubKnowledge",
    "ImageKnowledge",
    "ImageRegionSpec",
    "InstructionPageParseResult",
    "FigureBandSpec",
]
