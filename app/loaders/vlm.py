"""调用视觉语言模型处理图片与多模态输入。"""

from __future__ import annotations

from app.services.models.image_caption_service import (
    StructuredImageCaption,
    caption_image_bytes_structured,
    caption_image_structured,
    format_structured_image_caption,
)

__all__ = [
    "StructuredImageCaption",
    "caption_image_structured",
    "caption_image_bytes_structured",
    "format_structured_image_caption",
]

