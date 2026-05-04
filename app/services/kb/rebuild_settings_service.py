"""构建知识库重建时使用的运行配置。"""

from __future__ import annotations

from app.services.core.settings import AppSettings


def build_rebuild_settings(
    settings: AppSettings,
    *,
    enable_image_vlm_for_build: bool = False,
    force_full_rebuild: bool = False,
) -> AppSettings:
    kb_updates: dict[str, object] = {}
    model_updates: dict[str, object] = {}

    if force_full_rebuild or enable_image_vlm_for_build:
        kb_updates["ENABLE_INCREMENTAL_REBUILD"] = False

    if enable_image_vlm_for_build:
        model_updates["IMAGE_VLM_ENABLED"] = True
        model_updates["IMAGE_VLM_AUTO_CAPTION_ENABLED"] = True
        model_updates["IMAGE_VLM_AUTO_TRIGGER_BY_OCR"] = False
        model_updates["IMAGE_VLM_ONLY_WHEN_OCR_EMPTY"] = False

    if not kb_updates and not model_updates:
        return settings

    updates: dict[str, object] = {}
    if kb_updates:
        updates["kb"] = settings.kb.model_copy(update=kb_updates)
    if model_updates:
        updates["model"] = settings.model.model_copy(update=model_updates)
    return settings.model_copy(update=updates)


__all__ = ["build_rebuild_settings"]
