"""管理临时知识库的创建、清理与有效期检查。"""

from __future__ import annotations

import json
import os
import shutil
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.schemas.kb import (
    TempKnowledgeCleanupEntry,
    TempKnowledgeCleanupResult,
    TempKnowledgeManifest,
)
from app.services.core.settings import AppSettings


class TempKnowledgeBaseExpiredError(FileNotFoundError):
    pass


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def build_temp_cleanup_policy(
    ttl_minutes: int,
    *,
    cleanup_on_startup: bool,
    touch_on_access: bool,
) -> str:
    return (
        f"ttl_minutes={ttl_minutes};"
        f"cleanup_on_startup={str(cleanup_on_startup).lower()};"
        f"touch_on_access={str(touch_on_access).lower()}"
    )


def get_temp_manifest_path(settings: AppSettings, knowledge_id: str) -> Path:
    return settings.temp_knowledge_dir(knowledge_id) / "manifest.json"


def create_temp_manifest(
    settings: AppSettings,
    knowledge_id: str,
    *,
    saved_files: list[str],
    now: datetime | None = None,
) -> TempKnowledgeManifest:
    current = now or utc_now()
    return TempKnowledgeManifest(
        knowledge_id=knowledge_id,
        scope="temp",
        created_at=current,
        last_accessed_at=current,
        expires_at=current + timedelta(minutes=settings.kb.TEMP_KB_TTL_MINUTES),
        ttl_minutes=settings.kb.TEMP_KB_TTL_MINUTES,
        touch_on_access=settings.kb.TEMP_KB_TOUCH_ON_ACCESS,
        cleanup_policy=build_temp_cleanup_policy(
            settings.kb.TEMP_KB_TTL_MINUTES,
            cleanup_on_startup=settings.kb.TEMP_KB_CLEANUP_ON_STARTUP,
            touch_on_access=settings.kb.TEMP_KB_TOUCH_ON_ACCESS,
        ),
        cleanup_state="active",
        deleted_at=None,
        saved_files=saved_files,
    )


def write_temp_manifest(settings: AppSettings, manifest: TempKnowledgeManifest) -> Path:
    manifest_path = get_temp_manifest_path(settings, manifest.knowledge_id)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def load_temp_manifest(settings: AppSettings, knowledge_id: str) -> TempKnowledgeManifest:
    knowledge_dir = settings.temp_knowledge_dir(knowledge_id)
    if not knowledge_dir.exists():
        raise FileNotFoundError(f"临时知识库不存在: {knowledge_id}")

    manifest_path = get_temp_manifest_path(settings, knowledge_id)
    raw_payload: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            raw_payload = json.loads(manifest_path.read_text(encoding="utf-8")) or {}
        except json.JSONDecodeError:
            raw_payload = {}

    manifest = TempKnowledgeManifest.model_validate(
        _normalize_manifest_payload(settings, knowledge_id, knowledge_dir, raw_payload)
    )

    normalized_payload = manifest.model_dump(mode="json")
    if raw_payload != normalized_payload:
        write_temp_manifest(settings, manifest)

    return manifest


def record_temp_knowledge_access(
    settings: AppSettings,
    knowledge_id: str,
    *,
    manifest: TempKnowledgeManifest | None = None,
    now: datetime | None = None,
) -> TempKnowledgeManifest:
    current = now or utc_now()
    current_manifest = manifest or load_temp_manifest(settings, knowledge_id)
    updated = current_manifest.model_copy(
        update={
            "last_accessed_at": current,
            "expires_at": (
                current + timedelta(minutes=current_manifest.ttl_minutes)
                if current_manifest.touch_on_access
                else current_manifest.expires_at
            ),
            "cleanup_policy": build_temp_cleanup_policy(
                current_manifest.ttl_minutes,
                cleanup_on_startup=settings.kb.TEMP_KB_CLEANUP_ON_STARTUP,
                touch_on_access=current_manifest.touch_on_access,
            ),
        }
    )
    write_temp_manifest(settings, updated)
    return updated


def ensure_temp_knowledge_available(
    settings: AppSettings,
    knowledge_id: str,
    *,
    now: datetime | None = None,
) -> TempKnowledgeManifest:
    current = now or utc_now()
    manifest = load_temp_manifest(settings, knowledge_id)
    if manifest.cleanup_state == "disabled":
        raise TempKnowledgeBaseExpiredError(f"临时知识库已过期并已被清理: {knowledge_id}")
    if manifest.expires_at <= current:
        cleanup_temp_knowledge_bases(
            settings,
            knowledge_id=knowledge_id,
            expired_only=False,
            cleanup_reason="expired_on_access",
            now=current,
        )
        raise TempKnowledgeBaseExpiredError(f"临时知识库已过期并已被清理: {knowledge_id}")

    return record_temp_knowledge_access(settings, knowledge_id, manifest=manifest, now=current)


def cleanup_temp_knowledge_bases(
    settings: AppSettings,
    *,
    knowledge_id: str | None = None,
    expired_only: bool = True,
    cleanup_reason: str = "manual",
    now: datetime | None = None,
) -> TempKnowledgeCleanupResult:
    current = now or utc_now()
    temp_root = settings.temp_root.resolve()
    temp_root.mkdir(parents=True, exist_ok=True)

    entries: list[TempKnowledgeCleanupEntry] = []
    knowledge_ids = [knowledge_id] if knowledge_id else _list_temp_knowledge_ids(temp_root)

    for current_knowledge_id in knowledge_ids:
        knowledge_dir = settings.temp_knowledge_dir(current_knowledge_id)
        if not knowledge_dir.exists():
            if knowledge_id:
                entries.append(
                    TempKnowledgeCleanupEntry(
                        knowledge_id=current_knowledge_id,
                        knowledge_dir=knowledge_dir,
                        expired=False,
                        removed=False,
                        reason="not_found",
                    )
                )
            continue

        manifest = load_temp_manifest(settings, current_knowledge_id)
        expired = manifest.expires_at <= current
        if expired_only and not expired:
            entries.append(
                TempKnowledgeCleanupEntry(
                    knowledge_id=current_knowledge_id,
                    knowledge_dir=knowledge_dir,
                    expired=False,
                    removed=False,
                    reason="not_expired",
                )
            )
            continue

        try:
            remove_temp_knowledge_base(settings, current_knowledge_id)
        except Exception as exc:
            try:
                disable_temp_knowledge_base(
                    settings,
                    current_knowledge_id,
                    manifest=manifest,
                    now=current,
                )
            except Exception as fallback_exc:
                entries.append(
                    TempKnowledgeCleanupEntry(
                        knowledge_id=current_knowledge_id,
                        knowledge_dir=knowledge_dir,
                        expired=expired,
                        removed=False,
                        reason=f"cleanup_failed: {exc}; disable_failed: {fallback_exc}",
                    )
                )
                continue

            entries.append(
                TempKnowledgeCleanupEntry(
                    knowledge_id=current_knowledge_id,
                    knowledge_dir=knowledge_dir,
                    expired=expired,
                    removed=True,
                    reason="expired_soft_deleted" if expired else f"{cleanup_reason}_soft_deleted",
                )
            )
            continue

        entries.append(
            TempKnowledgeCleanupEntry(
                knowledge_id=current_knowledge_id,
                knowledge_dir=knowledge_dir,
                expired=expired,
                removed=True,
                reason="expired" if expired else cleanup_reason,
            )
        )

    removed = sum(1 for entry in entries if entry.removed)
    skipped = len(entries) - removed
    return TempKnowledgeCleanupResult(
        cleanup_reason=cleanup_reason,
        expired_only=expired_only,
        requested_knowledge_id=knowledge_id,
        ttl_minutes=settings.kb.TEMP_KB_TTL_MINUTES,
        scanned=len(entries),
        removed=removed,
        skipped=skipped,
        entries=entries,
    )


def maybe_run_startup_cleanup(
    settings: AppSettings,
    *,
    startup_name: str,
) -> TempKnowledgeCleanupResult | None:
    if not settings.kb.TEMP_KB_CLEANUP_ON_STARTUP:
        return None
    return cleanup_temp_knowledge_bases(
        settings,
        expired_only=True,
        cleanup_reason=f"startup_{startup_name}",
    )


def remove_temp_knowledge_base(settings: AppSettings, knowledge_id: str) -> None:
    knowledge_dir = _ensure_safe_temp_target(settings, settings.temp_knowledge_dir(knowledge_id))
    if not knowledge_dir.exists():
        return
    shutil.rmtree(knowledge_dir, onerror=_handle_remove_error)


def disable_temp_knowledge_base(
    settings: AppSettings,
    knowledge_id: str,
    *,
    manifest: TempKnowledgeManifest | None = None,
    now: datetime | None = None,
) -> TempKnowledgeManifest:
    current = now or utc_now()
    current_manifest = manifest or load_temp_manifest(settings, knowledge_id)
    knowledge_dir = settings.temp_knowledge_dir(knowledge_id)
    disabled_policy = (
        build_temp_cleanup_policy(
            current_manifest.ttl_minutes,
            cleanup_on_startup=settings.kb.TEMP_KB_CLEANUP_ON_STARTUP,
            touch_on_access=current_manifest.touch_on_access,
        )
        + ";cleanup_state=disabled"
    )

    for file_path in sorted(knowledge_dir.rglob("*")):
        if not file_path.is_file() or file_path.name == "manifest.json":
            continue
        _truncate_file(file_path)

    disabled_manifest = current_manifest.model_copy(
        update={
            "last_accessed_at": current,
            "expires_at": current,
            "cleanup_state": "disabled",
            "deleted_at": current,
            "saved_files": [],
            "cleanup_policy": disabled_policy,
        }
    )
    write_temp_manifest(settings, disabled_manifest)
    return disabled_manifest


def render_temp_cleanup_summary(result: TempKnowledgeCleanupResult) -> str:
    lines = [
        "临时知识库清理完成",
        f"清理原因: {result.cleanup_reason}",
        f"过期清理模式: {str(result.expired_only).lower()}",
        f"TTL 分钟数: {result.ttl_minutes}",
        f"扫描目录数: {result.scanned}",
        f"删除目录数: {result.removed}",
        f"跳过目录数: {result.skipped}",
    ]

    if result.requested_knowledge_id:
        lines.append(f"目标 knowledge_id: {result.requested_knowledge_id}")

    if result.entries:
        lines.append("清理详情:")
        for entry in result.entries:
            lines.append(
                f"- {entry.knowledge_id}: removed={str(entry.removed).lower()}, "
                f"expired={str(entry.expired).lower()}, reason={entry.reason}"
            )

    return "\n".join(lines)


def _normalize_manifest_payload(
    settings: AppSettings,
    knowledge_id: str,
    knowledge_dir: Path,
    raw_payload: dict[str, Any],
) -> dict[str, Any]:
    base_time = _manifest_base_time(knowledge_dir)
    created_at = _coerce_datetime(raw_payload.get("created_at"), default=base_time)
    last_accessed_at = _coerce_datetime(raw_payload.get("last_accessed_at"), default=created_at)

    raw_ttl = raw_payload.get("ttl_minutes", settings.kb.TEMP_KB_TTL_MINUTES)
    try:
        ttl_minutes = int(raw_ttl)
    except (TypeError, ValueError):
        ttl_minutes = settings.kb.TEMP_KB_TTL_MINUTES
    if ttl_minutes < 1:
        ttl_minutes = settings.kb.TEMP_KB_TTL_MINUTES

    if "touch_on_access" in raw_payload:
        touch_on_access = _coerce_bool(raw_payload.get("touch_on_access"))
    else:
        touch_on_access = settings.kb.TEMP_KB_TOUCH_ON_ACCESS

    expires_default = last_accessed_at + timedelta(minutes=ttl_minutes)
    expires_at = _coerce_datetime(raw_payload.get("expires_at"), default=expires_default)
    saved_files = _coerce_saved_files(raw_payload.get("saved_files"), knowledge_dir / "content")

    return {
        "knowledge_id": raw_payload.get("knowledge_id") or knowledge_id,
        "scope": "temp",
        "created_at": created_at,
        "last_accessed_at": last_accessed_at,
        "expires_at": expires_at,
        "ttl_minutes": ttl_minutes,
        "touch_on_access": touch_on_access,
        "cleanup_policy": raw_payload.get("cleanup_policy")
        or build_temp_cleanup_policy(
            ttl_minutes,
            cleanup_on_startup=settings.kb.TEMP_KB_CLEANUP_ON_STARTUP,
            touch_on_access=touch_on_access,
        ),
        "cleanup_state": raw_payload.get("cleanup_state") or "active",
        "deleted_at": _coerce_datetime(raw_payload.get("deleted_at"), default=None),
        "saved_files": saved_files,
    }


def _manifest_base_time(knowledge_dir: Path) -> datetime:
    return datetime.fromtimestamp(knowledge_dir.stat().st_mtime, tz=timezone.utc)


def _coerce_datetime(value: Any, *, default: datetime | None) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return default
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return default


def _coerce_saved_files(value: Any, content_dir: Path) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not content_dir.exists():
        return []
    return sorted(file.name for file in content_dir.iterdir() if file.is_file())


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _list_temp_knowledge_ids(temp_root: Path) -> list[str]:
    return [
        item.name
        for item in sorted(temp_root.iterdir(), key=lambda path: path.name.lower())
        if item.is_dir()
    ]


def _ensure_safe_temp_target(settings: AppSettings, target: Path) -> Path:
    temp_root = settings.temp_root.resolve()
    resolved_target = target.resolve(strict=False)
    try:
        resolved_target.relative_to(temp_root)
    except ValueError as exc:
        raise ValueError(f"拒绝清理临时目录之外的路径: {resolved_target}") from exc

    if resolved_target == temp_root:
        raise ValueError(f"拒绝清理临时目录根路径: {resolved_target}")

    return resolved_target


def _handle_remove_error(func, path, exc_info) -> None:
    del exc_info
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _truncate_file(file_path: Path) -> None:
    suffix = file_path.suffix.lower()
    if suffix in {".json", ".txt", ".md", ".docx"}:
        file_path.write_text("", encoding="utf-8")
        return
    file_path.write_bytes(b"")

