"""知识库上传文件保存与目录布局服务。"""

from __future__ import annotations

from pathlib import Path

from fastapi import UploadFile

from app.services.core.settings import AppSettings
from app.storage.vector_stores import vector_store_index_exists


def ensure_knowledge_base_layout(settings: AppSettings, knowledge_base_name: str) -> tuple[Path, Path]:
    content_dir = settings.knowledge_base_content_dir(knowledge_base_name)
    vector_store_dir = settings.vector_store_dir(knowledge_base_name)
    content_dir.mkdir(parents=True, exist_ok=True)
    vector_store_dir.mkdir(parents=True, exist_ok=True)
    return content_dir, vector_store_dir


def save_uploaded_files(
    *,
    files: list[UploadFile],
    target_dir: Path,
    supported_extensions: list[str],
    overwrite_existing: bool,
) -> dict[str, list[str]]:
    target_dir.mkdir(parents=True, exist_ok=True)
    supported = {ext.lower() for ext in supported_extensions}
    saved_files: list[str] = []
    overwritten_files: list[str] = []
    skipped_files: list[str] = []

    for upload in files:
        try:
            filename = Path(upload.filename or "").name
            if not filename:
                continue

            suffix = Path(filename).suffix.lower()
            if suffix not in supported:
                raise ValueError(
                    f"不支持的文件类型: {filename}。支持格式: {', '.join(supported_extensions)}"
                )

            target = safe_upload_target(target_dir, filename)
            file_bytes = upload.file.read()
            existed_before_write = target.exists()
            if existed_before_write and not overwrite_existing:
                skipped_files.append(filename)
                continue

            target.write_bytes(file_bytes)
            if existed_before_write:
                overwritten_files.append(filename)
            else:
                saved_files.append(filename)
        finally:
            upload.file.close()

    return {
        "saved_files": saved_files,
        "overwritten_files": overwritten_files,
        "skipped_files": skipped_files,
    }


def safe_upload_target(target_dir: Path, filename: str) -> Path:
    target = (target_dir / filename).resolve()
    target_root = target_dir.resolve()
    try:
        target.relative_to(target_root)
    except ValueError as exc:
        raise ValueError(f"非法文件名: {filename}") from exc
    return target


def requires_local_rebuild(
    *,
    content_dir: Path,
    vector_store_dir: Path,
    files_processed: int,
    auto_rebuild: bool,
) -> bool:
    if auto_rebuild:
        return False
    if files_processed > 0:
        return True
    has_content = any(path.is_file() for path in content_dir.rglob("*"))
    index_exists = vector_store_index_exists(vector_store_dir)
    return has_content and not index_exists
