"""管理知识库重建任务的提交与状态查询。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Lock
from uuid import uuid4

from app.schemas.kb import RebuildKnowledgeBaseRequest, RebuildTaskAccepted, RebuildTaskStatus
from app.services.kb.kb_ingestion_service import rebuild_knowledge_base
from app.services.core.settings import AppSettings


_TASK_RETENTION_HOURS = 24
_MAX_TASK_HISTORY = 200
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="kb-rebuild")
_task_lock = Lock()
_tasks: dict[str, RebuildTaskStatus] = {}


def submit_rebuild_task(
    *,
    settings: AppSettings,
    request: RebuildKnowledgeBaseRequest,
) -> RebuildTaskAccepted:
    _cleanup_finished_tasks()
    task_id = f"rebuild-{uuid4().hex[:12]}"
    created_at = datetime.now(UTC)
    task = RebuildTaskStatus(
        task_id=task_id,
        knowledge_base_name=request.knowledge_base_name,
        status="pending",
        progress=0.0,
        progress_message="等待重建任务开始",
        created_at=created_at,
    )
    with _task_lock:
        _tasks[task_id] = task
    _executor.submit(_run_rebuild_task, task_id, settings, request)
    return RebuildTaskAccepted(
        task_id=task_id,
        knowledge_base_name=request.knowledge_base_name,
        status=task.status,
        progress=task.progress,
        progress_message=task.progress_message,
        created_at=created_at,
    )


def get_rebuild_task(task_id: str) -> RebuildTaskStatus | None:
    with _task_lock:
        task = _tasks.get(task_id)
        if task is None:
            return None
        return task.model_copy(deep=True)


def _run_rebuild_task(
    task_id: str,
    settings: AppSettings,
    request: RebuildKnowledgeBaseRequest,
) -> None:
    started_at = datetime.now(UTC)
    with _task_lock:
        task = _tasks[task_id]
        _tasks[task_id] = task.model_copy(
            update={
                "status": "running",
                "progress": 0.02,
                "progress_message": "开始重建任务",
                "started_at": started_at,
                "error_message": None,
            }
        )

    try:
        result = rebuild_knowledge_base(
            settings=settings,
            knowledge_base_name=request.knowledge_base_name,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            embedding_model=request.embedding_model,
            enable_image_vlm_for_build=request.enable_image_vlm_for_build,
            force_full_rebuild=request.force_full_rebuild,
            progress_callback=lambda progress, message: _update_task_progress(
                task_id,
                progress=progress,
                progress_message=message,
            ),
        )
    except Exception as exc:
        finished_at = datetime.now(UTC)
        with _task_lock:
            task = _tasks[task_id]
            _tasks[task_id] = task.model_copy(
                update={
                    "status": "failed",
                    "progress": 1.0,
                    "progress_message": "重建失败",
                    "finished_at": finished_at,
                    "error_message": str(exc),
                }
            )
        return

    finished_at = datetime.now(UTC)
    with _task_lock:
        task = _tasks[task_id]
        _tasks[task_id] = task.model_copy(
            update={
                "status": "succeeded",
                "progress": 1.0,
                "progress_message": "重建完成",
                "finished_at": finished_at,
                "result": result,
            }
        )


def _update_task_progress(
    task_id: str,
    *,
    progress: float,
    progress_message: str,
) -> None:
    with _task_lock:
        task = _tasks.get(task_id)
        if task is None or task.status not in {"pending", "running"}:
            return
        _tasks[task_id] = task.model_copy(
            update={
                "status": "running",
                "progress": max(0.0, min(1.0, progress)),
                "progress_message": progress_message,
            }
        )


def _cleanup_finished_tasks() -> None:
    cutoff = datetime.now(UTC) - timedelta(hours=_TASK_RETENTION_HOURS)
    with _task_lock:
        finished = [
            task_id
            for task_id, task in _tasks.items()
            if task.finished_at is not None and task.finished_at < cutoff
        ]
        for task_id in finished:
            _tasks.pop(task_id, None)

        if len(_tasks) <= _MAX_TASK_HISTORY:
            return

        finished_ordered = sorted(
            (
                task
                for task in _tasks.values()
                if task.finished_at is not None
            ),
            key=lambda task: task.finished_at or task.created_at,
        )
        while len(_tasks) > _MAX_TASK_HISTORY and finished_ordered:
            oldest = finished_ordered.pop(0)
            _tasks.pop(oldest.task_id, None)

