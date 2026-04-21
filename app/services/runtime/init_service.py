"""初始化项目目录、默认配置与启动前资源。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.services.core.settings import (
    AppSettings,
    default_config_data,
    dump_yaml,
    load_settings,
)


@dataclass
class InitResult:
    project_root: Path
    config_root: Path
    settings: AppSettings
    created_files: list[Path]
    created_dirs: list[Path]


def ensure_default_configs(project_root: Path) -> list[Path]:
    config_root = project_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)

    created_files: list[Path] = []
    for filename, data in default_config_data().items():
        path = config_root / filename
        if not path.exists():
            path.write_text(dump_yaml(data), encoding="utf-8")
            created_files.append(path)
    return created_files


def ensure_runtime_dirs(settings: AppSettings) -> list[Path]:
    basic = settings.basic
    candidates = [
        settings.project_root / Path(basic.DATA_ROOT),
        settings.project_root / Path(basic.KB_ROOT_PATH),
        settings.project_root / Path(basic.TEMP_ROOT_PATH),
        settings.project_root / Path(basic.LOG_PATH),
        settings.project_root / Path(basic.VECTOR_STORE_PATH),
    ]

    created_dirs: list[Path] = []
    for directory in candidates:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            created_dirs.append(directory)
        else:
            directory.mkdir(parents=True, exist_ok=True)
    return created_dirs


def initialize_project(project_root: Path) -> InitResult:
    created_files = ensure_default_configs(project_root)
    settings = load_settings(project_root)
    created_dirs = ensure_runtime_dirs(settings)

    return InitResult(
        project_root=project_root,
        config_root=project_root / "configs",
        settings=settings,
        created_files=created_files,
        created_dirs=created_dirs,
    )


def render_init_summary(result: InitResult) -> str:
    basic = result.settings.basic
    model = result.settings.model
    kb = result.settings.kb

    created_files = ", ".join(path.name for path in result.created_files) or "无"
    created_dirs = ", ".join(str(path.relative_to(result.project_root)) for path in result.created_dirs) or "无"

    lines = [
        "初始化完成",
        f"项目根目录: {result.project_root}",
        f"配置目录: {result.config_root}",
        f"新建配置文件: {created_files}",
        f"新建数据目录: {created_dirs}",
        f"LLM 提供者: {model.LLM_PROVIDER}",
        f"默认 LLM 模型: {model.DEFAULT_LLM_MODEL}",
        f"默认 Embedding 模型: {model.DEFAULT_EMBEDDING_MODEL}",
        f"默认向量库类型: {kb.DEFAULT_VS_TYPE}",
        f"知识库目录: {basic.KB_ROOT_PATH}",
        f"临时目录: {basic.TEMP_ROOT_PATH}",
        f"日志目录: {basic.LOG_PATH}",
    ]
    return "\n".join(lines)

