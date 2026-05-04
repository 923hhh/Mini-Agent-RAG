"""提供知识库目录浏览与重建结果展示能力。"""

from __future__ import annotations

from app.schemas.kb import KnowledgeBaseSummary, RebuildKnowledgeBaseResult
from app.services.core.settings import AppSettings
from app.storage.vector_stores import vector_store_index_exists


def list_knowledge_bases(settings: AppSettings) -> list[KnowledgeBaseSummary]:
    root = settings.knowledge_base_root
    root.mkdir(parents=True, exist_ok=True)

    summaries: list[KnowledgeBaseSummary] = []
    for entry in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if not entry.is_dir():
            continue

        content_dir = settings.knowledge_base_content_dir(entry.name)
        vector_store_dir = settings.vector_store_dir(entry.name)
        content_dir.mkdir(parents=True, exist_ok=True)
        vector_store_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(
            [
                file.relative_to(content_dir).as_posix()
                for file in content_dir.rglob("*")
                if file.is_file()
            ]
        )
        summaries.append(
            KnowledgeBaseSummary(
                knowledge_base_name=entry.name,
                content_dir=content_dir,
                vector_store_dir=vector_store_dir,
                files=files,
                file_count=len(files),
                vector_store_type=settings.kb.DEFAULT_VS_TYPE,
                index_exists=vector_store_index_exists(vector_store_dir),
                metadata_exists=(vector_store_dir / "metadata.json").exists(),
            )
        )
    return summaries


def render_rebuild_summary(result: RebuildKnowledgeBaseResult) -> str:
    lines = [
        "知识库重建完成",
        f"知识库名称: {result.knowledge_base_name}",
        f"知识库内容目录: {result.content_dir}",
        f"向量库存储目录: {result.vector_store_dir}",
        f"元数据文件: {result.metadata_path}",
        f"构建清单文件: {result.build_manifest_path or '未生成'}",
        f"处理文件数: {result.files_processed}",
        f"原始文档数: {result.raw_documents}",
        f"最终切片数: {result.chunks}",
        f"增量重建: {'是' if result.incremental_rebuild else '否'}",
        f"索引构建模式: {result.index_mode}",
        f"复用文件数: {result.files_reused}",
        f"重建文件数: {result.files_rebuilt}",
        f"删除文件数: {result.files_deleted}",
        f"复用切片数: {result.chunks_reused}",
        f"新增向量切片数: {result.chunks_embedded}",
        f"向量存储类型: {result.vector_store_type}",
        f"本次启用图片 VLM: {'是' if result.image_vlm_enabled_for_build else '否'}",
        f"本次强制全量重建: {'是' if result.force_full_rebuild else '否'}",
    ]
    if result.stage_timings_seconds:
        lines.append("阶段耗时(秒):")
        for name, value in result.stage_timings_seconds.items():
            lines.append(f"  - {name}: {value:.4f}")
    return "\n".join(lines)


__all__ = [
    "list_knowledge_bases",
    "render_rebuild_summary",
]
