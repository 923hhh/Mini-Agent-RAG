"""时间序列扩展边界服务。"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from app.services.core.settings import AppSettings
from app.services.retrieval.candidate_common_service import RetrievalCandidate
from app.services.retrieval.candidate_retrieval_service import (
    retrieve_candidates,
    retrieve_candidates_with_timeseries_branching,
)
from app.services.retrieval.query_profile_service import (
    JointQueryProfile,
    QueryModalityProfile,
    extract_path_hint_terms_from_queries,
)
from app.services.retrieval.timeseries_retrieval_service import (
    TimeSeriesQueryProfile,
    infer_timeseries_query_profile,
)
from app.storage.bm25_index import LoadedBM25Index
from app.storage.filters import MetadataFilters
from app.storage.vector_stores import BaseVectorStoreAdapter


@dataclass(frozen=True)
class TimeseriesExtensionPlan:
    timeseries_query_profile: TimeSeriesQueryProfile
    joint_query_profile: JointQueryProfile


def is_timeseries_extension_enabled(settings: AppSettings) -> bool:
    return bool(getattr(settings.kb, "ENABLE_TIMESERIES_RETRIEVAL_EXTENSION", True))


def build_timeseries_query_modality_profile(
    query_bundle: list[str],
    timeseries_query_profile: TimeSeriesQueryProfile,
) -> QueryModalityProfile | None:
    if not timeseries_query_profile.is_timeseries_related:
        return None
    return QueryModalityProfile(
        query_type="timeseries_related",
        preferred_modalities=("timeseries", "text", "ocr", "ocr+vision", "vision", "image"),
        modality_bonus={"timeseries": 0.08, "text": 0.03, "ocr": 0.0, "ocr+vision": -0.01, "vision": -0.02, "image": -0.02},
        preferred_extensions=(".json", ".txt", ".md", ".pdf", ".docx", ".epub"),
        extension_bonus={".json": 0.07, ".txt": 0.02, ".md": 0.02, ".pdf": 0.01, ".docx": 0.01, ".epub": 0.01},
        path_hint_terms=extract_path_hint_terms_from_queries(query_bundle),
    )


def infer_timeseries_joint_query_profile(
    query_bundle: list[str],
    timeseries_query_profile: TimeSeriesQueryProfile,
) -> JointQueryProfile:
    import re

    from app.utils.text import deduplicate_strings

    combined = " ".join(item.strip() for item in query_bundle if item.strip())
    normalized = combined.lower()
    background_markers = ("原因", "背景", "事件", "为什么", "影响", "依据", "缓解", "加重", "改善", "恢复", "回滚", "发布", "风险", "风险来源", "业务判断", "业务简报", "值班", "解释")
    trend_markers = ("趋势", "变化", "回落", "下降", "上升", "波动", "峰值", "谷值", "监测", "浓度", "序列", "更高", "更低", "幅度", "同步", "比较", "判断", "解释", "对比", "哪一天", "风险", "异常", "抖动", "恢复", "恶化")
    location_terms = timeseries_query_profile.location_terms or tuple(
        deduplicate_strings(re.findall(r"[\u4e00-\u9fff]{2,}(?:市|省|区|县)?", combined))[:4]
    )
    channel_terms = tuple(
        deduplicate_strings(
            re.findall(
                r"(?:pm2\.5|pm25|pm10|so2|no2|co|o3|温度|湿度|风速|latency(?:_ms)?|error_rate|qps)",
                normalized,
                flags=re.IGNORECASE,
            )
        )[:6]
    )
    event_terms = tuple(
        deduplicate_strings(
            term
            for term in (
                "空气质量监测",
                "冷空气",
                "扩散条件",
                "污染物",
                "短时累积",
                "臭氧",
                "高温",
                "静稳",
                "海风",
                "阵雨",
                "订单服务",
                "在线服务监测",
                "在线服务",
                "缓存命中率",
                "缓存",
                "回滚",
                "版本变更",
                "发布",
                "服务抖动",
            )
            if term in combined
        )[:5]
    )
    domain_terms = timeseries_query_profile.domain_terms
    requires_timeseries = timeseries_query_profile.is_timeseries_related or any(marker in combined for marker in trend_markers)
    requires_text_background = any(marker in combined for marker in background_markers)
    has_guard_constraint = timeseries_query_profile.has_guard_constraint
    is_joint_query = (requires_timeseries and requires_text_background) or (
        has_guard_constraint and (requires_timeseries or requires_text_background)
    ) or (
        requires_timeseries and bool(location_terms or domain_terms or event_terms) and any(
            marker in combined for marker in ("原因", "为什么", "依据", "解释", "风险", "恢复", "改善")
        )
    )
    return JointQueryProfile(
        is_joint_query=is_joint_query,
        requires_timeseries=requires_timeseries,
        requires_text_background=requires_text_background,
        has_explicit_window=timeseries_query_profile.has_window_constraint,
        has_guard_constraint=has_guard_constraint,
        location_terms=location_terms,
        channel_terms=channel_terms,
        event_terms=event_terms,
        domain_terms=domain_terms,
    )


def build_timeseries_extension_plan(query_bundle: list[str]) -> TimeseriesExtensionPlan:
    timeseries_query_profile = infer_timeseries_query_profile(query_bundle)
    joint_query_profile = infer_timeseries_joint_query_profile(query_bundle, timeseries_query_profile)
    return TimeseriesExtensionPlan(
        timeseries_query_profile=timeseries_query_profile,
        joint_query_profile=joint_query_profile,
    )


def retrieve_candidates_with_timeseries_extension(
    *,
    settings: AppSettings,
    vector_store: BaseVectorStoreAdapter,
    sentence_vector_store: BaseVectorStoreAdapter | None,
    all_documents: dict[str, Document],
    query_bundle: list[str],
    dense_query_bundle: list[str] | None,
    bm25_index: LoadedBM25Index | None,
    top_k: int,
    metadata_filters: MetadataFilters | None = None,
    query_profile: QueryModalityProfile | None = None,
    extension_plan: TimeseriesExtensionPlan,
    diagnostics: dict[str, object] | None = None,
) -> list[RetrievalCandidate]:
    extension_enabled = is_timeseries_extension_enabled(settings)
    if diagnostics is not None:
        diagnostics["timeseries_extension_enabled"] = extension_enabled

    if not extension_enabled:
        if diagnostics is not None:
            diagnostics["timeseries_extension_bypassed"] = True
            diagnostics["timeseries_extension_bypass_reason"] = "disabled_by_setting"
            diagnostics["timeseries_branch_used"] = False
        return retrieve_candidates(
            settings=settings,
            vector_store=vector_store,
            sentence_vector_store=sentence_vector_store,
            all_documents=all_documents,
            query_bundle=query_bundle,
            dense_query_bundle=dense_query_bundle,
            bm25_index=bm25_index,
            top_k=top_k,
            metadata_filters=metadata_filters,
            query_profile=query_profile,
            diagnostics=diagnostics,
        )

    if diagnostics is not None:
        diagnostics["timeseries_extension_bypassed"] = False
    return retrieve_candidates_with_timeseries_branching(
        settings=settings,
        vector_store=vector_store,
        sentence_vector_store=sentence_vector_store,
        all_documents=all_documents,
        query_bundle=query_bundle,
        dense_query_bundle=dense_query_bundle,
        bm25_index=bm25_index,
        top_k=top_k,
        metadata_filters=metadata_filters,
        query_profile=query_profile,
        joint_query_profile=extension_plan.joint_query_profile,
        timeseries_query_profile=extension_plan.timeseries_query_profile,
        diagnostics=diagnostics,
    )
