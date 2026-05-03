"""查询画像与检索路由判断。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re

from app.constants import IMAGE_QUERY_HINTS
from app.services.core.settings import AppSettings
from app.utils.text import deduplicate_strings


TEXT_QUERY_HINTS = (
    "文档",
    "章节",
    "文章",
    "这本书",
    "说明书",
    "资料",
    "参数",
    "配置",
)
MULTI_DOC_QUERY_HINTS = (
    "共同",
    "分别",
    "各自",
    "区别",
    "不同",
    "相同",
    "比较",
    "相比",
    "对比",
    "优势",
    "特点",
    "目标",
    "哪些专业",
    "哪个专业",
    "哪几个专业",
)
MULTI_DOC_CONNECTORS = (
    "与",
    "和",
    "及",
    "以及",
    "、",
)
TEMPORAL_QUERY_HINTS = (
    "时间",
    "日期",
    "哪一年",
    "哪年",
    "何时",
    "什么时候",
    "几月",
    "几号",
    "多久",
    "截至",
    "截止",
    "开始",
    "结束",
    "报名",
    "查询",
)
TEMPORAL_RECENCY_HINTS = (
    "最新",
    "当前",
    "目前",
    "现任",
    "最近",
    "今年",
    "本年",
    "本年度",
)
YEAR_PATTERN = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
DATE_PATTERN = re.compile(
    r"(?<!\d)((?:19|20)\d{2})\s*(?:-|/|\.|年)\s*(\d{1,2})\s*(?:-|/|\.|月)\s*(\d{1,2})\s*(?:日|号)?(?!\d)"
)
LABELED_DATE_PATTERN = re.compile(
    r"(?:日期|发布时间|更新(?:时间)?|发布(?:时间)?)\s*[：: ]+\s*"
    r"((?:19|20)\d{2})\s*(?:-|/|\.|年)\s*(\d{1,2})\s*(?:-|/|\.|月)\s*(\d{1,2})\s*(?:日|号)?"
)
MONTH_DAY_PATTERN = re.compile(r"(?<!\d)(\d{1,2})\s*月\s*(\d{1,2})\s*(?:日|号)(?!\d)")
ANSWER_SEEKING_HINTS = (
    "多少",
    "几",
    "谁",
    "哪位",
    "何人",
    "哪一年",
    "哪年",
    "何时",
    "什么时候",
    "开始时间",
    "什么时间",
    "什么等级",
    "被评为",
    "哪三个",
    "哪几",
    "哪些",
    "哪个",
    "是什么",
    "是怎样",
    "理念",
    "愿景",
    "定位",
    "目标",
    "对象",
    "政策",
)
ANSWER_LABEL_TERMS = (
    "办学理念",
    "办学愿景",
    "发展定位",
    "人才培养目标",
    "降分政策",
    "招生人数",
    "招生计划",
    "开始时间",
    "招生对象",
    "定向招收对象",
    "办学目标",
    "等级",
)


@dataclass(frozen=True)
class QueryModalityProfile:
    query_type: str
    preferred_modalities: tuple[str, ...]
    modality_bonus: dict[str, float]
    preferred_extensions: tuple[str, ...]
    extension_bonus: dict[str, float]
    path_hint_terms: tuple[str, ...]


@dataclass(frozen=True)
class TemporalQueryProfile:
    is_temporal: bool
    prefers_recent: bool
    explicit_years: tuple[int, ...]
    explicit_dates: tuple[int, ...]


@dataclass(frozen=True)
class DiversityQueryProfile:
    prefer_family_diversity: bool


@dataclass(frozen=True)
class JointQueryProfile:
    is_joint_query: bool
    requires_timeseries: bool
    requires_text_background: bool
    has_explicit_window: bool
    has_guard_constraint: bool
    location_terms: tuple[str, ...]
    channel_terms: tuple[str, ...]
    event_terms: tuple[str, ...]
    domain_terms: tuple[str, ...]


@dataclass(frozen=True)
class RerankModelSelection:
    model_name: str
    route: str


def build_image_query_expansions(query_bundle: list[str]) -> list[str]:
    profile = infer_query_modality_profile(query_bundle)
    if profile.query_type not in {"image_related", "multimodal_joint"}:
        return []

    expansions: list[str] = []
    base_queries = [item.strip() for item in query_bundle if item.strip()]
    suffixes = ("图片内容", "图像描述", "图片文字", "ocr文字", "画面信息")
    for query in base_queries[:2]:
        lowered = query.lower()
        if any(suffix.lower() in lowered for suffix in suffixes):
            continue
        expansions.extend(f"{query} {suffix}" for suffix in suffixes[:3])
        if profile.query_type == "multimodal_joint":
            expansions.append(f"{query} 文档内容")
            expansions.append(f"{query} 图片内容")
    return deduplicate_strings(expansions)


def infer_query_modality_profile(
    query_bundle: list[str],
    *,
    timeseries_query_profile=None,
) -> QueryModalityProfile:
    combined = " ".join(item.strip().lower() for item in query_bundle if item.strip())
    primary_query = next((item.strip().lower() for item in query_bundle if item.strip()), "")
    if not combined:
        return QueryModalityProfile(
            query_type="text_related",
            preferred_modalities=("text",),
            modality_bonus={"text": 0.04, "ocr": 0.0, "vision": -0.02, "ocr+vision": -0.01, "image": -0.02},
            preferred_extensions=(".txt", ".md", ".pdf", ".docx", ".epub"),
            extension_bonus={".txt": 0.02, ".md": 0.02, ".pdf": 0.02, ".docx": 0.02, ".epub": 0.02},
            path_hint_terms=(),
        )

    if getattr(timeseries_query_profile, "is_timeseries_related", False):
        from app.services.retrieval.timeseries_extension_service import build_timeseries_query_modality_profile

        profile = build_timeseries_query_modality_profile(query_bundle, timeseries_query_profile)
        if profile is not None:
            return profile

    image_hits = sum(1 for keyword in IMAGE_QUERY_HINTS if keyword in combined)
    text_hits = sum(1 for keyword in TEXT_QUERY_HINTS if keyword in combined)
    if looks_like_multimodal_joint_query(primary_query):
        return QueryModalityProfile(
            query_type="multimodal_joint",
            preferred_modalities=("text", "vision", "image", "ocr", "ocr+vision"),
            modality_bonus={"text": 0.04, "ocr": 0.04, "ocr+vision": 0.05, "vision": 0.04, "image": 0.04},
            preferred_extensions=(".txt", ".md", ".pdf", ".docx", ".epub", ".png", ".jpg", ".jpeg", ".bmp", ".webp"),
            extension_bonus={".txt": 0.02, ".md": 0.02, ".pdf": 0.02, ".docx": 0.02, ".epub": 0.02, ".png": 0.02, ".jpg": 0.02, ".jpeg": 0.02, ".bmp": 0.02, ".webp": 0.02},
            path_hint_terms=("图片", "图像", "照片", "文档", "书籍", "内容"),
        )
    if image_hits > text_hits:
        return QueryModalityProfile(
            query_type="image_related",
            preferred_modalities=("ocr", "vision", "ocr+vision", "image", "text"),
            modality_bonus={"ocr": 0.05, "vision": 0.05, "ocr+vision": 0.06, "image": 0.03, "text": 0.0},
            preferred_extensions=(".png", ".jpg", ".jpeg", ".bmp", ".webp", ".pdf"),
            extension_bonus={".png": 0.05, ".jpg": 0.05, ".jpeg": 0.05, ".bmp": 0.05, ".webp": 0.05, ".pdf": 0.01, ".epub": -0.03, ".docx": -0.03},
            path_hint_terms=("图片", "图像", "截图", "照片", "ocr", "图中", "画面"),
        )
    return QueryModalityProfile(
        query_type="text_related",
        preferred_modalities=("text", "ocr", "ocr+vision", "vision", "image"),
        modality_bonus={"text": 0.05, "ocr": 0.02, "ocr+vision": 0.01, "vision": -0.02, "image": -0.02},
        preferred_extensions=(".txt", ".md", ".pdf", ".docx", ".epub"),
        extension_bonus={".txt": 0.02, ".md": 0.03, ".pdf": 0.03, ".docx": 0.03, ".epub": 0.03, ".png": -0.03, ".jpg": -0.03, ".jpeg": -0.03, ".bmp": -0.03, ".webp": -0.03},
        path_hint_terms=extract_path_hint_terms_from_queries(query_bundle),
    )


def infer_diversity_query_profile(query_bundle: list[str]) -> DiversityQueryProfile:
    combined = " ".join(item.strip().lower() for item in query_bundle if item.strip())
    connector_hits = sum(combined.count(connector) for connector in MULTI_DOC_CONNECTORS)
    hint_hits = sum(1 for hint in MULTI_DOC_QUERY_HINTS if hint in combined)
    return DiversityQueryProfile(prefer_family_diversity=hint_hits > 0 and connector_hits > 0)


def resolve_rerank_model_selection(
    settings: AppSettings,
    query_bundle: list[str],
    query_profile: QueryModalityProfile,
) -> RerankModelSelection:
    default_model = settings.model.RERANK_MODEL.strip()
    primary_query = next((item.lower().strip() for item in query_bundle if item.strip()), "")
    temporal_profile = infer_temporal_query_profile(query_bundle)
    diversity_profile = infer_diversity_query_profile(query_bundle)
    answer_focused_model = settings.model.RERANK_MODEL_ANSWER_FOCUSED.strip()
    multi_doc_model = settings.model.RERANK_MODEL_MULTI_DOC.strip()
    temporal_model = settings.model.RERANK_MODEL_TEMPORAL.strip()

    if temporal_profile.is_temporal and temporal_model:
        return RerankModelSelection(model_name=temporal_model, route="temporal")
    if diversity_profile.prefer_family_diversity and multi_doc_model:
        return RerankModelSelection(model_name=multi_doc_model, route="multi_doc")
    if should_focus_answer_window(primary_query) and answer_focused_model:
        return RerankModelSelection(model_name=answer_focused_model, route="answer_focused")
    return RerankModelSelection(model_name=default_model, route=query_profile.query_type or "default")


def infer_temporal_query_profile(query_bundle: list[str]) -> TemporalQueryProfile:
    combined = " ".join(item.strip().lower() for item in query_bundle if item.strip())
    explicit_years = tuple(sorted(set(extract_years_from_text(combined))))
    explicit_dates = tuple(sorted(set(extract_date_ordinals_from_text(combined))))
    has_month_day_reference = bool(MONTH_DAY_PATTERN.search(combined))
    prefers_recent = any(term in combined for term in TEMPORAL_RECENCY_HINTS)
    is_temporal = prefers_recent or bool(explicit_years) or bool(explicit_dates) or has_month_day_reference or any(
        term in combined for term in TEMPORAL_QUERY_HINTS
    )
    return TemporalQueryProfile(
        is_temporal=is_temporal,
        prefers_recent=prefers_recent,
        explicit_years=explicit_years,
        explicit_dates=explicit_dates,
    )


def infer_joint_query_profile(query_bundle: list[str], timeseries_query_profile) -> JointQueryProfile:
    from app.services.retrieval.timeseries_extension_service import infer_timeseries_joint_query_profile

    return infer_timeseries_joint_query_profile(query_bundle, timeseries_query_profile)


def should_focus_answer_window(primary_query: str) -> bool:
    query = str(primary_query or "").strip().lower()
    return bool(query) and any(hint in query for hint in ANSWER_SEEKING_HINTS)


def should_use_sentence_index(query_bundle: list[str]) -> bool:
    primary_query = next((item.lower().strip() for item in query_bundle if item.strip()), "")
    if not should_focus_answer_window(primary_query):
        return False
    if any(term in primary_query for term in TEMPORAL_QUERY_HINTS + TEMPORAL_RECENCY_HINTS):
        return False
    if any(hint in primary_query for hint in MULTI_DOC_QUERY_HINTS):
        return False
    connector_hits = sum(primary_query.count(connector) for connector in MULTI_DOC_CONNECTORS)
    if connector_hits >= 2:
        return False
    if any(term in primary_query for term in ANSWER_LABEL_TERMS):
        return True
    if "语言" in primary_query:
        return True
    strong_extractive_hints = ("多少", "几", "哪一年", "哪年", "何时", "什么时候", "开始时间", "什么时间", "什么等级", "哪三个")
    return any(hint in primary_query for hint in strong_extractive_hints)


def extract_years_from_text(text: str) -> list[int]:
    return [int(match.group(1)) for match in YEAR_PATTERN.finditer(str(text or ""))]


def extract_date_ordinals_from_text(text: str, *, prefer_labeled: bool = False) -> list[int]:
    source = str(text or "")
    pattern = LABELED_DATE_PATTERN if prefer_labeled else DATE_PATTERN
    ordinals: list[int] = []
    for match in pattern.finditer(source):
        ordinal = coerce_date_ordinal(match.group(1), match.group(2), match.group(3))
        if ordinal is not None:
            ordinals.append(ordinal)
    return ordinals


def coerce_date_ordinal(year_text: str, month_text: str, day_text: str) -> int | None:
    try:
        return date(int(year_text), int(month_text), int(day_text)).toordinal()
    except (TypeError, ValueError):
        return None


def looks_like_multimodal_joint_query(combined_query: str) -> bool:
    joint_markers = ("既有", "也有", "同时", "一起", "以及", "又有", "既包含", "也包含")
    image_markers = ("图片", "图像", "照片", "截图")
    text_markers = ("书籍", "文档", "内容", "正文", "资料")
    if any(marker in combined_query for marker in joint_markers) and any(marker in combined_query for marker in image_markers):
        return True
    return any(marker in combined_query for marker in image_markers) and any(marker in combined_query for marker in text_markers)


def resolve_rerank_cutoff(settings: AppSettings, query_profile: QueryModalityProfile, top_k: int) -> int:
    base_cutoff = max(settings.kb.HYBRID_RERANK_TOP_K, top_k, 1)
    if query_profile.query_type == "multimodal_joint":
        return max(base_cutoff, top_k * 4, 12)
    if query_profile.query_type == "image_related":
        return max(base_cutoff, top_k * 3, 10)
    return base_cutoff


def resolve_required_modalities_for_query(query_profile: QueryModalityProfile) -> tuple[str, ...]:
    if query_profile.query_type == "image_related":
        return ("ocr", "vision", "ocr+vision", "image")
    if query_profile.query_type == "multimodal_joint":
        return ("text", "vision", "image", "ocr", "ocr+vision")
    return ()


def extract_path_hint_terms_from_queries(query_bundle: list[str]) -> tuple[str, ...]:
    hints: list[str] = []
    for query in query_bundle:
        stripped = query.strip()
        if not stripped:
            continue
        hints.extend(token for token in re.findall(r"[\u4e00-\u9fff]{2,}", stripped) if len(token) >= 2)
    return tuple(deduplicate_strings(hints)[:8])
