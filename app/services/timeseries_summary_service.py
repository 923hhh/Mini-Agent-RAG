"""生成时间序列样本的可检索摘要文本。"""

from __future__ import annotations

from statistics import mean
from typing import NamedTuple

from app.schemas.kb import TimeSeriesKnowledgeUnit


class ChannelStats(NamedTuple):
    channel_name: str
    count: int
    start_value: float
    end_value: float
    delta_value: float
    mean_value: float
    peak_value: float
    peak_timestamp: str
    trough_value: float
    trough_timestamp: str


def build_timeseries_document_text(unit: TimeSeriesKnowledgeUnit) -> str:
    lines = [
        f"时间序列ID：{unit.series_id}",
        f"时间范围：{unit.start_time} 至 {unit.end_time}",
    ]
    if unit.location:
        lines.append(f"地点：{unit.location}")
    if unit.event_type:
        lines.append(f"事件类型：{unit.event_type}")
    if unit.channel_names:
        lines.append(f"通道：{', '.join(unit.channel_names)}")

    summary = build_timeseries_summary(unit)
    if summary:
        lines.append(f"序列摘要：{summary}")
    if unit.description:
        lines.append(f"描述：{unit.description}")
    if unit.event_background:
        lines.append(f"事件背景：{unit.event_background}")

    numeric_preview = build_timeseries_numeric_preview(unit)
    if numeric_preview:
        lines.append(f"数值预览：{numeric_preview}")
    return "\n".join(lines)


def build_timeseries_summary(unit: TimeSeriesKnowledgeUnit) -> str:
    if unit.ts_summary:
        return unit.ts_summary.strip()

    if not unit.points:
        return "未提供原始点位，仅保留时间范围与元数据。"

    channel_names = unit.channel_names or sorted(
        {
            key
            for point in unit.points
            for key in point.values
            if str(key).strip()
        }
    )
    if not channel_names:
        return f"{unit.start_time} 至 {unit.end_time} 的时间序列样本，未识别到有效通道值。"

    channel_summaries: list[str] = []
    for channel_name in channel_names[:3]:
        stats = extract_channel_stats(unit, channel_name)
        if stats is None:
            continue
        trend = infer_value_trend(stats.start_value, stats.end_value)
        channel_summaries.append(
            f"{channel_name}整体{trend}，起点{stats.start_value:.3f}，终点{stats.end_value:.3f}，变化{stats.delta_value:+.3f}，"
            f"峰值{stats.peak_value:.3f}（{stats.peak_timestamp}），谷值{stats.trough_value:.3f}（{stats.trough_timestamp}）"
        )

    if channel_summaries:
        return "；".join(channel_summaries)
    return f"{unit.start_time} 至 {unit.end_time} 的时间序列样本，共 {len(unit.points)} 个时间点。"


def build_timeseries_numeric_preview(unit: TimeSeriesKnowledgeUnit) -> str:
    if not unit.points:
        return ""

    channel_names = unit.channel_names or list(unit.points[0].values.keys())
    preview_parts: list[str] = []
    for channel_name in channel_names[:3]:
        stats = extract_channel_stats(unit, channel_name)
        if stats is None:
            continue
        preview_parts.append(
            f"{channel_name}: count={stats.count}, start={stats.start_value:.3f}, end={stats.end_value:.3f}, "
            f"delta={stats.delta_value:+.3f}, mean={stats.mean_value:.3f}, min={stats.trough_value:.3f}@{stats.trough_timestamp}, "
            f"max={stats.peak_value:.3f}@{stats.peak_timestamp}"
        )
    return " | ".join(preview_parts)


def extract_channel_stats(
    unit: TimeSeriesKnowledgeUnit,
    channel_name: str,
) -> ChannelStats | None:
    samples: list[tuple[str, float]] = []
    for point in unit.points:
        if channel_name not in point.values:
            continue
        value = point.values.get(channel_name)
        if value is None:
            continue
        samples.append((point.timestamp, float(value)))
    if len(samples) < 2:
        return None

    timestamps = [timestamp for timestamp, _ in samples]
    numeric_values = [value for _, value in samples]
    peak_index = max(range(len(numeric_values)), key=lambda index: numeric_values[index])
    trough_index = min(range(len(numeric_values)), key=lambda index: numeric_values[index])
    start_value = numeric_values[0]
    end_value = numeric_values[-1]
    return ChannelStats(
        channel_name=channel_name,
        count=len(numeric_values),
        start_value=start_value,
        end_value=end_value,
        delta_value=end_value - start_value,
        mean_value=mean(numeric_values),
        peak_value=numeric_values[peak_index],
        peak_timestamp=timestamps[peak_index],
        trough_value=numeric_values[trough_index],
        trough_timestamp=timestamps[trough_index],
    )


def infer_value_trend(start_value: float, end_value: float) -> str:
    delta = end_value - start_value
    if abs(delta) <= max(1e-6, abs(start_value) * 0.03):
        return "基本平稳"
    if delta > 0:
        return "上升"
    return "下降"
