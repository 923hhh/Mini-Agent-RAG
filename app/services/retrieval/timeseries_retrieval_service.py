"""时间序列问题识别与检索辅助。"""

from __future__ import annotations

from dataclasses import dataclass
import re


TIMESERIES_QUERY_HINTS = (
    "时间序列",
    "趋势",
    "变化",
    "波动",
    "异常",
    "峰值",
    "谷值",
    "最高",
    "最低",
    "更高",
    "更低",
    "拐点",
    "上升",
    "下降",
    "连续",
    "走势",
    "监测",
    "曲线",
    "浓度",
    "序列",
    "回落",
    "改善",
    "缓解",
    "异常",
    "抖动",
    "恶化",
    "恢复",
    "风险",
    "风险来源",
    "原因",
    "为什么",
    "解释",
    "依据",
    "峰值时间",
    "谷值时间",
    "最高值",
    "最低值",
    "幅度",
    "对比",
    "比较",
    "同步",
    "冲高",
    "改善幅度",
    "回落幅度",
)
TIMESERIES_WINDOW_HINTS = (
    "某时间段",
    "时间窗口",
    "时间范围",
    "期间",
    "区间",
    "至",
    "到",
    "前后",
    "之后",
    "以前",
    "以来",
)
TIMESERIES_CHANNEL_HINTS = (
    "pm25",
    "pm2.5",
    "pm10",
    "o3",
    "so2",
    "no2",
    "co",
    "latency",
    "latency_ms",
    "error_rate",
    "qps",
    "臭氧",
    "延迟",
    "错误率",
)
TIMESERIES_DOMAIN_HINTS = (
    "空气质量",
    "空气",
    "臭氧",
    "颗粒物",
    "pm2.5",
    "pm25",
    "pm10",
    "订单服务",
    "订单",
    "在线服务",
    "服务",
    "接口",
    "服务稳定性",
    "延迟",
    "错误率",
    "缓存命中率",
    "缓存",
    "回滚",
    "发布",
    "版本变更",
    "服务抖动",
    "高温",
    "静稳",
    "阵雨",
    "海风",
)
TIMESERIES_GUARD_HINTS = (
    "只看",
    "不要混入",
    "不要混进",
    "不要带入",
    "不要混",
    "不要掺入",
)
TIMESERIES_CAUSAL_HINTS = (
    "为什么",
    "原因",
    "由什么导致",
    "导致",
    "驱动",
    "解释",
    "依据",
    "风险来源",
    "业务判断",
    "抬头",
    "缓解",
    "改善",
    "恶化",
    "恢复",
    "回滚后",
    "发布后",
    "降雨后",
    "高温后",
)
TIMESERIES_COMPARISON_HINTS = (
    "更高",
    "更低",
    "大于",
    "小于",
    "高于",
    "低于",
    "比较",
    "对比",
    "是否",
    "哪一天",
    "哪个更",
    "哪段时间",
    "同步",
    "幅度",
)
LOCATION_PATTERN = re.compile(
    r"(?:北京|上海|苏州|南京|杭州|广州|深圳|天津|重庆|成都|武汉|西安|长沙|郑州|青岛|宁波)"
    r"|[\u4e00-\u9fff]{2,10}(?:市|省|区|县)"
)


@dataclass(frozen=True)
class TimeSeriesQueryProfile:
    is_timeseries_related: bool
    matched_keywords: tuple[str, ...]
    has_window_constraint: bool
    location_terms: tuple[str, ...]
    domain_terms: tuple[str, ...]
    has_guard_constraint: bool


def infer_timeseries_query_profile(query_bundle: list[str]) -> TimeSeriesQueryProfile:
    combined = " ".join(item.strip().lower() for item in query_bundle if item.strip())
    raw_combined = " ".join(item.strip() for item in query_bundle if item.strip())
    if not combined:
        return TimeSeriesQueryProfile(False, (), False, (), (), False)

    matched_keywords = tuple(
        keyword
        for keyword in (
            *TIMESERIES_QUERY_HINTS,
            *TIMESERIES_CHANNEL_HINTS,
            *TIMESERIES_DOMAIN_HINTS,
        )
        if keyword in combined
    )
    has_window_constraint = any(keyword in combined for keyword in TIMESERIES_WINDOW_HINTS)
    has_guard_constraint = any(keyword in combined for keyword in TIMESERIES_GUARD_HINTS)
    has_causal_intent = any(keyword in combined for keyword in TIMESERIES_CAUSAL_HINTS)
    has_comparison_intent = any(keyword in combined for keyword in TIMESERIES_COMPARISON_HINTS)
    location_terms = tuple(dict.fromkeys(LOCATION_PATTERN.findall(raw_combined)))[:4]
    domain_terms = tuple(
        keyword for keyword in TIMESERIES_DOMAIN_HINTS if keyword in combined
    )
    is_timeseries_related = bool(
        matched_keywords
        or has_window_constraint
        or ((has_causal_intent or has_comparison_intent) and (location_terms or domain_terms))
        or (has_guard_constraint and (location_terms or domain_terms))
    )
    return TimeSeriesQueryProfile(
        is_timeseries_related=is_timeseries_related,
        matched_keywords=matched_keywords,
        has_window_constraint=has_window_constraint,
        location_terms=location_terms,
        domain_terms=domain_terms,
        has_guard_constraint=has_guard_constraint,
    )
