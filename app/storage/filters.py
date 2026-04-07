from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FilterOperator(str, Enum):
    EQ = "=="
    GT = ">"
    LT = "<"
    NE = "!="
    GTE = ">="
    LTE = "<="
    IN = "in"
    NIN = "nin"
    EXISTS = "exists"


class FilterCondition(str, Enum):
    AND = "and"
    OR = "or"


class MetadataFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1)
    operator: FilterOperator = FilterOperator.EQ
    value: str | int | float | bool | list[str] | list[int] | list[float] | None = None


class MetadataFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    condition: FilterCondition = FilterCondition.AND
    filters: list[MetadataFilter] = Field(default_factory=list)


def matches_metadata_filters(
    metadata: dict[str, Any],
    filters: MetadataFilters | None,
) -> bool:
    if filters is None or not filters.filters:
        return True

    results = [_matches_filter(metadata, item) for item in filters.filters]
    if filters.condition == FilterCondition.OR:
        return any(results)
    return all(results)


def metadata_filters_to_chroma_where(
    filters: MetadataFilters | None,
) -> tuple[dict[str, Any] | None, bool]:
    if filters is None or not filters.filters:
        return None, False

    converted: list[dict[str, Any]] = []
    requires_post_filter = False
    for item in filters.filters:
        converted_item, supported = _convert_filter_to_chroma(item)
        if converted_item is not None:
            converted.append(converted_item)
        if not supported:
            requires_post_filter = True

    if not converted:
        return None, True
    if len(converted) == 1:
        return converted[0], requires_post_filter
    return {f"${filters.condition.value}": converted}, requires_post_filter


def _matches_filter(metadata: dict[str, Any], item: MetadataFilter) -> bool:
    value = metadata.get(item.key)
    operator = item.operator

    if operator == FilterOperator.EXISTS:
        expected = _coerce_exists_value(item.value)
        exists = value is not None
        return exists if expected else not exists

    if value is None:
        return False

    if operator == FilterOperator.EQ:
        return value == item.value
    if operator == FilterOperator.NE:
        return value != item.value
    if operator == FilterOperator.GT:
        return _compare_numbers(value, item.value, lambda left, right: left > right)
    if operator == FilterOperator.GTE:
        return _compare_numbers(value, item.value, lambda left, right: left >= right)
    if operator == FilterOperator.LT:
        return _compare_numbers(value, item.value, lambda left, right: left < right)
    if operator == FilterOperator.LTE:
        return _compare_numbers(value, item.value, lambda left, right: left <= right)
    if operator == FilterOperator.IN:
        return _contains_value(item.value, value)
    if operator == FilterOperator.NIN:
        return not _contains_value(item.value, value)
    return False


def _convert_filter_to_chroma(item: MetadataFilter) -> tuple[dict[str, Any] | None, bool]:
    if item.operator == FilterOperator.EXISTS:
        return None, False
    if item.operator == FilterOperator.EQ:
        return {item.key: item.value}, True

    operator_map = {
        FilterOperator.NE: "$ne",
        FilterOperator.GT: "$gt",
        FilterOperator.GTE: "$gte",
        FilterOperator.LT: "$lt",
        FilterOperator.LTE: "$lte",
        FilterOperator.IN: "$in",
        FilterOperator.NIN: "$nin",
    }
    chroma_operator = operator_map.get(item.operator)
    if chroma_operator is None:
        return None, False
    return {item.key: {chroma_operator: item.value}}, True


def _compare_numbers(
    left: Any,
    right: Any,
    predicate,
) -> bool:
    try:
        return predicate(float(left), float(right))
    except (TypeError, ValueError):
        return False


def _contains_value(container: Any, value: Any) -> bool:
    if not isinstance(container, list):
        return False
    return value in container


def _coerce_exists_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return True
