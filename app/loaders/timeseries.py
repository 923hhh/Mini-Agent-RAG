"""加载结构化时间序列 JSON，并转成可检索文档。"""

from __future__ import annotations

import json

from langchain_core.documents import Document

from app.schemas.kb import TimeSeriesKnowledgeUnit
from app.services.timeseries_summary_service import build_timeseries_document_text, build_timeseries_summary

from .factory import BaseKnowledge


class TimeSeriesKnowledge(BaseKnowledge):
    supported_extensions = (".json",)

    def load(self) -> list[Document]:
        raw_text = self.path.read_text(encoding="utf-8-sig")
        payload = json.loads(raw_text)
        units = parse_timeseries_units(payload)
        if not units:
            return [
                Document(
                    page_content=raw_text,
                    metadata={
                        **self.base_metadata,
                        "doc_id": self.relative_path,
                    },
                )
            ]
        documents: list[Document] = []

        for index, unit in enumerate(units):
            ts_summary = build_timeseries_summary(unit)
            documents.append(
                Document(
                    page_content=build_timeseries_document_text(unit),
                    metadata={
                        **self.base_metadata,
                        "doc_id": f"{self.relative_path}::series::{unit.series_id}",
                        "source": f"{self.path.name}#{unit.series_id}",
                        "title": unit.series_id,
                        "content_type": "timeseries_text_evidence",
                        "source_modality": "timeseries",
                        "original_file_type": "json",
                        "evidence_summary": ts_summary,
                        "series_id": unit.series_id,
                        "start_time": unit.start_time,
                        "end_time": unit.end_time,
                        "ts_summary": ts_summary,
                        "event_type": unit.event_type,
                        "location": unit.location,
                        "channel_names": list(unit.channel_names),
                        "timeseries_index": index,
                    },
                )
            )
        return documents

    @classmethod
    def supports(cls, path) -> bool:
        return path.suffix.lower() == ".json"


def parse_timeseries_units(payload: object) -> list[TimeSeriesKnowledgeUnit]:
    if isinstance(payload, list):
        if not payload:
            return []
        if not all(isinstance(item, dict) and "series_id" in item for item in payload):
            return []
        return [build_timeseries_unit(item) for item in payload]
    if isinstance(payload, dict):
        if isinstance(payload.get("samples"), list):
            if not payload["samples"]:
                return []
            return [build_timeseries_unit(item) for item in payload["samples"]]
        if "series_id" in payload and "start_time" in payload and "end_time" in payload:
            return [build_timeseries_unit(payload)]
    return []


def build_timeseries_unit(payload: object) -> TimeSeriesKnowledgeUnit:
    if not isinstance(payload, dict):
        raise ValueError("时间序列样本必须是对象。")
    normalized = dict(payload)
    normalized["channel_names"] = normalize_channel_names(
        normalized.get("channel_names"),
        normalized.get("points"),
    )
    return TimeSeriesKnowledgeUnit.model_validate(normalized)


def normalize_channel_names(
    raw_channel_names: object,
    raw_points: object,
) -> list[str]:
    if isinstance(raw_channel_names, list):
        return [str(item).strip() for item in raw_channel_names if str(item).strip()]
    if isinstance(raw_points, list):
        ordered: dict[str, None] = {}
        for point in raw_points:
            if not isinstance(point, dict):
                continue
            values = point.get("values")
            if not isinstance(values, dict):
                continue
            for key in values:
                normalized = str(key).strip()
                if normalized:
                    ordered[normalized] = None
        return list(ordered)
    return []
