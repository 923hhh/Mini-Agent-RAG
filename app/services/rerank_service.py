from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import exp
from pathlib import Path
from typing import Literal

from app.services.settings import AppSettings


@dataclass(frozen=True)
class RerankTextInput:
    candidate_id: str
    text: str


@dataclass(frozen=True)
class RerankOutcome:
    applied: bool
    strategy: Literal[
        "model",
        "disabled",
        "dependency_missing",
        "model_load_failed",
        "prediction_failed",
        "no_candidates",
    ]
    scores: dict[str, float]
    message: str = ""


def rerank_texts(
    settings: AppSettings,
    query: str,
    items: list[RerankTextInput],
    top_n: int | None = None,
    model_name_override: str | None = None,
) -> RerankOutcome:
    if not settings.kb.ENABLE_MODEL_RERANK:
        return RerankOutcome(
            applied=False,
            strategy="disabled",
            scores={},
            message="ENABLE_MODEL_RERANK=false",
        )

    if not items:
        return RerankOutcome(
            applied=False,
            strategy="no_candidates",
            scores={},
            message="没有可供重排的候选。",
        )

    model_name = (model_name_override or settings.model.RERANK_MODEL).strip()
    if not model_name:
        return RerankOutcome(
            applied=False,
            strategy="disabled",
            scores={},
            message="RERANK_MODEL 为空。",
        )

    try:
        model = load_cross_encoder(
            model_name=resolve_rerank_model_path(settings, model_name),
            device=settings.model.RERANK_DEVICE.strip(),
        )
    except ImportError as exc:
        return RerankOutcome(
            applied=False,
            strategy="dependency_missing",
            scores={},
            message=str(exc),
        )
    except Exception as exc:
        return RerankOutcome(
            applied=False,
            strategy="model_load_failed",
            scores={},
            message=str(exc),
        )

    limit = min(len(items), max(top_n or settings.kb.RERANK_CANDIDATES_TOP_N, 1))
    selected_items = items[:limit]
    pairs = [(query, item.text) for item in selected_items]

    try:
        raw_scores = model.predict(pairs, show_progress_bar=False)
    except TypeError:
        raw_scores = model.predict(pairs)
    except Exception as exc:
        return RerankOutcome(
            applied=False,
            strategy="prediction_failed",
            scores={},
            message=str(exc),
        )

    scores = {
        item.candidate_id: normalize_rerank_score(raw_score)
        for item, raw_score in zip(selected_items, raw_scores, strict=False)
    }
    return RerankOutcome(
        applied=True,
        strategy="model",
        scores=scores,
    )


@lru_cache(maxsize=4)
def load_cross_encoder(model_name: str, device: str):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ImportError(
            "未安装 sentence-transformers；如需启用模型重排，请先安装可选依赖。"
        ) from exc

    kwargs: dict[str, str] = {}
    if device:
        kwargs["device"] = device
    return CrossEncoder(model_name, **kwargs)


def resolve_rerank_model_path(settings: AppSettings, model_name: str) -> str:
    raw = model_name.strip()
    candidate = Path(raw)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)

    resolved = settings.resolve_path(raw)
    if resolved.exists():
        return str(resolved)
    return raw


def normalize_rerank_score(raw_score: float) -> float:
    clipped = max(-18.0, min(18.0, float(raw_score)))
    return 1.0 / (1.0 + exp(-clipped))
