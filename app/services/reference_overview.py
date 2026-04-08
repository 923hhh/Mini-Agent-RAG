from __future__ import annotations

from app.schemas.chat import ReferenceOverview, RetrievedReference


def build_reference_overview(
    references: list[RetrievedReference],
) -> ReferenceOverview:
    source_modality_counts: dict[str, int] = {}
    evidence_type_counts: dict[str, int] = {}
    text_count = 0
    image_side_count = 0
    multimodal_count = 0

    for ref in references:
        source_modality = (ref.source_modality or "").strip() or "missing"
        evidence_type = (ref.evidence_type or "").strip() or "missing"
        source_modality_counts[source_modality] = source_modality_counts.get(source_modality, 0) + 1
        evidence_type_counts[evidence_type] = evidence_type_counts.get(evidence_type, 0) + 1

        if evidence_type == "text" or source_modality == "text":
            text_count += 1
        if evidence_type in {"ocr", "vision", "multimodal"} or source_modality in {
            "ocr",
            "vision",
            "ocr+vision",
            "image",
        }:
            image_side_count += 1
        if evidence_type == "multimodal" or source_modality == "ocr+vision":
            multimodal_count += 1

    return ReferenceOverview(
        reference_count=len(references),
        text_count=text_count,
        image_side_count=image_side_count,
        multimodal_count=multimodal_count,
        has_joint_text_image_coverage=text_count > 0 and image_side_count > 0,
        source_modality_counts=source_modality_counts,
        evidence_type_counts=evidence_type_counts,
    )
