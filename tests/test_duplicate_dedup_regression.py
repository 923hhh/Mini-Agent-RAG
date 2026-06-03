from __future__ import annotations

import unittest

from langchain_core.documents import Document

from app.services.retrieval.candidate_common_service import RetrievalCandidate
from app.services.retrieval.candidate_rerank_service import (
    apply_same_sample_group_rerank_adjustments,
    diversify_candidates,
)
from app.services.retrieval.query_profile_service import (
    DiversityQueryProfile,
    JointQueryProfile,
    infer_query_modality_profile,
)


def build_candidate(
    *,
    chunk_id: str,
    doc_id: str,
    family_id: str,
    rerank_score: float,
    answer_support_bonus: float = 0.0,
    answer_focus_score: float = 0.0,
    sentence_text: str = "",
    sample_id: str = "",
) -> RetrievalCandidate:
    return RetrievalCandidate(
        document=Document(
            page_content=f"{chunk_id} content",
            metadata={
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "reference_id": family_id,
                "source_modality": "text",
                "sample_id": sample_id,
            },
        ),
        rerank_score=rerank_score,
        answer_support_bonus=answer_support_bonus,
        answer_focus_score=answer_focus_score,
        sentence_text=sentence_text,
    )


class DuplicateDedupRegressionTest(unittest.TestCase):
    def test_diversify_prefers_new_family_before_same_family_filler(self) -> None:
        candidates = [
            build_candidate(
                chunk_id="family-a-primary",
                doc_id="doc-a-1",
                family_id="family-a",
                rerank_score=1.20,
            ),
            build_candidate(
                chunk_id="family-a-filler",
                doc_id="doc-a-2",
                family_id="family-a",
                rerank_score=1.15,
            ),
            build_candidate(
                chunk_id="family-b-unique",
                doc_id="doc-b-1",
                family_id="family-b",
                rerank_score=1.05,
            ),
            build_candidate(
                chunk_id="family-a-answer",
                doc_id="doc-a-3",
                family_id="family-a",
                rerank_score=0.98,
                answer_support_bonus=0.08,
                answer_focus_score=0.35,
                sentence_text="现任校长为某某。",
            ),
        ]

        selected = diversify_candidates(
            candidates,
            target_count=3,
            query_profile=infer_query_modality_profile(["南京航空航天大学校长是谁？"]),
            joint_query_profile=JointQueryProfile(False, False, False, False, False, (), (), (), ()),
            diversity_profile=DiversityQueryProfile(prefer_family_diversity=False),
            query_bundle=["南京航空航天大学校长是谁？"],
        )

        selected_ids = [item.document.metadata.get("chunk_id") for item in selected]
        self.assertEqual(
            selected_ids,
            ["family-a-primary", "family-b-unique", "family-a-answer"],
        )

    def test_same_sample_group_adjustment_stays_conservative(self) -> None:
        dominant_top = build_candidate(
            chunk_id="dom-1",
            doc_id="dom-doc-1",
            family_id="dom-family-1",
            rerank_score=1.00,
            sample_id="dominant",
        )
        dominant_second = build_candidate(
            chunk_id="dom-2",
            doc_id="dom-doc-2",
            family_id="dom-family-2",
            rerank_score=0.95,
            sample_id="dominant",
        )
        minority = build_candidate(
            chunk_id="minority-1",
            doc_id="minority-doc-1",
            family_id="minority-family",
            rerank_score=0.90,
            sample_id="minority",
        )
        candidates = [dominant_top, dominant_second, minority]
        before_scores = [item.rerank_score for item in candidates]

        apply_same_sample_group_rerank_adjustments(candidates)

        dominant_boost = candidates[0].rerank_score - before_scores[0]
        minority_penalty = before_scores[2] - candidates[2].rerank_score
        self.assertGreater(dominant_boost, 0.0)
        self.assertGreater(minority_penalty, 0.0)
        self.assertLess(dominant_boost, 0.15)
        self.assertLess(minority_penalty, 0.12)


if __name__ == "__main__":
    unittest.main()
