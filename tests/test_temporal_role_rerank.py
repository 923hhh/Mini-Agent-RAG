from __future__ import annotations

import unittest
from pathlib import Path

from langchain_core.documents import Document

from app.services.core.settings import load_settings
from app.services.retrieval.candidate_common_service import RetrievalCandidate
from app.services.retrieval.candidate_rerank_service import heuristic_rerank_candidates
from app.services.retrieval.query_profile_service import JointQueryProfile, infer_temporal_query_profile


ROOT_DIR = Path(__file__).resolve().parents[1]


class TemporalRoleRerankTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.settings = load_settings(ROOT_DIR)

    def test_current_role_query_is_profiled_as_temporal(self) -> None:
        profile = infer_temporal_query_profile(["中国人民大学的校长是谁？"])
        self.assertTrue(profile.is_temporal)
        self.assertTrue(profile.is_current_role_query)
        self.assertEqual(profile.asked_role_terms, ("校长",))

    def test_current_role_page_ranks_above_sibling_role_noise(self) -> None:
        query_bundle = ["中国人民大学的校长是谁？"]
        candidates = [
            RetrievalCandidate(
                document=Document(
                    page_content="2025年，中国人民大学校长为王某某，负责学校行政工作。",
                    metadata={
                        "chunk_id": "temporal-role-1",
                        "title": "中国人民大学校长介绍",
                        "section_title": "现任校长",
                        "source": "https://www.ruc.edu.cn/president",
                        "date": "2025-01-15",
                        "source_modality": "text",
                    },
                ),
                dense_relevance=0.82,
                lexical_score=0.80,
                fused_score=0.86,
            ),
            RetrievalCandidate(
                document=Document(
                    page_content="2025年，中国人民大学党委书记为李某某，负责党委工作。",
                    metadata={
                        "chunk_id": "temporal-role-2",
                        "title": "中国人民大学党委书记介绍",
                        "section_title": "现任书记",
                        "source": "https://www.ruc.edu.cn/party-secretary",
                        "date": "2025-01-15",
                        "source_modality": "text",
                    },
                ),
                dense_relevance=0.82,
                lexical_score=0.80,
                fused_score=0.86,
            ),
        ]

        ranked = heuristic_rerank_candidates(
            settings=self.settings,
            candidates=candidates,
            query_bundle=query_bundle,
            joint_query_profile=JointQueryProfile(False, False, False, False, False, (), (), (), ()),
            top_k=2,
        )

        self.assertEqual(ranked[0].document.metadata.get("chunk_id"), "temporal-role-1")
        self.assertGreater(ranked[0].rerank_score, ranked[1].rerank_score)


if __name__ == "__main__":
    unittest.main()
