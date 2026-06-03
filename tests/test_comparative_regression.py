from __future__ import annotations

import json
import unittest
from pathlib import Path

from app.retrievers.local_kb import search_local_knowledge_base
from app.services.core.settings import load_settings
from app.services.retrieval.query_rewrite_service import generate_multi_queries


ROOT_DIR = Path(__file__).resolve().parents[1]
SEED_CASES_PATH = ROOT_DIR / "data" / "eval" / "phase0" / "gold" / "phase0_gold_manual_seed.jsonl"
COMPARATIVE_CASE_PREFIXES = ("phase0-seed-domain-multi", "phase0-seed-domain-multidoc")
COMPARATIVE_REGRESSION_KB = "domainrag_full_corpus"
TOP_K = 5


def load_comparative_seed_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    with SEED_CASES_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            item = json.loads(payload)
            case_id = str(item.get("case_id", "")).strip()
            if not case_id.startswith(COMPARATIVE_CASE_PREFIXES):
                continue

            gold_titles: list[str] = []
            for document in item.get("gold_documents", []):
                title = str(document.get("title", "")).strip()
                if title and title not in gold_titles:
                    gold_titles.append(title)

            cases.append(
                {
                    "case_id": case_id,
                    "query": str(item["query"]).strip(),
                    "gold_titles": gold_titles,
                }
            )
    return cases


def title_hits_gold(title: str, gold_title: str) -> bool:
    normalized_title = title.strip()
    normalized_gold = gold_title.strip()
    return bool(normalized_title and normalized_gold) and (
        normalized_gold in normalized_title or normalized_title in normalized_gold
    )


class ComparativeRetrievalRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.settings = load_settings(ROOT_DIR)
        cls.cases = load_comparative_seed_cases()

    def test_expected_seed_cases_are_present(self) -> None:
        self.assertEqual(len(self.cases), 6, "Expected 6 comparative regression seed cases.")

    def test_comparative_seed_cases_cover_all_gold_titles_in_top5(self) -> None:
        for case in self.cases:
            with self.subTest(case_id=case["case_id"]):
                refs = search_local_knowledge_base(
                    settings=self.settings,
                    knowledge_base_name=COMPARATIVE_REGRESSION_KB,
                    query=str(case["query"]),
                    top_k=TOP_K,
                    score_threshold=0.0,
                    history=None,
                )
                top_titles = [str(ref.title).strip() for ref in refs]
                missing_titles = [
                    gold_title
                    for gold_title in case["gold_titles"]
                    if not any(title_hits_gold(title, gold_title) for title in top_titles)
                ]
                self.assertFalse(
                    missing_titles,
                    msg=(
                        f"{case['case_id']} missing gold titles in top{TOP_K}: {missing_titles}. "
                        f"Retrieved titles: {top_titles}"
                    ),
                )

    def test_employment_comparative_rewrite_stays_on_employment_dimension(self) -> None:
        employment_case = next(
            case for case in self.cases if case["case_id"] == "phase0-seed-domain-multi-doc-6"
        )
        rewrites = generate_multi_queries(self.settings, str(employment_case["query"]), None)
        rewrite_text = " ".join(rewrites)
        self.assertIn("就业方向", rewrite_text)
        self.assertIn("就业去向", rewrite_text)
        self.assertNotIn("人才培养目标", rewrite_text)

    def test_comparative_rewrite_preserves_entity_pair_and_aspects(self) -> None:
        query = "计算机专业与软件工程专业共同目标和独特特点是什么"
        rewrites = generate_multi_queries(self.settings, query, None)
        aspectful_rewrites = [
            item for item in rewrites
            if "计算机专业" in item and "软件工程专业" in item
        ]
        self.assertTrue(aspectful_rewrites, msg=f"Expected paired-entity rewrites, got: {rewrites}")
        self.assertTrue(
            any(("共同" in item or "共同点" in item) and ("特点" in item or "独特" in item or "区别" in item) for item in aspectful_rewrites),
            msg=f"Expected aspect-aware paired rewrite, got: {rewrites}",
        )


if __name__ == "__main__":
    unittest.main()
