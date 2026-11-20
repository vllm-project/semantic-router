"""Gemini 4 evidence stays versioned and separate from Intelligence 1.0."""

from __future__ import annotations

import json
import unittest

from test_generate_model_catalog import DEFAULT_INDEX_ID, catalog


class GeminiCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.snapshot = json.loads(catalog.render_outputs()[catalog.WEBSITE_OUTPUT])

    def test_published_evidence_uses_exact_efforts_and_benchmark_versions(self) -> None:
        expected_profiles = {
            "artificial-analysis/gdp-pdf@1.0.0": "independent-standard",
            "harbor/terminal-bench@4.0.0": "independent-agent",
            "artificial-analysis/briefcase@1.1.0": "independent-agent",
            "artificial-analysis/omniscience@1.1.0": "independent-standard",
            "cais/hle-diamond@1.0.0": "independent-full-1000-no-tools",
        }
        for model, effort in (
            ("argon", "high"),
            ("carbon", "high"),
            ("hydrogen", "xhigh"),
            ("neon", "xhigh"),
        ):
            with self.subTest(model=model):
                records = [
                    item
                    for item in self.snapshot["evaluations"]
                    if item["model"] == f"google/gemini-4-{model}"
                ]
                self.assertEqual(len(records), 5)
                self.assertEqual(
                    {item["benchmark"]: item["benchmark_profile"] for item in records},
                    expected_profiles,
                )
                for record in records:
                    self.assertEqual(record["reasoning_effort"], effort)
                    self.assertEqual(record["subject"]["run_kind"], "independent")
                    self.assertEqual(record["evidence"]["provenance"], "third_party")
                    self.assertEqual(record["evidence"]["verification"], "imported")
                    self.assertEqual(
                        record["evidence"]["source"],
                        f"https://artificialanalysis.ai/models/gemini-4-{model}",
                    )
                    self.assertIn("observed_at", record)
                    self.assertNotIn("measured_at", record)

    def test_new_evidence_does_not_fill_default_index_or_other_efforts(self) -> None:
        for model in ("argon", "carbon", "hydrogen", "neon"):
            with self.subTest(model=model):
                rows = [
                    item
                    for item in self.snapshot["index_results"]
                    if item["model"] == f"google/gemini-4-{model}"
                    and item["index"] == DEFAULT_INDEX_ID
                ]
                efforts = {"low", "medium", "high"}
                if model in {"hydrogen", "neon"}:
                    efforts.add("xhigh")
                self.assertEqual({row["reasoning_effort"] for row in rows}, efforts)
                for row in rows:
                    self.assertEqual(row["status"], "missing")
                    self.assertIsNone(row["score"])
                    self.assertEqual(row["coverage"], 0)


if __name__ == "__main__":
    unittest.main()
