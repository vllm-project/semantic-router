"""Qwen4-Max evidence stays on the xhigh bucket and the Gemini 4 benchmark versions."""

from __future__ import annotations

import json
import unittest

from test_generate_model_catalog import DEFAULT_INDEX_ID, catalog


class Qwen4CatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.snapshot = json.loads(catalog.render_outputs()[catalog.WEBSITE_OUTPUT])

    def test_dashscope_mapping_uses_the_native_id_and_provider_transport(self) -> None:
        providers = {item["id"]: item for item in self.snapshot["providers"]}
        dashscope = providers["dashscope"]
        self.assertEqual(
            dashscope["default_base_url"],
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        self.assertEqual(dashscope["reasoning_transport"], "top_level_effort")
        binding = next(
            item for item in dashscope["models"] if item["catalog"] == "qwen/qwen4-max"
        )
        self.assertEqual(binding["id"], "qwen4-max")
        self.assertEqual(binding["protocols"], ["openai/chat-completions@1"])
        self.assertEqual(
            binding["reasoning_transport"], "top_level_effort_boolean_switch"
        )
        families = {item["id"]: item for item in self.snapshot["reasoning_families"]}
        self.assertEqual(families["qwen4"]["activation_parameter"], "enable_thinking")
        self.assertEqual(families["qwen4"]["levels"], ["low", "high", "xhigh"])
        self.assertEqual(families["qwen4"]["default"], "high")
        self.assertEqual(families["qwen4"]["modes"], ["enabled", "disabled"])
        model = next(
            item for item in self.snapshot["models"] if item["id"] == "qwen/qwen4-max"
        )
        self.assertEqual(model["reasoning_family"], "qwen4")
        self.assertEqual(model["limits"]["context_window_size"], 1_000_000)
        self.assertEqual(model["limits"]["max_output_tokens"], 262_144)

    def test_published_xhigh_records_use_the_gemini_4_definitions(self) -> None:
        expected = {
            "artificial-analysis/gdp-pdf@1.0.0": (
                "published-standard",
                {"all_pass": 0.262},
            ),
            "harbor/terminal-bench@4.0.0": (
                "independent-agent",
                {"resolved": 0.484848484848485},
            ),
            "artificial-analysis/briefcase@1.1.0": (
                "independent-agent",
                {"elo": 1686.07, "elo_lower": 1675.59, "elo_upper": 1696.55},
            ),
            "artificial-analysis/omniscience@1.1.0": (
                "independent-standard",
                {
                    "accuracy": 0.437,
                    "net_score": 27.7,
                    "hallucination_rate": 0.2841918294849023,
                },
            ),
            "cais/hle-diamond@1.0.0": (
                "independent-full-1000-no-tools",
                {"accuracy": 0.449},
            ),
        }
        records = [
            item
            for item in self.snapshot["evaluations"]
            if item["model"] == "qwen/qwen4-max"
        ]
        self.assertEqual(
            {item["benchmark"]: item["benchmark_profile"] for item in records},
            {benchmark: profile for benchmark, (profile, _) in expected.items()},
        )
        for record in records:
            profile, metrics = expected[record["benchmark"]]
            self.assertEqual(record["benchmark_profile"], profile)
            self.assertEqual(record["reasoning_effort"], "xhigh")
            self.assertEqual(record["metrics"], metrics)
            self.assertEqual(record["subject"]["run_kind"], "independent")
            self.assertEqual(record["subject"]["source_model_slug"], "qwen4-max")
            self.assertEqual(record["evidence"]["provenance"], "third_party")
            self.assertEqual(record["evidence"]["verification"], "imported")
            self.assertEqual(
                record["evidence"]["source"],
                "https://artificialanalysis.ai/models/qwen4-max",
            )

    def test_xhigh_records_do_not_fill_the_default_index(self) -> None:
        rows = [
            item
            for item in self.snapshot["index_results"]
            if item["model"] == "qwen/qwen4-max" and item["index"] == DEFAULT_INDEX_ID
        ]
        self.assertEqual(
            {row["reasoning_effort"] for row in rows},
            {"low", "high", "xhigh"},
        )
        for row in rows:
            self.assertEqual(row["status"], "missing")
            self.assertIsNone(row["score"])
            self.assertEqual(row["coverage"], 0)


if __name__ == "__main__":
    unittest.main()
