import json
import unittest
from pathlib import Path


class OpenAIExactEffortTests(unittest.TestCase):
    def test_published_scores_belong_to_the_measured_variant(self) -> None:
        root = Path(__file__).resolve().parents[3]
        resources = json.loads(
            (root / "website/static/model-catalog/catalog.json").read_text()
        )
        cases = {
            ("gpt-6.2-astra", "xhigh"): {
                "arc-prize/arc-agi-1@1.0.0": ("semi-private", "accuracy", 0.985),
                "arc-prize/arc-agi-2@1.0.0": (
                    "semi-private",
                    "accuracy",
                    0.9583333333333334,
                ),
                "duellab/gamebench@2.0.0": ("public-games", "score", 97.146404),
                "harbor/terminal-bench@4.0.0": (
                    "independent-agent",
                    "resolved",
                    0.641414141414141,
                ),
                "artificial-analysis/gdp-pdf@1.0.0": (
                    "published-standard",
                    "all_pass",
                    0.28,
                ),
            },
            ("gpt-6.2-sol", "xhigh"): {
                "arc-prize/arc-agi-1@1.0.0": ("semi-private", "accuracy", 0.97),
                "arc-prize/arc-agi-2@1.0.0": (
                    "semi-private",
                    "accuracy",
                    0.8583333333333333,
                ),
                "duellab/gamebench@2.0.0": ("public-games", "score", 94.330527),
                "harbor/terminal-bench@4.0.0": (
                    "independent-agent",
                    "resolved",
                    0.560606060606061,
                ),
                "artificial-analysis/gdp-pdf@1.0.0": (
                    "published-standard",
                    "all_pass",
                    0.284,
                ),
            },
            ("gpt-6.1-terra", "high"): {
                "scicode-bench/scicode@1.0.0": (
                    "independent-test-288",
                    "score",
                    0.516203703703704,
                ),
                "cais/humanitys-last-exam@1.0.0": (
                    "independent-text-only",
                    "accuracy",
                    0.432808155699722,
                ),
                "artificial-analysis/gdp-pdf@1.0.0": (
                    "published-standard",
                    "all_pass",
                    0.28,
                ),
                "artificial-analysis/critpt@1.0.0": (
                    "independent-standard",
                    "score",
                    0.257142857142857,
                ),
                "artificial-analysis/lcr@1.1.0": (
                    "independent-standard",
                    "score",
                    0.823333333333333,
                ),
            },
        }
        for (model, effort), expected in cases.items():
            with self.subTest(model=model, effort=effort):
                records = [
                    r
                    for r in resources["evaluations"]
                    if r["model"] == f"openai/{model}"
                    and r["reasoning_effort"] == effort
                    and r["status"] == "available"
                ]
                self.assertEqual({r["benchmark"] for r in records}, set(expected))
                for record in records:
                    profile, metric, score = expected[record["benchmark"]]
                    self.assertEqual(record["benchmark_profile"], profile)
                    self.assertEqual(record["metrics"][metric], score)
                    if record["benchmark"] == "duellab/gamebench@2.0.0":
                        self.assertIn(
                            "gb2-20261119-235af8c04037", record["evidence"]["source"]
                        )
