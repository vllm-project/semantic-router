from __future__ import annotations

import unittest
from pathlib import Path

import yaml

PROVIDER_PATH = (
    Path(__file__).resolve().parents[3] / "config/catalog/resources/providers/groq.yaml"
)
SOURCE = "https://console.groq.com/docs/models"
OBSERVED_AT = "2026-09-22"
EXPECTED_MAPPINGS = {
    "openai/gpt-oss-120b": {
        "id": "openai/gpt-oss-120b",
        "prompt_per_1m": 0.15,
        "completion_per_1m": 0.60,
        "lifecycle": "active",
    },
    "openai/gpt-oss-20b": {
        "id": "openai/gpt-oss-20b",
        "prompt_per_1m": 0.075,
        "completion_per_1m": 0.30,
        "lifecycle": "active",
    },
    "qwen/qwen3.8-27b": {
        "id": "qwen/qwen3.8-27b",
        "prompt_per_1m": 0.80,
        "completion_per_1m": 4.00,
        "lifecycle": "experimental",
    },
}


class GroqProviderMappingsTests(unittest.TestCase):
    def test_initial_mappings_match_the_dated_groq_catalog_snapshot(self) -> None:
        provider = yaml.safe_load(PROVIDER_PATH.read_text())
        mappings = provider["models"]

        self.assertEqual(provider["conformance"]["status"], "live_verified")
        self.assertEqual(str(provider["conformance"]["verified_at"]), OBSERVED_AT)

        self.assertEqual(len(mappings), len(EXPECTED_MAPPINGS))
        self.assertEqual(
            {mapping["catalog"] for mapping in mappings}, set(EXPECTED_MAPPINGS)
        )
        self.assertEqual(
            {mapping["id"] for mapping in mappings},
            {expected["id"] for expected in EXPECTED_MAPPINGS.values()},
        )

        for mapping in mappings:
            with self.subTest(catalog=mapping["catalog"]):
                expected = EXPECTED_MAPPINGS[mapping["catalog"]]
                self.assertEqual(mapping["relationship"], "managed_cloud")
                self.assertEqual(mapping["id"], expected["id"])
                self.assertEqual(mapping["protocols"], ["openai/chat-completions@1"])
                self.assertEqual(mapping["lifecycle"], expected["lifecycle"])
                self.assertEqual(mapping["pricing"]["currency"], "USD")
                self.assertEqual(
                    mapping["pricing"]["prompt_per_1m"], expected["prompt_per_1m"]
                )
                self.assertEqual(
                    mapping["pricing"]["completion_per_1m"],
                    expected["completion_per_1m"],
                )
                self.assertEqual(mapping["verification"]["status"], "reproduced")
                self.assertEqual(
                    str(mapping["verification"]["verified_at"]), OBSERVED_AT
                )
                self.assertEqual(mapping["verification"]["source"], SOURCE)
