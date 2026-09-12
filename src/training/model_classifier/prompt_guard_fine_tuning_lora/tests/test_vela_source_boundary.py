"""Contrast semantics and source-family isolation are part of the data contract."""

import copy
import json
import unittest
from collections import Counter

from src.training.model_classifier.prompt_guard_fine_tuning_lora.vela_source_boundary import (
    DEFAULT_CONFIG,
    build_rows,
)


class SourceBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(DEFAULT_CONFIG.read_text())
        self.rows = build_rows(self.config)

    def test_frozen_family_counts_and_language_pairing(self):
        self.assertEqual(len(self.rows), 300)
        self.assertEqual(
            Counter(row["split"] for row in self.rows),
            {"train": 200, "development": 100},
        )
        for group in {row["group_id"] for row in self.rows}:
            rows = [row for row in self.rows if row["group_id"] == group]
            self.assertEqual(len(rows), 10)
            self.assertEqual({row["language"] for row in rows}, {"en", "zh"})
            self.assertEqual(len({row["split"] for row in rows}), 1)

    def test_ordinary_harm_is_not_a_control_attack(self):
        rows = [row for row in self.rows if row["condition"] == "ordinary_harmful"]
        self.assertTrue(rows)
        self.assertEqual({row["label"] for row in rows}, {"benign"})
        self.assertEqual({row["content_risk"] for row in rows}, {"harmful"})

    def test_control_attack_crosses_both_content_risk_classes(self):
        rows = [row for row in self.rows if row["label"] == "jailbreak"]
        self.assertEqual({row["content_risk"] for row in rows}, {"benign", "harmful"})
        self.assertEqual(len(rows), 120)

    def test_quoted_controls_remain_benign(self):
        rows = [row for row in self.rows if row["condition"] == "quoted_control"]
        self.assertEqual(len(rows), 60)
        self.assertEqual({row["label"] for row in rows}, {"benign"})

    def test_duplicate_family_and_changed_quote_fail(self):
        config = copy.deepcopy(self.config)
        config["families"].append(config["families"][0])
        with self.assertRaises(ValueError):
            build_rows(config)
        config = copy.deepcopy(self.config)
        config["families"][0]["languages"]["en"]["quoted_control"] = "Unrelated text."
        with self.assertRaises(ValueError):
            build_rows(config)

    def test_duplicate_text_and_missing_translation_fail(self):
        config = copy.deepcopy(self.config)
        config["families"][1]["languages"]["en"]["ordinary_safe"] = config["families"][
            0
        ]["languages"]["en"]["ordinary_safe"]
        with self.assertRaises(ValueError):
            build_rows(config)
        config = copy.deepcopy(self.config)
        del config["families"][0]["languages"]["zh"]
        with self.assertRaises(ValueError):
            build_rows(config)


if __name__ == "__main__":
    unittest.main()
