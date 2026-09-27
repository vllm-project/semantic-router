"""Synthetic CPU tests for aggregate 0.6B exposure auditing."""

from __future__ import annotations

import unittest

from training.data.audit_small06_train_balance import native_parts, summarize
from training.model.data import canonical


class CharacterTokenizer:
    def encode(self, value: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return list(value.encode())


def row(kind: str, label: int, family: str) -> dict:
    return {
        "state": "Two facts.",
        "instructions": "Pick the supported answer.",
        "options": [
            {"key": "0", "description": "no"},
            {"key": "1", "description": "maybe"},
            {"key": "2", "description": "yes"},
        ],
        "task_type": kind,
        "label": label,
        "family": family,
        "group_id": "group-a",
        "source": "synthetic",
        "language": "en",
    }


class BalanceTests(unittest.TestCase):
    def test_native_parts_match_frozen_format(self) -> None:
        item = row("choice", 0, "test")
        parts = native_parts(item)
        self.assertTrue(
            parts[0].startswith("Context:\nTwo facts.\n\nTask type: choice")
        )
        self.assertIn(canonical(item["options"][0]), parts[1])
        self.assertEqual(len(parts), 5)

    def test_aggregate_counts_and_no_item_content(self) -> None:
        items = [row("choice", 1, "decision"), row("score", 2, "ordinal")]
        report = summarize(items, CharacterTokenizer())
        self.assertEqual(report["groups"], 1)
        self.assertEqual(report["score_level_rows"], {3: 1})
        self.assertEqual(report["score_three_family_rows"], {"ordinal": 1})
        self.assertEqual(report["language_rows"], {"en": 2})
        self.assertEqual(report["language_tokens"]["en"], report["native_tokens"])
        self.assertEqual(report["type_language_rows"], {"choice/en": 1, "score/en": 1})
        self.assertEqual(
            report["source_type_rows"], {"synthetic/choice": 1, "synthetic/score": 1}
        )
        self.assertEqual(report["non_three_score_group_sizes"], {})
        self.assertEqual(
            report["score_three_family_tokens"]["ordinal"],
            report["score_three_tokens"],
        )
        self.assertGreater(report["score_three_tokens"], 0)
        self.assertNotIn("Two facts", str(report))

    def test_invalid_label_fails(self) -> None:
        with self.assertRaises(ValueError):
            summarize([row("choice", 3, "test")], CharacterTokenizer())


if __name__ == "__main__":
    unittest.main()
