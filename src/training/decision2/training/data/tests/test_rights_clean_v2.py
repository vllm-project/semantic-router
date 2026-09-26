"""Check human annotation projection and whole-comment quarantine."""

from __future__ import annotations

import unittest

from training.data.build_rights_clean_v2 import (
    _make_pair,
    _remove_overlaps,
    _select_balanced,
)
from training.model.data import validate_row


class GoEmotionsProjectionTest(unittest.TestCase):
    def test_pair_is_a_valid_group_in_every_partition(self) -> None:
        source = {
            "id": "abc123",
            "text": "Thanks for the help; I appreciate it.",
            "emotion": "gratitude",
        }
        for role in ("train", "select", "cal"):
            first = _make_pair(source, role, "fixed-seed")
            self.assertEqual(first, _make_pair(source, role, "fixed-seed"))
            self.assertEqual(
                {row["group_id"] for row in first}, {"goemotions-official-train:abc123"}
            )
            self.assertEqual({row["task_type"] for row in first}, {"choice", "noul"})
            self.assertEqual(first[0]["options"][first[0]["label"]]["key"], "gratitude")
            self.assertEqual(len(first[0]["options"]), 4)
            for row in first:
                validate_row(row, role)

    def test_exact_context_quarantines_whole_comment_group(self) -> None:
        sources = [
            {
                "id": "first",
                "text": "Thanks for the help; I appreciate it.",
                "emotion": "gratitude",
            },
            {
                "id": "second",
                "text": "The result makes me wonder what happened.",
                "emotion": "curiosity",
            },
        ]
        protected = _make_pair(sources[0], "train", "fixed-seed")
        kept, receipt = _remove_overlaps(sources, protected, "fixed-seed", "select")
        self.assertEqual([row["id"] for row in kept], ["second"])
        self.assertEqual(receipt["removed_comments"]["id_group"], 1)

    def test_balanced_source_selection(self) -> None:
        sources = [
            {"id": f"{label}-{index}", "emotion": label}
            for label in ("anger", "joy", "surprise")
            for index in range(3)
        ]
        chosen = _select_balanced(sources, 6, "fixed-seed")
        self.assertEqual(len({row["id"] for row in chosen}), 6)
        self.assertEqual(
            {
                label: sum(row["emotion"] == label for row in chosen)
                for label in ("anger", "joy", "surprise")
            },
            {"anger": 2, "joy": 2, "surprise": 2},
        )


if __name__ == "__main__":
    unittest.main()
