"""Checks for frozen human/anchor source identity and deterministic merge."""

from __future__ import annotations

import unittest

from training.data import build_human_anchor as human_anchor


class HumanAnchorTests(unittest.TestCase):
    def test_ordered_merge_keeps_every_source_id_once(self) -> None:
        anchor = [{"id": f"a-{index}"} for index in range(1024)]
        human = [{"id": f"h-{index}"} for index in range(3600)]
        first = human_anchor.ordered_merge(anchor, human, "fixed")
        second = human_anchor.ordered_merge(
            list(reversed(anchor)), list(reversed(human)), "fixed"
        )
        self.assertEqual([row["id"] for row in first], [row["id"] for row in second])
        self.assertEqual(len(first), 4624)
        self.assertEqual(
            {row["id"] for row in first}, {row["id"] for row in anchor + human}
        )

    def test_repeated_source_id_is_rejected(self) -> None:
        anchor = [{"id": f"a-{index}"} for index in range(1024)]
        human = [{"id": f"h-{index}"} for index in range(3599)] + [{"id": "a-0"}]
        with self.assertRaises(ValueError):
            human_anchor.ordered_merge(anchor, human, "fixed")

    def test_blank_seed_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            human_anchor.ordered_merge([], [], " ")


if __name__ == "__main__":
    unittest.main()
