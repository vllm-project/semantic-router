"""Source-group and deterministic merge checks for the 3,024-row candidate."""

from __future__ import annotations

import unittest

from training.data import build_targeted_anchor as anchor
from training.data import build_targeted_candidate as targeted


class TargetedAnchorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fresh = targeted.generate("anchor-test-source")

    def test_complete_pair_and_source_groups(self) -> None:
        old = [
            {"id": "old-a", "group_id": "old-group"},
            {"id": "old-b", "group_id": "old-group"},
        ]
        receipt = anchor.verify_group_integrity(old, old, self.fresh)
        self.assertEqual(receipt["frozen_anchor_groups"], 1)
        self.assertEqual(receipt["targeted_complete_pair_groups"], 1000)
        with self.assertRaisesRegex(ValueError, "split frozen source groups"):
            anchor.verify_group_integrity(old[:1], old, self.fresh)

    def test_fixed_seed_order_preserves_all_ids(self) -> None:
        old = [{"id": f"old-{i}", "group_id": f"old-group-{i}"} for i in range(1024)]
        first = anchor.ordered_merge(old, self.fresh, "fixed-seed")
        second = anchor.ordered_merge(old, self.fresh, "fixed-seed")
        self.assertEqual([row["id"] for row in first], [row["id"] for row in second])
        self.assertEqual(len(first), 3024)
        self.assertEqual(
            {row["group_id"] for row in first if row["id"].startswith("d2t_")},
            {row["group_id"] for row in self.fresh},
        )
        with self.assertRaisesRegex(AssertionError, "3,024 unique rows"):
            anchor.ordered_merge(old[:-1] + [old[0]], self.fresh, "fixed-seed")


if __name__ == "__main__":
    unittest.main()
