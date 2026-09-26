"""Whole-group typed supplement selection and protected-context filtering."""

from __future__ import annotations

import unittest

from training.data import build_balanced_human as balanced
from training.data import build_pilot as pilot


def row(name: str, group: str, state: str, task_type: str = "noul") -> dict:
    return {
        "id": name,
        "group_id": group,
        "state": state,
        "input_sha256": pilot.sha_bytes(f"input:{name}".encode()),
        "task_type": task_type,
    }


class BalancedHumanTests(unittest.TestCase):
    def test_filter_excludes_complete_group_on_one_exact_context(self) -> None:
        source = [
            row("a", "group-a", "Same protected context."),
            row("b", "group-a", "Different group member."),
            row("c", "group-c", "Safe and independent example."),
        ]
        protected = [row("protected", "other", "Same protected context.")]
        groups, receipt = balanced.filter_groups(source, protected)
        self.assertEqual(set(groups), {"group-c"})
        self.assertEqual(receipt["rejected_rows_by_reason"]["exact_context"], 2)

    def test_selection_never_splits_pair_group(self) -> None:
        source = {
            "pair-a": [row("a1", "pair-a", "a1"), row("a2", "pair-a", "a2")],
            "pair-b": [row("b1", "pair-b", "b1"), row("b2", "pair-b", "b2")],
            "single": [row("s", "single", "s")],
        }
        picked, groups = balanced.choose_groups(source, "noul", 3, "seed", "source")
        self.assertEqual(len(picked), 3)
        self.assertEqual(sum(group.startswith("pair") for group in groups), 1)
        self.assertEqual(set(groups) & {"single"}, {"single"})

    def test_insufficient_group_quota_fails(self) -> None:
        source = {"pair": [row("a", "pair", "a"), row("b", "pair", "b")]}
        with self.assertRaises(ValueError):
            balanced.choose_groups(source, "noul", 1, "seed", "source")


if __name__ == "__main__":
    unittest.main()
