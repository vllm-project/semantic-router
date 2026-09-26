"""Whole-group and holdout-quarantine checks for Nox-4B replay selection."""

from __future__ import annotations

import unittest

from training.data.build_nox4b_structured_mix import (
    QUOTAS,
    _group_candidates,
    _select_source,
)


def row(identifier: str, group: str, source: str, family: str, state: str) -> dict:
    return {
        "id": identifier,
        "group_id": group,
        "source": source,
        "family": family,
        "state": state,
        "input_sha256": identifier,
    }


class Nox4BStructuredMixTests(unittest.TestCase):
    def test_protected_context_quarantines_complete_counterfactual_group(self):
        source = next(iter(QUOTAS))
        candidates = [
            row(
                "counterfactual-a",
                "g1",
                source,
                "stage4_automaton",
                "The west switch is on.",
            ),
            row(
                "counterfactual-b",
                "g1",
                source,
                "stage4_automaton",
                "The west switch is off.",
            ),
            row(
                "distinct",
                "g2",
                source,
                "stage4_scope",
                "A document about three blue tools.",
            ),
        ]
        protected = [row("holdout", "held", "other", "other", "The west switch is on.")]
        eligible, audit = _group_candidates(candidates, protected)
        self.assertNotIn("g1", eligible)
        self.assertIn("g2", eligible)
        self.assertEqual(audit["rejected_rows"]["exact_context"], 2)

    def test_deterministic_selection_never_splits_group(self):
        source = next(iter(QUOTAS))
        candidates = {
            "pair": [
                row("a", "pair", source, "stage4_automaton", "A"),
                row("b", "pair", source, "stage4_automaton", "B"),
            ],
            "single": [row("c", "single", source, "stage4_scope", "C")],
        }
        first, groups = _select_source(candidates, source, 3, "fixed-seed")
        second, repeat = _select_source(candidates, source, 3, "fixed-seed")
        self.assertEqual(groups, repeat)
        self.assertEqual(first, second)
        self.assertEqual({item["id"] for item in first}, {"a", "b", "c"})
        with self.assertRaisesRegex(ValueError, "Not enough safe whole groups"):
            _select_source(candidates, source, 4, "fixed-seed")


if __name__ == "__main__":
    unittest.main()
