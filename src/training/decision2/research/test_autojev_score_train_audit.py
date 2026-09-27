"""CPU contracts for an all-row, TRAIN-only Score teacher audit."""

from __future__ import annotations

import copy
import unittest

from research.autojev_score_train_audit import (
    EXPECTED_ROWS,
    _internal_source,
    aggregate,
    profile,
    roster_sha256,
    score_rows,
)
from research.test_eikos_teacher_train_pilot import row


def score_row(index: int) -> dict:
    result = row("score", index)
    result.update(
        {
            "source": "decision2_targeted_programmatic_v1",
            "family": "targeted_quantized_median",
            "language": "en",
            "audit_metadata": {"generator": "targeted-oracle-v1"},
        }
    )
    result["options"] = [{"key": str(i), "description": f"level {i}"} for i in range(3)]
    return result


class ScoreTrainAuditTest(unittest.TestCase):
    def test_full_roster_is_gold_free_and_preserves_grouping(self) -> None:
        rows = [score_row(i) for i in range(EXPECTED_ROWS)]
        rows[-1]["group_id"] = rows[0]["group_id"]
        selected = score_rows(rows)
        self.assertEqual(len(selected), EXPECTED_ROWS)
        self.assertEqual(profile(selected)["groups"], EXPECTED_ROWS - 1)
        changed = copy.deepcopy(rows)
        for item in changed:
            item["label"] = 2
        self.assertEqual(roster_sha256(rows), roster_sha256(changed))
        self.assertNotEqual(
            profile(rows)["by_gold_class"], profile(changed)["by_gold_class"]
        )

    def test_source_and_ordering_are_strict(self) -> None:
        item = score_row(0)
        self.assertTrue(_internal_source(item))
        item["audit_metadata"] = {}
        self.assertFalse(_internal_source(item))
        rows = [score_row(i) for i in range(EXPECTED_ROWS)]
        rows[0]["options"][0]["key"] = "2"
        with self.assertRaisesRegex(ValueError, "ordering"):
            score_rows(rows)

    def test_aggregate_has_gold_free_distributions_and_score_bins(self) -> None:
        items = [score_row(0), score_row(1)]
        items[0]["label"] = 0
        items[1]["label"] = 2

        def decide(state: str, question: dict) -> tuple[dict, int]:
            return {
                "type": "score",
                "probabilities": {"0": 0.6, "1": 0.2, "2": 0.2},
            }, 80

        results, distributions = aggregate(items, decide)
        self.assertEqual(results["overall"]["correct"], 1)
        self.assertEqual(results["overall"]["valid"], 2)
        self.assertEqual(results["by_gold_class"]["2"]["correct"], 0)
        self.assertEqual(len(distributions), 2)
        self.assertNotIn("label", str(distributions))
        self.assertNotIn("state", str(distributions))

    def test_documented_overflow_keeps_denominator(self) -> None:
        def decide(state: str, question: dict) -> tuple[dict, int]:
            raise ValueError(
                "Question branch exceeds the 8192-token limit; no input was truncated."
            )

        results, distributions = aggregate([score_row(0)], decide)
        self.assertEqual(results["overall"]["n"], 1)
        self.assertEqual(results["overall"]["invalid"], 1)
        self.assertEqual(results["overall"]["overflow"], 1)
        self.assertEqual(distributions, [])


if __name__ == "__main__":
    unittest.main()
