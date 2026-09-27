"""CPU contracts for the full native Choice/Noul TRAIN teacher source screen."""

from __future__ import annotations

import copy
import unittest

from research.autojev_choice_noul_train_audit import (
    EXPECTED,
    aggregate,
    gold_bucket,
    profile,
    roster_sha256,
    train_rows,
)
from research.test_eikos_teacher_train_pilot import row


def train_row(kind: str, index: int) -> dict:
    result = row(kind, index)
    result.update(
        {
            "source": "decision2_programmatic_original_v1",
            "family": "fixture",
            "language": "en",
        }
    )
    return result


class ChoiceNoulTrainAuditTest(unittest.TestCase):
    def test_all_rows_and_gold_free_roster(self) -> None:
        rows = [
            train_row(kind, index)
            for kind, count in EXPECTED.items()
            for index in range(count)
        ]
        selected = train_rows(rows)
        self.assertEqual(len(selected), sum(EXPECTED.values()))
        self.assertEqual(profile(selected)["by_type"], EXPECTED)
        changed = copy.deepcopy(rows)
        for item in changed:
            item["label"] = 1
        self.assertEqual(roster_sha256(rows), roster_sha256(changed))
        self.assertNotEqual(
            profile(rows)["by_type_gold_bucket"],
            profile(changed)["by_type_gold_bucket"],
        )

    def test_native_contract_and_class_bins(self) -> None:
        items = [train_row("choice", 0), train_row("noul", 0)]
        self.assertEqual(gold_bucket(items[0]), "first")
        self.assertEqual(gold_bucket(items[1]), "true")

        def decide(state: str, question: dict) -> tuple[dict, int]:
            if question["type"] == "noul":
                return {"type": "noul", "noul": 0.8}, 40
            return {"type": "choice", "probabilities": {"left": 0.8, "right": 0.2}}, 40

        stats, artifact = aggregate(items, decide)
        self.assertEqual(stats["overall"]["correct"], 2)
        self.assertEqual(stats["overall"]["valid"], 2)
        self.assertEqual(stats["by_type"]["choice"]["n"], 1)
        self.assertEqual(stats["by_type"]["noul"]["n"], 1)
        self.assertEqual(len(artifact), 2)
        self.assertNotIn("label", str(artifact))
        self.assertNotIn("state", str(artifact))

    def test_overflow_remains_failure_without_teacher_vector(self) -> None:
        def overflow(state: str, question: dict) -> tuple[dict, int]:
            raise ValueError(
                "Question branch exceeds the 8192-token limit; no input was truncated."
            )

        stats, artifact = aggregate([train_row("choice", 0)], overflow)
        self.assertEqual(stats["overall"]["invalid"], 1)
        self.assertEqual(stats["overall"]["context_overflow"], 1)
        self.assertEqual(stats["overall"]["correct"], 0)
        self.assertEqual(artifact, [])

    def test_unexpected_source_or_keys_stop_preflight(self) -> None:
        rows = [
            train_row(kind, index)
            for kind, count in EXPECTED.items()
            for index in range(count)
        ]
        rows[0]["source"] = "unrecorded-source"
        with self.assertRaisesRegex(ValueError, "source"):
            train_rows(rows)
        rows[0]["source"] = "decision2_programmatic_original_v1"
        rows[EXPECTED["choice"]]["options"][0]["key"] = "maybe"
        with self.assertRaisesRegex(ValueError, "option"):
            train_rows(rows)


if __name__ == "__main__":
    unittest.main()
