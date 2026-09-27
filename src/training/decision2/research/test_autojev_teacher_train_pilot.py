"""Small CPU contracts for the frozen native AutoJev TRAIN source screen."""

from __future__ import annotations

import unittest

from research.autojev_teacher_train_pilot import aggregate
from research.test_eikos_teacher_train_pilot import row


class AutoJevTeacherPilotTest(unittest.TestCase):
    def test_all_three_native_types_and_score_gold_key(self) -> None:
        rows = [row(kind, 0) for kind in ("choice", "noul", "score")]

        def decide(state: str, item: dict) -> tuple[dict, int]:
            kind = item["type"]
            if kind == "noul":
                return {"type": kind, "noul": 0.8}, 20
            if kind == "choice":
                return {"type": kind, "probabilities": {"left": 0.7, "right": 0.3}}, 20
            return {"type": kind, "probabilities": {"0": 0.2, "1": 0.2, "2": 0.6}}, 20

        result = aggregate(rows, decide)
        self.assertEqual([result[k]["valid"] for k in result], [1, 1, 1])
        self.assertEqual([result[k]["correct"] for k in result], [1, 1, 1])
        self.assertEqual([result[k]["ties"] for k in result], [0, 0, 0])

    def test_documented_overflow_is_invalid_unexpected_error_stops(self) -> None:
        def overflow(state: str, item: dict) -> tuple[dict, int]:
            raise ValueError(
                "Question branch exceeds the 8192-token limit; no input was truncated."
            )

        result = aggregate([row("choice", 0)], overflow)
        self.assertEqual(result["choice"]["invalid"], 1)
        self.assertEqual(result["choice"]["correct"], 0)

        def bad(state: str, item: dict) -> tuple[dict, int]:
            raise ValueError("Unexpected native return")

        with self.assertRaisesRegex(ValueError, "Unexpected"):
            aggregate([row("choice", 0)], bad)


if __name__ == "__main__":
    unittest.main()
