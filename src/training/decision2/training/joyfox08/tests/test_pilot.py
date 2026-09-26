"""Gold isolation, native conversion, and deterministic sampling contracts."""

from __future__ import annotations

import unittest

from training.joyfox08 import pilot
from training.joyfox08.train import metric_summary, schedule


def row(kind: str, index: int) -> dict:
    keys = (
        ["x", "y"]
        if kind == "choice"
        else ["true", "false"] if kind == "noul" else ["2", "0", "1"]
    )
    return {
        "id": f"{kind}-{index}",
        "task_type": kind,
        "state": {"beta": 2, "alpha": 1},
        "instructions": {"goal": "Choose"},
        "options": [{"key": key, "description": {"name": key}} for key in keys],
        "label": 0,
    }


class PilotTest(unittest.TestCase):
    def test_native_mapping_preserves_choice_order_and_score_level_order(self) -> None:
        choice = pilot.to_record(row("choice", 1))["questions"]["decision"]
        self.assertEqual(list(choice["criteria"]), ["x", "y"])
        self.assertEqual(choice["criteria"]["x"], '{"name":"x"}')
        self.assertEqual(choice["instructions"], '{"goal":"Choose"}')
        score = pilot.to_record(row("score", 1))["questions"]["decision"]
        self.assertEqual(
            score["criteria"], ['{"name":"0"}', '{"name":"1"}', '{"name":"2"}']
        )
        noul = pilot.to_record(row("noul", 1))["questions"]["decision"]
        self.assertNotIn("criteria", noul)
        self.assertEqual(pilot.target_key(row("noul", 1)), "true")

    def test_fixed_sample_is_order_invariant_and_fails_shortfall(self) -> None:
        rows = [row(kind, index) for kind in pilot.QUOTAS for index in range(210)]
        first = pilot.choose_rows(rows)
        self.assertEqual(
            [item["id"] for item in first],
            [item["id"] for item in pilot.choose_rows(list(reversed(rows)))],
        )
        self.assertEqual(len(first), 512)
        with self.assertRaisesRegex(ValueError, "Insufficient native-1024 score"):
            pilot.choose_rows([item for item in rows if item["task_type"] != "score"])

    def test_native_rejections_are_explicit(self) -> None:
        rows = [row("choice", 0), row("noul", 1), row("score", 2)]

        def validate(record):
            if record["questions"]["decision"]["type"] == "score":
                raise ValueError("at most 128 candidates are allowed")

        def encode(_tokenizer, record, _cutoff):
            if record["questions"]["decision"]["type"] == "noul":
                raise ValueError("input requires 1025 tokens, cutoff_len=1024")
            return {"segments": [0, 1, 1]}

        eligible, audit = pilot.assess_eligibility(rows, encode, validate, object())
        self.assertEqual([item["id"] for item in eligible], ["choice-0"])
        self.assertEqual(audit["overflow_by_type"], {"noul": 1})
        self.assertEqual(audit["native_schema_invalid_by_type"], {"score": 1})

    def test_frozen_schedule_and_family_macro(self) -> None:
        self.assertAlmostEqual(schedule(1), 1.25e-6)
        self.assertAlmostEqual(schedule(8), 1e-5)
        self.assertAlmostEqual(schedule(64), 1e-6)
        with self.assertRaisesRegex(ValueError, "outside frozen budget"):
            schedule(65)
        summary = metric_summary(
            {"a": [(True, 0.0), (False, 1.0)], "b": [(True, 0.1)]},
            {"choice": [True, False], "noul": [True]},
            1,
        )
        self.assertEqual(summary["correct"], 2)
        self.assertEqual(summary["total"], 3)
        self.assertAlmostEqual(summary["family_macro_accuracy"], 0.75)


if __name__ == "__main__":
    unittest.main()
