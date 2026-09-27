"""CPU-only tests for prospective Gemma development scheduling and gates."""

from __future__ import annotations

import unittest

from training.model.gemma4_full_plan import (
    TRAIN_COUNT,
    TRAIN_TOKENS,
    checkpoint_choice,
    schedule_receipt,
    select_stop_reason,
    select_summary,
)


class GemmaFullPlanTest(unittest.TestCase):
    def test_schedule_accounts_for_every_example_and_token(self) -> None:
        base, remainder = divmod(TRAIN_TOKENS, TRAIN_COUNT)
        lengths = [base + int(i < remainder) for i in range(TRAIN_COUNT)]
        result = schedule_receipt(lengths)
        self.assertEqual(result["updates"], 456)
        self.assertEqual(result["unpadded_tokens"], TRAIN_TOKENS)
        self.assertEqual(result["checkpoint_steps"], [16, 64, 128, 256, 456])
        self.assertEqual(result["select_steps"], [64, 128, 256, 456])
        with self.assertRaises(ValueError):
            schedule_receipt(lengths[:-1])

    def test_select_metrics_and_collapse_policy(self) -> None:
        records = []
        for kind, count, width in (
            ("choice", 320, 4),
            ("noul", 290, 2),
            ("score", 90, 5),
        ):
            for index in range(count):
                label = index % width
                records.append(
                    {
                        "task_type": kind,
                        "family": f"{kind}-family",
                        "probabilities": [
                            0.8 if i == label else 0.2 / (width - 1)
                            for i in range(width)
                        ],
                        "label": label,
                        "predicted": label,
                    }
                )
        result = select_summary(records)
        self.assertEqual(result["count"], 700)
        self.assertEqual(result["family_macro_accuracy"], 1.0)
        self.assertEqual(result["score_predicted_levels"], 5)
        self.assertIsNone(select_stop_reason(256, result))
        collapsed = {
            **result,
            "score_predicted_levels": 1,
            "score_max_predicted_share": 1.0,
        }
        self.assertIsNone(select_stop_reason(64, collapsed))
        self.assertEqual(
            select_stop_reason(256, collapsed), "score_prediction_collapse"
        )
        weak = {**result, "family_macro_accuracy": 0.69}
        self.assertEqual(select_stop_reason(256, weak), "futility_below_0.70")
        records[0]["predicted"] = -1
        self.assertEqual(
            select_summary(records)["by_type"]["choice"]["accuracy"], 319 / 320
        )
        records[0]["probabilities"] = [0.1, 0.1, 0.1, 0.1]
        with self.assertRaises(ValueError):
            select_summary(records)

    def test_checkpoint_selection_accuracy_then_brier_then_earliest(self) -> None:
        rows = {
            64: {"family_macro_accuracy": 0.8, "family_macro_brier": 0.2},
            128: {"family_macro_accuracy": 0.8, "family_macro_brier": 0.1},
            256: {"family_macro_accuracy": 0.8, "family_macro_brier": 0.1},
        }
        self.assertEqual(checkpoint_choice(rows), 128)


if __name__ == "__main__":
    unittest.main()
