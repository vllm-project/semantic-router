"""CPU-only checks for the gold-free factorial comparison."""

from __future__ import annotations

import copy
import unittest

from training.model.zero_step_factorial import CELLS, compare_cells


def cells() -> dict:
    rows = [
        {
            "id": f"select-{index}",
            "prompt_sha256": f"prompt-{index}",
            "token_ids_sha256": f"tokens-{index}",
            "task_type": "choice",
            "prediction_key": "a",
            "probabilities": {"a": 0.75, "b": 0.25},
        }
        for index in range(32)
    ]
    return {
        name: {
            "lock_sha256": "lock",
            "cell": name,
            "source_model_sha256": "source",
            "historical": {"status": "PASS"},
            "predictions": copy.deepcopy(rows),
        }
        for name in CELLS
    }


class FactorialTests(unittest.TestCase):
    def test_all_four_axes_and_numeric_failure(self):
        records = cells()
        self.assertTrue(
            all(
                value["within_1e-4"]
                for value in compare_cells(records)["comparisons"].values()
            )
        )
        records["reference-lora"]["predictions"][0]["probabilities"]["a"] += 0.02
        report = compare_cells(records)["comparisons"]
        self.assertFalse(report["lora_effect_reference"]["within_1e-4"])
        self.assertFalse(report["backend_effect_lora"]["within_1e-4"])
        self.assertTrue(report["lora_effect_fla"]["within_1e-4"])

    def test_rejects_missing_or_different_cells(self):
        records = cells()
        del records["fla-bare"]
        with self.assertRaisesRegex(ValueError, "four frozen cells"):
            compare_cells(records)
        records = cells()
        records["reference-bare"]["predictions"][0]["token_ids_sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "input"):
            compare_cells(records)
        records = cells()
        records["reference-bare"]["lock_sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "source or protocol"):
            compare_cells(records)


if __name__ == "__main__":
    unittest.main()
