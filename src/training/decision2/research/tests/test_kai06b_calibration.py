"""Guard probability validity and group-based Kai calibration behavior."""

from __future__ import annotations

import json
import math
import unittest

from inference.run import digest
from research.kai06b_calibration import (
    _correct,
    _prior_bias,
    _probabilities,
    _stable_fold,
)
from research.kai06b_css_cal import _encode


def prediction(
    a: float, b: float, *, choice: str = "A", invalid: str | None = None
) -> dict:
    return {
        "answers": {"label": {"choice": choice, "probabilities": {"A": a, "B": b}}},
        "invalid_reason": invalid,
    }


class KaiCalibrationTests(unittest.TestCase):
    def test_rejects_invalid_or_inconsistent_probabilities(self) -> None:
        self.assertEqual(_probabilities(prediction(0.6, 0.4), ["A", "B"]), [0.6, 0.4])
        for row in (
            prediction(0.6, 0.4, choice="B"),
            prediction(0.6, 0.4, invalid="context_overflow"),
            prediction(float("nan"), 0.4),
            prediction(0.9, 0.2),
            prediction(-0.1, 1.1),
        ):
            self.assertIsNone(_probabilities(row, ["A", "B"]))

    def test_prior_correction_uses_cal_labels_and_normalizes(self) -> None:
        rows = [({"gold": "B"}, prediction(0.8, 0.2)) for _ in range(8)]
        rows += [({"gold": "A"}, prediction(0.8, 0.2)) for _ in range(2)]
        bias = _prior_bias(rows, ["A", "B"])
        adjusted = _correct([0.8, 0.2], bias=bias, strength=1.0)
        self.assertAlmostEqual(sum(adjusted), 1.0)
        self.assertTrue(all(math.isfinite(value) for value in adjusted))
        self.assertGreater(adjusted[1], adjusted[0])
        self.assertEqual(_stable_fold("same-lineage"), _stable_fold("same-lineage"))

    def test_cal_prompt_hash_survives_jsonl_serialization(self) -> None:
        payload = {
            "state": {"text": "示例"},
            "questions": {
                "label": {
                    "type": "choice",
                    "instructions": "Classify",
                    "criteria": {"A": "first", "B": "second"},
                }
            },
        }
        encoded = _encode([{"id": "cal-example", **payload}])
        reread = json.loads(encoded.decode().strip())
        self.assertEqual(
            digest(payload),
            digest({"state": reread["state"], "questions": reread["questions"]}),
        )


if __name__ == "__main__":
    unittest.main()
