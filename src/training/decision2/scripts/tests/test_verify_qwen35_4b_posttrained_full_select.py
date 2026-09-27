"""The sealed full-arm reload must honor the trainer's SELECT-only selector."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.verify_qwen35_4b_posttrained_full_select import _best

STEPS = (64, 128, 192, 256, 320, 384, 448, 466)


class BestReloadSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "COMPLETE.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "step": 466,
                    "planned_updates": 466,
                    "calibration_status": "untouched",
                    "best": "checkpoint-0000256",
                }
            ),
            encoding="utf-8",
        )
        (self.root / "BEST.json").write_text(
            json.dumps({"checkpoint": "checkpoint-0000256"}), encoding="utf-8"
        )
        for step in STEPS:
            checkpoint = self.root / f"checkpoint-{step:07d}"
            checkpoint.mkdir()
            accuracy = 0.7 if step in (192, 256, 320) else 0.6
            brier = 0.15 if step in (256, 320) else 0.2
            (checkpoint / "checkpoint.json").write_text(
                json.dumps(
                    {
                        "dev_metrics": {
                            "family_macro_accuracy": accuracy,
                            "family_macro_brier": brier,
                        }
                    }
                ),
                encoding="utf-8",
            )
        (self.root / "select-step-0000256-predictions.jsonl").write_text(
            "{}\n", encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_accuracy_then_brier_then_earliest_update(self) -> None:
        step, checkpoint, predictions = _best(self.root)
        self.assertEqual(step, 256)
        self.assertEqual(checkpoint.name, "checkpoint-0000256")
        self.assertEqual(predictions.name, "select-step-0000256-predictions.jsonl")

    def test_rejects_override_of_select_best(self) -> None:
        (self.root / "BEST.json").write_text(
            json.dumps({"checkpoint": "checkpoint-0000320"}), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "BEST differs"):
            _best(self.root)

    def test_rejects_incomplete_or_calibrated_run(self) -> None:
        complete = self.root / "COMPLETE.json"
        value = json.loads(complete.read_text(encoding="utf-8"))
        value["calibration_status"] = "fitted"
        complete.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "did not complete"):
            _best(self.root)


if __name__ == "__main__":
    unittest.main()
