"""The post-key strict HOLD receipt binds immutable v3 reports."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.create_strict_hold_receipt_v3 import receipt


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


class StrictHoldReceiptTest(unittest.TestCase):
    def test_exact_score_regression_and_frozen_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            freeze = {
                "schema_version": "jevarena-v3-freeze/2",
                "panels": {"typed_gold_sha256": "a" * 64},
                "models": {
                    key: {
                        "model_id": f"org/{key}",
                        "revision": f"revision-{key}",
                        "predictions_sha256": {"typed": digit * 64},
                    }
                    for key, digit in (("new", "1"), ("old", "2"))
                },
            }
            write(root / "freeze.json", freeze)
            for key, digit, accuracy in (("new", "1", 0.4175), ("old", "2", 0.445)):
                write(
                    root / f"{key}.json",
                    {
                        "schema_version": "typed-decision-report/2",
                        "split": "final",
                        "items": 1600,
                        "overall": {"n": 2000},
                        "gold_sha256": "a" * 64,
                        "predictions_sha256": digit * 64,
                        "model": {"id": f"org/{key}", "revision": f"revision-{key}"},
                        "by_type": {"score": {"accuracy_all": accuracy}},
                    },
                )
            result = receipt(root / "freeze.json", root / "new.json", root / "old.json")
            self.assertEqual(result["status"], "HOLD")
            self.assertEqual(result["candidate_score_accuracy_all"], 0.4175)
            old = json.loads((root / "old.json").read_text())
            old["predictions_sha256"] = "3" * 64
            write(root / "old.json", old)
            with self.assertRaisesRegex(ValueError, "frozen model predictions"):
                receipt(root / "freeze.json", root / "new.json", root / "old.json")


if __name__ == "__main__":
    unittest.main()
