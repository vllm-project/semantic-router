"""Gold-blind pilot temperature transform and receipt checks."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.reweight_calibration import build, reweight_answer
from training.model.data import file_sha256
from training.model.infer import ADAPTER_VERSION, normalized_answer

TEMPERATURES = {"choice": 2.0, "noul": 0.8, "score": 3.0}


class ReweightTests(unittest.TestCase):
    def test_matches_direct_logit_temperature_for_all_types(self) -> None:
        cases = [
            ("choice", ["a", "b", "c"], [1.2, -0.4, 0.8]),
            ("noul", ["false", "true"], [-0.5, 1.1]),
            ("score", ["0", "1", "2", "3"], [-1.0, 0.2, 1.4, -0.3]),
        ]
        for kind, keys, logits in cases:
            with self.subTest(kind=kind):
                raw = normalized_answer(kind, keys, logits, 1.0)
                direct = normalized_answer(kind, keys, logits, TEMPERATURES[kind])
                reconstructed = reweight_answer(raw, TEMPERATURES)
                self.assertEqual(reconstructed.keys(), direct.keys())
                for key in direct:
                    if isinstance(direct[key], dict):
                        for option in direct[key]:
                            self.assertAlmostEqual(
                                reconstructed[key][option],
                                direct[key][option],
                                places=12,
                            )
                    elif isinstance(direct[key], float):
                        self.assertAlmostEqual(
                            reconstructed[key], direct[key], places=12
                        )
                    else:
                        self.assertEqual(reconstructed[key], direct[key])

    def test_refuses_probability_underflow(self) -> None:
        with self.assertRaisesRegex(ValueError, "underflow"):
            reweight_answer(
                {
                    "type": "choice",
                    "choice": "a",
                    "probabilities": {"a": 1.0, "b": 0.0},
                },
                TEMPERATURES,
            )
        with self.assertRaisesRegex(ValueError, "underflow"):
            reweight_answer({"type": "noul", "noul": 0.0}, TEMPERATURES)

    def test_receipt_binds_checkpoint_and_original_prediction_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction = root / "source.jsonl"
            row = {
                "id": "x",
                "model_sha256": "m",
                "adapter_sha256": "a",
                "answers": {
                    "q": normalized_answer("choice", ["a", "b"], [0.3, 0.1], 1.0)
                },
            }
            prediction.write_text(json.dumps(row) + "\n", encoding="utf-8")
            manifest = {
                "adapter_version": ADAPTER_VERSION,
                "predictions_sha256": file_sha256(prediction),
                "model_sha256": "m",
                "model_revision": "checkpoint-0000016",
                "adapter_sha256": "a",
                "counts": {"items": 1, "questions": 1},
            }
            receipt = prediction.with_name(prediction.name + ".manifest.json")
            receipt.write_text(json.dumps(manifest), encoding="utf-8")
            cal = root / "cal.json"
            cal.write_text(
                json.dumps(
                    {
                        "model_sha256": "m",
                        "selected_checkpoint": "checkpoint-0000016",
                        "cal_sha256": "c",
                        "temperature_by_type": TEMPERATURES,
                    }
                ),
                encoding="utf-8",
            )
            transformed, output_manifest = build(prediction, cal)
            self.assertEqual(len(transformed), 1)
            self.assertEqual(
                output_manifest["posthoc"]["source_predictions_sha256"],
                file_sha256(prediction),
            )
            self.assertEqual(transformed[0]["calibration_sha256"], file_sha256(cal))
            cal.write_text(
                json.dumps(
                    {
                        "model_sha256": "other",
                        "selected_checkpoint": "checkpoint-0000016",
                        "cal_sha256": "c",
                        "temperature_by_type": TEMPERATURES,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not bound"):
                build(prediction, cal)
            manifest["predictions_sha256"] = "corrupted"
            receipt.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "intact"):
                build(prediction, cal)


if __name__ == "__main__":
    unittest.main()
