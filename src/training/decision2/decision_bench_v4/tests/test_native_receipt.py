"""The public scorer accepts the actual native inference sidecar contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from decision_bench_v4.bench import (
    BUILD_VERSION,
    compact,
    input_digest,
    score,
    sha_file,
)


class NativeReceiptTest(unittest.TestCase):
    def test_sidecar_identity_and_calibration_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            panel = root / "panel"
            panel.mkdir()
            prompts, targets, predictions = [], [], []
            for index in range(1041):
                item_id = f"case-{index:04d}"
                state = {"record": f"Evidence {index}"}
                questions = {
                    "decision": {
                        "type": "choice",
                        "instructions": "Choose the supported option.",
                        "criteria": {"a": "supported", "b": "unsupported"},
                    }
                }
                digest = input_digest(state, questions)
                prompts.append({"id": item_id, "state": state, "questions": questions})
                targets.append(
                    {
                        "id": item_id,
                        "task": "ENG-1",
                        "category": "engineering",
                        "modality": "text",
                        "labels": ["a", "b"],
                        "gold": "a",
                        "source_input_sha256": digest,
                    }
                )
                predictions.append(
                    {
                        "id": item_id,
                        "source_input_sha256": digest,
                        "input_sha256": digest,
                        "model_sha256": "model-fingerprint",
                        "adapter_sha256": "adapter-fingerprint",
                        "calibration_sha256": "calibration-file-hash",
                        "answers": {
                            "decision": {
                                "type": "choice",
                                "choice": "a",
                                "probabilities": {"a": 0.9, "b": 0.1},
                            }
                        },
                    }
                )
            for name, rows in (
                ("prompts", prompts),
                ("targets", targets),
                ("ineligible", [{"id": "visual-only"}]),
            ):
                (panel / f"{name}.jsonl").write_bytes(
                    b"".join(compact(row) for row in rows)
                )
            panel_manifest = {
                "build_version": BUILD_VERSION,
                "eligible_items": 1041,
                "ineligible_items": 30,
                **{
                    f"{name}_sha256": sha_file(panel / f"{name}.jsonl")
                    for name in ("prompts", "targets", "ineligible")
                },
                "scope": "synthetic test fixture",
            }
            (panel / "manifest.json").write_bytes(compact(panel_manifest))
            prediction_file = root / "predictions.jsonl"
            prediction_file.write_bytes(b"".join(compact(row) for row in predictions))
            receipt = {
                "input_sha256": panel_manifest["prompts_sha256"],
                "predictions_sha256": sha_file(prediction_file),
                "model_id": "candidate",
                "model_revision": "checkpoint-1",
                "model_sha256": "model-fingerprint",
                "adapter_sha256": "adapter-fingerprint",
                "calibration": {"file_sha256": "calibration-file-hash"},
                "input_items": 1041,
                "counts": {
                    "items": 1041,
                    "questions": 1041,
                    "valid_questions": 1041,
                    "invalid_questions": 0,
                },
            }
            receipt_file = root / "predictions.jsonl.manifest.json"
            receipt_file.write_bytes(compact(receipt))

            def run(name: str, *, manifest: bool = True) -> dict:
                return score(
                    panel,
                    prediction_file,
                    "candidate",
                    "checkpoint-1",
                    root / f"{name}.json",
                    prediction_manifest=receipt_file if manifest else None,
                )

            result = run("valid")
            self.assertEqual(result["correct"], 1041)
            self.assertEqual(result["invalid"], 0)
            self.assertEqual(
                result["prediction_manifest_sha256"], sha_file(receipt_file)
            )
            with self.assertRaisesRegex(ValueError, "input/model identity"):
                run("without-receipt", manifest=False)

            receipt["calibration"]["file_sha256"] = "wrong-calibration"
            receipt_file.write_bytes(compact(receipt))
            with self.assertRaisesRegex(ValueError, "packaged model identity"):
                run("wrong-calibration")
            receipt["calibration"]["file_sha256"] = "calibration-file-hash"

            receipt["counts"]["valid_questions"] = 1040
            receipt_file.write_bytes(compact(receipt))
            with self.assertRaisesRegex(ValueError, "Prediction manifest differs"):
                run("wrong-count")
            receipt["counts"]["valid_questions"] = 1041

            predictions[0]["model_sha256"] = "wrong-model"
            prediction_file.write_bytes(b"".join(compact(row) for row in predictions))
            receipt["predictions_sha256"] = sha_file(prediction_file)
            receipt_file.write_bytes(compact(receipt))
            with self.assertRaisesRegex(ValueError, "packaged model identity"):
                run("wrong-model")


if __name__ == "__main__":
    unittest.main()
