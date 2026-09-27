import json
import tempfile
import unittest
from pathlib import Path

from research.eos08_temperature_audit import (
    check_invariance,
    check_source,
    check_transformed,
)


class Eos08TemperatureAuditTest(unittest.TestCase):
    def test_source_binding_and_probability_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            row = {
                "id": "one",
                "model_sha256": "a" * 64,
                "calibration_sha256": "b" * 64,
                "adapter_sha256": "c" * 64,
                "adapter_status": "ok",
                "truncated_questions": 0,
                "answers": {
                    "decision": {
                        "type": "choice",
                        "choice": "a",
                        "probabilities": {"a": 0.8, "b": 0.2},
                    }
                },
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            manifest = {
                "model_sha256": "a" * 64,
                "model_revision": "checkpoint-0000001",
                "predictions_sha256": "d" * 64,
                "adapter_sha256": "c" * 64,
                "calibration": {
                    "file_sha256": "b" * 64,
                    "temperature_by_type": {
                        "choice": 2.0,
                        "noul": 1.0,
                        "score": 1.0,
                    },
                },
                "counts": {
                    "items": 1,
                    "questions": 1,
                    "valid_questions": 1,
                    "invalid_questions": 0,
                    "over_budget_questions": 0,
                    "truncated_questions": 0,
                },
            }
            calibration = {
                "temperature_by_type": {
                    "choice": 2.0,
                    "noul": 1.0,
                    "score": 1.0,
                }
            }
            expected = {
                "model_sha256": "a" * 64,
                "checkpoint": "checkpoint-0000001",
            }
            panel = {"predictions_sha256": "d" * 64, "n": 1}
            check_source(path, manifest, calibration, "b" * 64, expected, panel)
            row["answers"]["decision"]["probabilities"] = {"a": 1.0, "b": 0.0}
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "zero/one"):
                check_source(path, manifest, calibration, "b" * 64, expected, panel)
            with self.assertRaisesRegex(ValueError, "sidecar"):
                check_source(
                    path,
                    {**manifest, "model_sha256": "x"},
                    calibration,
                    "b" * 64,
                    expected,
                    panel,
                )

    def test_point_decision_invariance_fails_closed(self):
        before = {
            "overall": {"n": 1, "valid_n": 1, "correct_n": 1},
            "by_type": {
                kind: {"n": 1, "valid_n": 1, "correct_n": 1}
                for kind in ("choice", "noul", "score")
            },
        }
        after = json.loads(json.dumps(before))
        check_invariance(before, after, "dev")
        after["by_type"]["score"]["correct_n"] = 0
        with self.assertRaisesRegex(ValueError, "changed point decisions"):
            check_invariance(before, after, "dev")

    def test_individual_answer_invariance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, derived = root / "before.jsonl", root / "after.jsonl"
            row = {
                "id": "one",
                "answers": {
                    "decision": {
                        "type": "score",
                        "score": 0.8,
                        "probabilities": {"0": 0.2, "1": 0.8},
                    }
                },
            }
            original.write_text(json.dumps(row) + "\n")
            row["answers"]["decision"] = {
                "type": "score",
                "score": 0.9,
                "probabilities": {"0": 0.1, "1": 0.9},
            }
            derived.write_text(json.dumps(row) + "\n")
            check_transformed(original, derived, 1)
            row["answers"]["decision"] = {
                "type": "score",
                "score": 0.1,
                "probabilities": {"0": 0.9, "1": 0.1},
            }
            derived.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "categorical answer"):
                check_transformed(original, derived, 1)


if __name__ == "__main__":
    unittest.main()
