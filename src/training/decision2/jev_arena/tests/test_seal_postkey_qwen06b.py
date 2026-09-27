"""Focused gold-free prediction-provenance checks for the 0.6B formal seal."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from inference.run import digest, file_digest
from jev_arena.seal_postkey_qwen06b import _candidate_rows, _control_rows


class PostKeyQwen06bSealTest(unittest.TestCase):
    def test_candidate_requires_full_answer_and_manifest_identity(self) -> None:
        prompt = {
            "id": "one",
            "state": "State",
            "questions": {
                "decision": {
                    "type": "choice",
                    "instructions": "Pick",
                    "criteria": {"a": "A", "b": "B"},
                }
            },
        }
        panel_sha = "a" * 64
        model = {
            "model_id": "research/model",
            "model_revision": "b" * 64,
            "model_sha256": "b" * 64,
            "adapter_sha256": "c" * 64,
            "adapter_version": "native-v1",
            "calibration_sha256": "d" * 64,
            "max_length": 8192,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            row = {
                "id": "one",
                "answers": {"decision": {"type": "choice", "choice": "a"}},
                "model_sha256": model["model_sha256"],
                "adapter_sha256": model["adapter_sha256"],
                "calibration_sha256": model["calibration_sha256"],
                "source_input_sha256": digest(
                    {"state": "State", "questions": prompt["questions"]}
                ),
            }
            path.write_text(json.dumps(row) + "\n")
            manifest = {
                "model_id": model["model_id"],
                "model_revision": model["model_revision"],
                "model_sha256": model["model_sha256"],
                "adapter_sha256": model["adapter_sha256"],
                "adapter_version": model["adapter_version"],
                "input_sha256": panel_sha,
                "max_length": 8192,
                "calibration": {"file_sha256": model["calibration_sha256"]},
                "predictions_sha256": file_digest(path),
                "input_items": 1,
                "counts": {"items": 1, "valid_questions": 1},
            }
            meta = path.with_name(path.name + ".manifest.json")
            meta.write_text(json.dumps(manifest))
            self.assertEqual(
                _candidate_rows(path, [prompt], model, panel_sha)["items"], 1
            )
            row["answers"] = {}
            path.write_text(json.dumps(row) + "\n")
            manifest["predictions_sha256"] = file_digest(path)
            meta.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "provenance or answers"):
                _candidate_rows(path, [prompt], model, panel_sha)

    def test_control_rejects_mismatched_question_identity(self) -> None:
        prompt = {
            "id": "one",
            "state": "S",
            "questions": {
                "q": {
                    "type": "noul",
                    "instructions": "Q",
                    "criteria": {"false": "F", "true": "T"},
                }
            },
        }
        model = {
            "model_id": "existing/control",
            "weight_revision": "rev",
            "adapter_version": "native-v2",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prior.jsonl"
            row = {
                "id": "one",
                "answers": {"q": {"type": "noul", "noul": 0.1}},
                "model_id": "existing/control",
                "model_revision": "rev",
                "adapter_version": "native-v2",
                "source_input_sha256": digest(
                    {"state": "S", "questions": prompt["questions"]}
                ),
            }
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(_control_rows(path, [prompt], model)["items"], 1)
            row["source_input_sha256"] = "0" * 64
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "provenance"):
                _control_rows(path, [prompt], model)


if __name__ == "__main__":
    unittest.main()
