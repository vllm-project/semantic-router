"""Gold-free public runtime parity rejects identity and answer drift."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from inference.run import digest
from publication import export_hf_v3 as export
from scripts import verify_public_hf_runtime_parity_v3 as parity


class FakeDecider:
    def __init__(self, answer: dict) -> None:
        self.answer = answer

    def decide_all(self, *, state, questions):
        assert state == "Question state"
        assert set(questions) == {"q"}
        return {"q": (self.answer, 10)}


class PublicRuntimeParityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prompts = [
            {
                "id": "one",
                "state": "Question state",
                "questions": {"q": {"type": "choice", "options": {"A": "A", "B": "B"}}},
            }
        ]
        self.answer = {
            "type": "choice",
            "choice": "A",
            "probabilities": {"A": 0.8, "B": 0.2},
        }
        self.predictions = self.root / "sealed.jsonl"
        self.row = {
            "id": "one",
            "model_id": "llm-semantic-router/DEV2.0-4B",
            "model_revision": "checkpoint-0232",
            "adapter_version": "decision2-eikos-semif-native-v1",
            "model_sha256": "a" * 64,
            "calibration_sha256": "b" * 64,
            "source_input_sha256": digest(
                {
                    "state": self.prompts[0]["state"],
                    "questions": self.prompts[0]["questions"],
                }
            ),
            "answers": {"q": self.answer},
        }
        self.predictions.write_text(json.dumps(self.row) + "\n")
        self.prompt_file = self.root / "prompts.jsonl"
        self.prompt_file.write_text(json.dumps(self.prompts[0]) + "\n")
        self.manifest_file = self.root / "scored-manifest.json"
        self.manifest = {
            "model_id": export.MODEL_ID,
            "model_revision": "checkpoint-0232",
            "adapter_version": "decision2-eikos-semif-native-v1",
            "model_sha256": "a" * 64,
            "package_manifest_sha256": "a" * 64,
            "calibration_sha256": "b" * 64,
            "input_sha256": export._sha(self.prompt_file),
            "predictions_sha256": export._sha(self.predictions),
            "input_items": 1,
            "evaluated_items": 1,
            "max_items": None,
            "runtime": {"torch_deterministic_algorithms": True},
        }
        self.manifest_file.write_text(json.dumps(self.manifest))

    def test_exact_prediction_parity_passes(self) -> None:
        references = parity._references(
            self.predictions,
            self.prompts,
            "a" * 64,
            "b" * 64,
            "checkpoint-0232",
        )
        result = parity._compare(FakeDecider(self.answer), self.prompts, references)
        self.assertEqual(result["items"], 1)
        self.assertEqual(result["answers"], 1)
        self.assertEqual(result["categorical_mismatches"], 0)
        self.assertEqual(result["maximum_probability_drift"], 0.0)

    def test_wrong_scored_identity_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not frozen scored output"):
            parity._references(
                self.predictions,
                self.prompts,
                "c" * 64,
                "b" * 64,
                "checkpoint-0232",
            )

    def test_complete_scored_manifest_binding_passes(self) -> None:
        result = parity._scored_manifest(
            self.manifest_file,
            prompts_path=self.prompt_file,
            predictions_path=self.predictions,
            item_count=1,
            private={
                "model_revision": "checkpoint-0232",
                "native_model_sha256": "a" * 64,
            },
            calibration_sha="b" * 64,
        )
        self.assertEqual(result, self.manifest)

    def test_subset_scored_manifest_is_rejected(self) -> None:
        self.manifest["max_items"] = 1
        self.manifest_file.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "complete panel"):
            parity._scored_manifest(
                self.manifest_file,
                prompts_path=self.prompt_file,
                predictions_path=self.predictions,
                item_count=1,
                private={
                    "model_revision": "checkpoint-0232",
                    "native_model_sha256": "a" * 64,
                },
                calibration_sha="b" * 64,
            )

    def test_changed_prediction_is_rejected(self) -> None:
        changed = {**self.answer, "choice": "B"}
        with self.assertRaisesRegex(ValueError, "categorical changes"):
            parity._compare(FakeDecider(changed), self.prompts, {"one": self.row})

    def test_probability_drift_is_rejected(self) -> None:
        changed = {
            **self.answer,
            "probabilities": {"A": 0.8001, "B": 0.1999},
        }
        with self.assertRaisesRegex(ValueError, "probability drift"):
            parity._compare(FakeDecider(changed), self.prompts, {"one": self.row})


if __name__ == "__main__":
    unittest.main()
