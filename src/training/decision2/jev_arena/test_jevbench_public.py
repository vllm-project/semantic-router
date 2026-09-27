"""Tests for JevBench public scoring rules and gold isolation."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from jev_arena.jevbench_public import (
    _ece_15,
    _evaluate,
    _native_manifest,
    _native_row_identity,
    _valid_probs,
    input_digest,
)


class PublicScoreTest(unittest.TestCase):
    def test_exact_label_set_and_renormalization(self) -> None:
        self.assertIsNone(_valid_probs({"a": 0.6}, ["a", "b"]))
        self.assertIsNone(_valid_probs({"a": 0.6, "b": 0.4, "c": 0}, ["a", "b"]))
        self.assertIsNone(_valid_probs({"a": math.nan, "b": 1}, ["a", "b"]))
        self.assertIsNone(_valid_probs({"a": True, "b": 0}, ["a", "b"]))
        self.assertIsNone(_valid_probs({"a": 0.7, "b": 0.34}, ["a", "b"]))
        accepted = _valid_probs({"a": 0.601, "b": 0.403}, ["a", "b"])
        self.assertIsNotNone(accepted)
        probs, strict, renormalized = accepted
        self.assertFalse(strict)
        self.assertTrue(renormalized)
        self.assertAlmostEqual(sum(probs.values()), 1)

    def test_choice_uses_probabilities_and_lexicographic_ties(self) -> None:
        target = {"task_type": "choice", "labels": ["z", "a"], "expected": "a"}
        result = _evaluate(
            {"type": "choice", "choice": "z", "probabilities": {"z": 0.5, "a": 0.5}},
            target,
        )
        self.assertTrue(result["valid"])
        self.assertTrue(result["correct"])
        self.assertEqual(result["predicted"], "a")
        self.assertTrue(result["point_disagrees_with_argmax"])

    def test_noul_probability_and_score_argmax(self) -> None:
        binary = {"task_type": "noul", "labels": ["no", "yes"], "expected": "yes"}
        self.assertTrue(_evaluate({"type": "noul", "noul": 0.8}, binary)["correct"])
        self.assertFalse(_evaluate({"type": "noul", "noul": 1.1}, binary)["valid"])
        ordinal = {"task_type": "score", "labels": ["0", "1", "2"], "expected": 2}
        result = _evaluate(
            {
                "type": "score",
                "score": 1.1,
                "probabilities": {"0": 0.01, "1": 0.48, "2": 0.51},
            },
            ordinal,
        )
        self.assertTrue(result["correct"])

    def test_missing_counts_as_wrong_and_ece(self) -> None:
        target = {"task_type": "choice", "labels": ["a", "b"], "expected": "a"}
        missing = _evaluate(None, target)
        self.assertFalse(missing["valid"])
        self.assertFalse(missing["correct"])
        self.assertIsNone(_ece_15([missing]))
        correct = _evaluate({"probabilities": {"a": 0.8, "b": 0.2}}, target)
        self.assertAlmostEqual(_ece_15([correct]), 0.2)

    def test_input_hash_ignores_gold(self) -> None:
        state, questions = "example", {"decision": {"type": "noul"}}
        digest = input_digest(state, questions)
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest, input_digest(state, questions))
        self.assertNotEqual(digest, input_digest("other", questions))

    def test_native_companion_receipt_binds_prediction_and_prompt_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts = root / "prompts.jsonl"
            predictions = root / "predictions.jsonl"
            receipt_path = root / "predictions.jsonl.manifest.json"
            prompts.write_text('{"id":"a"}\n', encoding="utf-8")
            predictions.write_text('{"id":"a"}\n', encoding="utf-8")

            def digest(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest()

            receipt = {
                "model_id": "candidate",
                "model_revision": "checkpoint-1",
                "predictions_sha256": digest(predictions),
                "input_sha256": digest(prompts),
                "input_items": 1,
                "counts": {"items": 1},
                "model_sha256": "a" * 64,
                "adapter_sha256": "b" * 64,
            }
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            self.assertEqual(
                _native_manifest(
                    receipt_path, predictions, prompts, "candidate", "checkpoint-1", 1
                ),
                receipt,
            )
            predictions.write_text('{"id":"other"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "binding differs"):
                _native_manifest(
                    receipt_path, predictions, prompts, "candidate", "checkpoint-1", 1
                )

    def test_eikos_collector_receipt_requires_source_and_stable_full_run(self) -> None:
        from scripts.eikos_stable_runtime_v3 import FLA_BACKEND, TORCH_BACKEND

        collector = (
            Path(__file__).resolve().parents[1] / "training/eikos/published_infer.py"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts, predictions, receipt_path = (
                root / "prompts.jsonl",
                root / "predictions.jsonl",
                root / "receipt.json",
            )
            prompts.write_text("{}\n", encoding="utf-8")
            predictions.write_text("{}\n", encoding="utf-8")
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            receipt = {
                "model_id": "candidate",
                "model_revision": "checkpoint-1",
                "predictions_sha256": digest(predictions),
                "input_sha256": digest(prompts),
                "input_items": 1,
                "evaluated_items": 1,
                "counts": {"items": 1},
                "max_items": None,
                "model_sha256": "a" * 64,
                "calibration_sha256": "c" * 64,
                "calibration": {"file_sha256": "c" * 64},
                "collector_source_sha256": digest(collector),
                "adapter_version": "eikos-native/1",
                "runtime": {
                    "torch_deterministic_algorithms": True,
                    "gated_delta_backend_before": FLA_BACKEND,
                    "gated_delta_backend": TORCH_BACKEND,
                },
            }
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            native = _native_manifest(
                receipt_path, predictions, prompts, "candidate", "checkpoint-1", 1
            )
            row = {
                "model_id": "candidate",
                "model_revision": "checkpoint-1",
                "model_sha256": "a" * 64,
                "calibration_sha256": "c" * 64,
                "adapter_version": "eikos-native/1",
                "backend": "eikos-semif-native",
                "source_input_sha256": "d" * 64,
            }
            self.assertTrue(
                _native_row_identity(row, native, "d" * 64, "candidate", "checkpoint-1")
            )
            self.assertFalse(
                _native_row_identity(
                    row | {"model_sha256": "b" * 64},
                    native,
                    "d" * 64,
                    "candidate",
                    "checkpoint-1",
                )
            )
            receipt["runtime"]["torch_deterministic_algorithms"] = False
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "collector/runtime"):
                _native_manifest(
                    receipt_path, predictions, prompts, "candidate", "checkpoint-1", 1
                )


if __name__ == "__main__":
    unittest.main()
