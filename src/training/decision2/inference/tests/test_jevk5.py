from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from inference.jevk5 import completed_rows, native_admission_reason, shared_answer
from inference.run import digest


class JevK5CollectorTest(unittest.TestCase):
    def test_score_uses_native_expected_value(self) -> None:
        question = {"type": "score", "criteria": ["low", "mid", "high"]}
        native = {
            "type": "score",
            "score": 1.4,
            "confidence": 0.5,
            "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5},
            "input_tokens": 87,
        }
        self.assertIs(shared_answer(question, native), native)
        with self.assertRaisesRegex(ValueError, "not its native expected value"):
            shared_answer(question, {**native, "score": 2})

    def test_choice_and_noul_preserve_native_probabilities(self) -> None:
        choice = {"type": "choice", "criteria": {"a": "first", "b": "second"}}
        native_choice = {
            "type": "choice",
            "choice": "b",
            "confidence": 0.8,
            "probabilities": {"a": 0.2, "b": 0.8},
            "input_tokens": 55,
        }
        self.assertIs(shared_answer(choice, native_choice), native_choice)
        noul = {"type": "noul", "confidence": 0.7, "noul": 0.3, "input_tokens": 30}
        self.assertIs(shared_answer({"type": "noul"}, noul), noul)

    def test_resume_requires_same_inputs_and_release_identity(self) -> None:
        row = {"id": "one", "state": "case", "questions": {"x": {"type": "noul"}}}
        identity = {
            "model_revision": "pinned",
            "adapter_version": "v1",
            "release_files_sha256": "release",
            "runtime_revision": "runtime",
        }
        receipt = {
            "id": "one",
            "answers": {"x": {"type": "noul", "noul": 0.7}},
            "source_input_sha256": digest(
                {"state": row["state"], "questions": row["questions"]}
            ),
            **identity,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
            self.assertEqual(completed_rows(path, [row], identity), {"one"})
            with self.assertRaisesRegex(ValueError, "stale JevK5 identity"):
                completed_rows(path, [row], {**identity, "runtime_revision": "other"})

    def test_native_capacity_errors_are_explicit_and_other_errors_abort(self) -> None:
        self.assertEqual(
            native_admission_reason(ValueError("Maximum 16384 tokens exceeded")),
            "context_overflow",
        )
        self.assertEqual(
            native_admission_reason(ValueError("Too many candidate options")),
            "candidate_limit",
        )
        self.assertIsNone(
            native_admission_reason(ValueError("invalid option probability map"))
        )


if __name__ == "__main__":
    unittest.main()
