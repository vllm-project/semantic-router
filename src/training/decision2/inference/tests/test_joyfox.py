"""Native Joyfox collector contract without loading a GPU model."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from inference.joyfox import (
    ADAPTER_VERSION,
    EXTENDED_ADAPTER_VERSION,
    _predict,
    collector_identity,
    completed_ids,
)
from inference.run import digest

ROW = {
    "id": "one",
    "state": "An order was refunded.",
    "questions": {"refund": {"type": "noul", "instructions": "Was it refunded?"}},
}
IDENTITY = {"backend": "joyfox", "model_revision": "abc", "adapter_version": "v1"}


class FakeEngine:
    def __init__(self, answer=None, error=None):
        self.answer = answer
        self.error = error

    def predict(self, payload):
        if self.error:
            raise ValueError(self.error)
        return self.answer


class JoyfoxCollectorTest(unittest.TestCase):
    def test_context_ablation_has_distinct_identity(self) -> None:
        release = {"adapter_version": ADAPTER_VERSION, "model_revision": "abc"}
        self.assertIs(collector_identity(release, 1024), release)
        extended = collector_identity(release, 4096)
        self.assertEqual(extended["adapter_version"], EXTENDED_ADAPTER_VERSION)
        self.assertEqual(extended["cutoff_len"], 4096)
        with self.assertRaisesRegex(ValueError, "between 1 and 4096"):
            collector_identity(release, 4097)

    def test_native_answer_and_overflow(self) -> None:
        answer = {"refund": {"type": "noul", "noul": 0.9}}
        self.assertEqual(_predict(FakeEngine(answer), ROW), (answer, "ok"))
        overflow, status = _predict(
            FakeEngine(error="input requires 1200 tokens, cutoff_len=1024"), ROW
        )
        self.assertEqual(status, "context_overflow")
        self.assertEqual(overflow["refund"]["invalid_reason"], "context_overflow")
        with self.assertRaisesRegex(ValueError, "bad input"):
            _predict(FakeEngine(error="bad input"), ROW)
        with self.assertRaisesRegex(ValueError, "answer keys differ"):
            _predict(FakeEngine({"other": answer["refund"]}), ROW)

    def test_resume_requires_identical_model_input_and_answer_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.jsonl"
            result = {
                "id": ROW["id"],
                "answers": {"refund": {"noul": 0.9}},
                "source_input_sha256": digest(
                    {"state": ROW["state"], "questions": ROW["questions"]}
                ),
                **IDENTITY,
            }
            output.write_text(json.dumps(result) + "\n")
            self.assertEqual(completed_ids(output, [ROW], IDENTITY), {"one"})
            with self.assertRaisesRegex(ValueError, "stale Joyfox identity"):
                completed_ids(output, [ROW], {**IDENTITY, "model_revision": "other"})
            with self.assertRaisesRegex(ValueError, "stale input"):
                completed_ids(output, [{**ROW, "state": "Changed"}], IDENTITY)


if __name__ == "__main__":
    unittest.main()
