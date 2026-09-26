from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from inference.apus import completed_rows, native_request, shared_answer
from inference.run import digest


class APUSCollectorTest(unittest.TestCase):
    def test_choice_preserves_label_and_state_information(self) -> None:
        row = {
            "id": "one",
            "state": {"record": "x"},
            "questions": {
                "decision": {
                    "type": "choice",
                    "instructions": "Choose x",
                    "criteria": {"x": "Matching record", "y": "Other record"},
                }
            },
        }
        request = native_request(row, "decision")
        self.assertEqual(request["primitive"], "choice")
        self.assertEqual(request["state"], '{"record":"x"}')
        self.assertEqual(
            request["criteria"],
            [
                {"id": "x", "description": "x: Matching record"},
                {"id": "y", "description": "y: Other record"},
            ],
        )
        answer = shared_answer(
            "choice",
            {
                "choice": "x",
                "probabilities": {"x": 0.7, "y": 0.3},
                "effort": "high",
                "projection": "full_head",
                "calibrated": False,
            },
        )
        self.assertEqual(answer["choice"], "x")
        self.assertEqual(answer["probabilities"], {"x": 0.7, "y": 0.3})

    def test_noul_keeps_true_false_meanings_and_native_binary_contract(self) -> None:
        row = {
            "id": "one",
            "state": "facts",
            "questions": {
                "decision": {
                    "type": "noul",
                    "instructions": "Can proceed?",
                    "criteria": {"true": "allowed", "false": "denied"},
                }
            },
        }
        request = native_request(row, "decision")
        self.assertEqual(request["primitive"], "noul")
        self.assertEqual([c["id"] for c in request["criteria"]], ["yes", "no"])
        self.assertIn("Yes means: allowed", request["instructions"])
        self.assertIn("No means: denied", request["instructions"])
        self.assertEqual(
            shared_answer(
                "noul",
                {
                    "yes_probability": 0.8,
                    "effort": "high",
                    "projection": "full_head",
                    "calibrated": False,
                },
            )["noul"],
            0.8,
        )

    def test_ordinal_score_has_no_native_projection(self) -> None:
        row = {
            "id": "one",
            "state": "state",
            "questions": {
                "decision": {
                    "type": "score",
                    "instructions": "rate",
                    "criteria": ["low", "mid", "high"],
                }
            },
        }
        self.assertIsNone(native_request(row, "decision"))

    def test_resume_identity_and_input_hash(self) -> None:
        row = {
            "id": "one",
            "state": "state",
            "questions": {
                "decision": {
                    "type": "score",
                    "instructions": "rate",
                    "criteria": ["low", "high"],
                }
            },
        }
        identity = {
            "model_revision": "pinned",
            "adapter_version": "v1",
            "effort": "high",
        }
        receipt = {
            "id": "one",
            "source_input_sha256": digest(
                {"state": row["state"], "questions": row["questions"]}
            ),
            "answers": {
                "decision": {
                    "type": "score",
                    "error": "unsupported_native_ordinal_score",
                }
            },
            **identity,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
            self.assertEqual(completed_rows(path, [row], identity), {"one"})
            with self.assertRaisesRegex(ValueError, "stale APUS identity"):
                completed_rows(path, [row], {**identity, "effort": "low"})


if __name__ == "__main__":
    unittest.main()
