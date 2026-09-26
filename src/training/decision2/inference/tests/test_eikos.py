from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from inference.eikos import completed_rows, shared_answer
from inference.run import digest


class EikosCollectorTest(unittest.TestCase):
    def test_score_keeps_native_level_and_shared_expected_value(self) -> None:
        native = {
            "type": "score",
            "score": 2,
            "expected": 1.7,
            "probabilities": {"0": 0.1, "1": 0.2, "2": 0.6, "3": 0.1},
        }
        mapped = shared_answer({"type": "score"}, native)
        self.assertEqual(mapped["score"], 1.7)
        self.assertEqual(mapped["native_score"], 2)
        self.assertEqual(mapped["probabilities"], native["probabilities"])
        self.assertEqual(native["score"], 2)

    def test_resume_requires_same_inputs_and_model_identity(self) -> None:
        row = {"id": "one", "state": "case", "questions": {"x": {"type": "noul"}}}
        identity = {"model_revision": "pinned", "adapter_version": "v1"}
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
            with self.assertRaisesRegex(ValueError, "stale Eikos identity"):
                completed_rows(path, [row], {**identity, "model_revision": "other"})


if __name__ == "__main__":
    unittest.main()
