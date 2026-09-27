from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.candidate_repeat_smoke_v3 import compare
from training.model.infer import prompt_input_sha256


class CandidateRepeatSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.prompts = self.root / "prompts.jsonl"
        self.first = self.root / "first.jsonl"
        self.second = self.root / "second.jsonl"
        self.items = [
            {
                "id": f"item-{index}",
                "state": "A stated fact.",
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": "Choose one.",
                        "criteria": {"a": "The stated fact", "b": "Other"},
                    }
                },
            }
            for index in range(32)
        ]
        self.prompts.write_text(
            "".join(json.dumps(row) + "\n" for row in self.items), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.folder.cleanup()

    def _write(self, path: Path, *, changed: bool = False, drift: float = 0.0) -> None:
        rows = []
        for index, prompt in enumerate(self.items):
            a = 0.8 - (drift if index == 0 else 0.0)
            if changed and index == 0:
                a = 0.2
            rows.append(
                {
                    "id": prompt["id"],
                    "source_input_sha256": prompt_input_sha256(prompt),
                    "model_sha256": "a" * 64,
                    "adapter_sha256": "b" * 64,
                    "usage": {"input_tokens": 10},
                    "answers": {
                        "q": {
                            "type": "choice",
                            "choice": "a" if a > 0.5 else "b",
                            "probabilities": {"a": a, "b": 1 - a},
                        }
                    },
                }
            )
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        manifest = {
            "predictions_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "input_sha256": hashlib.sha256(self.prompts.read_bytes()).hexdigest(),
            "input_items": 32,
            "model_id": "llm-semantic-router/DEV2.0-2B",
            "model_revision": "checkpoint-0000320",
            "model_sha256": "a" * 64,
            "adapter_sha256": "b" * 64,
            "adapter_version": "decision2-typed-benchmark-adapter-v2-calibrated",
            "calibration": {"file_sha256": "c" * 64},
            "max_length": 8192,
        }
        path.with_name(path.name + ".manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

    def _compare(self) -> dict:
        return compare(
            self.prompts,
            self.first,
            self.second,
            model_id="llm-semantic-router/DEV2.0-2B",
            revision="checkpoint-0000320",
            max_drift=0.02,
        )

    def test_exact_two_process_outputs_pass(self) -> None:
        self._write(self.first)
        self._write(self.second)
        result = self._compare()
        self.assertTrue(result["gate_pass"])
        self.assertEqual(result["answers"], 32)

    def test_category_change_fails_even_under_numeric_limit(self) -> None:
        self._write(self.first)
        self._write(self.second, changed=True)
        result = self._compare()
        self.assertFalse(result["gate_pass"])
        self.assertEqual(result["category_mismatch_ids"], ["item-0:q"])

    def test_probability_drift_is_reported_and_fails(self) -> None:
        self._write(self.first)
        self._write(self.second, drift=0.03)
        result = self._compare()
        self.assertFalse(result["gate_pass"])
        self.assertAlmostEqual(result["max_probability_drift"], 0.03)


if __name__ == "__main__":
    unittest.main()
