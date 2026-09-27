"""The Lux current-package gate compares native runs, never stale answers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from inference.run import ADAPTER_VERSION, digest

from jev_arena.verify_lux9b_current_package import (
    BUNDLE_SHA256,
    MODEL_ID,
    REVISION,
    make_prompts,
    verify_pair,
)


class CurrentLuxPackageTest(unittest.TestCase):
    def test_gold_free_pair_and_drift_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "model-card-example.json").write_text(
                json.dumps(
                    {
                        "requests": {
                            "model_card": {
                                "state": "A",
                                "questions": {
                                    "c": {"type": "choice"},
                                    "n": {"type": "noul"},
                                },
                            },
                            "usage": {
                                "state": "B",
                                "questions": {
                                    "s": {"type": "score"},
                                    "c": {"type": "choice"},
                                    "n": {"type": "noul"},
                                },
                            },
                        },
                        "actual_responses": "deliberately invalid stale reference",
                    }
                ),
                encoding="utf-8",
            )
            prompts = root / "prompts.jsonl"
            with patch("jev_arena.verify_lux9b_current_package.verify_package"):
                self.assertEqual(make_prompts(model, prompts)["answer_slots"], 5)
            self.assertNotIn("actual_responses", prompts.read_text())
            rows = [json.loads(line) for line in prompts.read_text().splitlines()]
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            for path in (first, second):
                with path.open("x", encoding="utf-8") as stream:
                    for row in rows:
                        answers = {}
                        for key, question in row["questions"].items():
                            kind = question["type"]
                            answers[key] = (
                                {
                                    "type": "choice",
                                    "choice": "a",
                                    "probabilities": {"a": 0.8, "b": 0.2},
                                }
                                if kind == "choice"
                                else (
                                    {"type": "noul", "noul": 0.7}
                                    if kind == "noul"
                                    else {
                                        "type": "score",
                                        "score": 1.2,
                                        "probabilities": {"0": 0.2, "1": 0.4, "2": 0.4},
                                    }
                                )
                            )
                        stream.write(
                            json.dumps(
                                {
                                    "id": row["id"],
                                    "answers": answers,
                                    "backend": "lux",
                                    "model_id": MODEL_ID,
                                    "adapter_version": ADAPTER_VERSION,
                                    "model_revision": REVISION,
                                    "revision_attested": True,
                                    "model_config_sha256": BUNDLE_SHA256,
                                    "source_input_sha256": digest(
                                        {
                                            "state": row["state"],
                                            "questions": row["questions"],
                                        }
                                    ),
                                    "runtime_matches_validated": True,
                                    "runtime_differences": {},
                                }
                            )
                            + "\n"
                        )
            with patch("jev_arena.verify_lux9b_current_package.verify_package"):
                self.assertEqual(
                    verify_pair(model, prompts, first, second)[
                        "max_answer_numeric_drift"
                    ],
                    0.0,
                )
                edited = second.read_text().replace('"noul": 0.7', '"noul": 0.72')
                second.write_text(edited)
                with self.assertRaisesRegex(ValueError, "drift exceeds gate"):
                    verify_pair(model, prompts, first, second)
                second.write_text(
                    first.read_text().replace('"choice": "a"', '"choice": "b"')
                )
                with self.assertRaisesRegex(ValueError, "category"):
                    verify_pair(model, prompts, first, second)


if __name__ == "__main__":
    unittest.main()
