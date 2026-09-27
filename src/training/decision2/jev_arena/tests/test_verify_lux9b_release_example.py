"""Contract checks for Lux's own published native example gate."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jev_arena.verify_lux9b_release_example import make_prompts, verify


class ReleaseExampleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.example = {
            "requests": {
                "model_card": {"state": "A", "questions": {"c": {"type": "choice"}}},
                "usage": {
                    "state": "B",
                    "questions": {
                        "n": {"type": "noul"},
                        "s": {"type": "score"},
                        "c": {"type": "choice"},
                        "c2": {"type": "choice"},
                    },
                },
            },
            "actual_responses": {
                "model_card": {
                    "answers": {
                        "c": {
                            "type": "choice",
                            "choice": "yes",
                            "probabilities": {"yes": 0.9, "no": 0.1},
                        }
                    }
                },
                "usage": {
                    "answers": {
                        "n": {"type": "noul", "noul": 0.7},
                        "s": {
                            "type": "score",
                            "score": 1.0,
                            "probabilities": {"0": 0.1, "1": 0.9},
                        },
                        "c": {"type": "choice", "choice": "no"},
                        "c2": {"type": "choice", "choice": "yes"},
                    }
                },
            },
        }

    def test_build_and_verify(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch(
                "jev_arena.verify_lux9b_release_example._example",
                return_value=self.example,
            ):
                prompts = root / "prompts.jsonl"
                make_prompts(root, prompts)
                self.assertEqual(len(prompts.read_text().splitlines()), 2)
                predictions = root / "predictions.jsonl"
                with predictions.open("x", encoding="utf-8") as target:
                    for name, response in self.example["actual_responses"].items():
                        target.write(
                            json.dumps(
                                {
                                    "id": f"lux-release-example:{name}",
                                    "answers": response["answers"],
                                    "runtime_matches_validated": True,
                                    "revision_attested": True,
                                }
                            )
                            + "\n"
                        )
                self.assertEqual(verify(root, predictions)["answer_slots"], 5)
                value = predictions.read_text().replace("0.7", "0.75")
                predictions.write_text(value)
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    verify(root, predictions)


if __name__ == "__main__":
    unittest.main()
