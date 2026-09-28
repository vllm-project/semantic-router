"""Small gold-free validation of the separately versioned r4 prediction seal."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from inference.run import OVER_BUDGET_ADAPTER_VERSION, digest, file_digest
from jev_arena import seal_lux9b_r4 as seal_module


class LuxR4SealTest(unittest.TestCase):
    def test_complete_invalid_original_is_counted_without_dropping_next(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prompts_path = root / "prompts.jsonl"
            predictions_path = root / "predictions.jsonl"
            prompts = [
                {
                    "id": "long",
                    "state": "long",
                    "questions": {
                        "label": {
                            "type": "choice",
                        }
                    },
                },
                {
                    "id": "next",
                    "state": "next",
                    "questions": {
                        "label": {
                            "type": "noul",
                        }
                    },
                },
            ]
            prompts_path.write_text("".join(json.dumps(row) + "\n" for row in prompts))
            predictions = []
            for row in prompts:
                long = row["id"] == "long"
                predictions.append(
                    {
                        "id": row["id"],
                        "answers": {
                            "label": (
                                None
                                if long
                                else {
                                    "type": "noul",
                                    "noul": 0.8,
                                }
                            )
                        },
                        "source_input_sha256": digest(
                            {
                                "state": row["state"],
                                "questions": row["questions"],
                            }
                        ),
                        "backend": "lux",
                        "adapter_version": OVER_BUDGET_ADAPTER_VERSION,
                        "model_revision": seal_module.REVISION,
                        "model_config_sha256": seal_module.CONFIG_SHA256,
                        "model_id": seal_module.MODEL_ID,
                        "revision_attested": True,
                        "runtime_matches_validated": True,
                        "model": "Decision-1.0-Lux",
                        "native_error": (
                            {
                                "kind": "native_input_over_budget",
                                "question_id": "label",
                                "input_tokens": 18198,
                                "max_length": 16384,
                            }
                            if long
                            else None
                        ),
                    }
                )
            predictions_path.write_text(
                "".join(json.dumps(row) + "\n" for row in predictions)
            )
            with patch.dict(
                seal_module.PANELS,
                {
                    "test": (2, 2, file_digest(prompts_path)),
                },
            ):
                result = seal_module.validate_panel(
                    "test", prompts_path, predictions_path
                )
                self.assertEqual(result["native_over_budget_originals"], 1)
                self.assertEqual(result["native_over_budget_answer_slots"], 1)
                self.assertEqual(result["obviously_invalid_answer_slots"], 1)
                predictions[0]["answers"]["label"] = {
                    "type": "choice",
                    "choice": "a",
                }
                predictions_path.write_text(
                    "".join(json.dumps(row) + "\n" for row in predictions)
                )
                with self.assertRaisesRegex(ValueError, "malformed"):
                    seal_module.validate_panel("test", prompts_path, predictions_path)


if __name__ == "__main__":
    unittest.main()
