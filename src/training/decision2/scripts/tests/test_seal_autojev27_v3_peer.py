"""The peer prediction seal must reject changed inputs and incomplete output."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from inference.autojev27 import ADAPTER_VERSION, BACKEND, MODEL_ID, MODEL_REVISION
from inference.run import digest, file_digest
from scripts.seal_autojev27_v3_peer import (
    CONFIG_SHA,
    MODEL_SHA,
    PANELS,
    SOURCE_SHA,
    audit_panel,
)


class SealAutoJevPeerTest(unittest.TestCase):
    def test_panel_hashes_match_signed_prereg(self) -> None:
        prereg_path = (
            Path(__file__).resolve().parents[2]
            / "research/autojev27-v3-public-peer-prereg-2026-09-28.md"
        )
        prereg = prereg_path.read_text(encoding="utf-8")
        for _, (_, _, prompt_sha) in PANELS.items():
            self.assertIn(f"prompt SHA-256 `{prompt_sha}`", prereg)

    def test_exact_single_item_and_changed_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt_path, prediction_path = (
                root / "prompt.jsonl",
                root / "prediction.jsonl",
            )
            prompt = {
                "id": "test-one",
                "state": "The package arrived.",
                "questions": {
                    "decision": {"type": "noul", "criteria": "Did it arrive?"}
                },
            }
            prompt_path.write_text(json.dumps(prompt) + "\n", encoding="utf-8")
            prediction = {
                "id": prompt["id"],
                "answers": {"decision": {"type": "noul", "noul": 0.8}},
                "source_input_sha256": digest(
                    {"state": prompt["state"], "questions": prompt["questions"]}
                ),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "revision_attested": True,
                "backend": BACKEND,
                "adapter_version": ADAPTER_VERSION,
                "native_model_sha256": MODEL_SHA,
                "runtime_source_sha256": SOURCE_SHA,
                "model_config_sha256": CONFIG_SHA,
            }
            prediction_path.write_text(json.dumps(prediction) + "\n", encoding="utf-8")
            expected = file_digest(prompt_path)
            receipt = audit_panel(
                prompts=prompt_path,
                predictions=prediction_path,
                expected_items=1,
                expected_questions=1,
                expected_prompt_sha256=expected,
            )
            self.assertEqual(receipt["items"], 1)
            prompt_path.write_text(json.dumps({**prompt, "state": "Changed"}) + "\n")
            with self.assertRaisesRegex(ValueError, "Prompt bytes"):
                audit_panel(
                    prompts=prompt_path,
                    predictions=prediction_path,
                    expected_items=1,
                    expected_questions=1,
                    expected_prompt_sha256=expected,
                )

    def test_missing_answer_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt_path, prediction_path = (
                root / "prompt.jsonl",
                root / "prediction.jsonl",
            )
            prompt = {
                "id": "test-one",
                "state": "A",
                "questions": {"q": {"type": "noul"}},
            }
            prompt_path.write_text(json.dumps(prompt) + "\n")
            prediction_path.write_text(
                json.dumps({"id": "test-one", "answers": {}}) + "\n"
            )
            with self.assertRaisesRegex(ValueError, "Native prediction"):
                audit_panel(
                    prompts=prompt_path,
                    predictions=prediction_path,
                    expected_items=1,
                    expected_questions=1,
                    expected_prompt_sha256=file_digest(prompt_path),
                )


if __name__ == "__main__":
    unittest.main()
