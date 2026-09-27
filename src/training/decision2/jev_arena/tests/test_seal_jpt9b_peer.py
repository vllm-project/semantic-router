"""Tamper and completeness checks for the JPT same-panel peer seal."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from inference.jpt import (
    ADAPTER_VERSION,
    MODEL_ID,
    MODEL_REVISION,
    SOURCE_REVISION,
    TEMPERATURE,
)
from inference.run import digest, file_digest

from jev_arena.seal_jpt9b_peer import seal


class SealJptPeerTest(unittest.TestCase):
    def test_seal_binds_input_model_and_complete_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            prompts = root / "prompts.jsonl"
            predictions = root / "predictions.jsonl"
            manifest = root / "predictions.manifest.json"
            reference = root / "reference.manifest.json"
            output = root / "seal.json"
            question = {"q": {"type": "choice", "criteria": {"a": "A", "b": "B"}}}
            prompt = {"id": "x", "state": "example", "questions": question}
            prompts.write_text(json.dumps(prompt) + "\n", encoding="utf-8")
            identity = {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "source_revision": SOURCE_REVISION,
                "temperature": TEMPERATURE,
                "adapter_version": ADAPTER_VERSION,
                "model_files_sha256": {"config.json": "hash"},
            }
            answer = {
                "id": "x",
                "source_input_sha256": digest(
                    {"state": prompt["state"], "questions": prompt["questions"]}
                ),
                "answers": {"q": {"type": "choice", "choice": "a"}},
                "backend": "jpt-llm2jev-hf",
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "adapter_version": ADAPTER_VERSION,
            }
            predictions.write_text(json.dumps(answer) + "\n", encoding="utf-8")
            reference.write_text(json.dumps(identity), encoding="utf-8")
            manifest.write_text(
                json.dumps(
                    {
                        **identity,
                        "input_sha256": file_digest(prompts),
                        "output_sha256": file_digest(predictions),
                        "counts": {"items": 1, "questions": 1},
                    }
                ),
                encoding="utf-8",
            )
            kwargs = dict(
                panel="typed-final",
                prompts_path=prompts,
                predictions_path=predictions,
                manifest_path=manifest,
                reference_manifest_path=reference,
                output=output,
            )
            with patch(
                "jev_arena.seal_jpt9b_peer.PANELS",
                {"typed-final": (1, 1, file_digest(prompts))},
            ):
                result = seal(**kwargs)
                self.assertEqual((result["items"], result["answer_slots"]), (1, 1))
                with self.assertRaises(FileExistsError):
                    seal(**kwargs)
                output.unlink()
                tampered = json.loads(predictions.read_text(encoding="utf-8"))
                tampered["source_input_sha256"] = "0" * 64
                predictions.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
                changed = json.loads(manifest.read_text(encoding="utf-8"))
                changed["output_sha256"] = file_digest(predictions)
                manifest.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "identity or questions"):
                    seal(**kwargs)


if __name__ == "__main__":
    unittest.main()
