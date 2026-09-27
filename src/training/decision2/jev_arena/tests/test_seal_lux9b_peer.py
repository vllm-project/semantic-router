"""A complete native Lux 1.0 control must pass the fixed pre-score seal."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from inference.run import ADAPTER_VERSION, digest, file_digest

from jev_arena.seal_lux9b_peer import CONFIG_SHA256, MODEL_ID, REVISION, seal


class SealLuxPeerTest(unittest.TestCase):
    def test_complete_predictions_and_runtime_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "bundle-manifest.json").write_text("{}", encoding="utf-8")
            prompts = root / "prompts.jsonl"
            predictions = root / "predictions.jsonl"
            output = root / "seal.json"
            question = {"q": {"type": "choice", "criteria": {"a": "A", "b": "B"}}}
            prompt = {"id": "x", "state": "example", "questions": question}
            prompts.write_text(json.dumps(prompt) + "\n", encoding="utf-8")
            answer = {
                "id": "x",
                "source_input_sha256": digest(
                    {"state": prompt["state"], "questions": prompt["questions"]}
                ),
                "answers": {"q": {"type": "choice", "choice": "a"}},
                "backend": "lux",
                "model_id": MODEL_ID,
                "model_revision": REVISION,
                "model_config_sha256": CONFIG_SHA256,
                "revision_attested": True,
                "adapter_version": ADAPTER_VERSION,
                "runtime_matches_validated": True,
            }
            predictions.write_text(json.dumps(answer) + "\n", encoding="utf-8")
            kwargs = dict(
                panel="typed-final",
                model_path=model,
                prompts_path=prompts,
                predictions_path=predictions,
                output=output,
            )
            with (
                patch(
                    "jev_arena.seal_lux9b_peer.PANELS",
                    {"typed-final": (1, 1, file_digest(prompts))},
                ),
                patch(
                    "jev_arena.seal_lux9b_peer.file_digest",
                    side_effect=lambda p: (
                        CONFIG_SHA256
                        if p.name == "bundle-manifest.json"
                        else file_digest(p)
                    ),
                ),
                patch("jev_arena.seal_lux9b_peer.local_revision", return_value=True),
            ):
                self.assertEqual(seal(**kwargs)["items"], 1)
                output.unlink()
                answer["runtime_matches_validated"] = False
                predictions.write_text(json.dumps(answer) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "runtime"):
                    seal(**kwargs)


if __name__ == "__main__":
    unittest.main()
