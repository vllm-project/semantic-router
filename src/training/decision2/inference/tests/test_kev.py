"""Kev release-integrity and collector tests without local model inference."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from inference.kev import (
    KEV_BASE_REVISION,
    KEV_MODEL_REVISION,
    KEV_SOURCE_REVISION,
    collect,
    verify_provenance,
)


class KevCollectorTest(unittest.TestCase):
    def test_provenance_checks_release_source_and_adapter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, source = root / "model", root / "source"
            model.mkdir()
            (source / "kev").mkdir(parents=True)
            code = b"released source"
            adapter = b"released adapter"
            (source / "kev/api.py").write_bytes(code)
            (model / "adapter_model.safetensors").write_bytes(adapter)
            provenance = {
                "git_commit": KEV_SOURCE_REVISION,
                "config": {"base_revision": KEV_BASE_REVISION},
                "measured_checkpoint": {
                    "adapter_sha256": hashlib.sha256(adapter).hexdigest()
                },
                "source_hashes": {"kev/api.py": hashlib.sha256(code).hexdigest()},
            }
            (model / "provenance.json").write_text(json.dumps(provenance))
            with patch(
                "inference.kev.subprocess.check_output",
                return_value=KEV_SOURCE_REVISION + "\n",
            ):
                self.assertEqual(verify_provenance(model, source), provenance)
                (source / "kev/api.py").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "source differs"):
                    verify_provenance(model, source)

    def test_gold_free_collection_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, source = root / "model", root / "source"
            model.mkdir()
            source.mkdir()
            for name in ("provenance.json", "head.pt", "adapter_model.safetensors"):
                (model / name).write_bytes(name.encode())
            metadata = (
                model / ".cache/huggingface/download/adapter_model.safetensors.metadata"
            )
            metadata.parent.mkdir(parents=True)
            metadata.write_text(KEV_MODEL_REVISION + "\netag\n")
            prompts = root / "prompts.jsonl"
            row = {
                "id": "x",
                "state": "Context",
                "questions": {
                    "choice": {
                        "type": "choice",
                        "instructions": "Pick",
                        "criteria": {"a": "A", "b": "B"},
                    }
                },
            }
            prompts.write_text(json.dumps(row) + "\n")
            response = {
                "model": "kev",
                "answers": {
                    "choice": {
                        "type": "choice",
                        "choice": "a",
                        "probabilities": {"a": 0.8, "b": 0.2},
                    }
                },
            }
            output = root / "prediction.jsonl"
            with patch(
                "inference.kev.load_native",
                return_value=(
                    object(),
                    lambda **_kw: response,
                    {"calibration_temperature": 2.4},
                ),
            ):
                first = collect(
                    model_path=model,
                    source_path=source,
                    revision=KEV_MODEL_REVISION,
                    prompts=prompts,
                    output=output,
                    device="cpu",
                )
                second = collect(
                    model_path=model,
                    source_path=source,
                    revision=KEV_MODEL_REVISION,
                    prompts=prompts,
                    output=output,
                    device="cpu",
                    resume=True,
                )
            self.assertEqual(first["collected_now"], 1)
            self.assertEqual(second["collected_now"], 0)
            receipt = json.loads(output.read_text())
            self.assertTrue(receipt["revision_attested"])
            self.assertEqual(receipt["answers"], response["answers"])
            self.assertEqual(receipt["model_id"], "jaredpalmer/kev-4b")


if __name__ == "__main__":
    unittest.main()
