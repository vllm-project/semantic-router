"""JPT native adapter release identity checks without a GPU."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from inference.jpt import (
    MODEL_REVISION,
    SOURCE_REVISION,
    collect,
    model_fingerprint,
    verify_source,
)


class JptCollectorTest(unittest.TestCase):
    def test_requires_exact_clean_native_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            with patch(
                "inference.jpt.subprocess.check_output",
                side_effect=[SOURCE_REVISION + "\n", ""],
            ):
                verify_source(source)
            with (
                patch(
                    "inference.jpt.subprocess.check_output",
                    return_value="another-revision\n",
                ),
                self.assertRaisesRegex(ValueError, "revision mismatch"),
            ):
                verify_source(source)
            with (
                patch(
                    "inference.jpt.subprocess.check_output",
                    side_effect=[SOURCE_REVISION + "\n", " M llm2jev/prompt.py\n"],
                ),
                self.assertRaisesRegex(ValueError, "modified"),
            ):
                verify_source(source)

    def test_model_fingerprint_requires_weights_and_hashes_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary)
            (model / "config.json").write_bytes(b"config")
            (model / "tokenizer.json").write_bytes(b"tokenizer")
            with self.assertRaisesRegex(ValueError, "lacks"):
                model_fingerprint(model)
            shard = model / "model-00001-of-00001.safetensors"
            shard.write_bytes(b"weights")
            hashes = model_fingerprint(model)
            self.assertEqual(hashes[shard.name], hashlib.sha256(b"weights").hexdigest())
            shard.write_bytes(b"changed")
            self.assertNotEqual(model_fingerprint(model), hashes)

    def test_rejects_other_model_revision_before_loading(self) -> None:
        with self.assertRaisesRegex(ValueError, "pinned model revision"):
            collect(
                model_path=Path("missing-model"),
                source_path=Path("missing-source"),
                model_revision=MODEL_REVISION + "-changed",
                prompts=Path("missing-prompts"),
                output=Path("missing-output"),
            )


if __name__ == "__main__":
    unittest.main()
