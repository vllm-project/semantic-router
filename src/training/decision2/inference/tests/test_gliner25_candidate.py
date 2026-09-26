"""Pin a continuation checkpoint by bytes before native collection."""

from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from inference import gliner25
from inference.gliner25_candidate import identity


class CandidateIdentityTest(unittest.TestCase):
    def test_requires_pinned_source_runtime_and_checkpoint_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            for name in ("model.safetensors", "config.json", "tokenizer.json"):
                (checkpoint / name).write_bytes(name.encode())
            weights = hashlib.sha256(b"model.safetensors").hexdigest()
            with patch.dict(os.environ, {"GLINER2_SOURCE_COMMIT": "wrong"}):
                with self.assertRaisesRegex(RuntimeError, "not pinned"):
                    identity(checkpoint, weights)
            with patch.dict(
                os.environ, {"GLINER2_SOURCE_COMMIT": gliner25.LIBRARY_COMMIT}
            ):
                with self.assertRaisesRegex(ValueError, "weights mismatch"):
                    identity(checkpoint, "0" * 64)
                receipt = identity(checkpoint, weights)
                self.assertEqual(receipt["model_weights_sha256"], weights)
                self.assertEqual(receipt["adapter_version"], gliner25.ADAPTER_VERSION)


if __name__ == "__main__":
    unittest.main()
