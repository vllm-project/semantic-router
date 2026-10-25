"""Small offline fixture checks for the frozen image asset packager."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "prepare_assets", Path(__file__).with_name("prepare_assets.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PrepareAssetsTest(unittest.TestCase):
    def test_validates_all_inputs_before_copying(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fixture.png"
            source.write_bytes(b"small owned fixture")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            entry = {
                "source": source.name,
                "asset": digest + ".png",
                "sha256": digest,
                "role": "positive",
            }
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"version": 1, "assets": [entry]}))
            output = root / "prepared"
            self.assertEqual(module.prepare(root, manifest, output), 1)
            self.assertEqual(
                (output / entry["asset"]).read_bytes(), source.read_bytes()
            )
            source.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                module.prepare(root, manifest, root / "failed")
            self.assertFalse((root / "failed").exists())

    def test_rejects_repository_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "assets": [
                            {
                                "source": "../private.png",
                                "asset": "x",
                                "sha256": "x",
                                "role": "positive",
                            }
                        ],
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "repository relative"):
                module.prepare(root, manifest, root / "out")


if __name__ == "__main__":
    unittest.main()
