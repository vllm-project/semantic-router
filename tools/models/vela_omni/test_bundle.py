"""Small artifact receipt/inventory tests; no models, ML packages, or network."""

import json
import tempfile
import unittest
from pathlib import Path

from bundle import stage, verify
from contract import MANIFEST, artifact_manifest, inventory, write_json


def fixture(root: Path) -> dict:
    manifest = artifact_manifest("nano", "right", 0)
    names = [
        "text/0",
        "text/1",
        "text/2",
        "text/3",
        "text/overflow-rejected",
        "text/padding",
    ]
    names += ["image/0", "image/1", "image/2"]
    names += [
        f"audio/{index}/{part}" for index in range(4) for part in ("end-to-end", "clap")
    ]
    for name in (
        "onnx/text.onnx",
        "onnx/image.onnx",
        "onnx/clap.onnx",
        "onnx/audio.onnx",
        "components/text/tokenizer.json",
        "processors/audio.json",
        "golden/fixture.f32",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"tiny test data; never an executable model")
    write_json(
        root / "reference_parity.json",
        {
            "passed": True,
            "source": manifest["source"],
            "variant": "nano",
            "tests": [{"name": name, "passed": True} for name in names],
        },
    )
    manifest["reference_parity"]["passed"] = True
    manifest["files"] = inventory(root)
    write_json(root / MANIFEST, manifest)
    return manifest


class BundleTests(unittest.TestCase):
    def test_stage_removes_only_golden_and_revalidates_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, destination = root / "source", root / "destination"
            source.mkdir()
            fixture(source)
            stage(source, destination)
            verify(destination)
            self.assertTrue((source / "golden/fixture.f32").exists())
            self.assertFalse((destination / "golden").exists())
            with self.assertRaisesRegex(ValueError, "already exists"):
                stage(source, destination)

    def test_incomplete_modality_receipt_is_never_staged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = fixture(root)
            report = json.loads((root / "reference_parity.json").read_text())
            report["tests"] = [
                test for test in report["tests"] if "clap" not in test["name"]
            ]
            write_json(root / "reference_parity.json", report)
            manifest["files"] = inventory(root)
            write_json(root / MANIFEST, manifest)
            with self.assertRaisesRegex(ValueError, "omits a required modality"):
                verify(root)

    def test_missing_graph_and_unlisted_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = fixture(root)
            (root / "unexpected").write_bytes(b"not inventoried")
            with self.assertRaisesRegex(ValueError, "absent from its inventory"):
                verify(root)
            (root / "unexpected").unlink()
            (root / "onnx/clap.onnx").unlink()
            manifest["files"] = inventory(root)
            write_json(root / MANIFEST, manifest)
            with self.assertRaisesRegex(ValueError, "omits a required graph"):
                verify(root)

    def test_native_code_and_pending_exports_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = fixture(root)
            (root / "model.py").write_text("raise RuntimeError('must not execute')")
            manifest["files"] = inventory(root)
            write_json(root / MANIFEST, manifest)
            with self.assertRaisesRegex(ValueError, "native source or weights"):
                verify(root)
            (root / "model.py").unlink()
            manifest["reference_parity"]["passed"] = False
            manifest["files"] = inventory(root)
            write_json(root / MANIFEST, manifest)
            with self.assertRaisesRegex(ValueError, "pinned tensor contract"):
                verify(root)


if __name__ == "__main__":
    unittest.main()
