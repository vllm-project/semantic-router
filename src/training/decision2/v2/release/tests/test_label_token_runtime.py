"""Label-token readout in the release builder and runtime (decoder M10): vendoring and dispatch (CPU)."""

from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path

from v2.release import build

IDENTITY = {"head_variant": "shared", "dec_residual": False, "readout": "label_token"}


class LabelTokenVendorTest(unittest.TestCase):
    def test_label_token_module_is_vendored_with_relative_imports(self):
        with tempfile.TemporaryDirectory() as scratch:
            stage = Path(scratch)
            records = build.vendor_runtime({"profile": "qwen-full"}, stage, IDENTITY)
            vendored = stage / "decision2/_vendor/dev2model/label_token.py"
            text = vendored.read_text(encoding="utf-8")
            self.assertNotIn("training.model", text)
            relative = {
                node.module
                for node in ast.walk(ast.parse(text))
                if isinstance(node, ast.ImportFrom) and node.level == 1
            }
            self.assertTrue({"data", "decision_model", "lora", "source"} <= relative)
            for name in relative:
                self.assertTrue((vendored.parent / f"{name}.py").is_file(), name)
            self.assertEqual(
                records["decision2/_vendor/dev2model/label_token.py"]["source"],
                "v2/dec/label_token.py",
            )

    def test_head_packages_do_not_vendor_it(self):
        with tempfile.TemporaryDirectory() as scratch:
            stage = Path(scratch)
            build.vendor_runtime(
                {"profile": "qwen-full"}, stage, {**IDENTITY, "readout": "head"}
            )
            self.assertFalse(
                (stage / "decision2/_vendor/dev2model/label_token.py").exists()
            )

    def test_runtime_dispatches_on_the_readout(self):
        source = (build.SOURCE_ROOT / "v2/release/runtime/qwen.py").read_text()
        self.assertIn('metadata.get("readout") == "label_token"', source)
        self.assertIn("load_label_checkpoint", source)
        self.assertIn("(self.encode_fn or encode)", source)


if __name__ == "__main__":
    unittest.main()
