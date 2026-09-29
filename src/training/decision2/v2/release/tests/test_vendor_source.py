"""vendor_source: the exact inference sources come from a pinned mirror tree (stdlib)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from v2.release import build, layout

IDENTITY = {"head_variant": "shared", "dec_residual": False}


def mirror_tree(scratch: Path, marker: bool = True) -> Path:
    mirror = scratch / f"{'a' * 40}-src_training_decision2"
    tree = mirror / "src/training/decision2"
    (tree / "training/model").mkdir(parents=True)
    for name in build.QWEN_MODULES:
        (tree / "training/model" / name).write_text(f"# scored {name}\n")
    if marker:
        (mirror / ".dev2-mirror.json").write_text(
            json.dumps({"commit": "a" * 40, "tree": "b" * 40})
        )
    return tree


class VendorSourceTest(unittest.TestCase):
    def test_default_is_the_builder_tree(self):
        self.assertEqual(build.vendor_root({}), (build.SOURCE_ROOT, None))

    def test_mirror_tree_supplies_the_inference_sources(self):
        with tempfile.TemporaryDirectory() as scratch:
            tree = mirror_tree(Path(scratch))
            spec = {"profile": "qwen-adapter", "vendor_source": str(tree)}
            self.assertEqual(build.vendor_root(spec)[1]["commit"], "a" * 40)
            stage = Path(scratch) / "stage"
            stage.mkdir()
            records = build.vendor_runtime(spec, stage, IDENTITY)
            for name in build.QWEN_MODULES:
                vendored = stage / "decision2/_vendor/dev2model" / name
                self.assertEqual(vendored.read_text(), f"# scored {name}\n")
                self.assertEqual(
                    records[f"decision2/_vendor/dev2model/{name}"]["source_sha256"],
                    layout.sha_file(tree / "training/model" / name),
                )
            self.assertEqual(
                records["decision2/qwen.py"]["source_sha256"],
                layout.sha_file(build.SOURCE_ROOT / "v2/release/runtime/qwen.py"),
            )

    def test_a_tree_outside_a_mirror_is_refused(self):
        with tempfile.TemporaryDirectory() as scratch:
            tree = mirror_tree(Path(scratch), marker=False)
            with self.assertRaises(ValueError):
                build.vendor_root({"vendor_source": str(tree)})


if __name__ == "__main__":
    unittest.main()
