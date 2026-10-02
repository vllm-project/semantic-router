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


def runtime_tree(scratch: Path, marker: bool = True) -> Path:
    tree = mirror_tree(scratch, marker)
    (tree / "v2/release/runtime").mkdir(parents=True)
    for name in ("__init__.py", "api.py", "qwen.py"):
        (tree / "v2/release/runtime" / name).write_text(f"# released {name}\n")
    return tree


class RuntimeSourceTest(unittest.TestCase):
    def test_default_is_the_builder_tree(self):
        self.assertEqual(build.runtime_root({}), (build.SOURCE_ROOT, None))

    def test_mirror_tree_supplies_the_runtime_template_only(self):
        with tempfile.TemporaryDirectory() as scratch:
            tree = runtime_tree(Path(scratch))
            spec = {"profile": "qwen-full", "runtime_source": str(tree)}
            self.assertEqual(build.runtime_root(spec)[1]["tree"], "b" * 40)
            stage = Path(scratch) / "stage"
            stage.mkdir()
            records = build.vendor_runtime(spec, stage, IDENTITY)
            for name in ("__init__.py", "api.py", "qwen.py"):
                self.assertEqual(
                    (stage / "decision2" / name).read_text(), f"# released {name}\n"
                )
                self.assertEqual(
                    records[f"decision2/{name}"]["source_sha256"],
                    layout.sha_file(tree / "v2/release/runtime" / name),
                )
            infer = "decision2/_vendor/dev2model/infer.py"
            self.assertEqual(
                records[infer]["source_sha256"],
                layout.sha_file(build.SOURCE_ROOT / "training/model/infer.py"),
            )

    def test_the_fast_path_ships_only_with_trees_that_have_it(self):
        with tempfile.TemporaryDirectory() as scratch:
            tree = runtime_tree(Path(scratch))
            for source, expected in (
                (str(tree), set()),
                (None, set(build.FAST_RUNTIME)),
            ):
                spec = {"profile": "qwen-full"}
                if source:
                    spec["runtime_source"] = source
                stage = Path(scratch) / f"stage-{bool(source)}"
                stage.mkdir()
                records = build.vendor_runtime(spec, stage, IDENTITY)
                shipped = {
                    name.removeprefix("decision2/")
                    for name in records
                    if name.removeprefix("decision2/") in build.FAST_RUNTIME
                }
                self.assertEqual(shipped, expected)
                for name in expected:
                    self.assertEqual(
                        records[f"decision2/{name}"]["source_sha256"],
                        layout.sha_file(
                            build.SOURCE_ROOT / "v2/release/runtime" / name
                        ),
                    )

    def test_a_tree_without_the_runtime_or_outside_a_mirror_is_refused(self):
        with tempfile.TemporaryDirectory() as scratch:
            with self.assertRaises(ValueError):
                build.runtime_root({"runtime_source": str(mirror_tree(Path(scratch)))})
        with tempfile.TemporaryDirectory() as scratch:
            tree = runtime_tree(Path(scratch), marker=False)
            with self.assertRaises(ValueError):
                build.runtime_root({"runtime_source": str(tree)})


if __name__ == "__main__":
    unittest.main()
