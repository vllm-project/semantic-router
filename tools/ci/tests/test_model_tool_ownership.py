"""The model Make targets cover ONNX tests once, with separate export/storage owners."""

import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/ci"))

from classify_pr_changes import classify  # noqa: E402


def onnx_inventory(target: str) -> set[Path]:
    # Ask Make for its actual discovery commands without installing model tooling.
    with tempfile.TemporaryDirectory() as directory:
        makefile = Path(directory) / "Makefile"
        makefile.write_text(
            f"include {ROOT / 'tools/make/models.mk'}\n"
            "harness-venv-install:\n"
            "ck-rewrite-deps:\n"
        )
        result = subprocess.run(
            ["make", "-sn", "-f", str(makefile), "AGENT_PYTHON=python3", target],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    files = set()
    for line in result.stdout.splitlines():
        if " -m unittest discover " not in line:
            continue
        args = shlex.split(line)
        if args[:4] != ["python3", "-m", "unittest", "discover"]:
            continue
        directory = args[args.index("-s") + 1]
        pattern = args[args.index("-p") + 1]
        if directory.startswith("onnx-binding/scripts/"):
            files.update((ROOT / directory).glob(pattern))
    return files


class ModelToolOwnershipTests(unittest.TestCase):
    def test_changed_onnx_files_select_their_actual_test_owner(self):
        selections = {
            "pack_shared_weights.py": {"onnx-artifacts"},
            "artifact_tests/test_pack_shared_weights.py": {"onnx-artifacts"},
            "export_classifier.py": {"training"},
            "export_2d_matryoshka.py": {"training"},
            "modernbert_inputs.py": {"training"},
            "model_precision.py": {"training"},
            "onnx_portable_ops.py": {"training"},
            "onnx_shape_simplification.py": {"training"},
            "tests/test_export_classifier.py": {"training"},
            "onnx_artifacts.py": {"onnx-artifacts", "training"},
        }
        for path, owners in selections.items():
            with self.subTest(path=path):
                selected = set(classify([f"onnx-binding/scripts/{path}"]).selected_jobs)
                self.assertEqual(selected & {"training", "onnx-artifacts"}, owners)

    def test_export_and_storage_targets_cover_each_onnx_test_module_once(self):
        training = onnx_inventory("test-training-contracts")
        artifacts = onnx_inventory("onnx-artifact-test")
        sources = set((ROOT / "onnx-binding/scripts").rglob("test_*.py"))
        self.assertTrue(training)
        self.assertTrue(artifacts)
        self.assertFalse(training & artifacts, "model tests must have one CI owner")
        self.assertEqual(
            training | artifacts, sources, "no ONNX test may lose its owner"
        )
        self.assertEqual(
            {path.name for path in artifacts}, {"test_pack_shared_weights.py"}
        )


if __name__ == "__main__":
    unittest.main()
