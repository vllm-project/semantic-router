import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

structure_check = importlib.import_module("structure_check")


class RootPlacementTests(unittest.TestCase):
    def test_allows_declared_root_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").touch()
            with mock.patch.object(structure_check, "REPO_ROOT", root):
                findings = structure_check.evaluate_root_placement(
                    "README.md", {"root_files": {"allowed": ["README.md"]}}
                )

        self.assertEqual(findings, [])

    def test_rejects_unowned_root_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "notes.md").touch()
            with mock.patch.object(structure_check, "REPO_ROOT", root):
                findings = structure_check.evaluate_root_placement(
                    "notes.md", {"root_files": {"allowed": ["README.md"]}}
                )

        self.assertEqual(len(findings), 1)
        self.assertIn("not allowlisted", findings[0].message)

    def test_ignores_deleted_and_nested_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "tools").mkdir()
            (root / "tools" / "notes.md").touch()
            rules = {"root_files": {"allowed": []}}
            with mock.patch.object(structure_check, "REPO_ROOT", root):
                nested = structure_check.evaluate_root_placement(
                    "tools/notes.md", rules
                )
                deleted = structure_check.evaluate_root_placement("removed.md", rules)

        self.assertEqual(nested, [])
        self.assertEqual(deleted, [])


if __name__ == "__main__":
    unittest.main()
