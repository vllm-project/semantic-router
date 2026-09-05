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


class DependencyRuleTests(unittest.TestCase):
    def test_rejects_new_forbidden_dependency(self) -> None:
        rules = {
            "dependency_rules": [
                {
                    "name": "layer-boundary",
                    "policy": "no-new",
                    "applies_to": ["src/*.py"],
                    "forbidden_literals": ["forbidden.layer"],
                }
            ]
        }
        with mock.patch.object(
            structure_check, "load_baseline_source", return_value=""
        ):
            findings = structure_check.evaluate_dependency_rules(
                "src/example.py", "import forbidden.layer\n", rules, "base"
            )

        self.assertEqual(findings[0].level, "ERROR")

    def test_warns_for_unchanged_forbidden_dependency(self) -> None:
        rules = {
            "dependency_rules": [
                {
                    "name": "layer-boundary",
                    "policy": "no-new",
                    "applies_to": ["src/*.py"],
                    "forbidden_literals": ["forbidden.layer"],
                }
            ]
        }
        source = "import forbidden.layer\n"
        with mock.patch.object(
            structure_check, "load_baseline_source", return_value=source
        ):
            findings = structure_check.evaluate_dependency_rules(
                "src/example.py", source, rules, "base"
            )

        self.assertEqual(findings[0].level, "WARN")


class TypeScriptStructureTests(unittest.TestCase):
    def test_collects_arrow_function_metrics(self) -> None:
        parser = structure_check.build_parser("typescript")
        source = b"const choose = () => {\n  if (ready) {\n    return 1\n  }\n  return 0\n}\n"

        metrics = structure_check.collect_function_metrics(
            parser.parse(source), "typescript", source
        )

        self.assertEqual(len(metrics), 1)
        metric = next(iter(metrics.values()))
        self.assertEqual(metric.lines, 6)
        self.assertEqual(metric.nesting, 1)


if __name__ == "__main__":
    unittest.main()
