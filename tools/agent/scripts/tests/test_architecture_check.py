import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

architecture_check = importlib.import_module("architecture_check")


class DependencyGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.python_scope = {
            "name": "python-example",
            "language": "python",
            "root": "src/pkg",
            "module_root": "src",
            "include": ["src/pkg/*.py"],
            "test_patterns": ["**/test_*.py"],
            "cycle_policy": "no-new",
        }

    def test_rejects_new_python_cycle(self) -> None:
        current = {
            "src/pkg/a.py": "from pkg import b\n",
            "src/pkg/b.py": "from pkg import a\n",
        }
        baseline = {
            "src/pkg/a.py": "from pkg import b\n",
            "src/pkg/b.py": "VALUE = 1\n",
        }

        findings = architecture_check.evaluate_dependency_graph(
            self.python_scope,
            current,
            baseline,
            {"src/pkg/b.py"},
        )

        errors = [finding for finding in findings if finding.level == "ERROR"]
        self.assertEqual(len(errors), 1)
        self.assertIn("new dependency cycle", errors[0].message)

    def test_warns_for_unchanged_python_cycle(self) -> None:
        sources = {
            "src/pkg/a.py": "from pkg import b\n",
            "src/pkg/b.py": "from pkg import a\n",
        }

        findings = architecture_check.evaluate_dependency_graph(
            self.python_scope,
            sources,
            sources,
            {"src/pkg/a.py"},
        )

        warnings = [finding for finding in findings if finding.level == "WARN"]
        self.assertEqual(len(warnings), 1)
        self.assertIn("pre-existing dependency cycle", warnings[0].message)

    def test_detects_typescript_reexport_cycle(self) -> None:
        scope = {
            "name": "typescript-example",
            "language": "typescript",
            "root": "ui",
            "include": ["ui/*.ts"],
            "test_patterns": [],
            "cycle_policy": "no-new",
        }
        current = {
            "ui/a.ts": "import { b } from './b'\nexport const a = b\n",
            "ui/b.ts": "export { a as b } from './a'\n",
        }
        baseline = {
            "ui/a.ts": "import { b } from './b'\nexport const a = b\n",
            "ui/b.ts": "export const b = 1\n",
        }

        findings = architecture_check.evaluate_dependency_graph(
            scope,
            current,
            baseline,
            {"ui/b.ts"},
        )

        self.assertTrue(any(finding.level == "ERROR" for finding in findings))

    def test_rejects_new_forbidden_graph_edge(self) -> None:
        scope = dict(self.python_scope)
        scope["forbidden_edges"] = [
            {
                "name": "domain-must-not-depend-on-entrypoint",
                "policy": "no-new",
                "from": ["src/pkg/a.py"],
                "to": ["src/pkg/b.py"],
            }
        ]
        current = {
            "src/pkg/a.py": "from pkg import b\n",
            "src/pkg/b.py": "VALUE = 1\n",
        }
        baseline = {
            "src/pkg/a.py": "VALUE = 1\n",
            "src/pkg/b.py": "VALUE = 1\n",
        }

        findings = architecture_check.evaluate_dependency_graph(
            scope,
            current,
            baseline,
            {"src/pkg/a.py"},
        )

        errors = [finding for finding in findings if finding.level == "ERROR"]
        self.assertEqual(len(errors), 1)
        self.assertIn("domain-must-not-depend-on-entrypoint", errors[0].message)

    def test_removing_cycle_edges_can_split_existing_component(self) -> None:
        baseline = {
            "src/pkg/a.py": "from pkg import b, c\n",
            "src/pkg/b.py": "from pkg import a\n",
            "src/pkg/c.py": "from pkg import a\n",
        }
        current = dict(baseline)
        current["src/pkg/a.py"] = "from pkg import b\n"
        findings = architecture_check.evaluate_dependency_graph(
            self.python_scope, current, baseline, {"src/pkg/a.py"}
        )
        self.assertTrue(any(f.level == "WARN" for f in findings))
        self.assertFalse(any(f.level == "ERROR" for f in findings))

    def test_new_edge_inside_existing_cycle_is_rejected(self) -> None:
        baseline = {
            "src/pkg/a.py": "from pkg import b\n",
            "src/pkg/b.py": "from pkg import c\n",
            "src/pkg/c.py": "from pkg import a\n",
        }
        current = dict(baseline)
        current["src/pkg/a.py"] = "from pkg import b, c\n"
        findings = architecture_check.evaluate_dependency_graph(
            self.python_scope, current, baseline, {"src/pkg/a.py"}
        )
        self.assertTrue(any(f.level == "ERROR" for f in findings))


class ImportBoundaryTests(unittest.TestCase):
    def check_import(self, path, source, forbidden, baseline=""):
        rules = {
            "dependency_rules": [
                {
                    "name": "domain-boundary",
                    "applies_to": ["src/*"],
                    "forbidden_imports": [forbidden],
                    "policy": "no-new",
                }
            ]
        }
        with mock.patch.object(
            architecture_check, "load_baseline_source", return_value=baseline
        ):
            return architecture_check.evaluate_dependency_rules(path, source, rules)

    def test_python_comments_and_data_do_not_create_dependencies(self):
        source = '# import tools.agent\nPATH = "tools.agent"\n'
        self.assertEqual(self.check_import("src/app.py", source, "tools.agent"), [])
        self.assertEqual(
            self.check_import("src/app.py", "from tools import agent\n", "tools.agent")[
                0
            ].level,
            "ERROR",
        )

    def test_go_imports_are_checked_but_comments_and_path_strings_are_not(self):
        source = 'package app\n// import "example.org/handlers"\nvar path = "example.org/handlers"\n'
        self.assertEqual(
            self.check_import("src/app.go", source, "example.org/handlers"), []
        )
        source = 'package app\nimport alias "example.org/handlers"\n'
        self.assertEqual(
            self.check_import("src/app.go", source, "example.org/handlers")[0].level,
            "ERROR",
        )
        self.assertEqual(
            self.check_import("src/app.go", source, "example.org/handlers", source)[
                0
            ].level,
            "WARN",
        )

    def test_rust_grouped_imports_and_crate_prefix_are_checked(self):
        source = '// use tools::agent;\nconst PATH: &str = "tools::agent";\n'
        self.assertEqual(self.check_import("src/app.rs", source, "tools::agent"), [])
        for source in (
            "use tools::{agent::run, other};",
            "use crate::tools::agent as helper;",
        ):
            self.assertEqual(
                self.check_import("src/app.rs", source, "tools::agent")[0].level,
                "ERROR",
            )

    def test_prefix_does_not_match_unrelated_module(self):
        self.assertEqual(
            self.check_import("src/app.py", "import tools.agentic\n", "tools.agent"), []
        )

    def test_relative_python_import_is_not_an_external_dependency(self):
        self.assertEqual(
            self.check_import(
                "src/app.py", "from .tools import agent\n", "tools.agent"
            ),
            [],
        )


class RootPlacementTests(unittest.TestCase):
    def test_only_existing_unowned_root_files_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").touch()
            (root / "notes.md").touch()
            (root / "tools").mkdir()
            (root / "tools/notes.md").touch()
            rules = {"root_files": {"allowed": ["README.md"]}}
            with mock.patch.object(architecture_check, "REPO_ROOT", root):
                for name in ("README.md", "tools/notes.md", "deleted.md"):
                    self.assertEqual(
                        architecture_check.evaluate_root_placement(name, rules), []
                    )
                self.assertEqual(
                    architecture_check.evaluate_root_placement("notes.md", rules)[
                        0
                    ].level,
                    "ERROR",
                )


if __name__ == "__main__":
    unittest.main()
