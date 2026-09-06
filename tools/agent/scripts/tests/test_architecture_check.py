import importlib
import sys
import unittest
from pathlib import Path

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

    def test_health_summary_is_diagnostic(self) -> None:
        sources = {
            "src/pkg/a.py": "VALUE = 1\n",
            "src/pkg/test_a.py": "def test_value():\n    assert True\n",
        }

        message = architecture_check.health_message(self.python_scope, sources)

        self.assertIn("production_files=1", message)
        self.assertIn("test_files=1", message)
        self.assertIn("source_lines=3", message)


if __name__ == "__main__":
    unittest.main()
