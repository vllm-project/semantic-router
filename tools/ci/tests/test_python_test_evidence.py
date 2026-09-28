from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/ci"))
from ci_results import collection_errors  # noqa: E402
from python_test_evidence import summarize  # noqa: E402


class FrameworkObserverTests(unittest.TestCase):
    def run_framework(self, source, framework):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sitecustomize.py").write_text(
                "from python_test_evidence import observe_unittest\nobserve_unittest()\nobserve_unittest()\n"
            )
            test = root / "test_observed.py"
            test.write_text(source)
            events = root / "events.jsonl"
            env = {
                **os.environ,
                "PYTHONPATH": str(root) + os.pathsep + str(ROOT / "tools/ci"),
                "CI_REPO_ROOT": str(root),
                "CI_PYTHON_TEST_EVENTS": str(events),
                "PYTEST_PLUGINS": "python_test_evidence",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            }
            command = (
                [sys.executable, "-m", "unittest", "test_observed"]
                if framework == "unittest"
                else [sys.executable, "-m", "pytest", "-q", str(test)]
            )
            process = subprocess.run(
                command,
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            return summarize(events)

    def test_nested_negative_tests_are_not_selected_inventory(self):
        report = self.run_framework(
            """import io
import unittest
class Outer(unittest.TestCase):
    def test_regression(self):
        class Negative(unittest.TestCase):
            def test_bad(self): self.fail("expected negative fixture")
            @unittest.skip("negative fixture")
            def test_skip(self): pass
        outcome = unittest.TextTestRunner(stream=io.StringIO()).run(unittest.defaultTestLoader.loadTestsFromTestCase(Negative))
        self.assertEqual(len(outcome.failures), 1)
        self.assertEqual(len(outcome.skipped), 1)
""",
            "unittest",
        )
        self.assertEqual(len(report["cases"]), 1)
        self.assertEqual(collection_errors(report, "test"), [])

    def test_genuine_framework_skips_are_rejected(self):
        report = self.run_framework(
            """import unittest
class Selected(unittest.TestCase):
    @unittest.skip("missing dependency")
    def test_selected(self): pass
""",
            "unittest",
        )
        self.assertTrue(
            any("skipped" in error for error in collection_errors(report, "test"))
        )

    def test_pytest_root_directory_keeps_collection_and_execution_identity(self):
        report = self.run_framework(
            "def test_real():\n    assert 2 + 2 == 4\n", "pytest"
        )
        self.assertEqual(len(report["cases"]), 1)
        self.assertEqual(collection_errors(report, "test"), [])


if __name__ == "__main__":
    unittest.main()
