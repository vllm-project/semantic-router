"""Vitest's framework IDs preserve parameterized cases without accepting reruns."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_results import collection_errors

ROOT = Path(__file__).resolve().parents[3]
REPORTER = ROOT / "tools/ci/vitest_evidence_reporter.mjs"
DRIVER = """
const { default: Reporter } = await import(process.argv[1]);
const scenario = JSON.parse(process.argv[2]);
const reporter = new Reporter();
reporter.onInit();
reporter.onTestModuleCollected({children: {allTests: () => scenario.collected}});
for (const test of scenario.executed) {
  reporter.onTestCaseResult({...test, result: () => ({state: test.status})});
}
reporter.onTestRunEnd([], scenario.errors || [], scenario.reason || 'passed');
"""


class VitestEvidenceTests(unittest.TestCase):
    def run_reporter(self, collected, executed, **options):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "frontend.json"
            result = subprocess.run(
                [
                    "node",
                    "--input-type=module",
                    "-e",
                    DRIVER,
                    REPORTER.as_uri(),
                    json.dumps(
                        {"collected": collected, "executed": executed, **options}
                    ),
                ],
                env={**os.environ, "VITEST_EVIDENCE_PATH": str(output)},
                capture_output=True,
                text=True,
                check=False,
            )
            return result, json.loads(output.read_text())

    def test_parameterized_cases_keep_distinct_framework_ids(self):
        collected = [{"id": "file_0_0"}, {"id": "file_0_1"}]
        executed = [
            {**test, "fullName": "same parameterized title", "status": "passed"}
            for test in collected
        ]
        result, evidence = self.run_reporter(collected, executed)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            evidence["expected_cases"], ["vitest:file_0_0", "vitest:file_0_1"]
        )
        self.assertEqual(collection_errors(evidence, "test"), [])
        self.assertEqual(len(evidence["cases"]), 2)

    def test_duplicate_missing_extra_skipped_and_failed_execution_rejected(self):
        first = {"id": "first", "fullName": "case", "status": "passed"}
        second = {"id": "second", "fullName": "case", "status": "passed"}
        for collected, executed, message in (
            ([first, first], [first], "duplicate"),
            ([first], [first, first], "duplicate"),
            ([first, second], [first], "inventory"),
            ([first], [first, second], "inventory"),
            ([], [], "empty"),
            ([first], [{**first, "status": "skipped"}], "did not pass"),
            ([first], [{**first, "status": "failed"}], "did not pass"),
            ([first], [{**first, "status": "pending"}], "did not pass"),
        ):
            with self.subTest(collected=collected, executed=executed):
                result, _ = self.run_reporter(collected, executed)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_run_level_failure_cannot_be_hidden_by_passing_cases(self):
        case = {"id": "first", "fullName": "case", "status": "passed"}
        for options in ({"reason": "failed"}, {"errors": ["suite cleanup failure"]}):
            with self.subTest(options=options):
                result, _ = self.run_reporter([case], [case], **options)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Vitest run did not pass", result.stderr)

    def test_dashboard_consumes_framework_ids_and_keeps_junit_diagnostics(self):
        target = (ROOT / "tools/make/dashboard.mk").read_text()
        workflow = (ROOT / ".github/workflows/dashboard-test.yml").read_text()
        self.assertIn("--reporter=../../tools/ci/vitest_evidence_reporter.mjs", target)
        self.assertIn(
            '--outputFile.junit="$(DASHBOARD_TEST_REPORT_DIR)/frontend.xml"', target
        )
        self.assertIn("--evidence .agent-harness/dashboard/frontend.json", workflow)
        self.assertNotIn("--junit .agent-harness/dashboard/frontend.xml", workflow)


if __name__ == "__main__":
    unittest.main()
