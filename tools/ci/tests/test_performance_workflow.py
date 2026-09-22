"""Numeric regressions are advisory without hiding execution or evidence failures."""

import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github/workflows/performance-test.yml"


class PerformanceWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.job = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"][
            "component-benchmarks"
        ]
        cls.steps = cls.job["steps"]
        cls.by_id = {step["id"]: step for step in cls.steps if "id" in step}

    def comparison(self, regression: bool) -> str:
        return json.dumps(
            {
                "has_regressions": regression,
                "results": [
                    {
                        "BenchmarkName": "BenchmarkClassifyBatch_Parallel",
                        "RegressionDetected": regression,
                        "Baseline": {"allocs_per_op": 176, "bytes_per_op": 22325},
                        "Current": {
                            "allocs_per_op": 180 if regression else 176,
                            "bytes_per_op": 25259 if regression else 22325,
                        },
                        "AllocsPerOpChange": 2.273 if regression else 0,
                        "BytesPerOpChange": 13.142 if regression else 0,
                        "NsAdvisory": regression,
                        "NsPerOpChange": 20.5 if regression else 0,
                    }
                ],
            }
        )

    def run_summary(self, report: str | None, **outcomes: str):
        expressions = {
            "steps.bench.outcome": "success",
            "steps.looper.outcome": "success",
            "steps.comparison.outcome": "success",
            "github.server_url": "https://github.com",
            "github.repository": "example/router",
            "github.run_id": "123",
        }
        expressions.update(
            {f"steps.{step}.outcome": outcome for step, outcome in outcomes.items()}
        )
        script = re.sub(
            r"\$\{\{\s*(.*?)\s*\}\}",
            lambda match: expressions[match[1]],
            self.by_id["summary"]["run"],
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports"
            reports.mkdir()
            if report is not None:
                (reports / "comparison.json").write_text(report, encoding="utf-8")
            (reports / "bench-output.txt").write_text(
                "BenchmarkClassifyBatch_Parallel-4 15 1 ns/op 25259 B/op 180 allocs/op\n"
                "BenchmarkEvaluate-4 20 1 ns/op\n"
                "BenchmarkCache-4 20 1 ns/op\n"
                "BenchmarkReMoM-4 20 1 ns/op\n",
                encoding="utf-8",
            )
            summary_path = root / "github-step-summary.md"
            result = subprocess.run(
                ["bash", "-e", "-o", "pipefail", "-c", script],
                cwd=root,
                env={**os.environ, "GITHUB_STEP_SUMMARY": str(summary_path)},
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            summary = (
                summary_path.read_text(encoding="utf-8")
                if summary_path.exists()
                else ""
            )
            return result, summary

    def test_allocation_regression_warns_and_preserves_measurements(self) -> None:
        result, summary = self.run_summary(self.comparison(True))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("::warning::Performance regressions detected", result.stdout)
        self.assertIn("Performance regressions detected (advisory)", summary)
        self.assertIn("numerical regressions do not block CI", summary)
        self.assertIn("BenchmarkClassifyBatch_Parallel", summary)
        self.assertIn("allocs/op 176→180 (+2%)", summary)
        self.assertIn("B/op 22325→25259 (+13%)", summary)
        self.assertIn("Advisory timing regressions", summary)
        self.assertNotIn("**FAILED**", summary)

    def test_clean_comparison_does_not_claim_regression(self) -> None:
        result, summary = self.run_summary(self.comparison(False))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("::warning::", result.stdout)
        self.assertIn("no allocation regressions beyond thresholds", summary)
        self.assertNotIn("Performance regressions detected", summary)
        self.assertNotIn("Advisory allocation regressions", summary)
        self.assertNotIn("**FAILED**", summary)

    def test_execution_and_comparison_failures_are_not_reported_as_completed(self):
        for outcomes in (
            {"bench": "failure", "comparison": "skipped"},
            {"looper": "failure", "comparison": "skipped"},
            {"comparison": "failure"},
        ):
            with self.subTest(outcomes=outcomes):
                result, summary = self.run_summary(None, **outcomes)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("**FAILED**", summary)
                self.assertNotIn("measurements completed", summary)
                self.assertNotIn("no allocation regressions beyond thresholds", summary)

    def test_success_outcome_requires_readable_structured_comparison(self) -> None:
        for report in (
            None,
            "{invalid json",
            "{}",
            '{"results": null, "has_regressions": false}',
            '{"results": {}, "has_regressions": false}',
            '{"results": [], "has_regressions": "false"}',
        ):
            with self.subTest(report=report):
                result, summary = self.run_summary(report)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertNotIn("no allocation regressions beyond thresholds", summary)
                self.assertNotIn("measurements completed", summary)

    def test_advisory_comparison_keeps_inventory_and_model_baseline(self) -> None:
        comparison = self.by_id["comparison"]["run"]
        self.assertIn("--inventory=", comparison)
        self.assertIn("--model-baseline=", comparison)
        self.assertNotIn("--fail-on-regression", comparison)
        self.assertFalse(self.job.get("continue-on-error", False))
        for identifier in ("bench", "looper", "current", "comparison", "summary"):
            with self.subTest(step=identifier):
                self.assertFalse(self.by_id[identifier].get("continue-on-error", False))

        evidence = next(
            step
            for step in self.steps
            if "tools/ci/workflow_evidence.py" in step.get("run", "")
        )
        self.assertFalse(evidence.get("continue-on-error", False))
        self.assertIn("--benchmarks reports/current.json", evidence["run"])
        self.assertIn(
            "--inventory perf/config/benchmark-inventory.json", evidence["run"]
        )
        receipt_upload = next(
            step
            for step in self.steps
            if step.get("with", {}).get("name") == "ci-result-performance"
        )
        self.assertFalse(receipt_upload.get("continue-on-error", False))
        self.assertEqual(receipt_upload["with"]["if-no-files-found"], "error")


if __name__ == "__main__":
    unittest.main()
