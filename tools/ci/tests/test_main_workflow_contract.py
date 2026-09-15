from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


class MainWorkflowContractTests(unittest.TestCase):
    def test_main_validation_runs_cannot_be_coalesced(self) -> None:
        workflow = yaml.safe_load(
            (REPO_ROOT / ".github" / "workflows" / "main.yml").read_text(
                encoding="utf-8"
            )
        )

        self.assertNotIn(
            "concurrency",
            workflow,
            "main push runs must remain per-commit so change classification cannot skip a commit",
        )


class PullRequestGateContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = yaml.safe_load(
            (REPO_ROOT / ".github" / "workflows" / "pr.yml").read_text(encoding="utf-8")
        )
        self.gate = self.workflow["jobs"]["pr-gate"]

    def test_gate_aggregates_failures_without_retaining_cancelled_runs(self) -> None:
        self.assertEqual(
            self.gate["if"],
            "${{ !cancelled() }}",
            "an explicit status condition must run after failure but stop on cancellation",
        )
        self.assertTrue(self.workflow["concurrency"]["cancel-in-progress"])

    def test_actual_gate_script_rejects_unsuccessful_selected_domains(self) -> None:
        script = self.gate["steps"][0]["run"]
        for result, expected_exit in (
            ("success", 0),
            ("skipped", 0),
            ("failure", 1),
            ("cancelled", 1),
        ):
            with self.subTest(
                result=result
            ), tempfile.TemporaryDirectory() as directory:
                summary = Path(directory) / "summary.md"
                actual = subprocess.run(
                    ["bash", "-e", "-c", script],
                    env={
                        **os.environ,
                        "GITHUB_STEP_SUMMARY": str(summary),
                        "DOMAIN_RESULTS": json.dumps(
                            {
                                "changes": {"result": "success"},
                                "core-tests": {"result": result},
                            }
                        ),
                    },
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=10,
                )
                self.assertEqual(actual.returncode, expected_exit, actual.stderr)
                self.assertIn(f"- core-tests: {result}", summary.read_text())


if __name__ == "__main__":
    unittest.main()
