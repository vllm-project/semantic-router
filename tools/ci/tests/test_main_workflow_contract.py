from __future__ import annotations

import os
import subprocess
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

    def test_pypi_accepts_only_qualified_main_dispatch_or_stable_tag(self) -> None:
        workflow = yaml.safe_load(
            (REPO_ROOT / ".github/workflows/pypi-publish.yml").read_text()
        )
        condition = workflow["jobs"]["pypi"]["if"]
        # Evaluate the workflow's actual guard across publication boundaries.
        for event, ref, channel, prebuilt, allowed in (
            ("workflow_dispatch", "refs/heads/main", "dev", True, True),
            ("workflow_dispatch", "refs/heads/main", "dev", False, False),
            ("workflow_dispatch", "refs/heads/feature", "dev", True, False),
            ("push", "refs/heads/main", "dev", True, False),
            ("pull_request", "refs/heads/main", "dev", True, False),
            ("push", "refs/tags/v0.5.0", "stable", True, True),
            ("workflow_dispatch", "refs/tags/v0.5.0", "stable", True, False),
        ):
            with self.subTest(event=event, ref=ref, channel=channel, prebuilt=prebuilt):
                expression = condition
                for key, value in {
                    "inputs.build-only": False,
                    "inputs.prebuilt-dist": prebuilt,
                    "inputs.channel": channel,
                    "needs.build.result": "success",
                    "github.repository": "vllm-project/semantic-router",
                    "github.event_name": event,
                    "github.ref": ref,
                }.items():
                    expression = expression.replace(key, repr(value))
                expression = expression.replace("&&", " and ").replace("||", " or ")
                expression = expression.replace("!", "not ")
                self.assertEqual(
                    eval(
                        expression,
                        {"__builtins__": {}},
                        {
                            "always": lambda: True,
                            "cancelled": lambda: False,
                            "failure": lambda: False,
                            "startsWith": lambda value, prefix: value.startswith(
                                prefix
                            ),
                        },
                    ),
                    allowed,
                )


class PullRequestGateContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = yaml.safe_load(
            (REPO_ROOT / ".github" / "workflows" / "pr.yml").read_text(encoding="utf-8")
        )
        self.gate = self.workflow["jobs"]["pr-gate"]

    def test_gates_evaluate_failed_or_cancelled_prerequisites(self) -> None:
        for filename, job in (
            ("pr.yml", "pr-gate"),
            ("ci.yml", "gate"),
        ):
            with self.subTest(workflow=filename):
                workflow = yaml.safe_load(
                    (REPO_ROOT / ".github" / "workflows" / filename).read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(
                    workflow["jobs"][job]["if"],
                    "always()",
                    "skipping a required gate must not hide failed or cancelled prerequisites",
                )
        self.assertTrue(self.workflow["concurrency"]["cancel-in-progress"])

    def test_stable_gate_rejects_unsuccessful_shared_qualification(self) -> None:
        script = self.gate["steps"][0]["run"]
        self.assertEqual(self.gate["needs"], "ci")
        for result, expected_exit in (
            ("success", 0),
            ("skipped", 1),
            ("failure", 1),
            ("cancelled", 1),
        ):
            with self.subTest(result=result):
                actual = subprocess.run(
                    ["bash", "-e", "-c", script],
                    env={**os.environ, "RESULT": result},
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(actual.returncode, expected_exit, actual.stderr)


if __name__ == "__main__":
    unittest.main()
