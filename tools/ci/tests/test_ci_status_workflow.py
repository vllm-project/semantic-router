"""The privileged status publisher must never execute a PR's contents."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/ci"))

from validate_workflows import load_workflows  # noqa: E402
from workflow_policy_validation import (  # noqa: E402
    validate_gate_transport,
    validate_pr_contract,
    validate_publisher_bootstrap,
    validate_status_publisher,
)


class CIStatusWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        errors: list[str] = []
        self.workflows = load_workflows(errors)
        self.assertEqual(errors, [])
        self.publisher = self.workflows["ci-status.yml"]

    def test_current_workflows_preserve_the_pr_gate_contract(self) -> None:
        errors: list[str] = []
        validate_pr_contract(self.workflows, errors)
        self.assertEqual(errors, [])

    def test_publisher_checkout_cannot_use_the_observed_pr_head(self) -> None:
        checkout = self.publisher.jobs["publish"]["steps"][0]
        checkout["with"]["ref"] = "${{ github.event.workflow_run.head_sha }}"
        errors: list[str] = []
        validate_status_publisher(self.workflows, errors)
        self.assertTrue(any("never check out" in error for error in errors), errors)

    def test_publisher_cannot_execute_untrusted_artifacts(self) -> None:
        self.publisher.jobs["publish"]["steps"].append(
            {"run": "python3 artifact/run.py"}
        )
        errors: list[str] = []
        validate_status_publisher(self.workflows, errors)
        self.assertTrue(
            any("trusted publisher entrypoint" in error for error in errors)
        )

    def test_publisher_dispatch_cannot_run_a_topic_branch(self) -> None:
        self.publisher.jobs["publish"]["if"] = "always()"
        errors: list[str] = []
        validate_status_publisher(self.workflows, errors)
        self.assertTrue(any("trusted main code" in error for error in errors))

    def test_publisher_observes_reruns_not_only_initial_requests(self) -> None:
        self.publisher.events["workflow_run"]["types"] = ["requested", "completed"]
        errors: list[str] = []
        validate_status_publisher(self.workflows, errors)
        self.assertTrue(any("reruns and completions" in error for error in errors))

    def test_publisher_cannot_gain_source_write_permission(self) -> None:
        self.publisher.jobs["publish"]["permissions"]["contents"] = "write"
        errors: list[str] = []
        validate_status_publisher(self.workflows, errors)
        self.assertTrue(any("token permissions" in error for error in errors))

    def test_issue_write_does_not_replace_pr_feedback_write_permission(self) -> None:
        self.publisher.jobs["publish"]["permissions"]["pull-requests"] = "read"
        errors: list[str] = []
        validate_status_publisher(self.workflows, errors)
        self.assertTrue(any("token permissions" in error for error in errors))

    def test_bootstrap_probe_cannot_use_the_checkout_tree(self) -> None:
        steps = self.workflows["ci-changes.yml"].jobs["plan"]["steps"]
        probe = next(step for step in steps if step.get("id") == "publisher")
        probe["run"] = probe["run"].replace("$base:", "HEAD:")
        errors: list[str] = []
        validate_publisher_bootstrap(self.workflows, errors)
        self.assertTrue(any("actual merge first parent" in error for error in errors))

    def test_gate_requires_publisher_readiness_before_deferring(self) -> None:
        gate = copy.deepcopy(self.workflows["ci.yml"].jobs["gate"])
        step = next(
            step for step in gate["steps"] if "check_ci_gate.py" in step.get("run", "")
        )
        step["run"] = step["run"].replace(
            'if [[ "$PUBLISHER_READY" == true ]]; then flags+=(--allow-deferred); fi',
            "flags+=(--allow-deferred)",
        )
        errors: list[str] = []
        validate_gate_transport(gate, errors)
        self.assertTrue(
            any("only with a trusted base publisher" in error for error in errors)
        )

    def test_failed_gate_outcome_must_still_be_uploaded(self) -> None:
        gate = self.workflows["ci.yml"].jobs["gate"]
        upload = next(
            step
            for step in gate["steps"]
            if step.get("with", {}).get("name") == "ci-gate"
        )
        upload.pop("if")
        errors: list[str] = []
        validate_gate_transport(gate, errors)
        self.assertTrue(any("always publish" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
