from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools" / "ci"))

from validate_workflows import Workflow, validate_local_call  # noqa: E402


def workflow(name: str, data: dict[str, object]) -> Workflow:
    return Workflow(REPO_ROOT / ".github" / "workflows" / name, data)


class ReusableWorkflowPermissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.publisher = workflow(
            "publisher.yml",
            {
                "on": {"workflow_call": {}},
                "permissions": {"contents": "read"},
                "jobs": {
                    "publish": {
                        "permissions": {
                            "contents": "read",
                            "packages": "write",
                        }
                    }
                },
            },
        )

    def test_rejects_callee_write_above_caller_ceiling(self) -> None:
        caller = workflow(
            "caller.yml",
            {
                "permissions": {"contents": "read"},
                "jobs": {
                    "publish": {
                        "uses": "./.github/workflows/publisher.yml",
                    }
                },
            },
        )
        errors: list[str] = []

        validate_local_call(
            caller,
            "publish",
            caller.jobs["publish"],
            {"publisher.yml": self.publisher},
            errors,
        )

        self.assertEqual(len(errors), 1)
        self.assertIn("callee requests write permissions", errors[0])
        self.assertIn("packages", errors[0])

    def test_accepts_explicit_caller_write_permission(self) -> None:
        caller = workflow(
            "caller.yml",
            {
                "permissions": {"contents": "read"},
                "jobs": {
                    "publish": {
                        "uses": "./.github/workflows/publisher.yml",
                        "permissions": {
                            "contents": "read",
                            "packages": "write",
                        },
                    }
                },
            },
        )
        errors: list[str] = []

        validate_local_call(
            caller,
            "publish",
            caller.jobs["publish"],
            {"publisher.yml": self.publisher},
            errors,
        )

        self.assertEqual(errors, [])


class CommunityLabelSyncConcurrencyTests(unittest.TestCase):
    def setUp(self) -> None:
        path = REPO_ROOT / ".github" / "workflows" / "community-labels.yml"
        self.data = yaml.safe_load(path.read_text(encoding="utf-8"))

    def test_pr_events_and_scheduled_sweep_use_independent_job_concurrency(
        self,
    ) -> None:
        self.assertNotIn("concurrency", self.data)
        jobs = self.data["jobs"]
        pr_concurrency = jobs["pull-request-state"]["concurrency"]
        sweep_concurrency = jobs["pull-request-sweep"]["concurrency"]

        self.assertIn(
            "github.event.workflow_run.pull_requests[0].number",
            pr_concurrency["group"],
        )
        self.assertIn(
            "github.event.check_suite.pull_requests[0].number",
            pr_concurrency["group"],
        )
        self.assertEqual(sweep_concurrency["group"], "community-label-sync-sweep")
        self.assertNotEqual(pr_concurrency["group"], sweep_concurrency["group"])
        self.assertFalse(pr_concurrency["cancel-in-progress"])
        self.assertFalse(sweep_concurrency["cancel-in-progress"])


if __name__ == "__main__":
    unittest.main()
