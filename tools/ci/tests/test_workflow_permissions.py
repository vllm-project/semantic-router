from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools" / "ci"))

from validate_workflows import (  # noqa: E402
    UniqueKeyLoader,
    Workflow,
    validate_local_call,
)


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

    def test_duplicate_workflow_output_keys_are_rejected(self) -> None:
        with self.assertRaises(yaml.YAMLError):
            yaml.load(
                "outputs:\n  native: true\n  native: matrix\n", Loader=UniqueKeyLoader
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

    def test_acceptance_backfill_runs_one_sweep_at_a_time_off_events(self) -> None:
        job = self.data["jobs"]["issue-acceptance-sweep"]

        self.assertEqual(
            job["concurrency"],
            {
                "group": "community-label-sync-acceptance-sweep",
                "cancel-in-progress": False,
            },
        )
        self.assertIn("github.event_name == 'schedule'", job["if"])
        self.assertIn("github.event_name == 'workflow_dispatch'", job["if"])
        self.assertNotIn("workflow_run", job["if"])
        self.assertNotIn("check_suite", job["if"])
        self.assertIn("accept-issues", job["steps"][-1]["run"])


class CommunityValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        path = REPO_ROOT / ".github" / "workflows" / "community.yml"
        self.data = yaml.safe_load(path.read_text(encoding="utf-8"))

    def test_linked_issue_validation_only_runs_for_ready_human_pr_events(self) -> None:
        condition = self.data["jobs"]["linked-issue"]["if"]
        for event in ("pull_request", "pull_request_review", "issues", "issue_comment"):
            for draft in (False, True):
                for author_type in ("User", "Bot"):
                    with self.subTest(event=event, draft=draft, author=author_type):
                        expression = condition
                        for name, value in {
                            "github.event_name": event,
                            "github.event.pull_request.draft": draft,
                            "github.event.pull_request.user.type": author_type,
                        }.items():
                            expression = expression.replace(name, repr(value))
                        expression = expression.replace("&&", " and ").replace(
                            "||", " or "
                        )
                        expression = " ".join(
                            re.sub(r"!(?!=)", "not ", expression).split()
                        )
                        self.assertEqual(
                            eval(expression, {"__builtins__": {}}),
                            event == "pull_request"
                            and not draft
                            and author_type == "User",
                        )

    def test_only_read_only_jobs_cancel_superseded_checks(self) -> None:
        self.assertNotIn("concurrency", self.data)
        jobs = self.data["jobs"]
        self.assertEqual(
            {name for name, job in jobs.items() if "concurrency" in job},
            {"title-format", "linked-issue"},
        )
        for name in ("title-format", "linked-issue"):
            with self.subTest(job=name):
                job = jobs[name]
                concurrency = job["concurrency"]
                self.assertTrue(concurrency["cancel-in-progress"])
                group = concurrency["group"]
                self.assertTrue(group.startswith(f"community-{name}-"))
                self.assertIn("github.event.pull_request.number", group)
                self.assertIn("github.run_id", group)
                self.assertNotIn("github.sha", group)
                self.assertTrue(
                    all(value == "read" for value in job["permissions"].values())
                )
        self.assertIn(
            "github.event.issue.number", jobs["title-format"]["concurrency"]["group"]
        )

    def test_accept_comments_reconcile_one_issue_not_the_repository(self) -> None:
        job = self.data["jobs"]["issue-acceptance"]
        commands = " ".join(step.get("run", "") for step in job["steps"])

        self.assertIn("github.event.comment.body == '/accept'", job["if"])
        self.assertIn("community_lifecycle.py accept-issue ", commands)
        self.assertNotIn("accept-issues", commands)

    def test_review_events_keep_the_trusted_label_sync_bridge(self) -> None:
        events = self.data.get("on", self.data.get(True))
        self.assertEqual(
            events["pull_request_review"]["types"], ["submitted", "dismissed"]
        )
        self.assertNotIn("synchronize", events["pull_request"]["types"])
        self.assertIn("edited", events["pull_request"]["types"])
        path = REPO_ROOT / ".github" / "workflows" / "community-labels.yml"
        sync = yaml.safe_load(path.read_text(encoding="utf-8"))
        sync_events = sync.get("on", sync.get(True))
        self.assertNotIn("pull_request_review", sync_events)
        self.assertEqual(sync_events["workflow_run"]["workflows"], ["Community"])
        self.assertIn(
            "github.event.workflow_run.event == 'pull_request_review'",
            sync["jobs"]["pull-request-state"]["if"],
        )


if __name__ == "__main__":
    unittest.main()
