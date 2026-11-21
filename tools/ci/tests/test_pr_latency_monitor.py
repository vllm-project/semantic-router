from __future__ import annotations

import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pr_latency_monitor import (
    alerts_for_approved_prs,
    alerts_for_queued_runs,
    reconcile,
)

NOW = datetime(2026, 11, 12, 18, 0, tzinfo=timezone.utc)
HEAD = "a" * 40
REPO = "example/router"


class Client:
    def __init__(self, queue=None, gate=None):
        self.queue = queue
        self.check = gate
        self.runs = []

    def pages(self, endpoint, key=None):
        if "actions/workflows/pr.yml/runs" in endpoint:
            if "status=queued" in endpoint:
                return iter(run for run in self.runs if run["status"] == "queued")
            return iter(self.runs)
        if endpoint.startswith("search/issues?"):
            return iter([{"number": 42}])
        if endpoint.endswith("/reviews"):
            return iter(
                [
                    {
                        "state": "APPROVED",
                        "user": {"login": "a"},
                        "submitted_at": "2026-11-12T17:40:00Z",
                    },
                    {
                        "state": "APPROVED",
                        "user": {"login": "b"},
                        "submitted_at": "2026-11-12T17:41:00Z",
                    },
                ]
            )
        if endpoint.endswith("/comments"):
            if self.queue is None:
                return iter([])
            return iter(
                [
                    {
                        "user": {"login": "mergify[bot]"},
                        "body": (
                            "-*- Mergify Payload -*-\n"
                            + json.dumps(self.queue)
                            + "\n-*- Mergify Payload End -*-"
                        ),
                    }
                ]
            )
        raise AssertionError(endpoint)

    def get(self, endpoint):
        if "/actions/runs/" in endpoint and "/jobs?" in endpoint:
            return {"total_count": 0 if "/101/" in endpoint else 1}
        if endpoint.endswith("/pulls/42"):
            return {
                "state": "open",
                "draft": False,
                "head": {"sha": HEAD},
                "html_url": "https://github.com/example/router/pull/42",
            }
        if "/check-runs?" in endpoint:
            return {"check_runs": [self.check] if self.check else []}
        raise AssertionError(endpoint)


class PRLatencyMonitorTest(unittest.TestCase):
    def test_zero_job_run_is_distinct_from_runner_wait(self):
        client = Client()
        client.runs = [
            {
                "id": 101,
                "status": "queued",
                "event": "pull_request",
                "created_at": "2026-11-12T17:45:00Z",
                "head_sha": HEAD,
                "head_branch": "one",
                "head_repository": {"full_name": "contributor/router"},
                "html_url": "https://github.com/example/router/actions/runs/101",
            },
            {
                "id": 102,
                "status": "queued",
                "event": "pull_request",
                "created_at": "2026-11-12T17:45:00Z",
                "head_sha": HEAD,
                "head_branch": "two",
                "head_repository": {"full_name": "contributor/router"},
                "html_url": "https://github.com/example/router/actions/runs/102",
            },
        ]
        alerts = alerts_for_queued_runs(client, REPO, NOW)
        self.assertEqual(list(alerts), ["zero-jobs:101"])

    def test_newer_run_on_same_head_supersedes_stuck_run(self):
        client = Client()
        client.runs = [
            {
                "id": 101,
                "status": "queued",
                "event": "pull_request",
                "created_at": "2026-11-12T17:30:00Z",
                "head_sha": HEAD,
                "head_branch": "one",
                "head_repository": {"full_name": "contributor/router"},
                "html_url": "https://github.com/example/router/actions/runs/101",
            },
            {
                "id": 103,
                "status": "completed",
                "event": "pull_request",
                "created_at": "2026-11-12T17:40:00Z",
                "head_sha": HEAD,
                "head_branch": "one",
                "head_repository": {"full_name": "contributor/router"},
            },
        ]
        self.assertEqual(alerts_for_queued_runs(client, REPO, NOW), {})

    def test_green_gate_without_queue_entry_alerts(self):
        client = Client(
            gate={
                "id": 1,
                "name": "PR Gate",
                "status": "completed",
                "conclusion": "success",
                "completed_at": "2026-11-12T17:42:00Z",
            }
        )
        alerts = alerts_for_approved_prs(client, REPO, NOW)
        self.assertEqual(list(alerts), [f"queue-admission:42:{HEAD}"])

    def test_expected_committer_approval_does_not_alert_as_missing_gate(self):
        client = Client(
            gate={
                "id": 1,
                "name": "PR Gate",
                "status": "in_progress",
                "output": {"title": "Waiting for committer approval"},
            }
        )
        self.assertEqual(alerts_for_approved_prs(client, REPO, NOW), {})

    def test_active_gate_does_not_alert_as_missing(self):
        client = Client(
            gate={
                "id": 1,
                "name": "PR Gate",
                "status": "in_progress",
                "output": {"title": "Checks running"},
            }
        )
        self.assertEqual(alerts_for_approved_prs(client, REPO, NOW), {})

    def test_missing_gate_alerts_after_approval_threshold(self):
        self.assertEqual(
            list(alerts_for_approved_prs(Client(), REPO, NOW)),
            [f"gate-missing:42:{HEAD}"],
        )

    def test_queue_age_alert_and_deduplicated_recovery(self):
        client = Client(
            queue={
                "state": "queued",
                "queued_at": "2026-11-12T16:20:00Z",
                "speculative_check_pr": 50,
            },
            gate={
                "id": 1,
                "name": "PR Gate",
                "status": "completed",
                "conclusion": "success",
                "completed_at": "2026-11-12T17:42:00Z",
            },
        )
        alerts = alerts_for_approved_prs(client, REPO, NOW)
        self.assertEqual(list(alerts), [f"queue-age:42:{HEAD}"])
        self.assertEqual(reconcile(alerts, alerts), [])
        self.assertIn("recovered incidents", reconcile(alerts, {})[0])

    def test_old_dequeue_is_not_reported_as_current_queue_wait(self):
        client = Client(
            queue={
                "state": "dequeued",
                "queued_at": "2026-11-07T10:48:00Z",
                "left_at": "2026-11-07T10:50:00Z",
                "dequeue_reason": {"code": "INCOMPATIBILITY_WITH_BRANCH_PROTECTIONS"},
            },
            gate={
                "id": 1,
                "name": "PR Gate",
                "status": "completed",
                "conclusion": "success",
                "completed_at": "2026-11-12T17:42:00Z",
            },
        )
        self.assertEqual(alerts_for_approved_prs(client, REPO, NOW), {})


if __name__ == "__main__":
    unittest.main()
