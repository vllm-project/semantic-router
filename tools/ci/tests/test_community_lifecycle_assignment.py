"""Tests for reconciling issue delivery state after assignment commands."""

import copy
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import unquote, urlparse

SCRIPT = Path(__file__).resolve().parents[1] / "community_lifecycle.py"
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("community_lifecycle", SCRIPT)
assert SPEC and SPEC.loader
community_lifecycle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = community_lifecycle
SPEC.loader.exec_module(community_lifecycle)
community_lifecycle_github = sys.modules["community_lifecycle_github"]

REPO = "acme/router"


def labels(*names: str) -> list[dict[str, str]]:
    return [{"name": name} for name in names]


def assignees(*logins: str) -> list[dict[str, str]]:
    return [{"login": login} for login in logins]


def issue(number: int = 7, **overrides):
    value = {
        "number": number,
        "state": "open",
        "title": "[Feature] Improve context routing",
        "body": "",
        "labels": labels("accepted", "wg/mom-routing"),
        "assignees": assignees(),
        "milestone": None,
    }
    value.update(overrides)
    return value


class FakeAssignmentApi:
    """Serve the issue endpoints a command reconciliation calls.

    Label and assignee writes change the stored issue, so assertions can check
    the final delivery state rather than the request count.
    """

    def __init__(self, issues, *, permissions=None) -> None:
        self.issues = {int(item["number"]): copy.deepcopy(item) for item in issues}
        self.permissions = permissions or {}
        self.requests: list[tuple[str, str]] = []
        self.fetched: list[int] = []
        self.added: dict[int, set[str]] = {}
        self.removed: dict[int, set[str]] = {}
        self.removed_assignees: dict[int, set[str]] = {}
        self.comments: dict[int, list[str]] = {}

    def request(
        self,
        endpoint: str,
        *,
        method: str = "GET",
        payload=None,
        ignore_not_found: bool = False,
    ):
        self.requests.append((method, endpoint))
        parsed = urlparse(endpoint)
        parts = parsed.path.split("/")
        if method == "GET" and parts[3] == "collaborators":
            return {"permission": self.permissions.get(unquote(parts[4]), "read")}

        number = int(parts[4])
        item = self.issues[number]
        tail = parts[5:]
        if not tail:
            self.fetched.append(number)
            return copy.deepcopy(item)
        if tail == ["labels"] and method == "POST":
            current = {label["name"] for label in item["labels"]}
            for name in payload["labels"]:
                if name not in current:
                    item["labels"].append({"name": name})
                    current.add(name)
                    self.added.setdefault(number, set()).add(name)
            return None
        if tail[0] == "labels" and method == "DELETE":
            name = unquote(tail[1])
            remaining = [label for label in item["labels"] if label["name"] != name]
            if len(remaining) != len(item["labels"]):
                item["labels"] = remaining
                self.removed.setdefault(number, set()).add(name)
            return None
        if tail == ["assignees"] and method == "DELETE":
            removed = set(payload["assignees"])
            item["assignees"] = [
                assignee
                for assignee in item["assignees"]
                if assignee["login"] not in removed
            ]
            self.removed_assignees.setdefault(number, set()).update(removed)
            return None
        if tail == ["comments"]:
            if method == "GET":
                return [{"body": body} for body in self.comments.get(number, [])]
            if method == "POST":
                self.comments.setdefault(number, []).append(payload["body"])
                return {"id": len(self.comments[number])}
        return None

    def label_set(self, number: int = 7) -> set[str]:
        return {label["name"] for label in self.issues[number]["labels"]}

    def assignee_set(self, number: int = 7) -> set[str]:
        return {assignee["login"] for assignee in self.issues[number]["assignees"]}


class AssignmentCommandParsingTests(unittest.TestCase):
    def test_only_assignment_commands_at_the_start_of_a_comment_count(self) -> None:
        cases = {
            "/assign": "/assign",
            "/assign @contributor": "/assign",
            " /unassign me\n": "/unassign",
            "/unassign": "/unassign",
            "/cc\n/assign": "/assign",
            "/cc\n/unassign @contributor": "/unassign",
            "  /assign\n/cc": "/assign",
            "/assignment": None,
            "please /assign me": None,
            "/cc\nplease /assign": None,
            "": None,
        }
        for body, expected in cases.items():
            with self.subTest(body=body):
                self.assertEqual(community_lifecycle.assignment_command(body), expected)


class AssignmentCommandReconciliationTests(unittest.TestCase):
    def setUp(self) -> None:
        repository_environment = mock.patch.dict(
            os.environ,
            {"GITHUB_REPOSITORY": REPO},
        )
        repository_environment.start()
        self.addCleanup(repository_environment.stop)

    def event(self, body: str = "/assign", *, number: int = 7, pull_request=None):
        issue_payload = {"number": number}
        if pull_request is not None:
            issue_payload["pull_request"] = pull_request
        return {
            "repository": {"full_name": REPO},
            "issue": issue_payload,
            "comment": {"body": body},
            "sender": {"login": "contributor"},
        }

    def sync(self, client, body: str = "/assign", **kwargs) -> None:
        community_lifecycle_github.sync_issue_assignment_event(
            client, self.event(body, **kwargs)
        )

    def test_assign_moves_accepted_work_to_in_progress(self) -> None:
        client = FakeAssignmentApi(
            [
                issue(
                    labels=labels(
                        "accepted",
                        "ready-for-dev",
                        "help wanted",
                        "good first issue",
                        "wg/mom-routing",
                    ),
                    assignees=assignees("contributor"),
                )
            ]
        )

        self.sync(client)

        self.assertEqual(
            client.label_set(), {"accepted", "wg/mom-routing", "in-progress"}
        )
        self.assertEqual(client.assignee_set(), {"contributor"})
        self.assertEqual(client.fetched, [7])

    def test_assign_on_a_later_comment_line_is_reconciled(self) -> None:
        client = FakeAssignmentApi(
            [
                issue(
                    labels=labels("accepted", "ready-for-dev", "wg/mom-routing"),
                    assignees=assignees("contributor"),
                )
            ]
        )

        self.sync(client, "/cc\n/assign")

        self.assertEqual(
            client.label_set(), {"accepted", "wg/mom-routing", "in-progress"}
        )

    def test_unassign_on_a_later_comment_line_is_reconciled(self) -> None:
        client = FakeAssignmentApi(
            [issue(labels=labels("accepted", "in-progress", "wg/mom-routing"))]
        )

        self.sync(client, "/cc\n/unassign")

        self.assertEqual(client.label_set(), {"accepted", "wg/mom-routing"})

    def test_stale_event_payload_does_not_decide_the_delivery_state(self) -> None:
        client = FakeAssignmentApi(
            [
                issue(
                    labels=labels("accepted", "ready-for-dev", "wg/mom-routing"),
                    assignees=assignees("contributor"),
                )
            ]
        )
        event = self.event()
        # The issue_comment payload predates Prow's assignment write.
        event["issue"]["assignees"] = []

        community_lifecycle_github.sync_issue_assignment_event(client, event)

        self.assertEqual(
            client.label_set(), {"accepted", "wg/mom-routing", "in-progress"}
        )

    def test_unassign_clears_in_progress_after_the_last_assignee(self) -> None:
        client = FakeAssignmentApi(
            [issue(labels=labels("accepted", "in-progress", "wg/mom-routing"))]
        )

        self.sync(client, "/unassign")

        self.assertEqual(client.label_set(), {"accepted", "wg/mom-routing"})
        self.assertNotIn("ready-for-dev", client.label_set())

    def test_unassign_keeps_in_progress_while_an_assignee_remains(self) -> None:
        client = FakeAssignmentApi(
            [
                issue(
                    labels=labels("accepted", "in-progress", "wg/mom-routing"),
                    assignees=assignees("alice"),
                )
            ]
        )

        self.sync(client, "/unassign")

        self.assertEqual(
            client.label_set(), {"accepted", "in-progress", "wg/mom-routing"}
        )
        self.assertEqual(client.assignee_set(), {"alice"})

    def test_rejected_assign_leaves_ready_for_dev_unchanged(self) -> None:
        client = FakeAssignmentApi(
            [issue(labels=labels("accepted", "ready-for-dev", "wg/mom-routing"))]
        )

        self.sync(client)

        self.assertEqual(
            client.label_set(), {"accepted", "ready-for-dev", "wg/mom-routing"}
        )
        self.assertNotIn("in-progress", client.label_set())
        self.assertEqual(client.added, {})
        self.assertEqual(client.removed, {})

    def test_repeated_assign_is_idempotent(self) -> None:
        client = FakeAssignmentApi(
            [
                issue(
                    labels=labels("accepted", "in-progress", "wg/mom-routing"),
                    assignees=assignees("contributor"),
                )
            ]
        )

        self.sync(client)

        self.assertEqual(
            client.label_set(), {"accepted", "in-progress", "wg/mom-routing"}
        )
        self.assertEqual(client.added, {})
        self.assertEqual(client.removed, {})
        self.assertEqual(client.removed_assignees, {})

    def test_repeated_unassign_is_idempotent(self) -> None:
        client = FakeAssignmentApi([issue(labels=labels("accepted", "wg/mom-routing"))])

        self.sync(client, "/unassign")

        self.assertEqual(client.label_set(), {"accepted", "wg/mom-routing"})
        self.assertEqual(client.added, {})
        self.assertEqual(client.removed, {})

    def test_serialized_assign_then_unassign_leaves_no_stale_in_progress(self) -> None:
        client = FakeAssignmentApi(
            [issue(labels=labels("accepted", "ready-for-dev", "wg/mom-routing"))]
        )

        # The /assign run reads the assignee Prow just wrote.
        client.issues[7]["assignees"] = assignees("contributor")
        self.sync(client, "/assign")
        self.assertIn("in-progress", client.label_set())

        # The queued /unassign run reads the issue after the removal.
        client.issues[7]["assignees"] = assignees()
        self.sync(client, "/unassign")

        self.assertEqual(client.label_set(), {"accepted", "wg/mom-routing"})
        self.assertEqual(client.assignee_set(), set())

    def test_serialized_unassign_then_assign_keeps_the_new_assignee_in_progress(
        self,
    ) -> None:
        client = FakeAssignmentApi(
            [
                issue(
                    labels=labels("accepted", "in-progress", "wg/mom-routing"),
                    assignees=assignees("contributor"),
                )
            ]
        )

        client.issues[7]["assignees"] = assignees()
        self.sync(client, "/unassign")
        self.assertNotIn("in-progress", client.label_set())

        client.issues[7]["assignees"] = assignees("other")
        self.sync(client, "/assign")

        self.assertEqual(
            client.label_set(), {"accepted", "wg/mom-routing", "in-progress"}
        )
        self.assertEqual(client.assignee_set(), {"other"})

    def test_assignment_before_acceptance_is_removed_without_re_seeding_the_form(
        self,
    ) -> None:
        client = FakeAssignmentApi(
            [
                issue(
                    body="### Proposed Workgroup\n\nEnterprise & Environment\n",
                    labels=labels("needs-acceptance"),
                    assignees=assignees("contributor"),
                )
            ]
        )

        self.sync(client)

        self.assertEqual(client.label_set(), {"needs-acceptance"})
        self.assertEqual(client.assignee_set(), set())
        self.assertEqual(client.removed_assignees, {7: {"contributor"}})
        self.assertNotIn("wg/enterprise-environment", client.added.get(7, set()))

    def test_manual_assignment_keeps_the_issues_event_sync(self) -> None:
        client = FakeAssignmentApi(
            [
                issue(
                    labels=labels("accepted", "ready-for-dev", "wg/mom-routing"),
                    assignees=assignees("contributor"),
                )
            ],
            permissions={"maintainer": "write"},
        )

        community_lifecycle_github.sync_issue_event(
            client,
            {
                "repository": {"full_name": REPO},
                "issue": {"number": 7},
                "action": "assigned",
                "sender": {"login": "maintainer"},
            },
        )

        self.assertEqual(
            client.label_set(), {"accepted", "wg/mom-routing", "in-progress"}
        )
        self.assertEqual(client.fetched, [7])

    def test_manual_assignment_then_unassign_of_another_user_keeps_in_progress(
        self,
    ) -> None:
        client = FakeAssignmentApi(
            [
                issue(
                    labels=labels("accepted", "in-progress", "wg/mom-routing"),
                    assignees=assignees("alice", "bob"),
                )
            ],
            permissions={"maintainer": "write"},
        )

        # The manual assignment sync runs first, then the queued /unassign.
        community_lifecycle_github.sync_issue_event(
            client,
            {
                "repository": {"full_name": REPO},
                "issue": {"number": 7},
                "action": "assigned",
                "sender": {"login": "maintainer"},
            },
        )
        self.assertIn("in-progress", client.label_set())

        client.issues[7]["assignees"] = assignees("bob")
        self.sync(client, "/cc\n/unassign")

        self.assertEqual(client.assignee_set(), {"bob"})
        self.assertIn("in-progress", client.label_set())

    def test_pull_request_comments_are_not_issue_claims(self) -> None:
        client = FakeAssignmentApi([issue(labels=labels("accepted", "wg/mom-routing"))])

        self.sync(client, pull_request={})

        self.assertEqual(client.requests, [])
        self.assertEqual(client.label_set(), {"accepted", "wg/mom-routing"})

    def test_a_fetched_pull_request_is_left_alone(self) -> None:
        client = FakeAssignmentApi(
            [
                issue(
                    labels=labels("accepted", "ready-for-dev", "wg/mom-routing"),
                    assignees=assignees("contributor"),
                    pull_request={},
                )
            ]
        )

        self.sync(client)

        self.assertEqual(client.removed, {})
        self.assertEqual(client.added, {})

    def test_other_comments_do_not_reconcile(self) -> None:
        client = FakeAssignmentApi(
            [issue(labels=labels("accepted", "ready-for-dev", "wg/mom-routing"))]
        )

        self.sync(client, "/assignment")

        self.assertEqual(client.requests, [])


if __name__ == "__main__":
    unittest.main()
