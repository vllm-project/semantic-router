"""Tests for repository-wide issue acceptance reconciliation."""

import contextlib
import importlib.util
import io
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


def labels(*names: str) -> list[dict[str, str]]:
    return [{"name": name} for name in names]


class FakeAcceptanceApi:
    """Serve acceptance endpoints and record every request and label write."""

    QUEUE_PAGE_SIZE = 50
    THREAD_PAGE_SIZE = 100

    def __init__(
        self,
        issues,
        comments,
        permissions,
        *,
        failing_requests=None,
        current_comments=None,
    ) -> None:
        self.issues = {int(issue["number"]): issue for issue in issues}
        self.comments = comments
        self.permissions = permissions
        self.failing_requests = set(failing_requests or ())
        self.current_comments = current_comments or {}
        self.requests: list[tuple[str, str]] = []
        self.fetched: list[int] = []
        self.added: dict[int, set[str]] = {}
        self.removed: dict[int, set[str]] = {}

    @property
    def graphql_requests(self) -> int:
        return sum(1 for _, endpoint in self.requests if endpoint == "graphql")

    def graphql(self, variables):
        if "number" in variables:
            thread = self.comments.get(variables["number"], [])
            start = int(variables["cursor"] or 0)
            end = start + self.THREAD_PAGE_SIZE
            page = {
                "pageInfo": {"hasNextPage": end < len(thread), "endCursor": str(end)},
                "nodes": thread[start:end],
            }
            return {"data": {"repository": {"issue": {"comments": page}}}}

        queue = sorted(
            number
            for number, issue in self.issues.items()
            if variables["label"] in {label["name"] for label in issue["labels"]}
        )
        start = int(variables["cursor"] or 0)
        end = start + self.QUEUE_PAGE_SIZE
        nodes = []
        for number in queue[start:end]:
            thread = self.comments.get(number, [])
            nodes.append(
                {
                    "number": number,
                    "comments": {
                        "totalCount": len(thread),
                        "nodes": thread[-self.THREAD_PAGE_SIZE :],
                    },
                }
            )
        page = {
            "pageInfo": {"hasNextPage": end < len(queue), "endCursor": str(end)},
            "nodes": nodes,
        }
        return {"data": {"repository": {"issues": page}}}

    def request(
        self,
        endpoint: str,
        *,
        method: str = "GET",
        payload=None,
        ignore_not_found: bool = False,
    ):
        self.requests.append((method, endpoint))
        if (method, endpoint) in self.failing_requests:
            raise RuntimeError(f"API failure for {method} {endpoint}")
        if endpoint == "graphql":
            return self.graphql(payload["variables"])

        parts = urlparse(endpoint).path.split("/")
        if parts[3] == "collaborators":
            return {"permission": self.permissions.get(unquote(parts[4]), "read")}
        if parts[4] == "comments":
            comment_id = int(parts[5])
            if comment_id in self.current_comments:
                return self.current_comments[comment_id]
            return {
                "id": comment_id,
                "body": "/accept",
                "user": {"login": "triggering-user"},
            }

        number = int(parts[4])
        tail = parts[5:]
        if not tail:
            self.fetched.append(number)
            return self.issues.get(number)
        if tail == ["labels"]:
            self.added.setdefault(number, set()).update(payload["labels"])
            return None
        if tail[0] == "labels":
            self.removed.setdefault(number, set()).add(unquote(tail[1]))
        return None


class AcceptanceReconciliationTests(unittest.TestCase):
    REPO = "acme/router"
    COMMENT_ID = 7001

    def setUp(self) -> None:
        repository_environment = mock.patch.dict(
            os.environ,
            {"GITHUB_REPOSITORY": self.REPO},
        )
        repository_environment.start()
        self.addCleanup(repository_environment.stop)

    def issue(self, number: int, **overrides):
        issue = {
            "number": number,
            "title": "[Feature] Improve context routing",
            "labels": labels("needs-acceptance", "wg/agentic-context"),
        }
        issue.update(overrides)
        return issue

    def comment(self, actor: str, body: str = "/accept"):
        return {"body": body, "author": {"login": actor}}

    def rest_comment(self, actor: str, body: str = "/accept"):
        return {"id": self.COMMENT_ID, "body": body, "user": {"login": actor}}

    def event(self, number: int, actor: str = "triggering-user", body: str = "/accept"):
        return {
            "repository": {"full_name": self.REPO},
            "issue": {"number": number},
            "comment": {
                "id": self.COMMENT_ID,
                "body": body,
                "user": {"login": actor},
            },
            "sender": {"login": actor},
        }

    def test_one_event_reads_the_pending_queue_and_repairs_every_stale_issue(
        self,
    ) -> None:
        client = FakeAcceptanceApi(
            [
                self.issue(1),
                self.issue(2),
                self.issue(3, labels=labels("accepted", "wg/agentic-context")),
                self.issue(4),
                self.issue(5),
            ],
            {
                1: [
                    self.comment("observer", "not /accept"),
                    self.comment("maintainer"),
                ],
                2: [
                    self.comment("maintainer"),
                    self.comment("another-maintainer"),
                ],
                3: [self.comment("maintainer")],
                4: [self.comment("maintainer", "/accept ")],
            },
            {"maintainer": "write", "another-maintainer": "maintain"},
        )

        community_lifecycle_github.accept_issue_event(client, self.event(1))

        self.assertEqual(client.graphql_requests, 1)
        self.assertEqual(client.fetched, [1, 2])
        self.assertEqual(client.added, {1: {"accepted"}, 2: {"accepted"}})
        self.assertEqual(
            client.removed,
            {1: {"needs-acceptance"}, 2: {"needs-acceptance"}},
        )

    def test_commented_issue_without_the_pending_label_is_accepted_with_the_queue(
        self,
    ) -> None:
        client = FakeAcceptanceApi(
            [self.issue(30, labels=labels("wg/agentic-context")), self.issue(31)],
            {30: [self.comment("maintainer")], 31: [self.comment("maintainer")]},
            {"maintainer": "write"},
            current_comments={self.COMMENT_ID: self.rest_comment("maintainer")},
        )
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            community_lifecycle_github.accept_issue_event(
                client, self.event(30, actor="maintainer")
            )

        self.assertEqual(client.graphql_requests, 1)
        self.assertEqual(client.fetched, [30, 31])
        self.assertEqual(client.added, {30: {"accepted"}, 31: {"accepted"}})
        self.assertIn("Issue #30 accepted", output.getvalue())

    def test_commented_issue_without_the_pending_label_keeps_every_check(
        self,
    ) -> None:
        cases = {
            "commenter cannot manage": (
                self.issue(40, labels=labels("wg/agentic-context")),
                "outsider",
                "requires repository write",
            ),
            "no owner": (
                self.issue(41, labels=labels()),
                "maintainer",
                "requires exactly one recognized owner",
            ),
            "two owners": (
                self.issue(
                    42, labels=labels("wg/agentic-context", "owner/maintainers")
                ),
                "maintainer",
                "requires exactly one recognized owner",
            ),
            "title without a category": (
                self.issue(
                    43,
                    title="Improve context routing",
                    labels=labels("wg/agentic-context"),
                ),
                "maintainer",
                "Title must begin",
            ),
        }
        for name, (issue, actor, expected) in cases.items():
            with self.subTest(name):
                client = FakeAcceptanceApi(
                    [issue],
                    {},
                    {"outsider": "read", "maintainer": "write"},
                    current_comments={self.COMMENT_ID: self.rest_comment(actor)},
                )
                output = io.StringIO()

                with contextlib.redirect_stdout(output):
                    community_lifecycle_github.accept_issue_event(
                        client, self.event(issue["number"], actor=actor)
                    )

                self.assertEqual(client.added, {})
                self.assertEqual(client.removed, {})
                self.assertIn(expected, output.getvalue())

    def test_withdrawn_comment_does_not_accept_its_issue_and_the_queue_still_runs(
        self,
    ) -> None:
        pending = labels("needs-acceptance", "wg/agentic-context")
        lost = labels("wg/agentic-context")
        cases = {
            "deleted, label missing": (lost, None),
            "edited, label missing": (lost, "/accept please"),
            "deleted, label present": (pending, None),
            "edited, label present": (pending, "/accept please"),
        }
        for name, (issue_labels, edited_body) in cases.items():
            with self.subTest(name):
                current = (
                    None
                    if edited_body is None
                    else self.rest_comment("maintainer", edited_body)
                )
                thread = (
                    []
                    if edited_body is None
                    else [self.comment("maintainer", edited_body)]
                )
                client = FakeAcceptanceApi(
                    [self.issue(50, labels=issue_labels), self.issue(51)],
                    {50: thread, 51: [self.comment("maintainer")]},
                    {"maintainer": "write"},
                    current_comments={self.COMMENT_ID: current},
                )
                output = io.StringIO()

                with contextlib.redirect_stdout(output):
                    community_lifecycle_github.accept_issue_event(
                        client, self.event(50, actor="maintainer")
                    )

                self.assertEqual(client.fetched, [51])
                self.assertEqual(client.added, {51: {"accepted"}})
                self.assertIn("edited or deleted", output.getvalue())

    def test_unreadable_comment_leaves_its_issue_alone_and_fails_the_job(self) -> None:
        comment_endpoint = f"repos/{self.REPO}/issues/comments/{self.COMMENT_ID}"
        client = FakeAcceptanceApi(
            [self.issue(60, labels=labels("wg/agentic-context")), self.issue(61)],
            {61: [self.comment("maintainer")]},
            {"maintainer": "write"},
            failing_requests={("GET", comment_endpoint)},
        )
        output = io.StringIO()

        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit):
            community_lifecycle_github.accept_issue_event(
                client, self.event(60, actor="maintainer")
            )

        self.assertEqual(client.fetched, [61])
        self.assertEqual(client.added, {61: {"accepted"}})
        self.assertIn("Triggering comment check failed", output.getvalue())
        self.assertIn("1 accepted", output.getvalue())
        self.assertIn("1 failed", output.getvalue())

    def test_queue_is_paged_until_every_pending_issue_is_read(self) -> None:
        numbers = range(1, 121)
        client = FakeAcceptanceApi(
            [self.issue(number) for number in numbers],
            {number: [self.comment("maintainer")] for number in numbers},
            {"maintainer": "write"},
        )

        with contextlib.redirect_stdout(io.StringIO()):
            community_lifecycle_github.accept_issue_event(client, self.event(1))

        self.assertEqual(client.graphql_requests, 3)
        self.assertEqual(client.fetched, list(numbers))
        self.assertEqual(set(client.added), set(numbers))

    def test_long_threads_are_read_in_full_before_matching(self) -> None:
        thread = [self.comment("maintainer")]
        thread.extend(self.comment("observer", f"note {index}") for index in range(129))
        client = FakeAcceptanceApi(
            [self.issue(7)],
            {7: thread},
            {"maintainer": "write"},
        )

        community_lifecycle_github.accept_issue_event(client, self.event(7))

        self.assertEqual(client.graphql_requests, 3)
        self.assertEqual(client.added, {7: {"accepted"}})

    def test_request_count_does_not_grow_with_accepted_history(self) -> None:
        history = range(100, 500)
        client = FakeAcceptanceApi(
            [self.issue(1)]
            + [
                self.issue(number, labels=labels("accepted", "wg/agentic-context"))
                for number in history
            ],
            {1: [self.comment("maintainer")]}
            | {number: [self.comment("maintainer")] for number in history},
            {"maintainer": "write"},
        )

        community_lifecycle_github.accept_issue_event(client, self.event(1))

        self.assertEqual(
            [endpoint for _, endpoint in client.requests if endpoint != "graphql"],
            [
                f"repos/{self.REPO}/issues/comments/{self.COMMENT_ID}",
                f"repos/{self.REPO}/issues/1",
                f"repos/{self.REPO}/collaborators/maintainer/permission",
                f"repos/{self.REPO}/issues/1/labels",
                f"repos/{self.REPO}/issues/1/labels/needs-acceptance",
            ],
        )
        self.assertEqual(client.graphql_requests, 1)

    def test_invalid_candidates_remain_unchanged_with_actionable_output(self) -> None:
        client = FakeAcceptanceApi(
            [
                self.issue(10),
                self.issue(
                    11,
                    labels=labels(
                        "needs-acceptance", "wg/agentic-context", "owner/maintainers"
                    ),
                ),
                self.issue(12, title="Improve context routing"),
            ],
            {
                10: [self.comment("outsider")],
                11: [self.comment("maintainer")],
                12: [self.comment("maintainer")],
            },
            {"outsider": "read", "maintainer": "admin"},
        )
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            community_lifecycle_github.accept_issue_event(client, self.event(10))

        self.assertEqual(client.added, {})
        self.assertEqual(client.removed, {})
        self.assertIn("Issue #10 not accepted", output.getvalue())
        self.assertIn("requires repository write", output.getvalue())
        self.assertIn("Issue #11 not accepted", output.getvalue())
        self.assertIn("requires exactly one recognized owner", output.getvalue())
        self.assertIn("Issue #12 not accepted", output.getvalue())
        self.assertIn("Title must begin", output.getvalue())

    def test_api_failure_is_reported_after_other_candidates_are_processed(self) -> None:
        failing_endpoint = f"repos/{self.REPO}/issues/20/labels"
        client = FakeAcceptanceApi(
            [self.issue(20), self.issue(21)],
            {20: [self.comment("maintainer")], 21: [self.comment("maintainer")]},
            {"maintainer": "write"},
            failing_requests={("POST", failing_endpoint)},
        )
        output = io.StringIO()

        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit):
            community_lifecycle_github.accept_issue_event(client, self.event(20))

        self.assertNotIn(20, client.removed)
        self.assertEqual(client.added[21], {"accepted"})
        self.assertEqual(client.removed[21], {"needs-acceptance"})
        self.assertIn("Issue #20 reconciliation failed", output.getvalue())
        self.assertIn("1 accepted", output.getvalue())
        self.assertIn("1 failed", output.getvalue())

    def test_trigger_command_must_also_be_exact(self) -> None:
        client = FakeAcceptanceApi([], {}, {})

        with self.assertRaises(SystemExit):
            community_lifecycle_github.accept_issue_event(
                client, self.event(1, body="/accept\n")
            )

        self.assertEqual(client.requests, [])


if __name__ == "__main__":
    unittest.main()
