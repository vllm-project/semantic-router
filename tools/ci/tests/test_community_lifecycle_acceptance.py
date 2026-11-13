"""Tests for issue acceptance: the per-comment path and the scheduled backfill."""

import contextlib
import importlib.util
import io
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import parse_qs, unquote, urlparse

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


def comment(actor: str, body: str = "/accept") -> dict:
    return {"body": body, "user": {"login": actor}}


def queued_comment(actor: str, body: str = "/accept", kind: str = "User") -> dict:
    return {"body": body, "author": {"__typename": kind, "login": actor}}


def cross_reference(number: int, *, state="open", pull_request=True, repo=REPO):
    source = {
        "number": number,
        "state": state,
        "repository_url": f"https://api.github.com/repos/{repo}",
    }
    if pull_request:
        source["pull_request"] = {"url": "example"}
    return {"event": "cross-referenced", "source": {"type": "issue", "issue": source}}


class FakeAcceptanceApi:
    """Serve acceptance endpoints, record every request, and spend a budget.

    ``queue`` maps each queued issue, newest first, to its comments as GraphQL
    returns them. Like the real client, GraphQL requests leave the REST budget
    untouched.
    """

    def __init__(
        self,
        issues,
        *,
        comments=None,
        timelines=None,
        queue=None,
        permissions=None,
        failing_requests=None,
        budget=None,
    ) -> None:
        self.issues = {int(issue["number"]): issue for issue in issues}
        self.comments = comments or {}
        self.timelines = timelines or {}
        self.queue = queue or {}
        self.permissions = permissions or {}
        self.failing_requests = set(failing_requests or ())
        self.rate_limit_remaining = budget
        self.requests: list[tuple[str, str]] = []
        self.queries: list[dict] = []
        self.fetched: list[int] = []
        self.added: dict[int, set[str]] = {}
        self.removed: dict[int, set[str]] = {}

    def queue_page(self, variables: dict) -> dict:
        self.queries.append(variables)
        start = int(variables["cursor"] or 0)
        numbers = list(self.queue)[start : start + 50]
        end = start + len(numbers)
        nodes = [
            {
                "number": number,
                "comments": {
                    "totalCount": len(self.queue[number]),
                    "nodes": self.queue[number][-100:],
                },
            }
            for number in numbers
        ]
        page_info = {"hasNextPage": end < len(self.queue), "endCursor": str(end)}
        return {
            "data": {"repository": {"issues": {"pageInfo": page_info, "nodes": nodes}}}
        }

    def request(
        self,
        endpoint: str,
        *,
        method: str = "GET",
        payload=None,
        ignore_not_found: bool = False,
    ):
        self.requests.append((method, endpoint))
        if self.rate_limit_remaining is not None and endpoint != "graphql":
            self.rate_limit_remaining -= 1
        if (method, endpoint) in self.failing_requests:
            raise RuntimeError(f"API failure for {method} {endpoint}")
        if endpoint == "graphql":
            return self.queue_page(payload["variables"])

        parsed = urlparse(endpoint)
        query = parse_qs(parsed.query)
        parts = parsed.path.split("/")
        if parts[3] == "collaborators":
            return {"permission": self.permissions.get(unquote(parts[4]), "read")}

        number = int(parts[4])
        tail = parts[5:]
        if not tail:
            self.fetched.append(number)
            return self.issues.get(number)
        if tail == ["comments"]:
            pages = self.comments.get(number, {})
            return pages.get(int(query["page"][0]), [])
        if tail == ["timeline"]:
            pages = self.timelines.get(number, {})
            return pages.get(int(query["page"][0]), [])
        if tail == ["labels"]:
            self.added.setdefault(number, set()).update(payload["labels"])
            return None
        if tail[0] == "labels":
            self.removed.setdefault(number, set()).add(unquote(tail[1]))
        return None

    def endpoints(self) -> list[str]:
        return [urlparse(endpoint).path for _, endpoint in self.requests]


def issue(number: int, **overrides):
    value = {
        "number": number,
        "title": "[Feature] Improve context routing",
        "labels": labels("needs-acceptance", "wg/agentic-context"),
        "comments": 1,
    }
    value.update(overrides)
    return value


class AcceptanceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        repository_environment = mock.patch.dict(
            os.environ,
            {"GITHUB_REPOSITORY": REPO},
        )
        repository_environment.start()
        self.addCleanup(repository_environment.stop)
        synced = mock.patch.object(community_lifecycle_github, "sync_pull_request")
        self.sync_pull_request = synced.start()
        self.addCleanup(synced.stop)

    def synced_pull_requests(self) -> list[int]:
        return [call.args[2] for call in self.sync_pull_request.call_args_list]


class CommentAcceptanceTests(AcceptanceTestCase):
    def event(self, number: int = 7, body: str = "/accept", actor="maintainer"):
        return {
            "repository": {"full_name": REPO},
            "issue": {"number": number},
            "comment": {"body": body, "user": {"login": actor}},
            "sender": {"login": actor},
        }

    def test_comment_reconciles_only_its_issue_and_linked_open_pull_requests(
        self,
    ) -> None:
        client = FakeAcceptanceApi(
            [issue(7), issue(8)],
            comments={
                7: {1: [comment("observer", "looks good"), comment("maintainer")]}
            },
            timelines={
                7: {
                    1: [
                        cross_reference(70),
                        cross_reference(71, state="closed"),
                        cross_reference(72, repo="someone/fork"),
                        cross_reference(73, pull_request=False),
                        {"event": "labeled"},
                    ]
                }
            },
            permissions={"maintainer": "write"},
        )

        community_lifecycle_github.accept_issue_event(client, self.event())

        self.assertEqual(client.fetched, [7])
        self.assertEqual(client.added, {7: {"accepted"}})
        self.assertEqual(client.removed, {7: {"needs-acceptance"}})
        self.assertEqual(self.synced_pull_requests(), [70])
        self.assertNotIn(f"/repos/{REPO}/issues/comments", client.endpoints())
        self.assertFalse(any("/issues/8" in path for path in client.endpoints()))
        self.assertLess(len(client.requests), 10)

    def test_earlier_maintainer_decision_counts_when_someone_else_comments(
        self,
    ) -> None:
        client = FakeAcceptanceApi(
            [issue(7)],
            comments={7: {1: [comment("maintainer"), comment("outsider")]}},
            permissions={"maintainer": "maintain", "outsider": "read"},
        )

        community_lifecycle_github.accept_issue_event(
            client, self.event(actor="outsider")
        )

        self.assertEqual(client.added, {7: {"accepted"}})

    def test_withdrawn_command_is_not_counted(self) -> None:
        client = FakeAcceptanceApi(
            [issue(7)],
            comments={7: {1: [comment("maintainer", "/accept after review")]}},
            timelines={7: {1: [cross_reference(70)]}},
            permissions={"maintainer": "write"},
        )
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            community_lifecycle_github.accept_issue_event(client, self.event())

        self.assertEqual(client.added, {})
        self.assertEqual(client.removed, {})
        self.assertEqual(self.synced_pull_requests(), [])
        self.assertEqual(client.fetched, [])
        self.assertIn("No exact `/accept` comment remains", output.getvalue())

    def test_comments_are_paged_for_the_issue(self) -> None:
        page_size = community_lifecycle_github.API_PAGE_SIZE
        client = FakeAcceptanceApi(
            [issue(7)],
            comments={
                7: {
                    1: [comment("observer", "+1")] * page_size,
                    2: [comment("maintainer")],
                }
            },
            permissions={"maintainer": "admin"},
        )

        community_lifecycle_github.accept_issue_event(
            client, self.event(actor="observer")
        )

        self.assertEqual(client.added, {7: {"accepted"}})

    def test_invalid_issue_is_left_unchanged_with_actionable_output(self) -> None:
        cases = {
            "requires repository write": (
                issue(10),
                {"outsider": "read"},
                "outsider",
            ),
            "requires exactly one recognized owner": (
                issue(
                    10,
                    labels=labels(
                        "needs-acceptance", "wg/agentic-context", "owner/maintainers"
                    ),
                ),
                {"maintainer": "admin"},
                "maintainer",
            ),
            "Title must begin": (
                issue(10, title="Improve context routing"),
                {"maintainer": "admin"},
                "maintainer",
            ),
        }
        for message, (candidate, permissions, actor) in cases.items():
            with self.subTest(message):
                self.sync_pull_request.reset_mock()
                client = FakeAcceptanceApi(
                    [candidate],
                    comments={10: {1: [comment(actor)]}},
                    timelines={10: {1: [cross_reference(70)]}},
                    permissions=permissions,
                )
                output = io.StringIO()

                with contextlib.redirect_stdout(output):
                    community_lifecycle_github.accept_issue_event(
                        client, self.event(10, actor=actor)
                    )

                self.assertEqual(client.added, {})
                self.assertEqual(client.removed, {})
                self.assertEqual(self.synced_pull_requests(), [])
                self.assertIn("Issue #10 not accepted", output.getvalue())
                self.assertIn(message, output.getvalue())

    def test_already_accepted_issue_still_refreshes_linked_pull_requests(
        self,
    ) -> None:
        client = FakeAcceptanceApi(
            [issue(7, labels=labels("accepted", "wg/agentic-context"))],
            comments={7: {1: [comment("maintainer")]}},
            timelines={7: {1: [cross_reference(70)]}},
            permissions={"maintainer": "write"},
        )

        community_lifecycle_github.accept_issue_event(client, self.event())

        self.assertEqual(client.added, {})
        self.assertEqual(self.synced_pull_requests(), [70])

    def test_pull_request_failure_is_reported_after_the_others_sync(self) -> None:
        self.sync_pull_request.side_effect = [RuntimeError("API failure"), None]
        client = FakeAcceptanceApi(
            [issue(7)],
            comments={7: {1: [comment("maintainer")]}},
            timelines={7: {1: [cross_reference(70), cross_reference(71)]}},
            permissions={"maintainer": "write"},
        )
        output = io.StringIO()

        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit):
            community_lifecycle_github.accept_issue_event(client, self.event())

        self.assertEqual(self.synced_pull_requests(), [70, 71])
        self.assertEqual(client.added, {7: {"accepted"}})
        self.assertIn("Pull request #70 sync failed", output.getvalue())

    def test_trigger_command_must_also_be_exact(self) -> None:
        client = FakeAcceptanceApi([issue(7)])

        with self.assertRaises(SystemExit):
            community_lifecycle_github.accept_issue_event(
                client, self.event(body="/accept\n")
            )

        self.assertEqual(client.requests, [])


class AcceptanceSweepTests(AcceptanceTestCase):
    def sweep(self, client, *, since=None, reserve=0) -> str:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            community_lifecycle_github.sweep_issue_acceptance(
                client, REPO, since=since, reserve=reserve
            )
        return output.getvalue()

    def test_sweep_reads_the_queue_in_pages_before_writing_and_accepts_valid_ones(
        self,
    ) -> None:
        queue = {number: [] for number in range(100, 153)}
        queue |= {
            1: [queued_comment("maintainer")],
            2: [queued_comment("observer", "/accept please")],
            4: [queued_comment("maintainer")],
        }
        client = FakeAcceptanceApi(
            [issue(1), issue(2), issue(4)],
            queue=queue,
            permissions={"maintainer": "write"},
        )

        output = self.sweep(client)

        self.assertEqual([query["cursor"] for query in client.queries], [None, "50"])
        last_read = max(
            index
            for index, (_, endpoint) in enumerate(client.requests)
            if endpoint == "graphql"
        )
        first_write = min(
            index
            for index, (method, endpoint) in enumerate(client.requests)
            if method != "GET" and endpoint != "graphql"
        )
        self.assertLess(last_read, first_write)
        self.assertEqual(client.added, {1: {"accepted"}, 4: {"accepted"}})
        self.assertEqual(client.fetched, [1, 4])
        self.assertFalse(any("/comments" in path for path in client.endpoints()))
        self.assertIn("56 open issue(s) awaiting acceptance", output)
        self.assertIn("2 accepted", output)
        self.assertIn("54 without `/accept`", output)

    def test_sweep_reads_a_thread_longer_than_the_queue_page_comment_by_comment(
        self,
    ) -> None:
        chatter = [f"comment {index}" for index in range(100)]
        client = FakeAcceptanceApi(
            [issue(7)],
            queue={
                7: [queued_comment("maintainer")]
                + [queued_comment("reporter", body) for body in chatter]
            },
            comments={
                7: {
                    1: [comment("maintainer")]
                    + [comment("reporter", body) for body in chatter[:99]],
                    2: [comment("reporter", chatter[99])],
                }
            },
            permissions={"maintainer": "write"},
        )

        self.sweep(client)

        self.assertEqual(client.added, {7: {"accepted"}})
        self.assertEqual(
            [path for path in client.endpoints() if path.endswith("/comments")],
            [f"repos/{REPO}/issues/7/comments"] * 2,
        )

    def test_sweep_keeps_the_login_rest_reports_for_bot_authors(self) -> None:
        client = FakeAcceptanceApi(
            [issue(1), issue(2)],
            queue={
                1: [queued_comment("github-actions", kind="Bot")],
                2: [queued_comment("github-actions")],
            },
        )

        self.sweep(client)

        self.assertEqual(client.added, {1: {"accepted"}})
        self.assertEqual(
            [path for path in client.endpoints() if "/collaborators/" in path],
            [f"repos/{REPO}/collaborators/github-actions/permission"],
        )

    def test_sweep_stops_at_the_reserve_and_leaves_the_rest_for_the_next_run(
        self,
    ) -> None:
        numbers = range(1, 6)
        client = FakeAcceptanceApi(
            [issue(number) for number in numbers],
            queue={number: [queued_comment("maintainer")] for number in numbers},
            permissions={"maintainer": "write"},
            budget=12,
        )

        output = self.sweep(client, reserve=5)

        self.assertEqual(sorted(client.added), [1, 2])
        self.assertLessEqual(client.rate_limit_remaining, 5)
        self.assertIn("Stopped with", output)
        self.assertIn("3 issue(s) remain for the next run", output)

    def test_sweep_asks_for_open_issues_with_the_label_and_the_lookback_window(
        self,
    ) -> None:
        client = FakeAcceptanceApi([])

        self.sweep(client, since="2026-11-01T00:00:00Z")
        self.sweep(client)

        self.assertEqual(
            [query["since"] for query in client.queries],
            ["2026-11-01T00:00:00Z", None],
        )
        self.assertEqual(
            {query["label"] for query in client.queries}, {"needs-acceptance"}
        )
        query = community_lifecycle_github.ACCEPTANCE_QUEUE_QUERY
        self.assertIn("states: OPEN", query)
        self.assertIn("orderBy: {field: UPDATED_AT, direction: DESC}", query)

    def test_sweep_fails_before_any_write_when_the_queue_cannot_be_read(self) -> None:
        client = FakeAcceptanceApi(
            [issue(1)],
            queue={1: [queued_comment("maintainer")]},
            permissions={"maintainer": "write"},
            failing_requests={("POST", "graphql")},
        )

        with self.assertRaises(RuntimeError):
            self.sweep(client)

        self.assertEqual(client.requests, [("POST", "graphql")])
        self.assertEqual(client.added, {})

    def test_sweep_reports_failures_after_processing_the_rest(self) -> None:
        failing = ("POST", f"repos/{REPO}/issues/1/labels")
        client = FakeAcceptanceApi(
            [issue(1), issue(2)],
            queue={
                1: [queued_comment("maintainer")],
                2: [queued_comment("maintainer")],
            },
            permissions={"maintainer": "write"},
            failing_requests={failing},
        )
        output = io.StringIO()

        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit):
            community_lifecycle_github.sweep_issue_acceptance(
                client, REPO, since=None, reserve=0
            )

        self.assertEqual(client.added, {2: {"accepted"}})
        self.assertIn("Issue #1 reconciliation failed", output.getvalue())
        self.assertIn("1 failed", output.getvalue())

    def test_scheduled_and_manual_runs_choose_their_window(self) -> None:
        for hours, expect_since in (("72", True), ("0", False), ("", False)):
            with self.subTest(hours=hours), mock.patch.dict(
                os.environ, {"ACCEPTANCE_SWEEP_LOOKBACK_HOURS": hours}
            ), mock.patch.object(
                community_lifecycle_github, "sweep_issue_acceptance"
            ) as sweep:
                community_lifecycle_github.accept_issue_queue(
                    FakeAcceptanceApi([]), {"repository": {"full_name": REPO}}
                )
                kwargs = sweep.call_args.kwargs
                self.assertEqual(kwargs["since"] is not None, expect_since)
                self.assertEqual(
                    kwargs["reserve"], community_lifecycle_github.API_RESERVE_REQUESTS
                )


class GitHubClientTests(unittest.TestCase):
    def run_client(self, stdout: str, *, returncode: int = 0, stderr: str = ""):
        client = community_lifecycle_github.GitHubClient()
        completed = subprocess.CompletedProcess([], returncode, stdout, stderr)
        with mock.patch.object(
            community_lifecycle_github.subprocess, "run", return_value=completed
        ) as run:
            result = client.request("repos/acme/router/issues/1", ignore_not_found=True)
        self.assertIn("--include", run.call_args.args[0])
        return client, result

    def test_reads_the_remaining_budget_from_the_response_headers(self) -> None:
        for separator in ("\r\n", "\n"):
            with self.subTest(separator=repr(separator)):
                headers = separator.join(
                    ["HTTP/2.0 200 OK", "X-Ratelimit-Remaining: 42", "Content-Type: x"]
                )
                client, result = self.run_client(
                    headers + separator + separator + '{"number": 1}'
                )
                self.assertEqual(result, {"number": 1})
                self.assertEqual(client.rate_limit_remaining, 42)

    def test_empty_bodies_and_missing_headers(self) -> None:
        client, result = self.run_client(
            "HTTP/2.0 204 No Content\nX-Ratelimit-Remaining: 7\n"
        )
        self.assertIsNone(result)
        self.assertEqual(client.rate_limit_remaining, 7)

        client, result = self.run_client('{"number": 1}')
        self.assertEqual(result, {"number": 1})
        self.assertIsNone(client.rate_limit_remaining)

    def test_not_found_still_records_the_budget(self) -> None:
        client, result = self.run_client(
            "HTTP/2.0 404 Not Found\nX-Ratelimit-Remaining: 3\n\n{}",
            returncode=1,
            stderr="gh: Not Found (HTTP 404)",
        )
        self.assertIsNone(result)
        self.assertEqual(client.rate_limit_remaining, 3)

    def test_only_the_rest_budget_is_recorded(self) -> None:
        client = community_lifecycle_github.GitHubClient()
        recorded = []
        for resource, remaining in (("graphql", 9876), ("core", 42), ("graphql", 9875)):
            stdout = (
                f"HTTP/2.0 200 OK\nX-Ratelimit-Resource: {resource}\n"
                f"X-Ratelimit-Remaining: {remaining}\n\n{{}}"
            )
            completed = subprocess.CompletedProcess([], 0, stdout, "")
            with mock.patch.object(
                community_lifecycle_github.subprocess, "run", return_value=completed
            ):
                client.request("graphql", method="POST", payload={})
            recorded.append(client.rate_limit_remaining)
        self.assertEqual(recorded, [None, 42, 42])


if __name__ == "__main__":
    unittest.main()
