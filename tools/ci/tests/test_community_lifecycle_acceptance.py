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
    """Serve acceptance endpoints, record every request, and spend a budget."""

    def __init__(
        self,
        issues,
        *,
        comments=None,
        timelines=None,
        listing=None,
        permissions=None,
        failing_requests=None,
        budget=None,
    ) -> None:
        self.issues = {int(issue["number"]): issue for issue in issues}
        self.comments = comments or {}
        self.timelines = timelines or {}
        self.listing = listing or {}
        self.permissions = permissions or {}
        self.failing_requests = set(failing_requests or ())
        self.rate_limit_remaining = budget
        self.requests: list[tuple[str, str]] = []
        self.fetched: list[int] = []
        self.added: dict[int, set[str]] = {}
        self.removed: dict[int, set[str]] = {}

    def request(
        self,
        endpoint: str,
        *,
        method: str = "GET",
        payload=None,
        ignore_not_found: bool = False,
    ):
        self.requests.append((method, endpoint))
        if self.rate_limit_remaining is not None:
            self.rate_limit_remaining -= 1
        if (method, endpoint) in self.failing_requests:
            raise RuntimeError(f"API failure for {method} {endpoint}")

        parsed = urlparse(endpoint)
        query = parse_qs(parsed.query)
        parts = parsed.path.split("/")
        if parts[3] == "collaborators":
            return {"permission": self.permissions.get(unquote(parts[4]), "read")}
        if parts[3:] == ["issues"]:
            return self.listing.get(int(query["page"][0]), [])

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

    def test_sweep_collects_candidates_before_writing_and_accepts_valid_ones(
        self,
    ) -> None:
        page_size = community_lifecycle_github.API_PAGE_SIZE
        first_page = [issue(100 + index, comments=0) for index in range(page_size - 3)]
        first_page += [issue(1), issue(2), issue(3, pull_request={"url": "example"})]
        client = FakeAcceptanceApi(
            [issue(1), issue(2), issue(4)],
            listing={1: first_page, 2: [issue(4)]},
            comments={
                1: {1: [comment("maintainer")]},
                2: {1: [comment("observer", "/accept please")]},
                4: {1: [comment("maintainer")]},
            },
            permissions={"maintainer": "write"},
        )

        output = self.sweep(client)

        listing = [
            endpoint for _, endpoint in client.requests if "/issues?" in endpoint
        ]
        self.assertEqual(len(listing), 2)
        last_listing = max(
            index
            for index, (_, endpoint) in enumerate(client.requests)
            if "/issues?" in endpoint
        )
        first_write = min(
            index
            for index, (method, _) in enumerate(client.requests)
            if method != "GET"
        )
        self.assertLess(last_listing, first_write)
        self.assertIn("labels=needs-acceptance", listing[0])
        self.assertIn("sort=updated&direction=desc", listing[0])
        self.assertEqual(client.added, {1: {"accepted"}, 4: {"accepted"}})
        self.assertEqual(client.fetched, [1, 4])
        self.assertFalse(
            any(path.endswith("/issues/100/comments") for path in client.endpoints())
        )
        self.assertNotIn(f"/repos/{REPO}/issues/comments", client.endpoints())
        self.assertIn("2 accepted", output)
        self.assertIn("1 without `/accept`", output)

    def test_sweep_stops_at_the_reserve_and_leaves_the_rest_for_the_next_run(
        self,
    ) -> None:
        candidates = [issue(number) for number in range(1, 6)]
        client = FakeAcceptanceApi(
            candidates,
            listing={1: candidates},
            comments={number: {1: [comment("maintainer")]} for number in range(1, 6)},
            permissions={"maintainer": "write"},
            budget=12,
        )

        output = self.sweep(client, reserve=5)

        self.assertEqual(sorted(client.added), [1, 2])
        self.assertLessEqual(client.rate_limit_remaining, 5)
        self.assertIn("Stopped with", output)
        self.assertIn("3 issue(s) remain for the next run", output)

    def test_sweep_passes_its_lookback_window_to_the_listing(self) -> None:
        client = FakeAcceptanceApi([], listing={1: []})

        self.sweep(client, since="2026-11-01T00:00:00Z")

        self.assertIn("since=2026-11-01T00%3A00%3A00Z", client.requests[0][1])

    def test_sweep_reports_failures_after_processing_the_rest(self) -> None:
        failing = ("POST", f"repos/{REPO}/issues/1/labels")
        client = FakeAcceptanceApi(
            [issue(1), issue(2)],
            listing={1: [issue(1), issue(2)]},
            comments={1: {1: [comment("maintainer")]}, 2: {1: [comment("maintainer")]}},
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


if __name__ == "__main__":
    unittest.main()
