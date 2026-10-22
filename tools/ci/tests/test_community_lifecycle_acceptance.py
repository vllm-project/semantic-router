"""Tests for repository-wide issue acceptance reconciliation."""

import contextlib
import importlib.util
import io
import os
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


def labels(*names: str) -> list[dict[str, str]]:
    return [{"name": name} for name in names]


class FakeAcceptanceApi:
    """Serve acceptance endpoints and record issue fetches and label writes."""

    def __init__(
        self,
        issues,
        comment_pages,
        permissions,
        *,
        failing_requests=None,
    ) -> None:
        self.issues = {int(issue["number"]): issue for issue in issues}
        self.comment_pages = comment_pages
        self.permissions = permissions
        self.failing_requests = set(failing_requests or ())
        self.fetched: list[int] = []
        self.comment_pages_fetched: list[int] = []
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
        if (method, endpoint) in self.failing_requests:
            raise RuntimeError(f"API failure for {method} {endpoint}")

        parsed = urlparse(endpoint)
        parts = parsed.path.split("/")
        if parts[3:] == ["issues", "comments"]:
            page = int(parse_qs(parsed.query)["page"][0])
            self.comment_pages_fetched.append(page)
            return self.comment_pages.get(page, [])
        if parts[3] == "collaborators":
            return {"permission": self.permissions.get(unquote(parts[4]), "read")}

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

    def comment(self, number: int, actor: str, body: str = "/accept"):
        return {
            "body": body,
            "issue_url": f"https://api.github.com/repos/{self.REPO}/issues/{number}",
            "user": {"login": actor},
        }

    def event(self, number: int = 999, body: str = "/accept"):
        return {
            "repository": {"full_name": self.REPO},
            "issue": {"number": number},
            "comment": {"body": body},
            "sender": {"login": "triggering-user"},
        }

    def test_one_event_pages_deduplicates_and_repairs_every_stale_issue(self) -> None:
        page_size = community_lifecycle_github.API_PAGE_SIZE
        first_page = [
            self.comment(700 + index, "observer", "not /accept")
            for index in range(page_size - 2)
        ]
        first_page.extend(
            [self.comment(2, "maintainer"), self.comment(1, "maintainer")]
        )
        second_page = [
            self.comment(2, "another-maintainer"),
            self.comment(3, "maintainer"),
            self.comment(4, "maintainer", "/accept "),
            self.comment(5, "maintainer"),
        ]
        client = FakeAcceptanceApi(
            [
                self.issue(1),
                self.issue(2),
                self.issue(3, labels=labels("accepted", "wg/agentic-context")),
                self.issue(5, pull_request={"url": "example"}),
            ],
            {1: first_page, 2: second_page},
            {"maintainer": "write", "another-maintainer": "maintain"},
        )

        community_lifecycle_github.accept_issue_event(client, self.event())

        self.assertEqual(client.comment_pages_fetched, [1, 2])
        self.assertEqual(client.fetched, [1, 2, 3, 5])
        self.assertEqual(client.added, {1: {"accepted"}, 2: {"accepted"}})
        self.assertEqual(
            client.removed,
            {1: {"needs-acceptance"}, 2: {"needs-acceptance"}},
        )
        self.assertNotIn(999, client.fetched)

    def test_invalid_candidates_remain_unchanged_with_actionable_output(self) -> None:
        client = FakeAcceptanceApi(
            [
                self.issue(10),
                self.issue(
                    11,
                    labels=labels(
                        "needs-acceptance",
                        "wg/agentic-context",
                        "owner/maintainers",
                    ),
                ),
                self.issue(12, title="Improve context routing"),
            ],
            {
                1: [
                    self.comment(10, "outsider"),
                    self.comment(11, "maintainer"),
                    self.comment(12, "maintainer"),
                ]
            },
            {"outsider": "read", "maintainer": "admin"},
        )
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            community_lifecycle_github.accept_issue_event(client, self.event())

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
            {1: [self.comment(20, "maintainer"), self.comment(21, "maintainer")]},
            {"maintainer": "write"},
            failing_requests={("POST", failing_endpoint)},
        )
        output = io.StringIO()

        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit):
            community_lifecycle_github.accept_issue_event(client, self.event())

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
                client, self.event(body="/accept\n")
            )

        self.assertEqual(client.comment_pages_fetched, [])


if __name__ == "__main__":
    unittest.main()
