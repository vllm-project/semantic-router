"""Tests for clearing an issue's delivery-state labels once it closes."""

import importlib.util
import sys
import unittest
from pathlib import Path
from urllib.parse import quote, unquote

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


class ClosedIssuePolicyTests(unittest.TestCase):
    def test_closed_issue_drops_delivery_state_labels_only(self) -> None:
        issue = {
            "title": "[Epic] Restore something",
            "state": "closed",
            "labels": labels(
                "needs-acceptance",
                "ready-for-dev",
                "in-progress",
                "help wanted",
                "good first issue",
                "accepted",
                "wg/mom-routing",
                "priority/P1",
                "backlog",
                "release-blocker",
                "epic",
            ),
            "assignees": [],
            "milestone": None,
        }
        plan = community_lifecycle.plan_issue(
            issue, event_action="closed", event_label=None, actor_can_manage=False
        )
        self.assertEqual(
            plan.remove_labels, set(community_lifecycle.ISSUE_DELIVERY_STATE_LABELS)
        )
        self.assertEqual(plan.add_labels, set())

    def test_closed_issue_with_no_delivery_label_is_a_no_op(self) -> None:
        issue = {
            "state": "closed",
            "labels": labels("accepted", "wg/mom-routing"),
            "assignees": [],
            "milestone": None,
        }
        plan = community_lifecycle.plan_issue(
            issue, event_action="closed", event_label=None, actor_can_manage=False
        )
        self.assertEqual(plan.remove_labels, set())

    def test_open_issue_keeps_its_delivery_state_label(self) -> None:
        issue = {
            "state": "open",
            "labels": labels("needs-acceptance"),
            "assignees": [],
            "milestone": None,
        }
        plan = community_lifecycle.plan_issue(
            issue, event_action="edited", event_label=None, actor_can_manage=False
        )
        self.assertNotIn("needs-acceptance", plan.remove_labels)


class FakeIssueApi:
    """Serve the endpoints issue reconciliation calls and record its writes."""

    def __init__(self, issues, listings=None) -> None:
        self.issues = {int(item["number"]): item for item in issues}
        self.listings = listings or {}
        self.fetched: list[int] = []
        self.added: set[str] = set()
        self.removed: set[str] = set()

    def request(self, endpoint, *, method="GET", payload=None, ignore_not_found=False):
        if "?" in endpoint:
            return self.listings.get(endpoint, [])
        parts = endpoint.split("/")
        number = int(parts[4])
        tail = parts[5:]
        if not tail:
            self.fetched.append(number)
            return self.issues.get(number)
        if tail == ["labels"]:
            self.added.update(payload["labels"])
            return None
        if tail[0] == "labels":
            self.removed.add(unquote(tail[1]))
        return None


class IssueReconciliationTests(unittest.TestCase):
    REPO = "acme/router"

    def closed_issue(self, **overrides):
        item = {
            "number": 20,
            "state": "closed",
            "labels": labels("needs-acceptance", "wg/mom-routing"),
            "assignees": [],
            "milestone": None,
        }
        item.update(overrides)
        return item

    def test_closed_issue_drops_its_delivery_state_label(self) -> None:
        client = FakeIssueApi([self.closed_issue()])

        community_lifecycle_github.sync_issue(client, self.REPO, 20)

        self.assertEqual(client.removed, {"needs-acceptance"})
        self.assertEqual(client.added, set())

    def closed_listing(self, label: str, page: int = 1) -> str:
        return (
            f"repos/{self.REPO}/issues?state=closed"
            f"&labels={quote(label, safe='')}"
            f"&per_page={community_lifecycle_github.API_PAGE_SIZE}&page={page}"
        )

    def test_closed_sweep_reconciles_only_labeled_issues(self) -> None:
        listings = {
            self.closed_listing("needs-acceptance"): [
                {"number": 21, "labels": labels("needs-acceptance")},
                {
                    "number": 22,
                    "labels": labels("needs-acceptance"),
                    "pull_request": {},
                },
            ],
        }
        client = FakeIssueApi(
            [self.closed_issue(number=21, labels=labels("needs-acceptance"))],
            listings=listings,
        )

        community_lifecycle_github.sync_labeled_closed_issues(client, self.REPO)

        self.assertEqual(client.fetched, [21])
        self.assertIn("needs-acceptance", client.removed)


if __name__ == "__main__":
    unittest.main()
