import importlib.util
import os
import re
import sys
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock
from urllib.parse import quote, unquote

import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "community_lifecycle.py"
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))
REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("community_lifecycle", SCRIPT)
assert SPEC and SPEC.loader
community_lifecycle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = community_lifecycle
SPEC.loader.exec_module(community_lifecycle)
community_lifecycle_github = sys.modules["community_lifecycle_github"]
community_reviewers = sys.modules["community_reviewers"]


def labels(*names: str) -> list[dict[str, str]]:
    return [{"name": name} for name in names]


class CommunityLifecycleTests(unittest.TestCase):
    def test_form_workgroup_is_parsed(self) -> None:
        body = """### Goal

Make routing better.

### Proposed Workgroup

MoM & Routing
"""
        self.assertEqual(
            community_lifecycle.proposed_workgroup(body),
            "wg/mom-routing",
        )

    def test_new_issue_enters_acceptance_with_proposed_owner(self) -> None:
        issue = {
            "body": "### Proposed Workgroup\n\nEvaluation & Quality\n",
            "labels": labels("bug"),
            "assignees": [],
            "milestone": None,
        }
        plan = community_lifecycle.plan_issue(
            issue,
            event_action="opened",
            event_label=None,
            actor_can_manage=False,
        )
        self.assertEqual(
            plan.add_labels,
            {"needs-acceptance", "wg/evaluation-quality"},
        )

    def test_maintainer_reclassification_overrides_form_workgroup(self) -> None:
        issue = {
            "body": "### Proposed Workgroup\n\nEnterprise & Environment\n",
            "labels": labels("needs-acceptance", "wg/data-plane-networking"),
            "assignees": [],
            "milestone": None,
        }
        plan = community_lifecycle.plan_issue(
            issue,
            event_action="labeled",
            event_label="wg/data-plane-networking",
            actor_can_manage=True,
        )
        self.assertNotIn("wg/enterprise-environment", plan.add_labels)
        self.assertNotIn("wg/data-plane-networking", plan.remove_labels)

    def test_maintainer_can_remove_stale_form_workgroup_before_relabeling(self) -> None:
        issue = {
            "body": "### Proposed Workgroup\n\nEnterprise & Environment\n",
            "labels": labels("needs-acceptance"),
            "assignees": [],
            "milestone": None,
        }
        plan = community_lifecycle.plan_issue(
            issue,
            event_action="unlabeled",
            event_label="wg/enterprise-environment",
            actor_can_manage=True,
        )
        self.assertNotIn("wg/enterprise-environment", plan.add_labels)

    def test_maintainer_owner_overrides_stale_form_workgroup(self) -> None:
        issue = {
            "body": "### Proposed Workgroup\n\nEnterprise & Environment\n",
            "labels": labels("needs-acceptance", "owner/maintainers"),
            "assignees": [],
            "milestone": None,
        }
        plan = community_lifecycle.plan_issue(
            issue,
            event_action="labeled",
            event_label="owner/maintainers",
            actor_can_manage=True,
        )
        self.assertNotIn("wg/enterprise-environment", plan.add_labels)
        self.assertNotIn("owner/maintainers", plan.remove_labels)

    def test_maintainer_can_remove_owner_without_form_race(self) -> None:
        issue = {
            "body": "### Proposed Workgroup\n\nEnterprise & Environment\n",
            "labels": labels("needs-acceptance"),
            "assignees": [],
            "milestone": None,
        }
        plan = community_lifecycle.plan_issue(
            issue,
            event_action="unlabeled",
            event_label="owner/maintainers",
            actor_can_manage=True,
        )
        self.assertNotIn("wg/enterprise-environment", plan.add_labels)

    def test_multiple_workgroups_are_not_resolved_from_stale_form_data(self) -> None:
        issue = {
            "body": "### Proposed Workgroup\n\nEnterprise & Environment\n",
            "labels": labels(
                "needs-acceptance",
                "wg/enterprise-environment",
                "wg/data-plane-networking",
            ),
            "assignees": [],
            "milestone": None,
        }
        plan = community_lifecycle.plan_issue(
            issue,
            event_action="labeled",
            event_label="wg/data-plane-networking",
            actor_can_manage=True,
        )
        self.assertFalse(
            plan.add_labels.intersection(community_lifecycle.WORKGROUP_LABELS)
        )
        self.assertFalse(
            plan.remove_labels.intersection(community_lifecycle.WORKGROUP_LABELS)
        )

    def test_unaccepted_issue_cannot_be_assigned_or_prioritized(self) -> None:
        issue = {
            "body": "",
            "labels": labels(
                "priority/P0",
                "ready-for-dev",
                "release-blocker",
                "backlog",
            ),
            "assignees": [{"login": "contributor"}],
            "milestone": {"number": 7},
        }
        plan = community_lifecycle.plan_issue(
            issue,
            event_action="assigned",
            event_label=None,
            actor_can_manage=True,
        )
        self.assertIn("needs-acceptance", plan.add_labels)
        self.assertTrue(
            {
                "priority/P0",
                "ready-for-dev",
                "release-blocker",
                "backlog",
            }
            <= plan.remove_labels
        )
        self.assertEqual(plan.remove_assignees, {"contributor"})
        self.assertTrue(plan.clear_milestone)

    def test_accepted_assigned_issue_becomes_in_progress(self) -> None:
        issue = {
            "body": "",
            "labels": labels(
                "accepted",
                "ready-for-dev",
                "help wanted",
                "stale",
                "wg/data-plane-networking",
            ),
            "assignees": [{"login": "contributor"}],
            "milestone": None,
        }
        plan = community_lifecycle.plan_issue(
            issue,
            event_action="assigned",
            event_label=None,
            actor_can_manage=True,
        )
        self.assertIn("in-progress", plan.add_labels)
        self.assertTrue({"ready-for-dev", "help wanted"} <= plan.remove_labels)
        self.assertIn("needs-acceptance", plan.remove_labels)
        self.assertIn("stale", plan.remove_labels)

    def test_unassigned_work_does_not_become_contributor_ready_implicitly(self) -> None:
        issue = {
            "body": "",
            "labels": labels(
                "accepted",
                "in-progress",
                "wg/data-plane-networking",
            ),
            "assignees": [],
            "milestone": None,
        }
        plan = community_lifecycle.plan_issue(
            issue,
            event_action="unassigned",
            event_label=None,
            actor_can_manage=True,
        )
        self.assertIn("in-progress", plan.remove_labels)
        self.assertNotIn("ready-for-dev", plan.add_labels)

    def test_acceptance_without_one_workgroup_is_rejected(self) -> None:
        issue = {
            "body": "",
            "labels": labels("accepted"),
            "assignees": [],
            "milestone": None,
        }
        plan = community_lifecycle.plan_issue(
            issue,
            event_action="labeled",
            event_label="accepted",
            actor_can_manage=True,
        )
        self.assertIn("accepted", plan.remove_labels)
        self.assertIn("needs-acceptance", plan.add_labels)

    def test_protected_label_from_non_maintainer_is_removed(self) -> None:
        issue = {
            "body": "",
            "labels": labels("accepted", "wg/evaluation-quality"),
            "assignees": [],
            "milestone": None,
        }
        plan = community_lifecycle.plan_issue(
            issue,
            event_action="labeled",
            event_label="accepted",
            actor_can_manage=False,
        )
        self.assertIn("accepted", plan.remove_labels)
        self.assertIn("needs-acceptance", plan.add_labels)

    def test_protected_label_removed_by_non_maintainer_is_restored(self) -> None:
        issue = {
            "body": "",
            "labels": labels("accepted", "wg/evaluation-quality"),
            "assignees": [],
            "milestone": None,
        }
        plan = community_lifecycle.plan_issue(
            issue,
            event_action="unlabeled",
            event_label="priority/P1",
            actor_can_manage=False,
        )
        self.assertIn("priority/P1", plan.add_labels)

    def test_write_collaborator_can_accept_one_owned_issue(self) -> None:
        evaluation = community_lifecycle.evaluate_issue_acceptance(
            {
                "title": "[Feature] Improve context routing",
                "labels": labels(
                    "needs-acceptance",
                    "wg/agentic-context",
                ),
            },
            actor_can_manage=True,
        )
        self.assertTrue(evaluation.valid)
        self.assertEqual(evaluation.owner_label, "wg/agentic-context")

    def test_accept_command_rejects_actor_without_write_permission(self) -> None:
        evaluation = community_lifecycle.evaluate_issue_acceptance(
            {
                "title": "[Feature] Improve context routing",
                "labels": labels("wg/agentic-context"),
            },
            actor_can_manage=False,
        )
        self.assertFalse(evaluation.valid)
        self.assertIn("write", evaluation.error or "")

    def test_accept_command_requires_exactly_one_workgroup(self) -> None:
        evaluation = community_lifecycle.evaluate_issue_acceptance(
            {
                "title": "[Feature] Improve context routing",
                "labels": labels(
                    "wg/agentic-context",
                    "wg/mom-routing",
                ),
            },
            actor_can_manage=True,
        )
        self.assertFalse(evaluation.valid)
        self.assertIn("exactly one", evaluation.error or "")

    def test_write_collaborator_can_accept_maintainer_owned_governance(self) -> None:
        evaluation = community_lifecycle.evaluate_issue_acceptance(
            {
                "title": "[Governance] Maintain the public roadmap",
                "labels": labels(
                    "needs-acceptance",
                    "owner/maintainers",
                ),
            },
            actor_can_manage=True,
        )
        self.assertTrue(evaluation.valid)
        self.assertEqual(evaluation.owner_label, "owner/maintainers")

    def test_accept_command_rejects_workgroup_plus_maintainer_owner(self) -> None:
        evaluation = community_lifecycle.evaluate_issue_acceptance(
            {
                "title": "[Governance] Maintain the public roadmap",
                "labels": labels(
                    "wg/evaluation-quality",
                    "owner/maintainers",
                ),
            },
            actor_can_manage=True,
        )
        self.assertFalse(evaluation.valid)
        self.assertIn("exactly one", evaluation.error or "")

    def test_title_format_accepts_one_open_category(self) -> None:
        for title in (
            "[Feature] Add standalone serving",
            "[CI/Build] Validate community titles",
            "[v0.5] Publish model cards",
        ):
            with self.subTest(title=title):
                self.assertIsNone(community_lifecycle.title_format_error(title))

    def test_title_format_rejects_missing_or_stacked_categories(self) -> None:
        for title in (
            "feature: add standalone serving",
            "Feature: add standalone serving",
            "[Router][Docs] Add guidance",
            "[Router] [Docs] Add guidance",
            "[] Add guidance",
            "[Bug]",
        ):
            with self.subTest(title=title):
                self.assertIsNotNone(community_lifecycle.title_format_error(title))

    def test_accept_command_requires_normalized_title(self) -> None:
        evaluation = community_lifecycle.evaluate_issue_acceptance(
            {
                "title": "feature: improve context routing",
                "labels": labels("wg/agentic-context"),
            },
            actor_can_manage=True,
        )
        self.assertFalse(evaluation.valid)
        self.assertIn("bracketed category", evaluation.error or "")

    def test_epic_title_adds_the_structural_label(self) -> None:
        plan = community_lifecycle.plan_issue_kind(
            {"title": "[Epic] Make routing behavior measurable", "labels": []}
        )
        self.assertEqual(plan.add_labels, {"epic"})
        self.assertEqual(plan.remove_labels, set())

    def test_non_epic_title_removes_the_structural_label(self) -> None:
        plan = community_lifecycle.plan_issue_kind(
            {
                "title": "[Feature] Make routing behavior measurable",
                "labels": labels("epic"),
            }
        )
        self.assertEqual(plan.add_labels, set())
        self.assertEqual(plan.remove_labels, {"epic"})

    def test_pr_requires_an_accepted_linked_issue(self) -> None:
        pull_request = {"draft": False, "user": {"type": "User"}}
        evaluation = community_lifecycle.evaluate_pull_request(
            pull_request,
            [{"labels": labels("needs-acceptance"), "milestone": None}],
        )
        self.assertFalse(evaluation.valid)
        self.assertEqual(evaluation.state_label, "pr/blocked")

    def test_pr_inherits_owner_milestone_and_release_blocker(self) -> None:
        pull_request = {"draft": False, "user": {"type": "User"}}
        linked = [
            {
                "labels": labels(
                    "accepted",
                    "release-blocker",
                    "wg/enterprise-environment",
                ),
                "milestone": {"number": 7},
            }
        ]
        evaluation = community_lifecycle.evaluate_pull_request(pull_request, linked)
        self.assertTrue(evaluation.valid)
        self.assertEqual(evaluation.state_label, "pr/needs-review")
        self.assertEqual(evaluation.owner_label, "wg/enterprise-environment")
        self.assertEqual(evaluation.milestone_number, 7)
        self.assertTrue(evaluation.release_blocker)

    def test_pr_can_inherit_maintainer_governance_owner(self) -> None:
        evaluation = community_lifecycle.evaluate_pull_request(
            {"draft": False, "user": {"type": "User"}},
            [
                {
                    "labels": labels("accepted", "owner/maintainers"),
                    "milestone": None,
                }
            ],
        )
        self.assertTrue(evaluation.valid)
        self.assertEqual(evaluation.owner_label, "owner/maintainers")

    def test_pr_review_and_runtime_signals_select_one_action_state(self) -> None:
        pull_request = {"draft": False, "user": {"type": "User"}}
        linked = [
            {
                "labels": labels("accepted", "wg/enterprise-environment"),
                "milestone": None,
            }
        ]
        cases = (
            ({"review_decision": "CHANGES_REQUESTED"}, "pr/needs-author"),
            ({"merge_state": "BEHIND"}, "pr/needs-rebase"),
            ({"checks_state": "FAILURE"}, "pr/blocked"),
            (
                {
                    "review_decision": "APPROVED",
                    "checks_state": "PENDING",
                    "merge_state": "CLEAN",
                },
                "pr/blocked",
            ),
            (
                {
                    "review_decision": "APPROVED",
                    "checks_state": "SUCCESS",
                    "merge_state": "CLEAN",
                },
                "pr/merge-ready",
            ),
        )
        for signals, expected in cases:
            with self.subTest(expected=expected):
                evaluation = community_lifecycle.evaluate_pull_request(
                    pull_request, linked, **signals
                )
                self.assertEqual(evaluation.state_label, expected)

    def test_pr_close_candidate_is_not_overwritten_by_reconciliation(self) -> None:
        evaluation = community_lifecycle.evaluate_pull_request(
            {
                "draft": False,
                "user": {"type": "User"},
                "labels": labels("pr/close-candidate"),
            },
            [
                {
                    "labels": labels("accepted", "wg/evaluation-quality"),
                    "milestone": None,
                }
            ],
            review_decision="REVIEW_REQUIRED",
            checks_state="SUCCESS",
            merge_state="CLEAN",
        )
        self.assertEqual(evaluation.state_label, "pr/close-candidate")

    def test_bot_pr_can_move_to_rebase_without_a_tracking_issue(self) -> None:
        evaluation = community_lifecycle.evaluate_pull_request(
            {"draft": False, "user": {"type": "Bot"}},
            [],
            merge_state="BEHIND",
        )
        self.assertTrue(evaluation.valid)
        self.assertEqual(evaluation.state_label, "pr/needs-rebase")
        self.assertEqual(evaluation.owner_label, "wg/evaluation-quality")

    def test_pr_rejects_multiple_accepted_workgroup_owners(self) -> None:
        pull_request = {"draft": False, "user": {"type": "User"}}
        linked = [
            {
                "labels": labels("accepted", "wg/enterprise-environment"),
                "milestone": None,
            },
            {
                "labels": labels("accepted", "wg/evaluation-quality"),
                "milestone": None,
            },
        ]
        evaluation = community_lifecycle.evaluate_pull_request(pull_request, linked)
        self.assertFalse(evaluation.valid)

    def test_draft_pr_is_author_work(self) -> None:
        evaluation = community_lifecycle.evaluate_pull_request(
            {"draft": True, "user": {"type": "User"}}, []
        )
        self.assertTrue(evaluation.valid)
        self.assertEqual(evaluation.state_label, "pr/needs-author")

    def test_draft_pr_keeps_accepted_owner_and_milestone_context(self) -> None:
        evaluation = community_lifecycle.evaluate_pull_request(
            {"draft": True, "user": {"type": "User"}},
            [
                {
                    "labels": labels(
                        "accepted",
                        "release-blocker",
                        "wg/evaluation-quality",
                    ),
                    "milestone": {"number": 7},
                }
            ],
        )
        self.assertEqual(evaluation.state_label, "pr/needs-author")
        self.assertEqual(evaluation.owner_label, "wg/evaluation-quality")
        self.assertEqual(evaluation.milestone_number, 7)
        self.assertTrue(evaluation.release_blocker)

    def test_related_issue_references_are_extracted(self) -> None:
        self.assertEqual(
            community_lifecycle.extract_related_issue_numbers(
                "[Router] change", "Related #123\nRelated: org/repo#456"
            ),
            {123, 456},
        )

    def test_foreign_related_issue_reference_is_ignored(self) -> None:
        self.assertEqual(
            community_lifecycle.extract_related_issue_numbers(
                "[Router] change",
                "Related: vllm-project/semantic-router#123\n"
                "Related: another/repository#456",
                repository="vllm-project/semantic-router",
            ),
            {123},
        )

    def test_policy_and_issue_forms_match_executable_workgroups(self) -> None:
        policy = yaml.safe_load(
            (REPO_ROOT / "tools/agent/maintainer-policy.yaml").read_text()
        )
        self.assertEqual(
            set(policy["labels"]["workgroups"]),
            set(community_lifecycle.WORKGROUP_LABELS),
        )
        for template in (
            ".github/ISSUE_TEMPLATE/001_feature_request.yaml",
            ".github/ISSUE_TEMPLATE/002_bug_report.yaml",
        ):
            issue_form = yaml.safe_load((REPO_ROOT / template).read_text())
            workgroup_field = next(
                field for field in issue_form["body"] if field.get("id") == "workgroup"
            )
            options = set(workgroup_field["attributes"]["options"])
            options.discard("Unsure / Maintainer triage")
            self.assertEqual(options, set(community_lifecycle.WORKGROUP_OPTIONS))
        self.assertEqual(
            policy["labels"]["maintainer_owner"],
            community_lifecycle.MAINTAINER_OWNER,
        )

    def test_policy_matches_executable_lifecycle_labels(self) -> None:
        policy = yaml.safe_load(
            (REPO_ROOT / "tools/agent/maintainer-policy.yaml").read_text()
        )
        lifecycle = policy["labels"]["lifecycle"]
        self.assertEqual(lifecycle["needs_acceptance"], "needs-acceptance")
        self.assertEqual(lifecycle["accepted"], "accepted")
        self.assertEqual(lifecycle["ready_for_dev"], "ready-for-dev")
        self.assertEqual(lifecycle["in_progress"], "in-progress")
        self.assertEqual(
            set(policy["labels"]["pr_state"].values()),
            set(community_lifecycle.PR_STATE_LABELS),
        )
        self.assertEqual(policy["labels"]["structure"]["epic"], "epic")

    def test_retired_parallel_taxonomies_are_not_declared(self) -> None:
        policy = yaml.safe_load(
            (REPO_ROOT / "tools/agent/maintainer-policy.yaml").read_text()
        )
        prow_labels = yaml.safe_load((REPO_ROOT / ".prowlabels.yaml").read_text())
        self.assertNotIn("release_tracks", policy)
        self.assertNotIn("area", prow_labels)
        self.assertNotIn("track", prow_labels)
        community_workflow = (REPO_ROOT / ".github/workflows/community.yml").read_text()
        self.assertNotIn("/area", community_workflow)


class FakePullRequestApi:
    """Serve the endpoints pull-request reconciliation calls and record its writes."""

    def __init__(self, pull_requests, issues=None, listings=None) -> None:
        self.pull_requests = {int(item["number"]): item for item in pull_requests}
        self.issues = {int(item["number"]): item for item in (issues or [])}
        self.listings = listings or {}
        self.requests: list[tuple[str, str]] = []
        self.writes: list[tuple[str, str, dict | None]] = []
        self.batched_merge_states: dict[int, str] = {}
        self.fetched: list[int] = []
        self.added: set[str] = set()
        self.removed: set[str] = set()
        self.requested_reviewers: dict[int, set[str]] = {}
        self.reviews: dict[int, list[dict]] = {}
        self.review_posts: list[tuple[int, str]] = []
        self.review_removals: list[tuple[int, str]] = []
        self.review_attempts: list[tuple[int, str]] = []
        self.non_collaborators: set[str] = set()
        self.comments: dict[int, list[dict]] = {}
        self.comment_posts: list[tuple[int, str]] = []
        self.comment_edits: list[tuple[int, str]] = []

    def request(
        self,
        endpoint: str,
        *,
        method: str = "GET",
        payload=None,
        ignore_not_found: bool = False,
        ignore_unprocessable: bool = False,
    ):
        self.requests.append((method, endpoint))
        if endpoint == "graphql":
            return self.graphql(payload)
        if method != "GET":
            self.writes.append((method, endpoint, payload))
        path, _, query = endpoint.partition("?")
        parts = path.split("/")
        if query and parts[3] == "issues" and parts[5:] == ["comments"]:
            return self.comments.get(int(parts[4]), [])
        if query:
            return self.listings.get(endpoint, [])
        if parts[3:5] == ["issues", "comments"]:
            comment_id = int(parts[5])
            for comment in (c for items in self.comments.values() for c in items):
                if comment["id"] == comment_id:
                    comment["body"] = payload["body"]
            self.comment_edits.append((comment_id, payload["body"]))
            return {"id": comment_id}
        number = int(parts[4])
        if parts[3:6] == ["issues", str(number), "comments"] and method == "POST":
            comment = {"id": 1000 + len(self.comment_posts), "body": payload["body"]}
            self.comments.setdefault(number, []).append(comment)
            self.comment_posts.append((number, payload["body"]))
            return comment
        if parts[3] == "pulls":
            self.fetched.append(number)
            if parts[5:] == ["requested_reviewers"]:
                if method == "POST":
                    reviewer = payload["reviewers"][0]
                    self.review_attempts.append((number, reviewer))
                    if reviewer in self.non_collaborators:
                        if not ignore_unprocessable:
                            raise SystemExit(1)
                        return None
                    self.requested_reviewers.setdefault(number, set()).add(reviewer)
                    self.review_posts.append((number, reviewer))
                elif method == "DELETE":
                    reviewer = payload["reviewers"][0]
                    self.requested_reviewers.setdefault(number, set()).discard(reviewer)
                    self.review_removals.append((number, reviewer))
                return {
                    "users": [
                        {"login": login}
                        for login in self.requested_reviewers.get(number, set())
                    ]
                }
            return self.pull_requests.get(number)
        tail = parts[5:]
        if not tail:
            return self.issues.get(number)
        if tail == ["labels"]:
            self.added.update(payload["labels"])
            return None
        if tail[0] == "labels":
            self.removed.add(unquote(tail[1]))
        return None

    def graphql(self, payload):
        if "number" not in payload["variables"]:
            nodes = {}
            for alias, number_text in re.findall(
                r"\b(pr\d+): pullRequest\(number: (\d+)\)", payload["query"]
            ):
                number = int(number_text)
                if number not in self.pull_requests:
                    nodes[alias] = None
                    continue
                response = self.graphql({"variables": {"number": number}})
                node = response["data"]["repository"]["pullRequest"]
                pr = self.pull_requests[number]
                node.update(
                    state=pr["state"].upper(),
                    isDraft=pr["draft"],
                    updatedAt=pr.get("updated_at"),
                    headRefOid=(pr.get("head") or {}).get("sha"),
                )
                if number in self.batched_merge_states:
                    node["mergeStateStatus"] = self.batched_merge_states[number]
                nodes[alias] = node
            return {"data": {"repository": nodes}}

        pull_request = self.pull_requests[payload["variables"]["number"]]
        nodes = [{"number": number} for number in pull_request.get("closes", ())]
        signals = pull_request.get("signals", {})
        rollup = {"state": signals.get("checks_state")}
        return {
            "data": {
                "repository": {
                    "pullRequest": {
                        "closingIssuesReferences": {"nodes": nodes},
                        "reviewDecision": signals.get("review_decision"),
                        "mergeStateStatus": signals.get("merge_state"),
                        "reviewRequests": {
                            "nodes": [
                                {"requestedReviewer": {"login": login}}
                                for login in self.requested_reviewers.get(
                                    int(pull_request["number"]), set()
                                )
                            ]
                        },
                        "latestReviews": {
                            "nodes": [
                                {"author": review.get("user")}
                                for review in self.reviews.get(
                                    int(pull_request["number"]), []
                                )
                            ]
                        },
                        "commits": {
                            "nodes": [{"commit": {"statusCheckRollup": rollup}}]
                        },
                    }
                }
            }
        }


class PullRequestReconciliationTests(unittest.TestCase):
    REPO = "acme/router"

    def setUp(self) -> None:
        repository_environment = mock.patch.dict(
            os.environ,
            {"GITHUB_REPOSITORY": self.REPO},
        )
        repository_environment.start()
        self.addCleanup(repository_environment.stop)

    def pull_request(self, **overrides):
        item = {
            "number": 7,
            "title": "[Feature] Update routing",
            "body": "",
            "state": "closed",
            "draft": False,
            "user": {"type": "User"},
            "labels": labels("pr/blocked", "wg/mom-routing"),
            "milestone": None,
            "closes": [5],
            "signals": {
                "review_decision": "APPROVED",
                "merge_state": "UNKNOWN",
                "checks_state": None,
            },
        }
        item.update(overrides)
        return item

    def accepted_issue(self):
        return {
            "number": 5,
            "labels": labels("accepted", "wg/mom-routing"),
            "milestone": None,
        }

    def test_merged_pull_request_drops_its_state_label(self) -> None:
        client = FakePullRequestApi([self.pull_request()], [self.accepted_issue()])

        community_lifecycle_github.sync_pull_request(client, self.REPO, 7)

        state_labels = set(community_lifecycle.PR_STATE_LABELS)
        self.assertEqual(client.removed & state_labels, {"pr/blocked"})
        self.assertEqual(client.added & state_labels, set())
        self.assertNotIn("wg/mom-routing", client.removed)

    def test_open_pull_request_still_carries_one_state_label(self) -> None:
        pull_request = self.pull_request(
            state="open",
            labels=labels("wg/mom-routing"),
            signals={
                "review_decision": "APPROVED",
                "merge_state": "CLEAN",
                "checks_state": "SUCCESS",
            },
        )
        client = FakePullRequestApi([pull_request], [self.accepted_issue()])

        community_lifecycle_github.sync_pull_request(client, self.REPO, 7)

        self.assertEqual(client.added, {"pr/merge-ready"})

    def test_each_workgroup_routes_to_active_maintainers_or_committers(self) -> None:
        self.assertEqual(
            set(community_reviewers.WORKGROUP_REVIEWERS),
            set(community_lifecycle.WORKGROUP_LABELS),
        )
        data = REPO_ROOT / "website/src/data"
        workgroups = (data / "workGroups.ts").read_text()
        member_types = {
            login.casefold(): member_type
            for login, member_type in re.findall(
                r"github: 'https://github\.com/([^']+)'.*?memberType: '(\w+)'",
                (data / "teamMembers.tsx").read_text(),
                flags=re.S,
            )
        }
        committer_status = {
            login.casefold(): status
            for login, status in re.findall(
                r'"login": "([^"]+)".*?"status": "(\w+)"',
                (data / "committerActivity.generated.ts").read_text(),
                flags=re.S,
            )
        }
        for owner, leads in community_reviewers.WORKGROUP_REVIEWERS.items():
            self.assertIn(f"label: '{owner}'", workgroups)
            self.assertTrue(leads, owner)
            keys = [login.casefold() for login in leads]
            self.assertEqual(len(keys), len(set(keys)), owner)
            for login in keys:
                member_type = member_types.get(login)
                self.assertTrue(
                    member_type == "maintainer"
                    or (
                        member_type == "committer"
                        and committer_status.get(login) == "active"
                    ),
                    f"{owner}: {login}",
                )

    def test_default_reviewers_match_codeowners(self) -> None:
        codeowners = (REPO_ROOT / ".github/CODEOWNERS").read_text().splitlines()
        owners = next(line.split()[1:] for line in codeowners if line.startswith("* "))
        self.assertEqual(
            {owner.removeprefix("@") for owner in owners},
            set(community_reviewers.DEFAULT_REVIEWERS),
        )

    def test_rotation_gives_every_eligible_lead_an_equal_share(self) -> None:
        for owner in community_reviewers.WORKGROUP_REVIEWERS:
            leads = community_reviewers.workgroup_reviewers(owner, "contributor")
            picks = Counter(
                community_reviewers.desired_workgroup_reviewer(
                    owner, number, "contributor"
                )
                for number in range(3 * len(leads))
            )
            self.assertEqual(picks, Counter(dict.fromkeys(leads, 3)), owner)

    def test_author_and_repository_wide_reviewers_are_skipped(self) -> None:
        reviewers = community_reviewers.workgroup_reviewers
        self.assertNotIn("Xunzhuo", reviewers("wg/mom-routing", "contributor"))
        self.assertEqual(
            reviewers("wg/evaluation-quality", "faust-benchou"), ("Zi-Yi-Ming",)
        )
        self.assertIsNone(
            community_reviewers.desired_workgroup_reviewer(
                "wg/enterprise-environment", 7, "AayushSaini101"
            )
        )
        self.assertEqual(reviewers(None, "contributor"), ())
        with mock.patch.dict(
            community_reviewers.WORKGROUP_REVIEWERS,
            {"wg/evaluation-quality": ("Xunzhuo", "FAUST-BENCHOU")},
        ):
            self.assertEqual(
                reviewers("wg/evaluation-quality", "FAUST-BENCHOU"), ("Xunzhuo",)
            )

    def test_requests_one_workgroup_reviewer_once(self) -> None:
        pull_request = self.pull_request(
            state="open",
            labels=[],
            user={"type": "User", "login": "contributor"},
        )
        client = FakePullRequestApi([pull_request], [self.accepted_issue()])

        community_lifecycle_github.sync_pull_request(client, self.REPO, 7)
        community_lifecycle_github.sync_pull_request(client, self.REPO, 7)

        self.assertEqual(len(client.review_posts), 1)
        self.assertIn(
            client.review_posts[0][1],
            community_reviewers.WORKGROUP_REVIEWERS["wg/mom-routing"],
        )

    def test_does_not_request_author_or_repeat_completed_review(self) -> None:
        owner = "wg/evaluation-quality"
        candidate = community_reviewers.desired_workgroup_reviewer(
            owner, 7, "FAUST-BENCHOU"
        )
        self.assertEqual(candidate, "Zi-Yi-Ming")
        pull_request = self.pull_request(
            state="open",
            labels=[],
            user={"type": "User", "login": "FAUST-BENCHOU"},
            closes=[5],
        )
        issue = {**self.accepted_issue(), "labels": labels("accepted", owner)}
        client = FakePullRequestApi([pull_request], [issue])
        client.reviews[7] = [{"user": {"login": candidate}, "state": "COMMENTED"}]

        community_lifecycle_github.sync_pull_request(client, self.REPO, 7)

        self.assertEqual(client.review_posts, [])

    def test_engaged_lead_is_not_joined_by_another(self) -> None:
        owner = "wg/mom-routing"
        reviewer = community_reviewers.desired_workgroup_reviewer(
            owner, 7, "contributor"
        )
        other = next(
            login
            for login in community_reviewers.workgroup_reviewers(owner, "contributor")
            if login != reviewer
        )
        pull_request = self.pull_request(
            state="open",
            labels=[],
            user={"type": "User", "login": "contributor"},
        )
        client = FakePullRequestApi([pull_request], [self.accepted_issue()])
        client.reviews[7] = [{"user": {"login": other}, "state": "APPROVED"}]

        community_lifecycle_github.sync_pull_request(client, self.REPO, 7)

        self.assertEqual(client.review_posts, [])

    def test_repository_wide_request_does_not_stand_in_for_the_group(self) -> None:
        pull_request = self.pull_request(
            state="open",
            labels=[],
            user={"type": "User", "login": "contributor"},
        )
        client = FakePullRequestApi([pull_request], [self.accepted_issue()])
        client.requested_reviewers[7] = set(community_reviewers.DEFAULT_REVIEWERS)

        community_lifecycle_github.sync_pull_request(client, self.REPO, 7)

        self.assertEqual(
            client.review_posts,
            [
                (
                    7,
                    community_reviewers.desired_workgroup_reviewer(
                        "wg/mom-routing", 7, "contributor"
                    ),
                )
            ],
        )

    def refused_review(self, roster, non_collaborators):
        pull_request = self.pull_request(
            state="open",
            labels=[],
            user={"type": "User", "login": "contributor"},
        )
        client = FakePullRequestApi([pull_request], [self.accepted_issue()])
        client.non_collaborators = set(non_collaborators)
        rosters = {"wg/mom-routing": roster}
        with mock.patch.dict(community_reviewers.WORKGROUP_REVIEWERS, rosters):
            community_lifecycle_github.sync_pull_request(client, self.REPO, 7)
        return client

    def test_refused_lead_passes_the_request_to_the_next(self) -> None:
        client = self.refused_review(("lead", "invited"), {"invited"})

        self.assertEqual(client.review_attempts, [(7, "invited"), (7, "lead")])
        self.assertEqual(client.review_posts, [(7, "lead")])
        self.assertEqual(
            client.comment_posts,
            [
                (
                    7,
                    community_lifecycle_github.workgroup_review_comment(
                        "wg/mom-routing", "lead", []
                    ),
                )
            ],
        )

    def test_refused_group_does_not_stop_the_sync(self) -> None:
        client = self.refused_review(("lead", "invited"), {"lead", "invited"})

        self.assertEqual(client.review_attempts, [(7, "invited"), (7, "lead")])
        self.assertEqual(client.review_posts, [])
        self.assertEqual(client.comment_posts, [])

    def test_requested_lead_is_announced_once(self) -> None:
        pull_request = self.pull_request(
            state="open",
            labels=[],
            user={"type": "User", "login": "contributor"},
        )
        client = FakePullRequestApi([pull_request], [self.accepted_issue()])
        rosters = {"wg/mom-routing": ("first", "second", "third", "fourth")}
        with mock.patch.dict(community_reviewers.WORKGROUP_REVIEWERS, rosters):
            community_lifecycle_github.sync_pull_request(client, self.REPO, 7)
            community_lifecycle_github.sync_pull_request(client, self.REPO, 7)

        self.assertEqual(client.review_posts, [(7, "fourth")])
        self.assertEqual(
            client.comment_posts,
            [
                (
                    7,
                    "<!-- community-lifecycle:workgroup-review -->\n"
                    "**Reviewer:** @fourth, MoM & Routing Lead\n\n"
                    "To speed up the review, mention them in a reply. "
                    "No response? Ask another Lead: `@first` or `@second`.",
                )
            ],
        )

    def test_review_comment_without_another_lead(self) -> None:
        body = community_lifecycle_github.workgroup_review_comment(
            "wg/enterprise-environment", "AayushSaini101", []
        )

        self.assertTrue(
            body.endswith(
                "**Reviewer:** @AayushSaini101, Enterprise & Environment Lead\n\n"
                "To speed up the review, mention them in a reply."
            )
        )

    def test_draft_or_unaccepted_pr_does_not_request_review(self) -> None:
        draft = self.pull_request(state="open", draft=True)
        unaccepted = self.pull_request(state="open", number=8, closes=[6])
        issue = {"number": 6, "labels": labels("needs-acceptance", "wg/mom-routing")}
        client = FakePullRequestApi([draft, unaccepted], [self.accepted_issue(), issue])

        community_lifecycle_github.sync_pull_request(client, self.REPO, 7)
        community_lifecycle_github.sync_pull_request(client, self.REPO, 8)

        self.assertEqual(client.review_posts, [])

    def owner_change(self, old_roster, new_roster, requested, comments=()):
        old_owner = "wg/mom-routing"
        new_owner = "wg/evaluation-quality"
        pull_request = self.pull_request(
            state="open",
            labels=labels(old_owner),
            user={"type": "User", "login": "contributor"},
        )
        issue = {**self.accepted_issue(), "labels": labels("accepted", new_owner)}
        client = FakePullRequestApi([pull_request], [issue])
        client.requested_reviewers[7] = set(requested)
        client.comments[7] = [dict(comment) for comment in comments]
        rosters = {old_owner: old_roster, new_owner: new_roster}
        with mock.patch.dict(community_reviewers.WORKGROUP_REVIEWERS, rosters):
            community_lifecycle_github.sync_pull_request(client, self.REPO, 7)
        return client

    def test_owner_change_replaces_outstanding_group_request(self) -> None:
        client = self.owner_change(("old-lead",), ("new-lead",), {"old-lead", "rootfs"})

        self.assertEqual(client.review_removals, [(7, "old-lead")])
        self.assertEqual(client.review_posts, [(7, "new-lead")])

    def test_owner_change_edits_the_review_comment(self) -> None:
        old_comment = community_lifecycle_github.workgroup_review_comment(
            "wg/mom-routing", "old-lead", []
        )
        client = self.owner_change(
            ("old-lead",),
            ("new-lead",),
            {"old-lead"},
            comments=[{"id": 1, "body": old_comment}],
        )

        self.assertEqual(client.comment_posts, [])
        self.assertEqual(
            client.comment_edits,
            [
                (
                    1,
                    community_lifecycle_github.workgroup_review_comment(
                        "wg/evaluation-quality", "new-lead", []
                    ),
                )
            ],
        )

    def test_owner_change_keeps_shared_and_repository_wide_requests(self) -> None:
        for previous, new_roster, posts in (
            ("shared-lead", ("new-lead", "shared-lead"), []),
            ("Xunzhuo", ("new-lead",), [(7, "new-lead")]),
        ):
            with self.subTest(previous=previous):
                client = self.owner_change((previous,), new_roster, {previous})

                self.assertEqual(client.review_removals, [])
                self.assertEqual(client.review_posts, posts)

    def test_synchronized_pull_request_does_not_repeat_review_request(self) -> None:
        pull_request = self.pull_request(
            state="open",
            labels=labels("pr/merge-ready", "wg/mom-routing"),
            signals={
                "review_decision": "APPROVED",
                "merge_state": "CLEAN",
                "checks_state": "SUCCESS",
            },
        )
        client = FakePullRequestApi([pull_request], [self.accepted_issue()])
        reviewer = community_reviewers.desired_workgroup_reviewer(
            "wg/mom-routing", 7, ""
        )
        client.requested_reviewers[7] = {reviewer}

        community_lifecycle_github.sync_pull_request(client, self.REPO, 7)

        self.assertEqual(
            client.requests,
            [
                ("GET", f"repos/{self.REPO}/pulls/7"),
                ("POST", "graphql"),
                ("GET", f"repos/{self.REPO}/issues/5"),
            ],
        )

    def test_skipped_review_signals_reconcile_live_admission_and_approval(self) -> None:
        for event_name, conclusion, decision, accepted, state in (
            ("pull_request_review", "skipped", "APPROVED", True, "pr/merge-ready"),
            (
                "pull_request_review",
                "skipped",
                "REVIEW_REQUIRED",
                True,
                "pr/needs-review",
            ),
            (
                "pull_request_review",
                "skipped",
                "CHANGES_REQUESTED",
                True,
                "pr/needs-author",
            ),
            ("pull_request_review", "skipped", "APPROVED", False, "pr/blocked"),
            ("pull_request_review", "failure", "APPROVED", True, "pr/blocked"),
            ("pull_request_review", "cancelled", "APPROVED", True, "pr/blocked"),
            ("pull_request", "skipped", "APPROVED", True, "pr/blocked"),
            ("pull_request", "failure", "APPROVED", True, "pr/blocked"),
            ("pull_request", "success", "APPROVED", True, "pr/merge-ready"),
        ):
            with self.subTest(
                event=event_name,
                conclusion=conclusion,
                decision=decision,
                accepted=accepted,
            ):
                pull_request = self.pull_request(
                    state="open",
                    labels=labels("wg/mom-routing"),
                    signals={
                        "review_decision": decision,
                        "merge_state": "CLEAN",
                        "checks_state": "SUCCESS",
                    },
                )
                issue = self.accepted_issue()
                if not accepted:
                    issue["labels"] = labels("needs-acceptance", "wg/mom-routing")
                client = FakePullRequestApi([pull_request], [issue])

                community_lifecycle_github.sync_pull_request_event(
                    client,
                    {
                        "repository": {"full_name": self.REPO},
                        "workflow_run": {
                            "event": event_name,
                            "conclusion": conclusion,
                            "pull_requests": [{"number": 7}],
                        },
                    },
                )

                self.assertEqual(
                    client.added & set(community_lifecycle.PR_STATE_LABELS), {state}
                )
                self.assertIn(("GET", f"repos/{self.REPO}/issues/5"), client.requests)

    def reconcile_previously_admitted(self, issue_labels):
        """Reconcile a merge-ready PR whose linked issue changed, then sweep again.

        The first pass follows an earlier successful Community run; the second is
        the hourly sweep over the labels the first pass wrote.
        """

        pull_request = self.pull_request(
            state="open",
            labels=labels("pr/merge-ready", "wg/mom-routing"),
            user={"type": "User", "login": "contributor"},
            signals={
                "review_decision": "APPROVED",
                "merge_state": "CLEAN",
                "checks_state": "SUCCESS",
            },
        )
        issue = {**self.accepted_issue(), "labels": labels(*issue_labels)}
        client = FakePullRequestApi([pull_request], [issue])
        rosters = {
            "wg/mom-routing": ("old-lead",),
            "wg/evaluation-quality": ("new-lead",),
        }
        with mock.patch.dict(community_reviewers.WORKGROUP_REVIEWERS, rosters):
            community_lifecycle_github.sync_pull_request_event(
                client,
                {
                    "repository": {"full_name": self.REPO},
                    "workflow_run": {
                        "event": "pull_request",
                        "conclusion": "success",
                        "pull_requests": [{"number": 7}],
                    },
                },
            )
            first = (set(client.added), set(client.removed), list(client.review_posts))
            current = {item["name"] for item in pull_request["labels"]}
            pull_request["labels"] = labels(
                *sorted((current - client.removed) | client.added)
            )
            client.added.clear()
            client.removed.clear()
            client.review_posts.clear()
            client.review_removals.clear()
            client.comment_posts.clear()
            client.comment_edits.clear()

            community_lifecycle_github.sync_pull_request(client, self.REPO, 7)

        self.assertEqual(client.added, set())
        self.assertEqual(client.removed, set())
        self.assertEqual(client.review_posts, [])
        self.assertEqual(client.review_removals, [])
        self.assertEqual(client.comment_posts, [])
        self.assertEqual(client.comment_edits, [])
        return first, {item["name"] for item in pull_request["labels"]}

    def test_reconciliation_blocks_on_current_admission_not_older_success(
        self,
    ) -> None:
        for name, issue_labels in (
            ("unaccepted", ("needs-acceptance", "wg/mom-routing")),
            ("missing owner", ("accepted",)),
            (
                "ambiguous owner",
                ("accepted", "wg/mom-routing", "wg/evaluation-quality"),
            ),
        ):
            with self.subTest(name):
                (added, removed, posts), final = self.reconcile_previously_admitted(
                    issue_labels
                )

                self.assertEqual(added, {"pr/blocked"})
                self.assertEqual(removed, {"pr/merge-ready", "wg/mom-routing"})
                self.assertEqual(posts, [])
                self.assertEqual(final, {"pr/blocked"})

    def test_reconciliation_moves_ownership_to_the_current_owner(self) -> None:
        (added, removed, posts), final = self.reconcile_previously_admitted(
            ("accepted", "wg/evaluation-quality")
        )

        self.assertEqual(added, {"wg/evaluation-quality"})
        self.assertEqual(removed, {"wg/mom-routing"})
        self.assertEqual(posts, [(7, "new-lead")])
        self.assertEqual(final, {"pr/merge-ready", "wg/evaluation-quality"})

    def test_release_blocker_label_follows_the_linked_issue(self) -> None:
        blocker = community_lifecycle_github.RELEASE_BLOCKER
        signals = {
            "review_decision": "APPROVED",
            "merge_state": "CLEAN",
            "checks_state": "SUCCESS",
        }
        blocking_issue = self.accepted_issue()
        blocking_issue["labels"] = labels("accepted", "wg/mom-routing", blocker)
        cases = [
            (
                labels("pr/merge-ready", "wg/mom-routing"),
                blocking_issue,
                {blocker},
                set(),
            ),
            (
                labels("pr/merge-ready", "wg/mom-routing", blocker),
                blocking_issue,
                set(),
                set(),
            ),
            (
                labels("pr/merge-ready", "wg/mom-routing", blocker),
                self.accepted_issue(),
                set(),
                {blocker},
            ),
        ]
        for current, issue, added, removed in cases:
            with self.subTest(current=current, added=added, removed=removed):
                pull_request = self.pull_request(
                    state="open", labels=current, signals=signals
                )
                client = FakePullRequestApi([pull_request], [issue])

                community_lifecycle_github.sync_pull_request(client, self.REPO, 7)

                self.assertEqual(client.added, added)
                self.assertEqual(client.removed, removed)

    def closed_listing(self, label: str, page: int = 1) -> str:
        return (
            f"repos/{self.REPO}/issues?state=closed"
            f"&labels={quote(label, safe='')}"
            f"&per_page={community_lifecycle_github.API_PAGE_SIZE}&page={page}"
        )

    def open_listing(self, page: int = 1) -> str:
        return (
            f"repos/{self.REPO}/pulls?state=open"
            f"&per_page={community_lifecycle_github.API_PAGE_SIZE}&page={page}"
        )

    def test_open_sweep_confirms_merge_state_even_without_a_label_change(self) -> None:
        # Like #3786, the batched DIRTY result agrees with the existing label.
        pull_request = self.pull_request(
            state="open",
            labels=labels("pr/needs-rebase", "wg/mom-routing"),
            signals={"review_decision": "REVIEW_REQUIRED", "merge_state": "CLEAN"},
        )
        client = FakePullRequestApi(
            [pull_request],
            [self.accepted_issue()],
            listings={self.open_listing(): [pull_request]},
        )
        client.batched_merge_states[7] = "DIRTY"

        community_lifecycle_github.sync_open_pull_requests(client, self.REPO)

        self.assertEqual(client.removed, {"pr/needs-rebase"})
        self.assertEqual(client.added, {"pr/needs-review"})
        self.assertEqual(
            client.requests[:5],
            [
                ("GET", self.open_listing()),
                ("POST", "graphql"),
                ("GET", f"repos/{self.REPO}/issues/5"),
                ("GET", f"repos/{self.REPO}/pulls/7"),
                ("POST", "graphql"),
            ],
        )

    def test_open_sweep_uses_fresh_linked_issue_after_confirmation(self) -> None:
        listed = self.pull_request(state="open", body="Related #5")
        current = {**listed, "closes": [], "body": "Related #6"}
        new_issue = {
            **self.accepted_issue(),
            "number": 6,
            "labels": labels("accepted", "wg/evaluation-quality"),
            "milestone": {"number": 9},
        }
        client = FakePullRequestApi(
            [current],
            [self.accepted_issue(), new_issue],
            listings={self.open_listing(): [listed]},
        )

        community_lifecycle_github.sync_open_pull_requests(client, self.REPO)

        self.assertIn("wg/evaluation-quality", client.added)
        self.assertIn("wg/mom-routing", client.removed)
        self.assertIn(
            ("PATCH", f"repos/{self.REPO}/issues/7", {"milestone": 9}), client.writes
        )
        self.assertIn(("GET", f"repos/{self.REPO}/issues/6"), client.requests)

    def test_open_sweep_missing_batch_node_uses_existing_lookup(self) -> None:
        missing = self.pull_request(number=7, state="open")
        draft = self.pull_request(number=8, state="open", draft=True, closes=[])
        client = FakePullRequestApi(
            [draft], listings={self.open_listing(): [missing, draft]}
        )

        community_lifecycle_github.sync_open_pull_requests(client, self.REPO)

        self.assertIn(("GET", f"repos/{self.REPO}/pulls/7"), client.requests)
        self.assertNotIn(("GET", f"repos/{self.REPO}/pulls/8"), client.requests)
        self.assertEqual(client.added, {"pr/needs-author"})

    def test_open_sweep_rechecks_a_pr_closed_since_listing(self) -> None:
        listed = self.pull_request(state="open", draft=True)
        closed = {**listed, "state": "closed"}
        client = FakePullRequestApi(
            [closed], [self.accepted_issue()], listings={self.open_listing(): [listed]}
        )

        community_lifecycle_github.sync_open_pull_requests(client, self.REPO)

        self.assertIn(("GET", f"repos/{self.REPO}/pulls/7"), client.requests)
        self.assertEqual(client.added, set())
        self.assertEqual(client.removed, {"pr/blocked"})

    @mock.patch.object(community_lifecycle_github, "API_PAGE_SIZE", 10)
    def test_pr_sweep_reduces_reads_and_preserves_mutations(self) -> None:
        first_open_number = 100
        first_unchanged_number = 103
        first_draft_number = 115
        open_prs = [
            self.pull_request(
                number=number,
                state="open",
                draft=number >= first_draft_number,
                closes=[5, 6] if number == first_open_number else [5],
                labels=labels(
                    "wg/mom-routing",
                    (
                        "pr/needs-author"
                        if number >= first_draft_number
                        else "pr/needs-rebase"
                    ),
                ),
                signals={
                    "review_decision": "REVIEW_REQUIRED",
                    "merge_state": (
                        "CLEAN" if number < first_unchanged_number else "DIRTY"
                    ),
                },
            )
            for number in range(first_open_number, 129)
        ]
        closed_prs = [self.pull_request(number=number) for number in range(129, 133)]
        open_listings = {
            self.open_listing(page): open_prs[start : start + 10]
            for page, start in enumerate(range(0, len(open_prs), 10), 1)
        }
        listings = {
            **open_listings,
            self.closed_listing("pr/blocked"): [
                {"number": pr["number"], "pull_request": {}} for pr in closed_prs
            ],
        }
        issues = [
            self.accepted_issue(),
            {**self.accepted_issue(), "number": 6, "milestone": {"number": 9}},
        ]
        legacy = FakePullRequestApi(open_prs + closed_prs, issues, listings)
        optimized = FakePullRequestApi(open_prs + closed_prs, issues, listings)
        for client in (legacy, optimized):
            lead = community_reviewers.WORKGROUP_REVIEWERS["wg/mom-routing"][0]
            client.requested_reviewers = {
                number: {lead}
                for number in range(first_open_number + 1, first_draft_number)
            }

        for endpoint in open_listings:
            for pr in legacy.request(endpoint):
                community_lifecycle_github.sync_pull_request(
                    legacy, self.REPO, pr["number"]
                )
        community_lifecycle_github.sync_labeled_closed_pull_requests(legacy, self.REPO)
        community_lifecycle_github.sync_pull_request_queue(
            optimized, {"repository": {"full_name": self.REPO}}
        )

        self.assertEqual(optimized.writes, legacy.writes)
        self.assertTrue(legacy.review_posts)
        self.assertIn(
            (
                "PATCH",
                f"repos/{self.REPO}/issues/{first_open_number}",
                {"milestone": 9},
            ),
            legacy.writes,
        )

        def reads(client):
            return sum(
                method == "GET" or endpoint == "graphql"
                for method, endpoint in client.requests
            )

        self.assertEqual(reads(legacy), 110)
        self.assertEqual(reads(optimized), 88)

    def test_closed_sweep_reconciles_only_labeled_pull_requests(self) -> None:
        listings = {
            self.closed_listing("pr/blocked"): [
                {"number": 11, "pull_request": {}},
                {"number": 9},
            ],
            self.closed_listing("pr/needs-review"): [
                {"number": 12, "pull_request": {}},
                {"number": 11, "pull_request": {}},
            ],
        }
        client = FakePullRequestApi(
            [
                self.pull_request(number=number, labels=labels("pr/blocked"), closes=[])
                for number in (11, 12)
            ],
            listings=listings,
        )

        community_lifecycle_github.sync_labeled_closed_pull_requests(client, self.REPO)

        self.assertEqual(client.fetched, [11, 12])
        self.assertIn("pr/blocked", client.removed)

    def test_closed_sweep_pages_through_full_listings(self) -> None:
        page_size = community_lifecycle_github.API_PAGE_SIZE
        numbers = list(range(100, 100 + page_size))
        listings = {
            self.closed_listing("pr/blocked"): [
                {"number": number, "pull_request": {}} for number in numbers
            ],
            self.closed_listing("pr/blocked", page=2): [
                {"number": 99, "pull_request": {}}
            ],
        }
        client = FakePullRequestApi(
            [
                self.pull_request(number=number, labels=labels("pr/blocked"), closes=[])
                for number in [99, *numbers]
            ],
            listings=listings,
        )

        community_lifecycle_github.sync_labeled_closed_pull_requests(client, self.REPO)

        self.assertEqual(client.fetched, [99, *numbers])


if __name__ == "__main__":
    unittest.main()
