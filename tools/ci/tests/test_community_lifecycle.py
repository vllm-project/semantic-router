import importlib.util
import sys
import unittest
from pathlib import Path
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

    def test_closed_issue_drops_its_delivery_state_labels(self) -> None:
        issue = {
            "state": "closed",
            "labels": labels(
                "needs-acceptance",
                "ready-for-dev",
                "in-progress",
                "help wanted",
                "good first issue",
                "wg/mom-routing",
            ),
            "assignees": [],
            "milestone": None,
        }
        plan = community_lifecycle.plan_issue(
            issue,
            event_action="closed",
            event_label=None,
            actor_can_manage=False,
        )
        self.assertEqual(
            plan.remove_labels,
            set(community_lifecycle.ISSUE_DELIVERY_STATE_LABELS),
        )
        self.assertEqual(plan.add_labels, set())
        self.assertNotIn("wg/mom-routing", plan.remove_labels)

    def test_closed_issue_keeps_owner_priority_and_record_labels(self) -> None:
        issue = {
            "title": "[Epic] Restore something",
            "state": "closed",
            "labels": labels(
                "needs-acceptance",
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
            issue,
            event_action="closed",
            event_label=None,
            actor_can_manage=False,
        )
        self.assertEqual(plan.remove_labels, {"needs-acceptance"})
        for durable in (
            "accepted",
            "wg/mom-routing",
            "priority/P1",
            "backlog",
            "release-blocker",
            "epic",
        ):
            self.assertNotIn(durable, plan.remove_labels)

    def test_closed_issue_with_no_delivery_state_label_is_a_no_op(self) -> None:
        issue = {
            "state": "closed",
            "labels": labels("accepted", "wg/mom-routing"),
            "assignees": [],
            "milestone": None,
        }
        plan = community_lifecycle.plan_issue(
            issue,
            event_action="closed",
            event_label=None,
            actor_can_manage=False,
        )
        self.assertEqual(plan.remove_labels, set())
        self.assertEqual(plan.add_labels, set())

    def test_open_issue_is_unaffected_by_the_closed_branch(self) -> None:
        issue = {
            "state": "open",
            "labels": labels("needs-acceptance"),
            "assignees": [],
            "milestone": None,
        }
        plan = community_lifecycle.plan_issue(
            issue,
            event_action="edited",
            event_label=None,
            actor_can_manage=False,
        )
        self.assertNotIn("needs-acceptance", plan.remove_labels)

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
        self.fetched: list[int] = []
        self.added: set[str] = set()
        self.removed: set[str] = set()

    def request(
        self,
        endpoint: str,
        *,
        method: str = "GET",
        payload=None,
        ignore_not_found: bool = False,
    ):
        if endpoint == "graphql":
            return self.graphql(payload)
        if "?" in endpoint:
            return self.listings.get(endpoint, [])
        parts = endpoint.split("/")
        number = int(parts[4])
        if parts[3] == "pulls":
            self.fetched.append(number)
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
        pull_request = self.pull_requests[payload["variables"]["number"]]
        if "closingIssuesReferences" in payload["query"]:
            nodes = [{"number": number} for number in pull_request.get("closes", ())]
            return {
                "data": {
                    "repository": {
                        "pullRequest": {"closingIssuesReferences": {"nodes": nodes}}
                    }
                }
            }
        signals = pull_request.get("signals", {})
        rollup = {"state": signals.get("checks_state")}
        return {
            "data": {
                "repository": {
                    "pullRequest": {
                        "reviewDecision": signals.get("review_decision"),
                        "mergeStateStatus": signals.get("merge_state"),
                        "commits": {
                            "nodes": [{"commit": {"statusCheckRollup": rollup}}]
                        },
                    }
                }
            }
        }


class PullRequestReconciliationTests(unittest.TestCase):
    REPO = "acme/router"

    def pull_request(self, **overrides):
        item = {
            "number": 7,
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

    def closed_listing(self, label: str, page: int = 1) -> str:
        return (
            f"repos/{self.REPO}/issues?state=closed"
            f"&labels={quote(label, safe='')}"
            f"&per_page={community_lifecycle_github.API_PAGE_SIZE}&page={page}"
        )

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


class FakeIssueApi:
    """Serve the endpoints issue reconciliation calls and record its writes."""

    def __init__(self, issues, listings=None) -> None:
        self.issues = {int(item["number"]): item for item in issues}
        self.listings = listings or {}
        self.fetched: list[int] = []
        self.added: set[str] = set()
        self.removed: set[str] = set()

    def request(
        self,
        endpoint: str,
        *,
        method: str = "GET",
        payload=None,
        ignore_not_found: bool = False,
    ):
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

    def test_open_issue_keeps_its_delivery_state_label(self) -> None:
        client = FakeIssueApi(
            [self.closed_issue(state="open", labels=labels("needs-acceptance"))]
        )

        community_lifecycle_github.sync_issue(client, self.REPO, 20)

        self.assertNotIn("needs-acceptance", client.removed)

    def test_missing_issue_is_a_no_op(self) -> None:
        client = FakeIssueApi([])

        community_lifecycle_github.sync_issue(client, self.REPO, 404)

        self.assertEqual(client.removed, set())
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

    def test_closed_sweep_covers_every_delivery_state_label(self) -> None:
        listings = {
            self.closed_listing("help wanted"): [
                {"number": 30, "labels": labels("help wanted", "good first issue")}
            ],
            self.closed_listing("good first issue"): [
                {"number": 30, "labels": labels("help wanted", "good first issue")}
            ],
        }
        client = FakeIssueApi(
            [
                self.closed_issue(
                    number=30, labels=labels("help wanted", "good first issue")
                )
            ],
            listings=listings,
        )

        community_lifecycle_github.sync_labeled_closed_issues(client, self.REPO)

        self.assertEqual(client.fetched, [30])
        self.assertEqual(client.removed, {"help wanted", "good first issue"})

    def test_closed_sweep_pages_through_full_listings(self) -> None:
        page_size = community_lifecycle_github.API_PAGE_SIZE
        numbers = list(range(200, 200 + page_size))
        listings = {
            self.closed_listing("needs-acceptance"): [
                {"number": number} for number in numbers
            ],
            self.closed_listing("needs-acceptance", page=2): [{"number": 199}],
        }
        client = FakeIssueApi(
            [
                self.closed_issue(number=number, labels=labels("needs-acceptance"))
                for number in [199, *numbers]
            ],
            listings=listings,
        )

        community_lifecycle_github.sync_labeled_closed_issues(client, self.REPO)

        self.assertEqual(client.fetched, [199, *numbers])


if __name__ == "__main__":
    unittest.main()
