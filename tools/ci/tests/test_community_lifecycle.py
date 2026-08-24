import importlib.util
import sys
import unittest
from pathlib import Path

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


def labels(*names: str) -> list[dict[str, str]]:
    return [{"name": name} for name in names]


class CommunityLifecycleTests(unittest.TestCase):
    def test_form_workgroup_is_parsed(self) -> None:
        body = """### Goal

Make routing better.

### Proposed Workgroup

MoM & Routing Intelligence
"""
        self.assertEqual(
            community_lifecycle.proposed_workgroup(body),
            "wg/mom-routing-intelligence",
        )

    def test_new_issue_enters_acceptance_with_proposed_owner(self) -> None:
        issue = {
            "body": "### Proposed Workgroup\n\nQuality & Release\n",
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
            {"needs-acceptance", "wg/quality-release"},
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
                "wg/data-plane-deployment",
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
                "wg/data-plane-deployment",
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
            "labels": labels("accepted", "wg/quality-release"),
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
            "labels": labels("accepted", "wg/quality-release"),
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
                    "wg/platform-operations",
                ),
                "milestone": {"number": 7},
            }
        ]
        evaluation = community_lifecycle.evaluate_pull_request(pull_request, linked)
        self.assertTrue(evaluation.valid)
        self.assertEqual(evaluation.state_label, "pr/needs-review")
        self.assertEqual(evaluation.owner_label, "wg/platform-operations")
        self.assertEqual(evaluation.milestone_number, 7)
        self.assertTrue(evaluation.release_blocker)

    def test_pr_review_and_runtime_signals_select_one_action_state(self) -> None:
        pull_request = {"draft": False, "user": {"type": "User"}}
        linked = [
            {
                "labels": labels("accepted", "wg/platform-operations"),
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
                    "labels": labels("accepted", "wg/quality-release"),
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
        self.assertEqual(evaluation.owner_label, "wg/quality-release")

    def test_pr_rejects_multiple_accepted_workgroup_owners(self) -> None:
        pull_request = {"draft": False, "user": {"type": "User"}}
        linked = [
            {
                "labels": labels("accepted", "wg/platform-operations"),
                "milestone": None,
            },
            {
                "labels": labels("accepted", "wg/quality-release"),
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
                        "wg/quality-release",
                    ),
                    "milestone": {"number": 7},
                }
            ],
        )
        self.assertEqual(evaluation.state_label, "pr/needs-author")
        self.assertEqual(evaluation.owner_label, "wg/quality-release")
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


if __name__ == "__main__":
    unittest.main()
