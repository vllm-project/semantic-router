import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

maintainer_board = importlib.import_module("maintainer_board")
maintainer_ga_readiness = importlib.import_module("maintainer_ga_readiness")


def empty_snapshot() -> dict:
    return {
        "generated_at": "2026-05-31T00:00:00Z",
        "active_milestone": "v0.3 - Themis",
        "raw": {"milestones": []},
        "groups": {
            "pull_requests": {
                "merge-candidate": [],
                "review-now": [],
                "unblock": [],
                "needs-rebase": [],
                "close-candidate": [],
            },
            "issues": {
                "milestone-bound": [],
                "milestone-candidate": [],
                "incoming-triage": [],
                "backlog": [],
                "stale": [],
            },
        },
        "proposed_actions": [],
    }


class MaintainerBoardGAReadinessTests(unittest.TestCase):
    def test_gh_json_retries_transient_gateway_errors(self) -> None:
        transient = maintainer_board.subprocess.CompletedProcess(
            ["gh", "pr", "list"],
            1,
            stdout="",
            stderr="HTTP 502: 502 Bad Gateway (https://api.github.com/graphql)",
        )
        success = maintainer_board.subprocess.CompletedProcess(
            ["gh", "pr", "list"],
            0,
            stdout='{"ok": true}',
            stderr="",
        )

        with (
            mock.patch.object(
                maintainer_board.subprocess,
                "run",
                side_effect=[transient, success],
            ) as run,
            mock.patch.object(maintainer_board.time, "sleep") as sleep,
        ):
            payload = maintainer_board.gh_json(["pr", "list"])

        self.assertEqual(payload, {"ok": True})
        self.assertEqual(run.call_count, 2)
        sleep.assert_called_once_with(1)

    def test_fetch_github_state_fetches_pr_status_rollups_separately(self) -> None:
        policy = {"default_issue_limit": 100, "default_pr_limit": 50}
        prs = [
            {"number": 12, "title": "ready"},
            {"number": 13, "title": "blocked"},
        ]

        with mock.patch.object(
            maintainer_board,
            "gh_json",
            side_effect=[
                [],
                prs,
                {"statusCheckRollup": [{"conclusion": "SUCCESS"}]},
                {"statusCheckRollup": [{"conclusion": "FAILURE"}]},
                [{"title": "v0.4"}],
                [{"name": "bug"}],
            ],
        ) as gh_json:
            state = maintainer_board.fetch_github_state(policy)

        self.assertEqual(
            state["pull_requests"],
            [
                {
                    "number": 12,
                    "title": "ready",
                    "statusCheckRollup": [{"conclusion": "SUCCESS"}],
                },
                {
                    "number": 13,
                    "title": "blocked",
                    "statusCheckRollup": [{"conclusion": "FAILURE"}],
                },
            ],
        )
        pr_list_call = gh_json.call_args_list[1].args[0]
        pr_list_fields = pr_list_call[pr_list_call.index("--json") + 1]
        self.assertIn(
            "reviewDecision,mergeStateStatus,headRefName,baseRefName", pr_list_fields
        )
        self.assertNotIn("statusCheckRollup", pr_list_fields)
        self.assertEqual(
            gh_json.call_args_list[2].args[0],
            ["pr", "view", "12", "--json", "statusCheckRollup"],
        )
        self.assertEqual(
            gh_json.call_args_list[3].args[0],
            ["pr", "view", "13", "--json", "statusCheckRollup"],
        )

    def test_latest_ga_readiness_report_summarizes_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reports = root / ".agent-harness" / "reports" / "session-routing-ga"
            (reports / "20260530T000000Z").mkdir(parents=True)
            (reports / "20260530T000000Z" / "ga-readiness.json").write_text(
                "{not json",
                encoding="utf-8",
            )
            latest = reports / "20260531T000000Z"
            latest.mkdir()
            (latest / "ga-readiness.json").write_text(
                json.dumps(
                    {
                        "generated_at": "2026-05-31T00:00:00Z",
                        "ga_ready": False,
                        "blocker_count": 1,
                        "blockers": [
                            {
                                "id": "branch_image_amd_validation",
                                "title": "Branch-image AMD benchmark",
                                "status": "missing",
                                "failures": ["not shown in board summary"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(maintainer_ga_readiness, "REPO_ROOT", root),
                mock.patch.object(
                    maintainer_ga_readiness, "GA_READINESS_ROOT", reports
                ),
            ):
                summary = maintainer_ga_readiness.latest_ga_readiness_report()

        self.assertEqual(
            summary,
            {
                "path": (
                    ".agent-harness/reports/session-routing-ga/"
                    "20260531T000000Z/ga-readiness.json"
                ),
                "generated_at": "2026-05-31T00:00:00Z",
                "ga_ready": False,
                "blocker_count": 1,
                "blockers": [
                    {
                        "id": "branch_image_amd_validation",
                        "title": "Branch-image AMD benchmark",
                        "status": "missing",
                    }
                ],
            },
        )

    def test_latest_ga_readiness_report_prefers_newer_generated_at(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reports = root / ".agent-harness" / "reports" / "session-routing-ga"
            current = reports / "current"
            current.mkdir(parents=True)
            (current / "ga-readiness.json").write_text(
                json.dumps(
                    {
                        "generated_at": "2026-05-30T00:00:00Z",
                        "ga_ready": False,
                        "blocker_count": 1,
                        "requirements": [
                            {
                                "id": "old_current",
                                "title": "Old current",
                                "status": "blocked",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            latest = reports / "20260531T222239Z"
            latest.mkdir()
            (latest / "ga-readiness.json").write_text(
                json.dumps(
                    {
                        "generated_at": "2026-05-31T22:22:39Z",
                        "ga_ready": False,
                        "blocker_count": 1,
                        "blockers": [
                            {
                                "id": "branch_image_amd_validation",
                                "title": "Branch-image AMD benchmark",
                                "status": "missing",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(maintainer_ga_readiness, "REPO_ROOT", root),
                mock.patch.object(
                    maintainer_ga_readiness, "GA_READINESS_ROOT", reports
                ),
            ):
                summary = maintainer_ga_readiness.latest_ga_readiness_report()

        self.assertEqual(
            summary["path"],
            ".agent-harness/reports/session-routing-ga/20260531T222239Z/"
            "ga-readiness.json",
        )
        self.assertEqual(
            summary["blockers"],
            [
                {
                    "id": "branch_image_amd_validation",
                    "title": "Branch-image AMD benchmark",
                    "status": "missing",
                }
            ],
        )

    def test_today_and_release_report_include_ga_readiness(self) -> None:
        snapshot = empty_snapshot()
        snapshot["session_routing_ga_readiness"] = {
            "path": ".agent-harness/reports/session-routing-ga/latest/ga-readiness.json",
            "generated_at": "2026-05-31T00:00:00Z",
            "ga_ready": False,
            "blocker_count": 1,
            "blockers": [
                {
                    "id": "cache_token_reporting",
                    "title": "Cache-token reporting",
                    "status": "blocked",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            plan = Path(temp_dir) / "plan.md"
            plan.write_text("# Plan\n\n- [ ] Run GA evidence\n", encoding="utf-8")

            today = maintainer_board.render_today(snapshot)
            release = maintainer_board.render_release_readiness(snapshot, plan)

        for rendered in (today, release):
            self.assertIn("## Session-Aware GA Readiness", rendered)
            self.assertIn("- ga_ready: false", rendered)
            self.assertIn(
                "- blocked: Cache-token reporting (`cache_token_reporting`)",
                rendered,
            )
        self.assertIn("## Release Blockers", release)
        self.assertIn("### Session-Aware GA", release)

    def test_release_report_uses_plan_issue_anchors_when_snapshot_is_limited(
        self,
    ) -> None:
        snapshot = empty_snapshot()
        snapshot["groups"]["issues"]["milestone-bound"] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            plan = Path(temp_dir) / "plan.md"
            plan.write_text(
                "\n".join(
                    [
                        "# Plan",
                        "",
                        "## Current issue anchors",
                        "",
                        "- [#1753](https://github.com/vllm-project/semantic-router/issues/1753) session-aware model switching.",
                        "",
                        "## Task List",
                        "",
                        "- [ ] `V030004` Land the Session-Aware Agentic Routing GA evaluation package.",
                    ]
                ),
                encoding="utf-8",
            )

            release = maintainer_board.render_release_readiness(snapshot, plan)

        self.assertIn("### V030004", release)
        self.assertIn("#1753 session-aware model switching", release)
        self.assertIn(
            "## Plan Tasks Without Matching Milestone Issue\n\n- none", release
        )

    def test_blocker_summary_falls_back_to_blocked_requirements(self) -> None:
        summary = maintainer_ga_readiness.blocker_summaries(
            {
                "requirements": [
                    {
                        "id": "synthetic_policy_matrix",
                        "title": "Synthetic policy matrix",
                        "status": "passed",
                    },
                    {
                        "id": "cache_token_reporting",
                        "title": "Cache-token reporting",
                        "status": "blocked",
                    },
                    {
                        "id": "branch_image_amd_validation",
                        "title": "Branch-image AMD benchmark",
                        "status": "missing",
                    },
                ]
            }
        )

        self.assertEqual(
            summary,
            [
                {
                    "id": "cache_token_reporting",
                    "title": "Cache-token reporting",
                    "status": "blocked",
                },
                {
                    "id": "branch_image_amd_validation",
                    "title": "Branch-image AMD benchmark",
                    "status": "missing",
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
