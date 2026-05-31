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
