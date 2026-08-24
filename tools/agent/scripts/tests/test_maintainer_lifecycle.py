import importlib
import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

maintainer_lifecycle = importlib.import_module("maintainer_lifecycle")


class MaintainerLifecycleTests(unittest.TestCase):
    def test_proposals_cover_only_missing_queue_labels(self) -> None:
        snapshot = {
            "raw": {
                "issues": [
                    {"number": 1, "labels": []},
                    {"number": 2, "labels": [{"name": "accepted"}]},
                ]
            },
            "groups": {
                "issues": {
                    "stale": [
                        {"number": 1, "labels": []},
                        {"number": 3, "labels": [{"name": "stale"}]},
                    ]
                },
                "pull_requests": {
                    "needs-rebase": [],
                    "review-now": [
                        {"number": 4, "labels": []},
                        {"number": 5, "labels": [{"name": "pr/needs-review"}]},
                    ],
                    "needs-author": [],
                    "unblock": [],
                    "merge-candidate": [],
                    "close-candidate": [],
                },
            },
        }
        policy = {
            "labels": {
                "lifecycle": {
                    "accepted": "accepted",
                    "needs_acceptance": "needs-acceptance",
                    "stale": "stale",
                },
                "pr_state": {
                    "needs_rebase": "pr/needs-rebase",
                    "needs_review": "pr/needs-review",
                    "needs_author": "pr/needs-author",
                    "blocked": "pr/blocked",
                    "merge_ready": "pr/merge-ready",
                    "close_candidate": "pr/close-candidate",
                },
            }
        }

        self.assertEqual(
            maintainer_lifecycle.proposed_actions(snapshot, policy),
            [
                {
                    "action": "label_issue",
                    "target": "#1",
                    "labels": ["needs-acceptance"],
                },
                {
                    "action": "label_issue",
                    "target": "#1",
                    "labels": ["stale"],
                },
                {
                    "action": "label_pr",
                    "target": "#4",
                    "labels": ["pr/needs-review"],
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
