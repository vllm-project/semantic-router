import argparse
import importlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

maintainer_board = importlib.import_module("maintainer_board")


class MaintainerBoardIssueCreationTests(unittest.TestCase):
    def test_release_issue_creation_has_no_parallel_track_taxonomy(self) -> None:
        args = argparse.Namespace(
            release_plan="unused.md",
            include_matched=True,
            labels="wg/evaluation-quality",
            prefix="[Release]",
            milestone=None,
        )
        policy = {
            "labels": {"lifecycle": {"needs_acceptance": "needs-acceptance"}},
            "public_artifact_policy": {},
        }
        with mock.patch.object(
            maintainer_board, "open_release_tasks", return_value=["Qualify v1"]
        ):
            actions = maintainer_board.create_issue_actions(args, policy)

        self.assertEqual(
            actions[0]["labels"],
            ["needs-acceptance", "wg/evaluation-quality"],
        )
        self.assertNotIn("Release Track", actions[0]["body"])

    def test_track_argument_is_rejected_instead_of_kept_for_compatibility(self) -> None:
        parser = maintainer_board.build_parser()
        with mock.patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "create-issues",
                    "--release-plan",
                    "release.md",
                    "--track",
                    "ops",
                ]
            )


if __name__ == "__main__":
    unittest.main()
