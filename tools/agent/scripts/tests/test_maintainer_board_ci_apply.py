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

ci_apply = importlib.import_module("maintainer_board_ci_apply")
maintainer_board = importlib.import_module("maintainer_board")


def sample_policy() -> dict:
    return maintainer_board.load_policy()


class MaintainerBoardCIApplyValidationTests(unittest.TestCase):
    def test_validate_accepts_sync_generated_label_actions(self) -> None:
        policy = sample_policy()
        actions = [
            {
                "action": "label_issue",
                "target": "#123",
                "labels": [policy["labels"]["lifecycle"]["needs_triage"]],
            },
            {
                "action": "label_pr",
                "target": "#456",
                "labels": [policy["labels"]["pr_state"]["needs_rebase"]],
            },
        ]
        validated = ci_apply.validate_ci_apply_actions(actions, policy)
        self.assertEqual(validated, actions)

    def test_validate_rejects_create_issue(self) -> None:
        policy = sample_policy()
        with self.assertRaisesRegex(ValueError, "unsupported action 'create_issue'"):
            ci_apply.validate_ci_apply_actions(
                [
                    {
                        "action": "create_issue",
                        "title": "Synthetic",
                        "body": "Body",
                        "labels": ["help wanted"],
                    }
                ],
                policy,
            )

    def test_validate_rejects_malformed_target(self) -> None:
        policy = sample_policy()
        with self.assertRaisesRegex(ValueError, "invalid target"):
            ci_apply.validate_ci_apply_actions(
                [
                    {
                        "action": "label_issue",
                        "target": "123",
                        "labels": [policy["labels"]["lifecycle"]["stale"]],
                    }
                ],
                policy,
            )

    def test_validate_rejects_disallowed_label(self) -> None:
        policy = sample_policy()
        with self.assertRaisesRegex(ValueError, "disallowed label 'help wanted'"):
            ci_apply.validate_ci_apply_actions(
                [
                    {
                        "action": "label_issue",
                        "target": "#1",
                        "labels": ["help wanted"],
                    }
                ],
                policy,
            )

    def test_validate_rejects_non_list_root(self) -> None:
        policy = sample_policy()
        with self.assertRaisesRegex(ValueError, "must be a JSON list"):
            ci_apply.validate_ci_apply_actions({"action": "label_issue"}, policy)

    def test_validate_rejects_empty_labels(self) -> None:
        policy = sample_policy()
        with self.assertRaisesRegex(ValueError, "non-empty labels list"):
            ci_apply.validate_ci_apply_actions(
                [{"action": "label_issue", "target": "#1", "labels": []}],
                policy,
            )


class MaintainerBoardCIApplyExecutionTests(unittest.TestCase):
    def test_apply_reports_partial_failures(self) -> None:
        policy = sample_policy()
        actions = [
            {
                "action": "label_issue",
                "target": "#1",
                "labels": [policy["labels"]["lifecycle"]["needs_triage"]],
            },
            {
                "action": "label_pr",
                "target": "#2",
                "labels": [policy["labels"]["pr_state"]["close_candidate"]],
            },
        ]

        def fake_runner(command, cwd, check, capture_output, text):
            if command[-1] == policy["labels"]["pr_state"]["close_candidate"]:
                raise ci_apply.subprocess.CalledProcessError(
                    1, command, stderr="label add failed"
                )
            return mock.Mock(returncode=0, stdout="", stderr="")

        results = ci_apply.apply_ci_actions(actions, policy, runner=fake_runner)
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].success)
        self.assertFalse(results[1].success)
        self.assertIn("label add failed", results[1].error or "")

        summary = ci_apply.render_apply_summary(
            results, source_run_id="999", actions_file=Path("proposed-actions.json")
        )
        self.assertIn("Attempted: **2**", summary)
        self.assertIn("Succeeded: **1**", summary)
        self.assertIn("Failed: **1**", summary)
        self.assertIn("label add failed", summary)

    def test_run_ci_apply_returns_non_zero_on_partial_failure(self) -> None:
        policy = sample_policy()
        actions = [
            {
                "action": "label_issue",
                "target": "#1",
                "labels": [policy["labels"]["lifecycle"]["stale"]],
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            actions_file = Path(temp_dir) / "proposed-actions.json"
            actions_file.write_text(json.dumps(actions), encoding="utf-8")
            summary_file = Path(temp_dir) / "summary.md"

            def failing_runner(command, cwd, check, capture_output, text):
                raise ci_apply.subprocess.CalledProcessError(1, command, stderr="boom")

            exit_code = ci_apply.run_ci_apply(
                actions_file,
                source_run_id="12345",
                summary_path=summary_file,
                runner=failing_runner,
            )

            self.assertEqual(exit_code, 1)
            summary = summary_file.read_text(encoding="utf-8")
            self.assertIn("Failed: **1**", summary)
            self.assertIn("boom", summary)

    def test_run_ci_apply_empty_payload_is_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            actions_file = Path(temp_dir) / "proposed-actions.json"
            actions_file.write_text("[]", encoding="utf-8")
            summary_file = Path(temp_dir) / "summary.md"

            exit_code = ci_apply.run_ci_apply(
                actions_file,
                source_run_id="12345",
                summary_path=summary_file,
            )

            self.assertEqual(exit_code, 0)
            self.assertIn(
                "No proposed actions to apply.",
                summary_file.read_text(encoding="utf-8"),
            )


class MaintainerBoardCIApplyGuardTests(unittest.TestCase):
    def test_validate_source_run_id_requires_numeric_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "source_run_id is required"):
            ci_apply.validate_source_run_id("")
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            ci_apply.validate_source_run_id("abc")

    def test_main_rejects_invalid_source_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            actions_file = Path(temp_dir) / "proposed-actions.json"
            actions_file.write_text("[]", encoding="utf-8")
            exit_code = ci_apply.main(
                ["--actions", str(actions_file), "--source-run-id", "not-a-run"]
            )
        self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()
