import importlib
import sys
import unittest
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

review_gate = importlib.import_module("agent_memory_review_gate")


class AgentMemoryReviewGateTests(unittest.TestCase):
    def test_missing_required_brief_fails_before_ai_review(self) -> None:
        result = review_gate.evaluate_gate(
            classifier={
                "memory_required": True,
                "memory_present": False,
                "memory_invalid": False,
                "gate_passed": False,
                "gate_reason": "Review brief is required",
            },
            review_response=None,
        )

        self.assertFalse(result.gate_passed)
        self.assertEqual(result.review_brief_match, "invalid-or-missing")
        self.assertIn("FAIL", result.comment_body)
        self.assertIn("Review brief is required", result.gate_reason or "")

    def test_invalid_brief_fails_even_for_small_pr(self) -> None:
        result = review_gate.evaluate_gate(
            classifier={
                "memory_required": False,
                "memory_present": False,
                "memory_invalid": True,
                "invalid_reason": "Review brief path must match docs/agent/reviews",
            },
            review_response=None,
        )

        self.assertFalse(result.gate_passed)
        self.assertIn("must match", result.gate_reason or "")

    def test_matching_brief_passes_with_parseable_verdict(self) -> None:
        result = review_gate.evaluate_gate(
            classifier={
                "memory_required": True,
                "memory_present": True,
                "memory_invalid": False,
                "gate_passed": True,
            },
            review_response="\n".join(
                [
                    "## Brief / Diff Consistency",
                    "",
                    "- Review brief matches diff: yes",
                    "- Hard gate: pass",
                    "",
                    "The brief matches the touched files.",
                ]
            ),
        )

        self.assertTrue(result.gate_passed)
        self.assertEqual(result.review_brief_match, "match")

    def test_mismatched_brief_fails_with_parseable_verdict(self) -> None:
        result = review_gate.evaluate_gate(
            classifier={
                "memory_required": False,
                "memory_present": True,
                "memory_invalid": False,
                "gate_passed": True,
            },
            review_response="\n".join(
                [
                    "## Brief / Diff Consistency",
                    "",
                    "- Review brief matches diff: no",
                    "- Hard gate: fail",
                    "- Reason: brief describes workflow changes but diff changes E2E mock",
                    "",
                    "## Findings",
                    "",
                    "1. Brief/diff mismatch.",
                ]
            ),
        )

        self.assertFalse(result.gate_passed)
        self.assertEqual(result.review_brief_match, "mismatch")
        self.assertIn("workflow changes", result.gate_reason or "")
        self.assertIn("FAIL", result.comment_body)

    def test_present_brief_without_parseable_verdict_fails_closed(self) -> None:
        result = review_gate.evaluate_gate(
            classifier={
                "memory_required": False,
                "memory_present": True,
                "memory_invalid": False,
                "gate_passed": True,
            },
            review_response="The brief seems unrelated, but no verdict line is present.",
        )

        self.assertFalse(result.gate_passed)
        self.assertEqual(result.review_brief_match, "unknown")
        self.assertIn("parseable", result.gate_reason or "")

    def test_present_brief_not_applicable_verdict_fails_closed(self) -> None:
        result = review_gate.evaluate_gate(
            classifier={
                "memory_required": False,
                "memory_present": True,
                "memory_invalid": False,
                "gate_passed": True,
            },
            review_response="\n".join(
                [
                    "## Brief / Diff Consistency",
                    "",
                    "- Review brief matches diff: not-applicable",
                    "- Hard gate: pass",
                ]
            ),
        )

        self.assertFalse(result.gate_passed)
        self.assertEqual(result.review_brief_match, "unknown")
        self.assertIn("not-applicable", result.gate_reason or "")

    def test_classifier_only_present_brief_passes_without_ai_review(self) -> None:
        result = review_gate.evaluate_gate(
            classifier={
                "memory_required": True,
                "memory_present": True,
                "memory_invalid": False,
                "gate_passed": True,
            },
            review_response=None,
            classifier_only=True,
        )

        self.assertTrue(result.gate_passed)
        self.assertEqual(result.review_brief_match, "not-evaluated")

    def test_ai_review_unavailable_can_be_advisory_when_flag_enabled(self) -> None:
        result = review_gate.evaluate_gate(
            classifier={
                "memory_required": True,
                "memory_present": True,
                "memory_invalid": False,
                "gate_passed": True,
            },
            review_response=None,
            advisory_on_review_unavailable=True,
        )

        self.assertTrue(result.gate_passed)
        self.assertIsNone(result.gate_reason)
        self.assertEqual(result.review_brief_match, "unknown")
        self.assertIn("AI review unavailable", result.comment_body)

    def test_not_required_without_brief_passes(self) -> None:
        result = review_gate.evaluate_gate(
            classifier={
                "memory_required": False,
                "memory_present": False,
                "memory_invalid": False,
                "gate_passed": True,
            },
            review_response=None,
        )

        self.assertTrue(result.gate_passed)
        self.assertEqual(result.review_brief_match, "not-applicable")

    def test_review_workflow_only_runs_ai_steps_for_present_brief(self) -> None:
        workflow_path = (
            SCRIPT_DIR.parents[2] / ".github/workflows/agent-memory-review.yml"
        )
        workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
        self.assertEqual(
            workflow["concurrency"]["group"],
            "${{ github.workflow }}-${{ github.event.pull_request.number }}",
        )
        self.assertTrue(workflow["concurrency"]["cancel-in-progress"])
        steps = workflow["jobs"]["review"]["steps"]
        names = [step.get("name") for step in steps]

        self.assertLess(
            names.index("Classify review brief"),
            names.index("Fetch classified review brief"),
        )
        self.assertLess(
            names.index("Fetch classified review brief"),
            names.index("Build review context pack"),
        )
        self.assertLess(
            names.index("Evaluate memory review hard gate"),
            names.index("Publish review comment"),
        )
        self.assertNotIn("Enforce memory review hard gate", names)
        classify_step = steps[names.index("Classify review brief")]
        self.assertEqual(classify_step["id"], "classify")
        for step_name in (
            "Fetch classified review brief",
            "Build review context pack",
            "Run memory-assisted review",
            "Evaluate memory review hard gate",
            "Publish review comment",
            "Publish review fallback comment",
        ):
            with self.subTest(step=step_name):
                step = steps[names.index(step_name)]
                self.assertIn(
                    "steps.classify.outputs.memory_present == 'true'",
                    step["if"],
                )
        evaluate_step = steps[names.index("Evaluate memory review hard gate")]
        self.assertIn("--advisory-on-review-unavailable", evaluate_step["run"])
        self.assertNotIn("--fail-on-gate", evaluate_step["run"])
        workflow_run_text = "\n".join(
            str(step.get("run", "")) for step in steps if "run" in step
        )
        self.assertNotIn("--fail-on-gate", workflow_run_text)
        collect_step = steps[names.index("Collect PR metadata and diff")]
        fetch_step = steps[names.index("Fetch classified review brief")]
        self.assertNotIn("briefMatch", collect_step["with"]["script"])
        self.assertIn(
            "changed_file_entries: changedFileEntries", collect_step["with"]["script"]
        )
        self.assertIn("status: file.status", collect_step["with"]["script"])
        self.assertNotIn("previous_filename", collect_step["with"]["script"])
        self.assertNotIn("docs\\/agent\\/reviews", collect_step["with"]["script"])
        self.assertIn("classification.memory_path", fetch_step["with"]["script"])


if __name__ == "__main__":
    unittest.main()
