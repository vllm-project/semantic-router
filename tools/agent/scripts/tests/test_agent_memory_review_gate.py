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

    def test_review_workflow_comments_before_enforcing_gate(self) -> None:
        workflow_path = (
            SCRIPT_DIR.parents[2] / ".github/workflows/agent-memory-review.yml"
        )
        workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
        steps = workflow["jobs"]["review"]["steps"]
        names = [step.get("name") for step in steps]

        self.assertLess(
            names.index("Evaluate memory review hard gate"),
            names.index("Publish review comment"),
        )
        self.assertLess(
            names.index("Publish review comment"),
            names.index("Enforce memory review hard gate"),
        )
        enforce_step = steps[names.index("Enforce memory review hard gate")]
        self.assertIn("agent_memory_review_gate.py", enforce_step["run"])
        self.assertIn("--fail-on-gate", enforce_step["run"])


if __name__ == "__main__":
    unittest.main()
