import importlib
import re
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

classifier = importlib.import_module("agent_memory_classifier")


class AgentMemoryClassifierTests(unittest.TestCase):
    def classify(self, *, body="", additions=250, deletions=250, changed_files=None):
        with tempfile.TemporaryDirectory() as temp_dir:
            return classifier.classify_pr(
                body=body,
                additions=additions,
                deletions=deletions,
                changed_files=changed_files or [],
                repo_root=Path(temp_dir),
            )

    def test_large_pr_without_brief_is_missing(self) -> None:
        result = self.classify(additions=400, deletions=101)

        self.assertTrue(result.memory_required)
        self.assertFalse(result.memory_present)
        self.assertFalse(result.memory_invalid)
        self.assertIsNone(result.invalid_reason)
        self.assertFalse(result.gate_passed)
        self.assertIn("required", result.gate_reason or "")
        self.assertIn("Hard gate: failed", result.comment_body)
        self.assertIn("Memory context: missing", result.comment_body)
        self.assertNotIn("Memory context: invalid", result.comment_body)
        self.assertEqual(result.labels_to_add, ["agent-memory-missing"])

    def test_large_pr_with_valid_changed_brief_is_present(self) -> None:
        path = "docs/agent/reviews/2026/2026-05-21-router-review-brief.md"
        result = self.classify(
            body=f"Review brief: {path}",
            changed_files=[path],
        )

        self.assertTrue(result.memory_required)
        self.assertTrue(result.memory_present)
        self.assertEqual(result.memory_path, path)
        self.assertTrue(result.gate_passed)
        self.assertEqual(result.labels_to_add, ["agent-memory-present"])

    def test_small_pr_without_brief_is_not_required(self) -> None:
        result = self.classify(additions=10, deletions=20)

        self.assertFalse(result.memory_required)
        self.assertFalse(result.memory_invalid)
        self.assertTrue(result.gate_passed)
        self.assertEqual(result.labels_to_add, ["agent-memory-not-required"])

    def test_invalid_brief_path_is_invalid(self) -> None:
        result = self.classify(body="Review brief: docs/agent/reviews/nope.md")

        self.assertTrue(result.memory_invalid)
        self.assertFalse(result.gate_passed)
        self.assertIn("must match", result.invalid_reason or "")
        self.assertEqual(result.labels_to_add, ["agent-memory-missing"])

    def test_small_pr_with_invalid_brief_fails_gate(self) -> None:
        result = self.classify(
            body="Review brief: docs/agent/reviews/nope.md",
            additions=5,
            deletions=5,
        )

        self.assertFalse(result.memory_required)
        self.assertTrue(result.memory_invalid)
        self.assertFalse(result.gate_passed)
        self.assertEqual(result.labels_to_add, ["agent-memory-missing"])
        self.assertIn("Memory context: invalid", result.comment_body)
        self.assertNotIn("not required for this PR size", result.comment_body)

    def test_brief_outside_review_directory_is_invalid(self) -> None:
        result = self.classify(body="Review brief: memory.md")

        self.assertTrue(result.memory_invalid)
        self.assertIn("docs/agent/reviews", result.invalid_reason or "")

    def test_existing_base_branch_brief_is_present(self) -> None:
        path = "docs/agent/reviews/2026/2026-05-21-existing-brief.md"
        with tempfile.TemporaryDirectory() as temp_dir:
            brief_path = Path(temp_dir) / path
            brief_path.parent.mkdir(parents=True)
            brief_path.write_text("# Review Brief\n", encoding="utf-8")

            result = classifier.classify_pr(
                body=f"Review brief: {path}",
                additions=500,
                deletions=0,
                changed_files=[],
                repo_root=Path(temp_dir),
            )

        self.assertTrue(result.memory_present)
        self.assertTrue(result.gate_passed)
        self.assertEqual(result.labels_to_add, ["agent-memory-present"])

    def test_classifier_workflow_enforces_gate_after_commenting(self) -> None:
        workflow_path = (
            SCRIPT_DIR.parents[2] / ".github/workflows/agent-memory-classifier.yml"
        )
        workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
        self.assertEqual(
            workflow["concurrency"]["group"],
            "${{ github.workflow }}-${{ github.event.pull_request.number }}",
        )
        self.assertTrue(workflow["concurrency"]["cancel-in-progress"])
        steps = workflow["jobs"]["classify"]["steps"]
        names = [step.get("name") for step in steps]

        self.assertLess(
            names.index("Comment review brief status"),
            names.index("Enforce review brief hard gate"),
        )
        enforce_step = steps[names.index("Enforce review brief hard gate")]
        self.assertIn("agent_memory_review_gate.py", enforce_step["run"])
        self.assertIn("--classifier-only", enforce_step["run"])
        self.assertIn("--fail-on-gate", enforce_step["run"])

    def test_workflow_creates_labels_and_removes_only_current_agent_labels(
        self,
    ) -> None:
        workflow_path = (
            SCRIPT_DIR.parents[2] / ".github/workflows/agent-memory-classifier.yml"
        )
        workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
        update_step = next(
            step
            for step in workflow["jobs"]["classify"]["steps"]
            if step.get("name") == "Update memory labels"
        )
        script = update_step["with"]["script"]

        for label in classifier.LABELS:
            self.assertIn(f"'{label}'", script)
        self.assertIn("github.rest.issues.createLabel", script)
        self.assertIn("github.rest.issues.listLabelsOnIssue", script)
        self.assertIn("currentLabelNames.has(name)", script)
        self.assertNotRegex(
            script,
            re.compile(
                r"for \(const name of labelsToRemove\).*?try\s*{.*?removeLabel",
                re.DOTALL,
            ),
        )

    def test_workflows_request_pr_write_for_pr_label_and_comment_apis(self) -> None:
        workflow_paths = (
            SCRIPT_DIR.parents[2] / ".github/workflows/agent-memory-classifier.yml",
            SCRIPT_DIR.parents[2] / ".github/workflows/agent-memory-review.yml",
        )
        for workflow_path in workflow_paths:
            with self.subTest(workflow=workflow_path.name):
                workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
                self.assertEqual(workflow["permissions"]["issues"], "write")
                self.assertEqual(workflow["permissions"]["pull-requests"], "write")

    def test_workflows_checkout_same_commit_as_workflow_definition(self) -> None:
        workflow_paths = (
            SCRIPT_DIR.parents[2] / ".github/workflows/agent-memory-classifier.yml",
            SCRIPT_DIR.parents[2] / ".github/workflows/agent-memory-review.yml",
        )
        for workflow_path in workflow_paths:
            with self.subTest(workflow=workflow_path.name):
                workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
                checkout_step = workflow["jobs"][
                    "classify" if "classifier" in workflow_path.name else "review"
                ]["steps"][0]

                self.assertEqual(checkout_step["name"], "Checkout workflow tools")
                self.assertEqual(
                    checkout_step["with"]["ref"], "${{ github.workflow_sha }}"
                )


if __name__ == "__main__":
    unittest.main()
