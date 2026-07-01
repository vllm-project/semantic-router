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
    VALID_BRIEF_PATH = "docs/agent/reviews/2026/2026-05-21-router-review-brief.md"

    def classify(
        self,
        *,
        body="",
        additions=250,
        deletions=250,
        changed_files=None,
        changed_file_entries=None,
        review_brief_text_by_path=None,
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            return classifier.classify_pr(
                body=body,
                additions=additions,
                deletions=deletions,
                changed_files=changed_files or [],
                changed_file_entries=changed_file_entries,
                review_brief_text_by_path=review_brief_text_by_path,
                repo_root=Path(temp_dir),
            )

    def assert_missing_required_brief(self, result) -> None:
        self.assertTrue(result.memory_required)
        self.assertFalse(result.memory_present)
        self.assertFalse(result.memory_invalid)
        self.assertIsNone(result.memory_path)
        self.assertIsNone(result.invalid_reason)
        self.assertFalse(result.gate_passed)
        self.assertIn("required", result.gate_reason or "")
        self.assertEqual(result.labels_to_add, ["agent-memory-missing"])

    def assert_not_required_without_brief(self, result) -> None:
        self.assertFalse(result.memory_required)
        self.assertFalse(result.memory_present)
        self.assertFalse(result.memory_invalid)
        self.assertIsNone(result.memory_path)
        self.assertIsNone(result.invalid_reason)
        self.assertTrue(result.gate_passed)
        self.assertEqual(result.labels_to_add, ["agent-memory-not-required"])

    def assert_present_brief(self, result, path: str) -> None:
        self.assertTrue(result.memory_required)
        self.assertTrue(result.memory_present)
        self.assertFalse(result.memory_invalid)
        self.assertEqual(result.memory_path, path)
        self.assertIsNone(result.invalid_reason)
        self.assertTrue(result.gate_passed)
        self.assertEqual(result.labels_to_add, ["agent-memory-present"])

    def test_large_pr_without_brief_is_missing(self) -> None:
        result = self.classify(additions=400, deletions=101)

        self.assert_missing_required_brief(result)
        self.assertIn("Hard gate: failed", result.comment_body)
        self.assertIn("Memory context: missing", result.comment_body)
        self.assertNotIn("Memory context: invalid", result.comment_body)

    def test_review_brief_line_accepts_bare_and_backtick_paths(self) -> None:
        for body in (
            f"Review brief: {self.VALID_BRIEF_PATH}",
            f"Review brief: `{self.VALID_BRIEF_PATH}`",
        ):
            with self.subTest(body=body):
                result = self.classify(
                    body=body,
                    changed_files=[self.VALID_BRIEF_PATH],
                )

                self.assert_present_brief(result, self.VALID_BRIEF_PATH)

    def test_unsupported_or_missing_brief_values_are_no_usable_brief(self) -> None:
        cases = {
            "template-placeholder": (
                "Review brief: N/A\n\n"
                "<!-- If required, replace N/A with:\n"
                "docs/agent/reviews/YYYY/YYYY-MM-DD-<short-kebab-title>.md\n"
                "-->"
            ),
            "na": "Review brief: N/A",
            "none": "Review brief: none",
            "empty": "Review brief:",
            "garbage": "Review brief: garbage",
            "invalid-review-path": "Review brief: docs/agent/reviews/nope.md",
            "outside-review-dir": "Review brief: memory.md",
            "markdown-link": f"Review brief: [brief]({self.VALID_BRIEF_PATH})",
            "body-path-only": f"See the review context at {self.VALID_BRIEF_PATH}.",
        }
        for name, body in cases.items():
            with self.subTest(case=name):
                result = self.classify(
                    body=body,
                    additions=10,
                    deletions=10,
                    changed_files=[self.VALID_BRIEF_PATH],
                )

                self.assert_not_required_without_brief(result)

    def test_large_pr_with_na_still_fails_missing_required_brief(self) -> None:
        result = self.classify(
            body="Review brief: N/A",
            additions=500,
            deletions=0,
        )

        self.assert_missing_required_brief(result)

    def test_small_pr_without_brief_is_not_required(self) -> None:
        result = self.classify(additions=10, deletions=20)

        self.assert_not_required_without_brief(result)

    def test_existing_base_branch_brief_is_present(self) -> None:
        path = "docs/agent/reviews/2026/2026-05-21-existing-brief.md"
        with tempfile.TemporaryDirectory() as temp_dir:
            brief_path = Path(temp_dir) / path
            brief_path.parent.mkdir(parents=True)
            brief_path.write_text(
                "# Review Brief\n\nThis brief has confirmed body content.\n",
                encoding="utf-8",
            )

            result = classifier.classify_pr(
                body=f"Review brief: {path}",
                additions=500,
                deletions=0,
                changed_files=[],
                repo_root=Path(temp_dir),
            )

        self.assert_present_brief(result, path)

    def test_referenced_old_renamed_path_is_missing_without_base_file(self) -> None:
        old_path = "docs/agent/reviews/2026/2026-06-14-old-brief.md"
        new_path = "docs/agent/reviews/2026/2026-06-14-new-brief.md"
        result = self.classify(
            body=f"Review brief: {old_path}",
            changed_file_entries=[
                {
                    "filename": new_path,
                    "status": "renamed",
                }
            ],
        )

        self.assert_missing_required_brief(result)

    def test_changed_brief_status_and_metadata_content_control_usability(self) -> None:
        path = "docs/agent/reviews/2026/2026-06-14-added-brief.md"
        cases = (
            (
                "removed",
                [{"filename": path, "status": "removed"}],
                None,
                False,
            ),
            ("added-empty", [{"filename": path, "status": "added"}], {path: ""}, False),
            (
                "added-nonempty",
                [{"filename": path, "status": "added"}],
                {path: "# Review Brief\n\nThis brief describes the current PR.\n"},
                True,
            ),
            (
                "unknown-status-no-content",
                [{"filename": path, "status": "mystery"}],
                None,
                True,
            ),
        )
        for name, entries, text_by_path, expected_present in cases:
            with self.subTest(case=name):
                result = self.classify(
                    body=f"Review brief: {path}",
                    changed_file_entries=entries,
                    review_brief_text_by_path=text_by_path,
                )

                if expected_present:
                    self.assert_present_brief(result, path)
                else:
                    self.assert_missing_required_brief(result)

    def test_existing_empty_base_brief_is_missing(self) -> None:
        path = "docs/agent/reviews/2026/2026-06-14-empty-base-brief.md"
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

        self.assert_missing_required_brief(result)

    def test_changed_file_refs_prefers_status_entries_over_legacy_files(self) -> None:
        path = "docs/agent/reviews/2026/2026-06-14-removed-brief.md"
        refs = classifier.changed_file_refs(
            {
                "changed_files": [path],
                "changed_file_entries": [{"filename": path, "status": "removed"}],
            }
        )

        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].filename, path)
        self.assertEqual(refs[0].status, "removed")

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

    def test_classifier_workflow_metadata_includes_changed_file_entries(self) -> None:
        workflow_path = (
            SCRIPT_DIR.parents[2] / ".github/workflows/agent-memory-classifier.yml"
        )
        workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
        steps = workflow["jobs"]["classify"]["steps"]
        collect_step = next(
            step for step in steps if step.get("name") == "Collect PR metadata"
        )
        script = collect_step["with"]["script"]

        self.assertIn("changed_file_entries: changedFileEntries", script)
        self.assertIn("status: file.status", script)
        self.assertNotIn("previous_filename", script)
        self.assertIn("review_brief_text_by_path", script)
        self.assertNotIn("docs\\/agent\\/reviews", script)

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
