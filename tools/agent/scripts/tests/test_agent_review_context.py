import importlib
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

review_context = importlib.import_module("agent_review_context")


class AgentReviewContextTests(unittest.TestCase):
    def build_prompt(
        self,
        repo_root: Path,
        *,
        body="",
        classifier_data=None,
        diff=None,
        additions=600,
        deletions=10,
    ):
        metadata = {
            "number": 12,
            "title": "Router update",
            "body": body,
            "additions": additions,
            "deletions": deletions,
            "changed_files": ["src/semantic-router/pkg/example.go"],
        }
        inputs = review_context.ContextInputs(
            metadata=metadata,
            classifier=classifier_data or {},
            diff=diff
            or "diff --git a/src/semantic-router/pkg/example.go b/src/semantic-router/pkg/example.go\n",
            repo_root=repo_root,
        )
        return review_context.build_review_prompt(inputs)

    def test_current_brief_is_marked_author_provided(self) -> None:
        path = "docs/agent/reviews/2026/2026-05-21-router-brief.md"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            brief_path = root / path
            brief_path.parent.mkdir(parents=True)
            brief_path.write_text(
                "# Review Brief\n\nNot verified: tests.\n", encoding="utf-8"
            )

            prompt = self.build_prompt(
                root,
                body=f"Review brief: {path}",
                classifier_data={"memory_path": path, "memory_present": True},
            )

        self.assertIn("## Author-provided review brief", prompt)
        self.assertIn("Not verified: tests.", prompt)
        self.assertIn("Brief / Diff Consistency", prompt)
        self.assertIn("Review brief matches diff: yes|no|not-applicable", prompt)
        self.assertIn("Hard gate: pass|fail", prompt)
        self.assertIn("first finding", prompt)
        self.assertIn("trust the diff", prompt)

    def test_missing_memory_still_builds_limited_review_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prompt = self.build_prompt(Path(temp_dir))

        self.assertIn("- Memory context: missing", prompt)
        self.assertIn("## PR Diff", prompt)
        self.assertIn("## Findings", prompt)

    def test_invalid_memory_status_is_reported_in_brief_section(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prompt = self.build_prompt(
                Path(temp_dir),
                body="Review brief: docs/agent/reviews/nope.md",
                classifier_data={
                    "memory_required": False,
                    "memory_present": False,
                    "memory_invalid": True,
                    "invalid_reason": "Review brief path must match docs/agent/reviews",
                },
            )

        self.assertIn("- Memory context: invalid", prompt)
        self.assertIn("Memory context: invalid", prompt)
        self.assertIn("Invalid reason: Review brief path must match", prompt)

    def test_large_missing_memory_prompt_stays_under_context_limit(self) -> None:
        diff_lines = [
            "diff --git a/src/semantic-router/pkg/example.go b/src/semantic-router/pkg/example.go\n",
            "--- a/src/semantic-router/pkg/example.go\n",
            "+++ b/src/semantic-router/pkg/example.go\n",
        ]
        diff_lines.extend(
            f"+func generatedLine{index}() {{ return {index} }}\n"
            for index in range(551)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prompt = self.build_prompt(
                root,
                body="Large PR without a review brief.\n" * 200,
                classifier_data={
                    "memory_required": True,
                    "memory_present": False,
                    "memory_invalid": False,
                },
                diff="".join(diff_lines),
                additions=551,
                deletions=0,
            )

        self.assertLessEqual(
            len(prompt.encode("utf-8")), review_context.DEFAULT_MAX_CONTEXT_BYTES
        )
        self.assertIn("Brief / Diff Consistency", prompt)
        self.assertIn("- Memory context: missing", prompt)


if __name__ == "__main__":
    unittest.main()
