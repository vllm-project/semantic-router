from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools" / "ci"))

from validate_workflows import Workflow, load_workflows, needs  # noqa: E402


class RecipeConformanceWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        errors: list[str] = []
        workflows = load_workflows(errors)
        self.assertEqual(errors, [])
        self.workflow: Workflow = workflows["recipe-conformance.yml"]
        self.text = self.workflow.path.read_text(encoding="utf-8")
        self.makefile = (
            REPO_ROOT / "tools" / "make" / "recipe-conformance.mk"
        ).read_text(encoding="utf-8")
        self.runner = (
            REPO_ROOT / "e2e" / "testing" / "run_recipe_conformance.sh"
        ).read_text(encoding="utf-8")

    def test_all_recipe_sources_share_one_live_matrix(self) -> None:
        self.assertEqual(
            set(self.workflow.jobs),
            {"inventory", "live-cpu", "report"},
        )
        self.assertIn("plan-all", self.text)
        self.assertIn("matrix.recipes_root", self.text)
        self.assertIn("matrix.report_dir", self.text)
        self.assertNotIn("live-cpu-built-in", self.text)
        self.assertIn("static-all", self.makefile)
        self.assertIn("report-all", self.makefile)
        self.assertIn("sources --format pipe", self.makefile)
        self.assertIn('RECIPES_ROOT="${RECIPES_ROOT:-', self.runner)
        self.assertIn('--recipes-root "${RECIPES_ROOT}"', self.runner)
        self.assertIn("runtime-auth", self.runner)
        self.assertIn('--recipe "${recipe}"', self.runner)
        self.assertIn("VSR_MGMT_TOKEN", self.runner)

    def test_report_fails_closed_on_the_source_aware_live_matrix(self) -> None:
        report_needs = needs(self.workflow.jobs["report"])

        self.assertEqual(report_needs, {"inventory", "live-cpu"})
        self.assertIn("find .agent-harness/recipe-conformance", self.text)


if __name__ == "__main__":
    unittest.main()
