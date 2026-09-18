from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools" / "ci"))

from ci_plan import make_plan  # noqa: E402
from run_component_batch import commands  # noqa: E402
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
        self.assertEqual(
            self.workflow.jobs["report"]["if"],
            "${{ !cancelled() && needs.inventory.result == 'success' }}",
            "report failed probes without keeping a superseded run alive",
        )
        self.assertIn("make recipe-conformance-report", self.text)
        self.assertIn("**/conformance-report.md", self.text)
        self.assertIn("if-no-files-found: error", self.text)

    def test_inventory_validates_assets_without_repeating_authoring_units(self) -> None:
        steps = self.workflow.jobs["inventory"]["steps"]
        validation = next(
            step
            for step in steps
            if step.get("name") == "Validate maintained recipe contracts"
        )
        self.assertIn("recipe_conformance.py", validation["run"])
        self.assertIn(
            "--output-dir .agent-harness/recipe-conformance static-all",
            validation["run"],
        )
        self.assertNotIn("recipe-conformance-assets", validation["run"])
        self.assertNotIn("unittest", validation["run"])
        self.assertNotIn("if", validation)
        self.assertFalse(validation.get("continue-on-error", False))
        self.assertIn(
            ["make", "test-calibration"],
            commands("test-learning-tools", Path("reports")),
        )

    def test_recipe_assets_keep_static_and_live_checks_without_a_tool_worker(
        self,
    ) -> None:
        for path in (
            "config/recipes/built-in/latest/mom-v1/config.yaml",
            "config/recipes/built-in/latest/mom-v1/probes.yaml",
            "tools/make/recipe-conformance.mk",
            "e2e/testing/run_recipe_conformance.sh",
        ):
            with self.subTest(path=path):
                plan = make_plan([path], source_sha="a" * 40)
                ids = plan["expected_verification_ids"]
                self.assertEqual(ids.count("recipe-conformance"), 1)
                self.assertNotIn("learning-tools", ids)

    def test_recipe_tools_and_schema_contracts_select_the_single_unit_owner(
        self,
    ) -> None:
        for path in (
            "tools/calibration/recipe/recipe_conformance.py",
            "tools/calibration/recipe/router_calibration_fixture_test.py",
            "config/schemas/recipe-metadata-v1.schema.json",
            "config/schemas/recipe-metadata-v1.contract.yaml",
            "config/schemas/recipe-probes-v1.schema.json",
            "tools/calibration/tuning/tests/test_patterns.py",
            "tools/make/tooling.mk",
        ):
            with self.subTest(path=path):
                plan = make_plan([path], source_sha="a" * 40)
                ids = plan["expected_verification_ids"]
                self.assertEqual(ids.count("learning-tools"), 1)
                self.assertEqual(
                    sum(
                        record["id"] == "learning-tools"
                        for batch in plan["component_batches"]
                        for record in batch["verifications"]
                    ),
                    1,
                )

    def test_runner_owns_stack_cleanup_and_uses_source_image(self) -> None:
        self.assertIn("VLLM_SR_STACK_NAME=", self.runner)
        self.assertIn("VLLM_SR_STATE_ROOT_DIR=", self.runner)
        self.assertIn("resolve_runtime_stack", self.runner)
        self.assertIn('--router-image "${ROUTER_IMAGE}"', self.runner)
        self.assertIn('--runtime-config "${config}"', self.runner)
        self.assertNotIn("docker system prune", self.text)
        self.assertNotIn("vllm-sr stop", self.text)

    def test_plan_keeps_envoy_external_and_requires_the_candidate_router(self) -> None:
        plan = make_plan([], source_sha="a" * 40, requested=("recipe-conformance",))
        self.assertEqual(plan["images"], ["vllm-sr"])
        self.assertEqual(len(plan["verifications"]), 1)
        self.assertIn("envoy", plan["verifications"][0]["services"])


if __name__ == "__main__":
    unittest.main()
