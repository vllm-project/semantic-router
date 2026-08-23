from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools" / "ci"))

from validate_workflows import Workflow, load_workflows, needs  # noqa: E402


class RecipeDistributionWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        errors: list[str] = []
        workflows = load_workflows(errors)
        self.assertEqual(errors, [])
        self.workflow: Workflow = workflows["recipe-distribution.yml"]
        self.release_workflow: Workflow = workflows["release.yml"]
        self.text = self.workflow.path.read_text(encoding="utf-8")

    def test_workflow_validates_manually_and_after_recipe_merges(self) -> None:
        self.assertIn("workflow_dispatch", self.workflow.events)
        self.assertNotIn("release", self.workflow.events)
        self.assertIn("pull_request", self.workflow.events)
        self.assertIn("push", self.workflow.events)
        push = self.workflow.events["push"]
        self.assertEqual(push["branches"], ["main"])
        pull_request = self.workflow.events["pull_request"]
        self.assertEqual(
            set(pull_request["paths"]),
            set(push["paths"]),
            "release-critical Recipe inputs must run the same gate before and after merge",
        )
        for path in (
            "config/recipes/**",
            "tools/agent/schemas/recipe-probes-v1.schema.json",
            "tools/agent/scripts/recipe_conformance*.py",
            "tools/agent/scripts/recipe_metadata_schema.py",
            "tools/agent/scripts/router_calibration_*.py",
            "tools/make/recipe-conformance.mk",
            "tools/release/recipe_bundle.py",
            "tools/release/snapshot_builtin_recipes.py",
        ):
            self.assertIn(path, push["paths"])

    def test_workflow_rejects_changes_to_published_tag_snapshots(self) -> None:
        self.assertIn("fetch-depth: 0", self.text)
        self.assertIn("Protect snapshots already bound to release tags", self.text)
        self.assertIn("--check-published --base-ref", self.text)
        self.assertIn("github.event.pull_request.base.sha", self.text)
        self.assertIn("github.event.before", self.text)
        self.assertIn("github.sha", self.text)

    def test_workflow_is_read_only_and_never_publishes_recipe_assets(self) -> None:
        self.assertEqual(self.workflow.data["permissions"], {"contents": "read"})
        self.assertEqual(set(self.workflow.jobs), {"validate-built-in-recipes"})
        validator = self.workflow.jobs["validate-built-in-recipes"]
        self.assertEqual(validator["permissions"], {"contents": "read"})
        self.assertNotIn("contents: write", self.text)
        self.assertNotIn("gh release", self.text)
        self.assertNotIn("vllm-sr recipe pack", self.text)
        self.assertNotIn(".vllm-sr-recipe.zip", self.text)

    def test_workflow_checks_the_canonical_source_without_a_package_mirror(
        self,
    ) -> None:
        self.assertIn("make recipe-conformance-static", self.text)

    def test_workflow_keeps_only_a_short_lived_validation_receipt(self) -> None:
        self.assertIn("built-in-recipe-receipt-${{ github.run_id }}", self.text)
        self.assertIn("dist/built-in-recipe-receipt/", self.text)
        self.assertIn("source-commit.txt", self.text)
        self.assertIn("find config/recipes/built-in", self.text)
        self.assertIn("SHA256SUMS", self.text)
        self.assertIn("retention-days: 30", self.text)

    def test_generated_package_mirror_does_not_exist(self) -> None:
        attributes = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
        self.assertNotIn("src/vllm-sr/cli/recipes", attributes)
        self.assertFalse(any((REPO_ROOT / "src/vllm-sr/cli/recipes").rglob("*")))

    def test_canonical_release_does_not_attach_recipe_assets(self) -> None:
        release_path = REPO_ROOT / ".github" / "workflows" / "release.yml"
        release_text = release_path.read_text(encoding="utf-8")
        self.assertNotIn("Managed Recipe packages", release_text)
        self.assertNotIn("managed-recipe-release-assets", release_text)
        self.assertNotIn("release-assets/recipes", release_text)
        self.assertNotIn(".vllm-sr-recipe.zip", release_text)
        self.assertIn("needs: [validate, docker, helm, pypi, crate]", release_text)
        self.assertIn("not duplicated in the CLI wheel", release_text)
        self.assertIn("separate GitHub Release assets", release_text)
        self.assertIn(
            "recipe_snapshot: ${{ steps.contract.outputs.recipe_snapshot }}",
            release_text,
        )
        self.assertNotIn(
            "recipe_snapshot: ${{ needs.validate.outputs.recipe_snapshot }}",
            release_text,
        )
        self.assertIn("fetch-depth: 0", release_text)
        self.assertIn("--check-published --base-ref", release_text)
        self.assertIn('base-ref "$GITHUB_SHA"', release_text)
        for job_name in ("docker", "helm", "pypi", "crate", "release-notes"):
            self.assertIn(
                "validate",
                needs(self.release_workflow.jobs[job_name]),
                msg=f"{job_name} must fail closed behind release validation",
            )

    def test_pypi_wheel_excludes_router_recipe_distribution(self) -> None:
        publish_path = REPO_ROOT / ".github" / "workflows" / "pypi-publish.yml"
        publish_text = publish_path.read_text(encoding="utf-8")
        self.assertNotIn("snapshot_builtin_recipes.py", publish_text)
        self.assertNotIn("recipe_snapshot", publish_text)
        self.assertIn(
            "CLI wheel must not embed the Router Recipe distribution", publish_text
        )
        self.assertIn('grep -Fqx "$file" wheel-files.txt', publish_text)

    def test_release_make_target_snapshots_latest_without_overwrite_flag(self) -> None:
        release_makefile = (REPO_ROOT / "tools" / "make" / "release.mk").read_text(
            encoding="utf-8"
        )
        self.assertIn("built-in-recipe-snapshot:", release_makefile)
        self.assertIn(
            'snapshot_builtin_recipes.py --version "$(RELEASE_VERSION)"',
            release_makefile,
        )
        self.assertNotIn("--force", release_makefile)


if __name__ == "__main__":
    unittest.main()
