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

    def test_workflow_validates_manually_and_after_catalog_merges(self) -> None:
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
            "release-critical catalog inputs must run the same gate before and after merge",
        )
        for path in (
            "config/recipes/**",
            "src/vllm-sr/MANIFEST.in",
            "src/vllm-sr/cli/commands/model.py",
            "src/vllm-sr/cli/commands/model_rendering.py",
            "src/vllm-sr/cli/model_assets/**",
            "src/vllm-sr/cli/model_bundle.py",
            "src/vllm-sr/cli/model_catalog.py",
            "src/vllm-sr/cli/model_catalog_types.py",
            "src/vllm-sr/cli/model_catalog_validation.py",
            "src/vllm-sr/pyproject.toml",
            "tools/agent/schemas/recipe-probes-v1.schema.json",
            "tools/agent/scripts/recipe_conformance*.py",
            "tools/agent/scripts/recipe_metadata_schema.py",
            "tools/agent/scripts/router_calibration_*.py",
            "tools/make/recipe-conformance.mk",
            "tools/release/sync_model_catalog.py",
            "tools/release/snapshot_model_catalog.py",
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
        self.assertEqual(set(self.workflow.jobs), {"validate-built-in-models"})
        validator = self.workflow.jobs["validate-built-in-models"]
        self.assertEqual(validator["permissions"], {"contents": "read"})
        self.assertNotIn("contents: write", self.text)
        self.assertNotIn("gh release", self.text)
        self.assertNotIn("vllm-sr recipe pack", self.text)
        self.assertNotIn(".vllm-sr-recipe.zip", self.text)

    def test_workflow_checks_source_mirror_and_conformance(self) -> None:
        self.assertIn("sync_model_catalog.py --check", self.text)
        self.assertIn("make recipe-conformance-static", self.text)
        self.assertIn("vllm-sr model list --all-versions", self.text)
        self.assertIn("vllm-sr model show vllm-sr/mom-v1-blend", self.text)

    def test_workflow_keeps_only_a_short_lived_validation_receipt(self) -> None:
        self.assertIn("built-in-model-catalog-receipt-${{ github.run_id }}", self.text)
        self.assertIn("dist/model-catalog-receipt/", self.text)
        self.assertIn("source-commit.txt", self.text)
        self.assertIn(
            "find config/recipes/built-in src/vllm-sr/cli/model_assets",
            self.text,
        )
        self.assertIn("SHA256SUMS", self.text)
        self.assertIn("retention-days: 30", self.text)

    def test_generated_package_mirror_is_collapsed_for_code_review(self) -> None:
        attributes = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
        self.assertIn(
            "src/vllm-sr/cli/model_assets/** linguist-generated=true",
            attributes,
        )

    def test_canonical_release_does_not_attach_recipe_or_catalog_assets(self) -> None:
        release_path = REPO_ROOT / ".github" / "workflows" / "release.yml"
        release_text = release_path.read_text(encoding="utf-8")
        self.assertNotIn("Managed Recipe packages", release_text)
        self.assertNotIn("managed-recipe-release-assets", release_text)
        self.assertNotIn("release-assets/recipes", release_text)
        self.assertNotIn(".vllm-sr-recipe.zip", release_text)
        self.assertIn("needs: [validate, docker, helm, pypi, crate]", release_text)
        self.assertIn("They are not published", release_text)
        self.assertIn("as separate GitHub Release assets", release_text)
        self.assertIn(
            "catalog_snapshot: ${{ steps.contract.outputs.catalog_snapshot }}",
            release_text,
        )
        self.assertIn(
            "catalog_snapshot: ${{ needs.validate.outputs.catalog_snapshot }}",
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

    def test_pypi_wheel_dynamically_checks_the_bound_release_snapshot(self) -> None:
        publish_path = REPO_ROOT / ".github" / "workflows" / "pypi-publish.yml"
        publish_text = publish_path.read_text(encoding="utf-8")
        self.assertIn("sync_model_catalog.py --check", publish_text)
        self.assertIn("snapshot_model_catalog.py", publish_text)
        self.assertIn('--check --version "$RELEASE_VERSION"', publish_text)
        self.assertIn("INPUT_CATALOG_SNAPSHOT", publish_text)
        self.assertIn(
            'EXPECTED_CATALOG_SNAPSHOT="v${VERSION_MAJOR}.${VERSION_MINOR}"',
            publish_text,
        )
        self.assertIn('for catalog_version in latest "$CATALOG_SNAPSHOT"', publish_text)
        self.assertIn('find "$SNAPSHOT_DIR"', publish_text)
        self.assertIn(
            'package_file="cli/model_assets/$catalog_version/$file"', publish_text
        )
        self.assertIn('REQUIRED_FILES+=("$package_file")', publish_text)
        self.assertIn('grep -Fqx "$file" wheel-files.txt', publish_text)
        self.assertIn(
            'cmp -s "$SNAPSHOT_DIR/$file" "wheel-unpacked/$package_file"',
            publish_text,
        )

    def test_release_make_target_snapshots_latest_without_overwrite_flag(self) -> None:
        release_makefile = (REPO_ROOT / "tools" / "make" / "release.mk").read_text(
            encoding="utf-8"
        )
        self.assertIn("built-in-model-snapshot:", release_makefile)
        self.assertIn(
            'snapshot_model_catalog.py --version "$(RELEASE_VERSION)"',
            release_makefile,
        )
        self.assertIn("sync_model_catalog.py --check", release_makefile)
        self.assertNotIn("--force", release_makefile)


if __name__ == "__main__":
    unittest.main()
