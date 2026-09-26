from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools" / "ci"))

from classify_pr_changes import classify  # noqa: E402
from validate_workflows import Workflow, load_workflows, needs  # noqa: E402


class RecipeDistributionWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        errors: list[str] = []
        workflows = load_workflows(errors)
        self.assertEqual(errors, [])
        self.workflow: Workflow = workflows["package-check.yml"]
        self.release_workflow: Workflow = workflows["release.yml"]
        self.text = self.workflow.path.read_text(encoding="utf-8")

    def test_catalog_is_one_planned_read_only_package_verification(self) -> None:
        self.assertEqual(set(self.workflow.events), {"workflow_call"})
        self.assertFalse(
            (REPO_ROOT / ".github/workflows/recipe-distribution.yml").exists()
        )
        for path in (
            "config/recipes/built-in/latest/mom-v1/probes.yaml",
            "src/vllm-sr/MANIFEST.in",
            "src/vllm-sr/setup.py",
            "src/vllm-sr/cli/model_catalog.py",
            "config/schemas/recipe-probes-v1.schema.json",
            "tools/release/stage_model_catalog_package.py",
            "tools/release/snapshot_model_catalog.py",
        ):
            self.assertIn("cli-package", classify([path]).selected_jobs, path)
        self.assertEqual(self.workflow.data["permissions"], {"contents": "read"})
        self.assertNotIn("gh release", self.text)

    def test_unique_snapshot_compiler_install_and_resource_contracts_survive(
        self,
    ) -> None:
        source = (REPO_ROOT / "tools/ci/package_contract.py").read_text()
        for contract in (
            "catalog.compiler",
            "catalog.completeness",
            "catalog.immutable",
            "package.resources",
            "package.installed",
        ):
            self.assertIn(contract, source)
        self.assertIn('"--check-published"', source)
        self.assertIn("check_wheel(wheels[0])", source)
        self.assertIn("verify_resources(wheels[0], sources[0])", source)
        self.assertIn("fetch-depth: 0", self.text)
        self.assertIn("github.event.pull_request.base.sha", self.text)

    def test_release_package_uses_tagged_commit_as_base(self) -> None:
        qualification = next(
            step
            for step in self.workflow.jobs["package"]["steps"]
            if step.get("name") == "Qualify the final candidate"
        )
        self.assertIn(
            "inputs.mode == 'release' && github.sha",
            qualification["env"]["BASE_REF"],
        )

    def test_source_package_build_stages_catalog_assets_automatically(self) -> None:
        setup_text = (REPO_ROOT / "src" / "vllm-sr" / "setup.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("stage_model_catalog_package.py", setup_text)
        self.assertIn("runpy.run_path", setup_text)
        self.assertIn('"latest" / "catalog.yaml"', setup_text)
        self.assertIn("if not catalog.is_file()", setup_text)

    def test_catalog_make_targets_work_without_an_agent_virtualenv(self) -> None:
        make_text = (REPO_ROOT / "tools" / "make" / "model-catalog.mk").read_text(
            encoding="utf-8"
        )

        self.assertIn("MODEL_CATALOG_PYTHON ?=", make_text)
        self.assertIn(",python3)", make_text)
        self.assertNotIn("@.venv-agent/bin/python", make_text)

    def test_live_conformance_runs_the_image_built_for_the_source_tree(self) -> None:
        make_text = (REPO_ROOT / "tools" / "make" / "recipe-conformance.mk").read_text(
            encoding="utf-8"
        )
        runner_text = (
            REPO_ROOT / "e2e" / "testing" / "run_recipe_conformance.sh"
        ).read_text(encoding="utf-8")

        self.assertIn('ROUTER_IMAGE="$(VLLM_SR_ROUTER_IMAGE)"', make_text)
        self.assertIn('ROUTER_IMAGE="${ROUTER_IMAGE:-}"', runner_text)
        self.assertIn('--router-image "${ROUTER_IMAGE}"', runner_text)
        self.assertIn("8080 + $(VLLM_SR_PORT_OFFSET)", make_text)
        self.assertIn("8080 + VLLM_SR_PORT_OFFSET", runner_text)

    def test_package_staging_tree_is_ignored_instead_of_committed(self) -> None:
        ignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn(
            "src/vllm-sr/cli/model_assets/*/",
            ignore,
        )
        tracked = subprocess.run(
            ["git", "ls-files", "src/vllm-sr/cli/model_assets"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        self.assertEqual(tracked, ["src/vllm-sr/cli/model_assets/__init__.py"])

    def test_canonical_release_does_not_attach_recipe_or_catalog_assets(self) -> None:
        release_path = REPO_ROOT / ".github" / "workflows" / "release.yml"
        release_text = release_path.read_text(encoding="utf-8")
        self.assertNotIn("Managed Recipe packages", release_text)
        self.assertNotIn("managed-recipe-release-assets", release_text)
        self.assertNotIn("release-assets/recipes", release_text)
        self.assertNotIn(".vllm-sr-recipe.zip", release_text)
        self.assertEqual(
            needs(self.release_workflow.jobs["release-notes"]),
            {"validate", "gate", "docker", "helm", "pypi", "crate"},
        )
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
        self.assertIn("stage_model_catalog_package.py --check", publish_text)
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
        self.assertIn("model-catalog-package-check", release_makefile)
        self.assertNotIn("--force", release_makefile)


if __name__ == "__main__":
    unittest.main()
