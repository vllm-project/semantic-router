from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools" / "ci"))

from classify_pr_changes import (  # noqa: E402
    NIGHTLY_IMAGES,
    PRODUCTION_RELEASE_IMAGES,
    classify,
)


class PRChangeClassifierTests(unittest.TestCase):
    def assert_classification(
        self,
        path: str,
        jobs: tuple[str, ...],
        *,
        profiles: tuple[str, ...] = (),
        images: tuple[str, ...] = (),
    ) -> None:
        result = classify([path])
        self.assertEqual(result.selected_jobs, jobs)
        self.assertEqual(result.profiles, profiles)
        self.assertEqual(result.pr_images, images)

    def test_domains_select_only_their_coarse_ci_job(self) -> None:
        fixtures = {
            "website/docs/community/development.md": ("quality",),
            "tools/make/agent.mk": ("quality", "security"),
            "src/semantic-router/pkg/extproc/processor.go": (
                "quality",
                "security",
                "core-tests",
            ),
            "deploy/operator/controllers/semanticrouter_controller.go": (
                "quality",
                "security",
                "operator",
            ),
            "src/vllm-sr/cli/evaluation/scoring.py": (
                "quality",
                "security",
                "core-tests",
            ),
        }
        for path, jobs in fixtures.items():
            with self.subTest(path=path):
                self.assert_classification(path, jobs)

    def test_runtime_cli_surface_has_explicit_integration_escalation(self) -> None:
        self.assert_classification(
            "src/vllm-sr/cli/commands/runtime.py",
            ("quality", "security", "core-tests", "cli"),
        )

    def test_workflow_only_change_runs_only_its_reusable_workflow(self) -> None:
        fixtures = {
            ".github/workflows/performance-test.yml": "performance",
            ".github/workflows/operator-ci.yml": "operator",
            ".github/workflows/integration-test-memory.yml": "memory",
            ".github/workflows/openvino-binding-ci.yml": "openvino",
        }
        for path, selected in fixtures.items():
            with self.subTest(path=path):
                self.assert_classification(path, ("quality", "security", selected))

    def test_e2e_workflow_selects_only_its_baseline_profile(self) -> None:
        self.assert_classification(
            ".github/workflows/integration-test-k8s.yml",
            ("quality", "security", "e2e"),
            profiles=("envoy-ai-gateway",),
        )

    def test_unit_test_only_change_suppresses_e2e_and_images(self) -> None:
        result = classify(["src/semantic-router/pkg/extproc/processor_test.go"])

        self.assertTrue(result.test_only)
        self.assertEqual(result.selected_jobs, ("quality", "security", "core-tests"))
        self.assertEqual(result.profiles, ())
        self.assertEqual(result.pr_images, ())
        self.assertEqual(result.publish_images, ())

    def test_actual_e2e_test_selects_its_named_profile(self) -> None:
        result = classify(["e2e/testcases/istio_routes_test.go"])

        self.assertTrue(result.test_only)
        self.assertEqual(result.selected_jobs, ("quality", "security", "e2e"))
        self.assertEqual(result.profiles, ("istio",))

    def test_only_image_definition_changes_trigger_pr_builds(self) -> None:
        self.assert_classification(
            "tools/docker/Dockerfile.extproc-rocm",
            ("quality", "security", "images"),
            images=("extproc-rocm",),
        )
        result = classify(["src/semantic-router/pkg/extproc/processor.go"])
        self.assertEqual(result.pr_images, ())
        self.assertEqual(result.publish_images, ("extproc", "vllm-sr"))

    def test_generated_api_contracts_keep_their_drift_check(self) -> None:
        for path in (
            "website/docs/api/apiserver.md",
            "website/static/openapi/apiserver/apiserver.openapi.json",
            "tools/openapi-gen/main.go",
        ):
            with self.subTest(path=path):
                self.assertIn("core-tests", classify([path]).selected_jobs)

    def test_router_learning_parity_covers_runtime_and_replay_changes(self) -> None:
        paths = (
            "src/semantic-router/pkg/extproc/router_learning_sampling_score.go",
            "src/vllm-sr/cli/evaluation/router_learning_policy.py",
            "src/vllm-sr/cli/evaluation/resources/router_learning_core.v2.json",
            "src/vllm-sr/tests/test_router_learning_policy_parity.py",
            "dashboard/backend/evaluationplane/method_router_learning_reducer.go",
        )
        for path in paths:
            with self.subTest(path=path):
                self.assertIn("router-learning", classify([path]).selected_jobs)

    def test_recipe_change_selects_static_conformance_without_e2e(self) -> None:
        result = classify(["config/recipes/privacy/probes.yaml"])

        self.assertEqual(
            result.selected_jobs,
            ("quality", "security", "core-tests", "recipe-conformance"),
        )
        self.assertEqual(result.profiles, ())

    def test_full_label_runs_full_safety_net_except_performance(self) -> None:
        result = classify(["README.md"], full=True)

        self.assertEqual(
            result.selected_jobs,
            ("quality", "security", "core-tests", "e2e", "recipe-conformance"),
        )
        self.assertEqual(
            result.profiles,
            ("envoy-ai-gateway", "dashboard", "remote-embedding"),
        )
        self.assertNotIn("performance", result.selected_jobs)

    def test_ownership_files_are_documentation_only(self) -> None:
        for path in ("OWNER", ".github/CODEOWNERS", "dashboard/OWNER"):
            with self.subTest(path=path):
                result = classify([path])
                self.assertEqual(result.selected_jobs, ("quality",))
                self.assertTrue(result.signals["docs_only"])

    def test_docs_in_a_mixed_change_do_not_add_product_domains(self) -> None:
        result = classify(
            [
                "tools/ci/classify_pr_changes.py",
                "src/semantic-router/pkg/config/AGENTS.md",
                "deploy/operator/README.md",
            ]
        )

        self.assertEqual(result.domains, ("harness",))
        self.assertEqual(result.selected_jobs, ("quality", "security"))

    def test_release_and_nightly_image_inventories_are_distinct(self) -> None:
        self.assertEqual(
            PRODUCTION_RELEASE_IMAGES,
            (
                "dashboard",
                "extproc",
                "extproc-rocm",
                "operator",
                "operator-bundle",
                "vllm-sr",
                "vllm-sr-cuda",
                "vllm-sr-rocm",
            ),
        )
        self.assertEqual(
            set(NIGHTLY_IMAGES) - set(PRODUCTION_RELEASE_IMAGES),
            {"anthropic-shim", "llm-katan", "vllm-sr-sim"},
        )


if __name__ == "__main__":
    unittest.main()
