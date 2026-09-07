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

REPRESENTATIVE_FIXTURES = {
    "website docs": (
        ["website/docs/community/development.md"],
        ("quality",),
        (),
        (),
    ),
    "harness docs": (
        ["tools/agent/docs/testing-strategy.md"],
        ("quality",),
        (),
        (),
    ),
    "harness executable": (
        ["tools/make/agent.mk", "tools/ci/classify_pr_changes.py"],
        ("quality", "security"),
        (),
        (),
    ),
    "precommit harness": (
        [
            ".pre-commit-config.yaml",
            ".github/workflows/pre-commit.yml",
            "tools/docker/Dockerfile.precommit",
            "tools/make/pre-commit.mk",
        ],
        ("quality", "security"),
        (),
        (),
    ),
    "local dev flow": (
        [
            "deploy/local/envoy.yaml",
            "tools/dev/local-up-router.sh",
            "tools/smoke/test-local-up-router.sh",
        ],
        ("quality", "security", "core-tests"),
        (),
        (),
    ),
    "core router": (
        ["src/semantic-router/pkg/extproc/processor.go"],
        ("quality", "security", "core-tests", "e2e", "images"),
        ("envoy-ai-gateway",),
        ("vllm-sr",),
    ),
    "dashboard": (
        ["dashboard/frontend/src/App.tsx"],
        ("quality", "security", "dashboard", "images"),
        (),
        ("dashboard",),
    ),
    "operator": (
        ["deploy/operator/controllers/semanticrouter_controller.go"],
        ("quality", "security", "operator", "e2e"),
        ("envoy-ai-gateway",),
        (),
    ),
    "CLI": (
        ["src/vllm-sr/cli/main.py"],
        ("quality", "security", "core-tests", "cli"),
        (),
        (),
    ),
    "memory": (
        ["src/semantic-router/pkg/memory/inmemory_store.go"],
        ("quality", "security", "core-tests", "e2e", "memory"),
        ("envoy-ai-gateway",),
        (),
    ),
    "native binding": (
        ["candle-binding/src/lib.rs"],
        ("quality", "security", "core-tests", "e2e", "images"),
        ("envoy-ai-gateway",),
        ("vllm-sr",),
    ),
    "OpenVINO": (
        ["openvino-binding/src/lib.rs"],
        ("quality", "security", "openvino"),
        (),
        (),
    ),
    "workflow only": (
        [".github/workflows/performance-test.yml"],
        ("quality", "security", "e2e"),
        ("envoy-ai-gateway",),
        (),
    ),
    "Docker validator only": (
        [".github/workflows/docker-validate.yml"],
        ("quality", "security", "e2e", "images"),
        ("envoy-ai-gateway",),
        ("vllm-sr",),
    ),
    "Docker publisher only": (
        [".github/workflows/docker-publish.yml"],
        ("quality", "security", "e2e"),
        ("envoy-ai-gateway",),
        (),
    ),
    "Docker product path": (
        ["tools/docker/Dockerfile.extproc-rocm"],
        ("quality", "security", "images"),
        (),
        ("extproc-rocm",),
    ),
    "PR 2788 CI architecture": (
        [
            ".github/workflows/ci-changes.yml",
            ".github/workflows/docker-publish.yml",
            ".github/workflows/docker-validate.yml",
            ".github/workflows/pr.yml",
            "CONTRIBUTING.md",
            "tools/agent/docs/plans/pl-0039-domain-ci-architecture.md",
            "tools/ci/validate_workflows.py",
            "tools/ci/workflow_policy_validation.py",
        ],
        ("quality", "security", "core-tests", "e2e", "images"),
        ("envoy-ai-gateway",),
        ("vllm-sr",),
    ),
}


class PRChangeClassifierTests(unittest.TestCase):
    def test_representative_pr_shapes(self) -> None:
        for name, (paths, jobs, profiles, images) in REPRESENTATIVE_FIXTURES.items():
            with self.subTest(name=name):
                result = classify(paths)
                self.assertEqual(result.selected_jobs, jobs)
                self.assertEqual(result.profiles, profiles)
                self.assertEqual(result.pr_images, images)

    def test_domain_documentation_does_not_enable_heavy_suites(self) -> None:
        result = classify(
            [
                "tools/ci/workflow_policy_validation.py",
                "perf/README.md",
                "e2e/testing/llm-katan/README.md",
                "deploy/operator/README.md",
            ]
        )

        self.assertEqual(
            result.selected_jobs,
            ("quality", "security"),
        )
        self.assertEqual(result.profiles, ())
        self.assertEqual(result.pr_images, ())

    def test_ci_full_does_not_enable_performance(self) -> None:
        result = classify(["README.md"], full=True)

        self.assertNotIn("performance", result.selected_jobs)
        self.assertIn("recipe-conformance", result.selected_jobs)
        self.assertEqual(
            result.profiles,
            ("envoy-ai-gateway", "dashboard", "remote-embedding"),
        )

    def test_generated_api_docs_select_core_tests(self) -> None:
        """Hand-edited API docs must still reach the api-docs-check drift gate.

        Both artifacts are generated from the route catalog. They live under
        `website/` and end in `.md`/`.json`, so without an explicit rule they
        classify as docs-only and skip `core-tests`, letting a manual edit
        drift away from the catalog undetected.
        """
        for path in (
            "website/docs/api/apiserver.md",
            "website/static/openapi/apiserver/apiserver.openapi.json",
            "tools/openapi-gen/main.go",
        ):
            with self.subTest(path=path):
                result = classify([path])

                self.assertIn("core-tests", result.selected_jobs)

    def test_protection_corpus_changes_select_core_benchmark_job(self) -> None:
        result = classify(
            [
                "src/semantic-router/pkg/extproc/testdata/router_learning_sessions.v1.yaml"
            ]
        )
        self.assertIn("core-tests", result.selected_jobs)

    def test_unrelated_website_docs_do_not_select_core_tests(self) -> None:
        """The api-docs rule must not drag ordinary docs edits into core-tests."""
        result = classify(["website/docs/community/development.md"])

        self.assertNotIn("core-tests", result.selected_jobs)

    def test_recipe_changes_select_conformance_domain(self) -> None:
        result = classify(["config/recipes/privacy/probes.yaml"])

        self.assertTrue(result.signals["recipe_conformance"])
        self.assertIn("recipe-conformance", result.selected_jobs)
        self.assertIn("core-tests", result.selected_jobs)
        self.assertEqual(result.pr_images, ())

    def test_recipe_workflow_change_selects_conformance_domain(self) -> None:
        result = classify([".github/workflows/recipe-conformance.yml"])

        self.assertTrue(result.signals["recipe_conformance"])
        self.assertIn("recipe-conformance", result.selected_jobs)

    def test_workflow_edits_do_not_select_product_domains(self) -> None:
        workflow_paths = (
            ".github/workflows/integration-test-memory.yml",
            ".github/workflows/openvino-binding-ci.yml",
            ".github/workflows/operator-ci.yml",
            ".github/workflows/integration-test-vllm-sr-cli.yml",
            ".github/workflows/performance-test.yml",
            ".github/workflows/router-learning-eval.yml",
        )

        for path in workflow_paths:
            with self.subTest(path=path):
                result = classify([path])
                self.assertEqual(
                    result.selected_jobs,
                    ("quality", "security", "e2e"),
                )
                self.assertEqual(result.profiles, ("envoy-ai-gateway",))
                self.assertEqual(result.pr_images, ())

    def test_composite_action_changes_select_ci_contracts(self) -> None:
        result = classify([".github/actions/free-disk-space/action.yml"])

        self.assertEqual(
            result.selected_jobs,
            ("quality", "security", "core-tests", "e2e"),
        )
        self.assertEqual(result.profiles, ("envoy-ai-gateway",))

    def test_agent_text_paths_remain_docs_light(self) -> None:
        paths = (
            "tools/agent/docs/testing-strategy.md",
            "tools/agent/skills/harness-contract-change/SKILL.md",
            "src/vllm-sr/cli/AGENTS.md",
        )

        for path in paths:
            with self.subTest(path=path):
                result = classify([path])
                self.assertEqual(result.selected_jobs, ("quality",))
                self.assertEqual(result.profiles, ())
                self.assertEqual(result.pr_images, ())

    def test_harness_executable_patterns_set_agent_exec(self) -> None:
        for path in (
            "tools/agent/requirements.txt",
            "tools/make/linter.mk",
            "tools/make/pre-commit.mk",
            ".pre-commit-config.yaml",
        ):
            with self.subTest(path=path):
                result = classify([path])
                self.assertTrue(result.signals["agent_exec"])

    def test_ownership_metadata_paths_remain_lightweight(self) -> None:
        paths = (
            "OWNER",
            ".github/CODEOWNERS",
            "ml-binding/OWNER",
            "src/semantic-router/OWNER",
            "dashboard/OWNER",
            "website/OWNER",
        )

        for path in paths:
            with self.subTest(path=path):
                result = classify([path])
                self.assertEqual(result.selected_jobs, ("quality",))
                self.assertEqual(result.profiles, ())
                self.assertEqual(result.pr_images, ())
                self.assertTrue(result.signals["docs_only"])

    def test_ownership_metadata_does_not_enable_adjacent_product_domains(
        self,
    ) -> None:
        result = classify(
            [
                ".github/CODEOWNERS",
                "dashboard/OWNER",
                "deploy/operator/OWNER",
                "ml-binding/OWNER",
            ]
        )

        self.assertFalse(result.signals["dashboard"])
        self.assertFalse(result.signals["operator"])
        self.assertFalse(result.signals["core"])
        self.assertFalse(result.signals["e2e_ml_model_selection"])
        self.assertEqual(result.selected_jobs, ("quality",))

    def test_core_main_publish_excludes_platform_variants(self) -> None:
        result = classify(["src/semantic-router/pkg/extproc/processor.go"])

        self.assertEqual(result.publish_images, ("extproc", "vllm-sr"))
        self.assertNotIn("extproc-rocm", result.publish_images)
        self.assertNotIn("vllm-sr-cuda", result.publish_images)
        self.assertNotIn("vllm-sr-rocm", result.publish_images)

    def test_fixture_builds_follow_active_integration_receipts(self) -> None:
        llm_katan = classify(["e2e/testing/llm-katan/llm_katan/server.py"])
        anthropic = classify(["e2e/testing/anthropic-shim/anthropic_shim/app.py"])
        responses = classify(["tools/mock-vllm/app.py"])

        self.assertIn("memory", llm_katan.selected_jobs)
        self.assertEqual(llm_katan.pr_images, ())
        self.assertIn("anthropic-shim", anthropic.profiles)
        self.assertIn("e2e", anthropic.selected_jobs)
        self.assertEqual(anthropic.pr_images, ())
        self.assertEqual(responses.profiles, ("response-api",))
        self.assertEqual(responses.pr_images, ())
        self.assertEqual(llm_katan.publish_images, ())
        self.assertEqual(anthropic.publish_images, ())

    def test_protocol_codec_changes_select_every_protocol_profile(self) -> None:
        result = classify(["src/semantic-router/pkg/protocolcodec/stream_responses.go"])

        self.assertEqual(
            result.profiles,
            (
                "envoy-ai-gateway",
                "streaming",
                "anthropic-shim",
                "response-api",
            ),
        )

    def test_release_and_nightly_image_lifecycles_are_distinct(self) -> None:
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
