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
    "core router": (
        ["src/semantic-router/pkg/extproc/processor.go"],
        ("quality", "security", "core-tests", "e2e", "images"),
        ("kubernetes",),
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
        ("kubernetes",),
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
        ("kubernetes",),
        (),
    ),
    "native binding": (
        ["candle-binding/src/lib.rs"],
        ("quality", "security", "core-tests", "e2e", "images"),
        ("kubernetes",),
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
        ("kubernetes",),
        (),
    ),
    "Docker validator only": (
        [".github/workflows/docker-validate.yml"],
        ("quality", "security", "e2e", "images"),
        ("kubernetes",),
        ("vllm-sr",),
    ),
    "Docker publisher only": (
        [".github/workflows/docker-publish.yml"],
        ("quality", "security", "e2e"),
        ("kubernetes",),
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
        ("kubernetes",),
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
            ("quality", "security", "core-tests", "e2e"),
        )
        self.assertEqual(result.profiles, ("kubernetes",))
        self.assertEqual(result.pr_images, ())

    def test_ci_full_does_not_enable_performance(self) -> None:
        result = classify(["README.md"], full=True)

        self.assertNotIn("performance", result.selected_jobs)
        self.assertEqual(
            result.profiles,
            ("kubernetes", "dashboard", "remote-embedding"),
        )

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
                self.assertEqual(result.profiles, ("kubernetes",))
                self.assertEqual(result.pr_images, ())

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

    def test_core_main_publish_excludes_platform_variants(self) -> None:
        result = classify(["src/semantic-router/pkg/extproc/processor.go"])

        self.assertEqual(result.publish_images, ("extproc", "vllm-sr"))
        self.assertNotIn("extproc-rocm", result.publish_images)
        self.assertNotIn("vllm-sr-cuda", result.publish_images)
        self.assertNotIn("vllm-sr-rocm", result.publish_images)

    def test_fixture_builds_are_owned_by_active_integration_receipts(self) -> None:
        llm_katan = classify(["e2e/testing/llm-katan/llm_katan/server.py"])
        anthropic = classify(["e2e/testing/anthropic-shim/anthropic_shim/app.py"])

        self.assertIn("memory", llm_katan.selected_jobs)
        self.assertEqual(llm_katan.pr_images, ())
        self.assertIn("anthropic-shim", anthropic.profiles)
        self.assertEqual(anthropic.pr_images, ())
        self.assertEqual(llm_katan.publish_images, ())
        self.assertEqual(anthropic.publish_images, ())

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
