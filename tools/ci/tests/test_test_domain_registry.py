from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools" / "ci"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "agent" / "scripts"))

from agent_context_resolution import resolve_context  # noqa: E402
from classify_pr_changes import E2E_RULES, NON_PR_E2E_RULES  # noqa: E402
from test_domain_registry import (  # noqa: E402
    domain_records,
    profile_paths,
    registry_schema_errors,
    suite_domains,
)


class TestDomainRegistryTest(unittest.TestCase):
    def test_registry_schema_is_complete(self) -> None:
        self.assertEqual(registry_schema_errors(), [])

    def test_pr_profile_paths_match_classifier(self) -> None:
        self.assertEqual(
            profile_paths("pr"),
            {name: tuple(paths) for name, paths in E2E_RULES.items()},
        )

    def test_manual_profile_paths_match_classifier_reporting(self) -> None:
        manual = profile_paths("manual")
        self.assertTrue(set(NON_PR_E2E_RULES).issubset(manual))
        self.assertEqual(
            {name: manual[name] for name in NON_PR_E2E_RULES},
            {name: tuple(paths) for name, paths in NON_PR_E2E_RULES.items()},
        )

    def test_registry_models_every_pr_domain_job(self) -> None:
        self.assertEqual(
            {data["pr_job"] for data in domain_records().values()},
            {
                "quality",
                "security",
                "core-tests",
                "dashboard",
                "operator",
                "e2e",
                "cli",
                "memory",
                "recipe-conformance",
                "openvino",
                "images",
                "performance",
                "router-learning",
                "paper",
            },
        )

    def test_workflow_suites_are_unique(self) -> None:
        suites = suite_domains()
        self.assertEqual(
            set(suites),
            {
                "recipe-conformance-live",
                "vllm-sr-cli-integration",
                "memory-integration",
            },
        )
        self.assertEqual(
            len(suites),
            len({domain_name for domain_name, _ in suites.values()}),
        )

    def test_local_recipe_selection_uses_registry_paths_and_commands(self) -> None:
        context = resolve_context(["src/semantic-router/pkg/projectiontrace/trace.go"])

        self.assertIn("maintained-recipes", context.matched_rules)
        self.assertIn("make recipe-conformance-static", context.fast_tests)
        self.assertIn(
            "recipe-conformance-live",
            context.workflow_integration_suites,
        )
        self.assertIn("recipe-conformance", context.ci_test_domains)
        self.assertIn("recipe-conformance-report", context.expected_artifacts)

    def test_local_profiles_include_hosted_baseline_injection(self) -> None:
        context = resolve_context(
            ["deploy/operator/controllers/semanticrouter_controller.go"]
        )

        self.assertIn("envoy-ai-gateway", context.ci_e2e_profiles)
        self.assertIn("envoy-ai-gateway", context.local_e2e_profiles)

    def test_harness_make_change_stays_on_lightweight_local_path(self) -> None:
        context = resolve_context(["tools/make/agent.mk"])

        self.assertFalse(context.requires_local_smoke)
        self.assertEqual(context.local_e2e_profiles, [])
        self.assertEqual(context.ci_e2e_profiles, [])


if __name__ == "__main__":
    unittest.main()
