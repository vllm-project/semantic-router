from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools" / "ci"))

from classify_pr_changes import classify, select_profiles  # noqa: E402
from domain_registry import (  # noqa: E402
    commands_for_domains,
    domain_records,
    job_records,
    matching_domains,
    path_matches,
    profile_paths,
    registry_schema_errors,
)


class DomainRegistryTests(unittest.TestCase):
    def test_registry_schema_is_complete(self) -> None:
        self.assertEqual(registry_schema_errors(), [])

    def test_registry_is_the_only_changed_path_source(self) -> None:
        self.assertTrue((REPO_ROOT / "tools/agent/domains.yaml").is_file())
        self.assertFalse((REPO_ROOT / "tools/agent/test-domain-registry.yaml").exists())
        self.assertFalse((REPO_ROOT / "tools/agent/task-matrix.yaml").exists())

    def test_path_matching_distinguishes_one_level_from_recursive(self) -> None:
        self.assertTrue(path_matches("e2e/root.go", "e2e/*.go"))
        self.assertFalse(path_matches("e2e/testcases/root.go", "e2e/*.go"))
        self.assertTrue(path_matches("e2e/testcases/root.go", "e2e/**/*.go"))
        self.assertTrue(path_matches("src/file.py", "src/**/*.py"))

    def test_pr_profile_paths_are_used_directly_by_classifier(self) -> None:
        paths = profile_paths("pr")
        for name, patterns in paths.items():
            with self.subTest(profile=name):
                concrete = patterns[0].replace("**", "case").replace("*", "case")
                self.assertIn(
                    name,
                    select_profiles((concrete,), full=False, suppress_expensive=False),
                )

    def test_domain_matching_can_report_overlapping_owners(self) -> None:
        self.assertEqual(
            matching_domains(("config/recipes/privacy/probes.yaml",)),
            ("router-core", "maintained-recipes"),
        )

    def test_memory_implementation_and_split_suite_select_live_integration(
        self,
    ) -> None:
        for path in (
            "src/semantic-router/pkg/memory/store.go",
            "src/semantic-router/pkg/memory/retrieval/store.go",
            "e2e/testing/memory_tests/test_retrieval.py",
            "e2e/testing/memory_tests/helpers/client.py",
            "e2e/testing/run_memory_integration.sh",
            "tools/make/milvus.mk",
        ):
            with self.subTest(path=path):
                self.assertIn("memory", matching_domains((path,)))
                self.assertIn(
                    "local.memory",
                    classify((path,)).selected_jobs,
                )

    def test_domain_commands_are_deduplicated_in_registry_order(self) -> None:
        commands = commands_for_domains(
            ("router-core", "dashboard", "maintained-recipes"), "checks"
        )
        self.assertEqual(
            commands,
            (
                "make test-semantic-router",
                "make config-schema-check",
                "make dashboard-check",
                "make recipe-conformance-static",
            ),
        )

    def test_modelcompat_tool_keeps_test_and_ci_coverage(self) -> None:
        for path in (
            "tools/modelcompat/main.go",
            "tools/modelcompat/main_test.go",
            "src/semantic-router/pkg/modelruntime/compatibility/receipt.go",
            "tools/make/models.mk",
        ):
            with self.subTest(path=path):
                domains = matching_domains((path,))
                self.assertIn(
                    "make check-modelcompat", commands_for_domains(domains, "checks")
                )
                self.assertIn("core", commands_for_domains(domains, "verifications"))

    def test_every_domain_job_is_declared_once(self) -> None:
        jobs = job_records()
        for name, domain in domain_records().items():
            with self.subTest(domain=name):
                self.assertTrue(set(domain["verifications"]).issubset(jobs))

    def test_performance_checks_do_not_repeat_the_integration_gate(self) -> None:
        for path in (
            "perf/pkg/benchmark/model_identity.go",
            "perf/benchmarks/cache_bench_test.go",
            "tools/make/performance.mk",
            "tools/make/models.mk",
            "src/semantic-router/tools/model-test-assets/main.go",
        ):
            with self.subTest(path=path):
                result = classify([path])
                checks = commands_for_domains(result.domains, "checks")
                self.assertIn("make perf-test-unit", checks)
                self.assertNotIn("make perf-check", checks)
                self.assertIn(
                    "make perf-check",
                    commands_for_domains(result.domains, "verify"),
                )
                self.assertIn("performance", result.selected_jobs)

    def test_shared_model_inputs_select_every_artifact_consumer(self) -> None:
        for path in (
            "src/semantic-router/pkg/config/registry.go",
            "src/semantic-router/pkg/config/canonical_defaults.go",
            "src/semantic-router/pkg/config/canonical_global.go",
            "src/semantic-router/pkg/modeldownload/downloader.go",
            "src/semantic-router/pkg/modeldownload/revision_receipt.go",
            "src/semantic-router/pkg/modeldownload/validator.go",
        ):
            with self.subTest(path=path):
                result = classify([path])
                self.assertIn("model-artifacts", result.domains)
                self.assertTrue(
                    {
                        "native.candle-cpu",
                        "native.ort-cpu",
                        "native.openvino-cpu",
                        "performance",
                    }
                    <= set(result.selected_jobs)
                )
                self.assertNotIn(
                    "make perf-check", commands_for_domains(result.domains, "checks")
                )

    def test_shared_model_inputs_do_not_expand_unrelated_or_unit_only_changes(
        self,
    ) -> None:
        for path in (
            "src/semantic-router/pkg/config/tool_selection_plugin.go",
            "src/semantic-router/pkg/config/vela_defaults_test.go",
        ):
            with self.subTest(path=path):
                self.assertFalse(
                    {
                        "native.candle-cpu",
                        "native.ort-cpu",
                        "native.openvino-cpu",
                        "performance",
                    }
                    & set(classify([path]).selected_jobs)
                )

    def test_generated_contract_sources_and_outputs_select_the_drift_gate(self) -> None:
        for path in (
            "src/semantic-router/pkg/apiserver/route_config.go",
            "src/semantic-router/pkg/catalog/catalog.go",
            "src/semantic-router/pkg/config/canonical.go",
            "src/semantic-router/pkg/configschema/router-config-v0.3.schema.json",
            "dashboard/frontend/src/generated/routerConfigContract.ts",
            "tools/codegen/configschema/main.go",
            "tools/codegen/openapi/main.go",
            "tools/codegen/embed_generated_index.py",
            "tools/make/docs.mk",
            "tools/make/golang.mk",
            "website/static/openapi/apiserver/apiserver.openapi.json",
            "website/docs/api/apiserver.md",
        ):
            with self.subTest(path=path):
                domains = matching_domains((path,))
                self.assertIn(
                    "make generated-contract-check",
                    commands_for_domains(domains, "checks"),
                )
                self.assertIn(
                    "generated-contracts",
                    commands_for_domains(domains, "verifications"),
                )

    def test_skill_only_changes_keep_the_lightweight_gate(self) -> None:
        domains = matching_domains(
            (
                "tools/agent/skills/vllm-sr-agent-operations/references/configuration-loop.md",
                "website/static/install/agent/vllm-sr/references/configuration-loop.md",
            )
        )
        self.assertIn("make harness-check", commands_for_domains(domains, "checks"))
        self.assertNotIn(
            "make generated-contract-check", commands_for_domains(domains, "checks")
        )

    def test_relocated_tools_retain_unit_check_ownership(self) -> None:
        cases = {
            "tools/dev/dsl/main.go": "make go-tools-test",
            "tools/models/classifier-operating-point/main.go": "make go-tools-test",
            "tools/calibration/image-routing/main.go": "make go-tools-test",
            "bench/grounded_fusion/fusioneval/main.go": "make go-tools-test",
            "tools/calibration/tuning/engine.py": "make test-calibration",
            "tools/test/services/mock-vllm/app.py": "make test-provider-simulator",
        }
        for path, command in cases.items():
            with self.subTest(path=path):
                domains = matching_domains((path,))
                self.assertIn(command, commands_for_domains(domains, "checks"))
                verification = {
                    "make test-calibration": "learning-tools",
                    "make test-provider-simulator": "mock-provider",
                }.get(command, "core")
                self.assertIn(
                    verification, commands_for_domains(domains, "verifications")
                )

    def test_reference_sources_select_checks_without_editing_outputs(self) -> None:
        cases = {
            "config/recipes/built-in/latest/mom-v1/recipe.dsl": "make model-catalog-generated-check",
            "config/catalog/resources/models/virtual/vllm-sr.yaml": "make model-catalog-generated-check",
            "website/static/model-catalog/catalog.json": "make model-catalog-generated-check",
            "src/vllm-sr/cli/commands/request.py": "make docs-cli-check",
            "website/docs/api/cli.md": "make docs-cli-check",
            "config/fragments/signals/heuristic/keyword.yaml": "make docs-config-check",
            "website/docs/tutorials/signal/heuristic/keyword.md": "make docs-config-check",
            "deploy/operator/api/v1alpha1/semanticrouter_types.go": "make docs-crd-check",
            "website/docs/api/crd-reference.md": "make docs-crd-check",
            "website/scripts/generate-contributor-rank.mjs": "make docs-community-check",
            "website/src/data/teamMembers.tsx": "make docs-community-check",
            "website/src/data/committerActivity.generated.ts": "make docs-community-check",
        }
        for path, command in cases.items():
            with self.subTest(path=path):
                self.assertIn(
                    command, commands_for_domains(matching_domains((path,)), "checks")
                )


if __name__ == "__main__":
    unittest.main()
