from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/ci"))
from ci_plan import (  # noqa: E402
    github_outputs,
    make_plan,
    performance_base,
    previous_release,
)
from ci_plan import (  # noqa: E402
    main as plan_main,
)
from classify_pr_changes import (  # noqa: E402
    CLI_RUNTIME_IMAGES,
    PRODUCTION_RELEASE_IMAGES,
    classify,
    full_e2e_profiles,
)
from domain_registry import load_domain_registry, profile_records  # noqa: E402
from verification_catalog import (  # noqa: E402
    full_cpu_ids,
    load_catalog,
    profile_image_dependencies,
    verification_records,
)

SHA = "a" * 40
MODELS = "platform.models-cpu"


class SelectionTests(unittest.TestCase):
    def test_shared_e2e_oracles_select_their_live_consumers(self):
        expected_by_source = {
            "signal_routing_helpers.go": {
                "envoy-ai-gateway",
                "structure-routing",
                "conversation-routing",
                "complexity-routing",
                "long-context-routing",
                "preference-routing",
            },
            "failover_during_traffic.go": {"production-stack"},
            "streaming_body.go": {"streaming"},
            "response_api_basic.go": {"response-api"},
        }
        for source, profiles in expected_by_source.items():
            with self.subTest(source=source):
                selected = set(classify([f"e2e/testcases/{source}"]).selected_jobs)
                self.assertTrue({f"e2e.{name}" for name in profiles} <= selected)
                self.assertNotIn("e2e.response-api-redis", selected)
                self.assertNotIn("e2e.response-api-redis-cluster", selected)

    def test_no_route_contract_keeps_an_isolated_required_gateway_profile(self):
        for path in (
            "e2e/profiles/routing-errors/values.yaml",
            "e2e/testcases/routing_error_codes.go",
            "src/semantic-router/pkg/extproc/routing_failure.go",
            "src/semantic-router/pkg/extproc/req_filter_classification_runtime.go",
        ):
            with self.subTest(path=path):
                self.assertIn("e2e.routing-errors", classify([path]).selected_jobs)
        self.assertIn("e2e.routing-errors", full_cpu_ids())
        record = make_plan([], source_sha=SHA, requested=("e2e.routing-errors",))[
            "verifications"
        ][0]
        self.assertEqual(record["profile"], "routing-errors")
        self.assertEqual(set(record["images"]), {"vllm-sr", "provider-mocker"})

    def test_manual_platform_selection_uses_the_same_plan_contract_and_build(self):
        plan = make_plan([], source_sha=SHA, requested=(MODELS,))
        self.assertEqual(plan["expected_verification_ids"], [MODELS])
        self.assertEqual(plan["images"], [])
        self.assertFalse(plan["publish_images"])
        record = plan["verifications"][0]
        self.assertEqual(record["source_sha"], SHA)
        self.assertEqual(record["platform_id"], "model-runtime-cpu")
        self.assertEqual(record["workflow"], ".github/workflows/test-platform.yml")
        self.assertEqual(record["reasons"], ["manual-selection"])
        self.assertIn(MODELS, full_cpu_ids())
        for requested in (("unknown",), (MODELS, MODELS)):
            with self.assertRaises(ValueError):
                make_plan([], source_sha=SHA, requested=requested)
        for overrides in ({"full": True}, {"draft": True}, {"profile": "release"}):
            with self.assertRaises(ValueError):
                make_plan([], source_sha=SHA, requested=(MODELS,), **overrides)
        with self.assertRaises(ValueError):
            make_plan(["README.md"], source_sha=SHA, requested=(MODELS,))

    def test_manual_selection_is_generic_and_never_publishes(self):
        for name in (MODELS, "local.cli", "cli-package"):
            with self.subTest(name=name):
                plan = make_plan([], source_sha=SHA, requested=(name,))
                self.assertEqual(plan["expected_verification_ids"], [name])
                self.assertEqual(plan["profile"], "pr")
                self.assertFalse(plan["publish_images"])
                self.assertFalse(plan["publish_python"])
                self.assertFalse(plan["publish_helm"])
                record = plan["verifications"][0]
                self.assertEqual(set(plan["images"]), set(record["images"]))
        workflow = yaml.load(
            (ROOT / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader
        )
        self.assertEqual(
            workflow["on"]["workflow_dispatch"]["inputs"]["verification"]["type"],
            "string",
        )
        for job in (
            "image-router",
            "image-fixtures",
            "image-distribution",
            "package",
        ):
            self.assertEqual(
                workflow["jobs"][job]["with"]["mode"],
                "${{ fromJSON(needs.plan.outputs.plan).profile }}",
            )
        reconciliation = [
            step["run"]
            for step in workflow["jobs"]["gate"]["steps"]
            if "check_ci_gate.py" in step.get("run", "")
        ]
        self.assertEqual(len(reconciliation), 1)
        self.assertIn("--results", reconciliation[0])

    def test_docs_and_paper_do_not_schedule_models_or_containers(self):
        for path in ("README.md", "website/docs/overview.md", "paper/main.tex"):
            selected = classify([path])
            self.assertFalse(
                any(
                    name.startswith(("platform.", "e2e.", "local."))
                    for name in selected.selected_jobs
                )
            )
            self.assertEqual(selected.pr_images, ())
            self.assertNotIn("paper", selected.selected_jobs)

    def test_platform_entrypoints_select_their_consumers(self):
        fixtures = {
            "tools/make/models.mk": {MODELS, "performance"},
            "tools/make/common.mk": {"performance"},
            "tools/make/build-run-test.mk": {"performance"},
            "src/semantic-router/go.mod": {"core"},
        }
        for path, expected in fixtures.items():
            with self.subTest(path=path):
                self.assertTrue(expected <= set(classify([path]).selected_jobs))

    def test_shared_pins_and_downloads_select_all_consumers_even_for_tests(self):
        for path in (
            "src/semantic-router/pkg/config/registry.go",
            "src/semantic-router/pkg/config/canonical_defaults.go",
            "src/semantic-router/pkg/config/canonical_global.go",
            "src/semantic-router/pkg/modeldownload/revision_receipt_test.go",
        ):
            self.assertTrue(
                {"local.cli", "performance"} <= set(classify([path]).selected_jobs),
                path,
            )

    def test_test_names_cannot_downgrade_integration_boundaries(self):
        cases = {
            "e2e/testing/vllm-sr-cli/test_integration.py": {"local.cli"},
            "e2e/testing/memory_tests/test_retrieval.py": {"local.memory"},
            "src/semantic-router/pkg/cache/redis_exact_cache_integration_test.go": {
                "storage"
            },
            "e2e/testcases/authz_header_spoofing_test.go": {"e2e.authz-rbac"},
        }
        for path, expected in cases.items():
            self.assertTrue(classify([path]).test_only)
            self.assertTrue(expected <= set(classify([path]).selected_jobs), path)

    def test_cli_lifecycle_and_envoy_sources_select_live_container_contracts(self):
        paths = [
            str(path.relative_to(ROOT))
            for pattern in (
                "src/vllm-sr/cli/container_*.py",
                "src/vllm-sr/cli/runtime_*.py",
                "src/vllm-sr/cli/commands/runtime*.py",
                "src/vllm-sr/cli/templates/envoy*.yaml",
            )
            for path in ROOT.glob(pattern)
        ]
        paths.extend(
            (
                "src/vllm-sr/cli/core.py",
                "src/vllm-sr/cli/config_generator.py",
                "src/vllm-sr/cli/config_translator.py",
                "src/vllm-sr/cli/catalog_provider_projection.py",
                "src/vllm-sr/cli/envoy_backend_pool.py",
                "src/vllm-sr/cli/deployment_backend.py",
                "src/vllm-sr/cli/container_future_component.py",
                "src/vllm-sr/cli/runtime_future_component.py",
                "src/vllm-sr/cli/commands/runtime_future_component.py",
            )
        )
        for path in paths:
            for profile in ("pr", "main"):
                with self.subTest(path=path, profile=profile):
                    plan = make_plan(
                        [path],
                        source_sha=SHA,
                        profile=profile,
                        full=profile == "main",
                        publish=profile == "main",
                    )
                    self.assertIn("local.cli", plan["expected_verification_ids"])
                    self.assertTrue({"vllm-sr", "dashboard"} <= set(plan["images"]))
        for path in (
            "src/vllm-sr/README.md",
            "src/vllm-sr/tests/test_container_start.py",
            "src/vllm-sr/cli/evaluation/runtime_factors.py",
            "src/vllm-sr/cli/commands/chat.py",
        ):
            with self.subTest(unrelated_path=path):
                plan = make_plan([path], source_sha=SHA)
                self.assertNotIn("local.cli", plan["expected_verification_ids"])
                self.assertEqual(plan["images"], [])

    def test_operator_request_helper_selects_its_real_deployment(self):
        path = "tools/ci/check_operator_request.py"
        self.assertTrue((ROOT / path).is_file())
        plan = make_plan([path], source_sha=SHA)
        self.assertIn("operator", plan["expected_verification_ids"])
        self.assertTrue(
            {"operator", "operator-bundle", "vllm-sr", "provider-mocker"}
            <= set(plan["images"])
        )
        self.assertNotIn(
            "operator",
            classify(["tools/ci/tests/test_operator_request.py"]).selected_jobs,
        )

    def test_owning_workflow_edits_select_executor_contracts(self):
        cases = {
            "test-platform.yml": {MODELS},
            "test-local.yml": {"local.cli", "local.memory"},
            "performance-test.yml": {"performance"},
            "operator-ci.yml": {"operator"},
            "integration-test-k8s.yml": {"e2e.envoy-ai-gateway"},
        }
        for workflow, expected in cases.items():
            selected = set(classify([f".github/workflows/{workflow}"]).selected_jobs)
            self.assertTrue(expected <= selected)

    def test_component_tools_have_execution_owners_after_quality_split(self):
        cases = {
            "src/vllm-sr/cli/core.py": "cli-unit",
            "src/training/tests/test_export.py": "training",
            "tools/ci/training-test-requirements.txt": "training",
            "bench/redteam/test_datasets.py": "training",
            "tools/test/services/provider-mocker/tests/test_fixture_latency.py": "mock-provider",
            "bench/test_agentic_routing_experiment.py": "learning-tools",
            "bench/test_openai_fault_proxy.py": "soak-tools",
            "bench/looper_tts/test_contracts.py": "bench-tools",
        }
        for path, expected in cases.items():
            self.assertIn(expected, classify([path]).selected_jobs)
        self.assertNotIn(
            "learning-tools",
            classify(
                [
                    "src/semantic-router/pkg/extproc/router_learning_session_runner_test.go"
                ]
            ).selected_jobs,
        )

    def test_shared_mock_changes_select_declared_pr_consumers(self):
        for path in (
            "tools/test/services/provider-mocker/provider_mocker/app.py",
            "tools/test/services/provider-mocker/requirements.txt",
            "tools/test/services/provider-mocker/tests/test_fixture_latency.py",
        ):
            plan = make_plan([path], source_sha=SHA)
            expected = {
                "e2e." + name
                for name in profile_records(selection="pr")
                if "provider-mocker" in profile_image_dependencies()[name]
            }
            self.assertTrue(expected <= set(plan["expected_verification_ids"]))
            self.assertIn("operator", plan["expected_verification_ids"])
            self.assertIn("mock-provider", plan["expected_verification_ids"])
            self.assertIn("provider-mocker", plan["images"])
            self.assertNotIn(
                "e2e.response-api-redis", plan["expected_verification_ids"]
            )

    def test_full_cpu_profiles_share_explicit_inventory(self):
        plans = [
            make_plan(["README.md"], source_sha=SHA, profile=profile, full=True)
            for profile in ("pr", "nightly", "release")
        ]
        self.assertEqual(
            {tuple(plan["expected_verification_ids"]) for plan in plans},
            {tuple(plans[0]["expected_verification_ids"])},
        )
        self.assertTrue(
            set(full_cpu_ids()) <= set(plans[0]["expected_verification_ids"])
        )
        for profile in ("standalone", "envoy-ai-gateway", "streaming"):
            self.assertIn(profile, full_e2e_profiles())
        self.assertNotIn("e2e.dynamo", plans[0]["expected_verification_ids"])
        self.assertNotIn("e2e.router-replay", plans[0]["expected_verification_ids"])

    def test_retired_gateway_profiles_have_no_live_selection_entrypoint(self):
        records = verification_records(load_domain_registry())
        full = make_plan([], source_sha=SHA, full=True)
        for profile in ("agentgateway", "aibrix", "dynamo", "istio", "llm-d"):
            identity = f"e2e.{profile}"
            with self.subTest(profile=profile):
                self.assertNotIn(profile, profile_records())
                self.assertNotIn(identity, records)
                self.assertNotIn(identity, full["expected_verification_ids"])
                self.assertNotIn(
                    identity,
                    classify([f"e2e/profiles/{profile}/profile.go"]).selected_jobs,
                )
                with self.assertRaisesRegex(ValueError, "unknown or duplicated"):
                    make_plan([], source_sha=SHA, requested=(identity,))
        for path in (
            "e2e/testcases/full_duplex_routing.go",
            "src/semantic-router/pkg/extproc/processor_req_body_full_duplex.go",
        ):
            with self.subTest(path=path):
                self.assertIn("e2e.streaming", classify([path]).selected_jobs)

    def test_authoring_contracts_remain_affected_and_manual_without_full_obligation(
        self,
    ):
        cases = {
            "training": "src/training/tests/test_model_artifacts.py",
            "onnx-artifacts": "tools/models/onnx/onnx_artifacts.py",
            "bench-tools": "bench/grounded_fusion/evaluate.py",
            "learning-tools": "bench/test_agentic_routing_experiment.py",
            "soak-tools": "bench/test_openai_fault_proxy.py",
        }
        baseline = make_plan([], source_sha=SHA, full=True)
        self.assertTrue(set(cases).isdisjoint(baseline["expected_verification_ids"]))
        self.assertEqual(len(baseline["verifications"]), 25)
        self.assertEqual(len(baseline["component_batches"]), 5)
        for identity, path in cases.items():
            with self.subTest(identity=identity):
                self.assertIn(identity, classify([path]).selected_jobs)
                affected_full = make_plan([path], source_sha=SHA, full=True)
                self.assertIn(identity, affected_full["expected_verification_ids"])
                manual = make_plan([], source_sha=SHA, requested=(identity,))
                self.assertEqual(manual["expected_verification_ids"], [identity])
                self.assertEqual(len(manual["component_batches"]), 1)

    def test_agent_task_migration_keeps_native_and_cli_checks(self) -> None:
        for path in (
            "bench/agent_task_live_benchmark.py",
            "bench/test_agent_task_live_benchmark.py",
            "src/vllm-sr/cli/sr_bench/agent_task.py",
            "src/vllm-sr/cli/sr_bench/agent_task_kernel.py",
            "src/vllm-sr/tests/test_sr_bench_agent_task.py",
        ):
            with self.subTest(path=path):
                result = classify([path])
                self.assertIn("agent-task-bench", result.domains)
                self.assertIn("bench-tools", result.selected_jobs)
                if path.startswith("src/"):
                    self.assertIn("cli-unit", result.selected_jobs)
        domain = load_domain_registry()["domains"]["agent-task-bench"]
        self.assertEqual(domain["checks"], ["make test-agent-task-bench"])
        self.assertEqual(domain["verify"], ["make test-agent-task-bench"])

    def test_shared_component_execution_selects_every_component_consumer(self):
        components = {
            identity
            for identity, record in verification_records(load_domain_registry()).items()
            if record["executor"] == "tools"
        }
        for path in (
            ".github/workflows/test-tools.yml",
            ".github/workflows/ci.yml",
            "tools/ci/run_component_batch.py",
            "tools/ci/worker_process.py",
            "tools/ci/execution_batches.py",
            "tools/ci/execution_targets.yaml",
            "tools/ci/ci_plan.py",
            "tools/ci/ci_results.py",
            "tools/ci/verification_catalog.yaml",
            "tools/agent/domains.yaml",
            "tools/agent/requirements.txt",
            "tools/docs/requirements.txt",
            "tools/make/agent.mk",
        ):
            with self.subTest(path=path):
                self.assertTrue(components <= set(classify([path]).selected_jobs))
        python_consumers = components - {"e2e-unit", "soak-tools"}
        self.assertTrue(
            python_consumers
            <= set(classify(["tools/ci/python_test_evidence.py"]).selected_jobs)
        )

    def test_product_full_inventory_keeps_every_published_support_contract(self):
        supported = load_catalog()["full_cpu"]["support_contracts"]
        self.assertEqual(
            set(supported),
            {
                "Helm chart",
                "Local deployment",
                "Kubernetes Operator",
                "Agent Router",
                "Streaming with Agent Router",
            },
        )
        required = set(full_cpu_ids())
        self.assertTrue(
            {identity for ids in supported.values() for identity in ids} <= required
        )
        self.assertTrue({MODELS, "model-runtime", "storage"} <= required)

    def test_full_cpu_preserves_affected_pr_and_publication_verifications(self):
        for path in (
            "e2e/profiles/route-action/profile.go",
            "tools/test/services/provider-mocker/provider_mocker/app.py",
        ):
            affected = make_plan([path], source_sha=SHA)
            for options in (
                {"profile": "pr", "full": True},
                {"profile": "main", "full": True, "publish": True},
            ):
                with self.subTest(path=path, options=options):
                    complete = make_plan([path], source_sha=SHA, **options)
                    self.assertTrue(
                        set(affected["expected_verification_ids"])
                        <= set(complete["expected_verification_ids"])
                    )
                    self.assertTrue(set(affected["images"]) <= set(complete["images"]))
                    self.assertNotIn(
                        "e2e.response-api-redis", complete["expected_verification_ids"]
                    )

    def test_no_base_qualification_does_not_treat_all_files_as_changes(self):
        for profile, flags in (
            ("nightly", []),
            ("release", []),
            ("main", ["--publish", "--full"]),
        ):
            with (
                self.subTest(profile=profile),
                tempfile.TemporaryDirectory() as temporary,
            ):
                output = Path(temporary) / "plan.json"
                argv = [
                    "ci_plan.py",
                    "--profile",
                    profile,
                    "--head",
                    SHA,
                    "--output",
                    str(output),
                    *flags,
                ]
                with (
                    patch.object(sys, "argv", argv),
                    patch(
                        "ci_plan.git_changed_files",
                        return_value=["e2e/profiles/production-stack/profile.go"],
                    ) as changed,
                    patch("ci_plan.resolve_image_sources"),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(plan_main(), 0)
                changed.assert_not_called()
                plan = json.loads(output.read_text())
                self.assertEqual(
                    set(plan["expected_verification_ids"]), set(full_cpu_ids())
                )
                self.assertEqual(plan["paths"], [])

    def test_experimental_profile_remains_available_for_changes_and_manual_runs(self):
        identity = "e2e.production-stack"
        self.assertNotIn(identity, full_cpu_ids())
        affected = make_plan(
            ["e2e/profiles/production-stack/profile.go"], source_sha=SHA
        )
        self.assertIn(identity, affected["expected_verification_ids"])
        manual = make_plan([], source_sha=SHA, requested=(identity,))
        self.assertEqual(manual["expected_verification_ids"], [identity])

    def test_build_union_is_unique_and_separate_from_publication(self):
        plan = make_plan(
            [
                "e2e/testing/run_memory_integration.sh",
                "e2e/testing/run_recipe_conformance.sh",
            ],
            source_sha=SHA,
        )
        self.assertEqual(plan["images"], ["dashboard", "provider-mocker", "vllm-sr"])
        self.assertEqual(plan["publish_images"], [])
        baseline = make_plan(["e2e/profiles/ai-gateway/profile.go"], source_sha=SHA)
        self.assertIn("provider-mocker", baseline["images"])

    def test_unpublished_main_run_checks_source_without_artifacts(self):
        for path in (
            "README.md",
            "src/vllm-sr/cli/core.py",
            "e2e/testing/vllm-sr-cli/test_unit_serve.py",
            "src/semantic-router/pkg/config/loader.go",
        ):
            with self.subTest(path=path):
                plan = make_plan([path], source_sha=SHA, profile="main")
                self.assertFalse(plan["publish_python"])
                self.assertEqual(plan["publish_images"], [])
                self.assertEqual(plan["images"], [])
                self.assertFalse(plan["publish_helm"])
                self.assertEqual(plan["deferred_verifications"], [])
                self.assertFalse(plan["approval_required"])

    def test_main_push_publishes_its_complete_set_after_automatic_checks(self):
        records = verification_records(load_domain_registry())
        for path, helm in (
            ("README.md", False),
            ("src/semantic-router/pkg/config/loader.go", False),
            ("deploy/helm/semantic-router/values.yaml", True),
        ):
            with self.subTest(path=path):
                plan = make_plan([path], source_sha=SHA, profile="main", publish=True)
                self.assertFalse(plan["full_cpu"])
                self.assertTrue(plan["publication"])
                self.assertTrue(
                    all(
                        records[name]["execution"] == "automatic"
                        for name in plan["expected_verification_ids"]
                    )
                )
                self.assertIn("cli-package", plan["expected_verification_ids"])
                self.assertEqual(
                    plan["publish_images"], list(PRODUCTION_RELEASE_IMAGES)
                )
                self.assertEqual(plan["images"], sorted(PRODUCTION_RELEASE_IMAGES))
                self.assertTrue(plan["multiarch"])
                self.assertTrue(plan["publish_python"])
                self.assertEqual(plan["publish_helm"], helm)
        unpublished = make_plan(["README.md"], source_sha=SHA, profile="main")
        self.assertNotIn("cli-package", unpublished["expected_verification_ids"])

    def test_main_publication_dispatch_qualifies_all_commit_addressed_images(self):
        plan = make_plan(
            ["README.md"], source_sha=SHA, profile="main", full=True, publish=True
        )
        self.assertTrue(plan["publish_python"])
        self.assertTrue(plan["publish_helm"])
        self.assertTrue(plan["full_cpu"])
        self.assertLessEqual(set(CLI_RUNTIME_IMAGES), set(plan["publish_images"]))
        self.assertLessEqual(set(plan["publish_images"]), set(plan["images"]))
        self.assertTrue(plan["multiarch"])
        self.assertIn("cli-package", plan["expected_verification_ids"])
        self.assertIn("local.cli", plan["expected_verification_ids"])
        self.assertIn("recipe-conformance", plan["expected_verification_ids"])
        self.assertIn("model-runtime", plan["expected_verification_ids"])
        with self.assertRaisesRegex(ValueError, "main"):
            make_plan(["README.md"], source_sha=SHA, publish=True)

    def test_cli_runtime_images_are_the_first_party_images_the_cli_pulls(self):
        consts = (ROOT / "src/vllm-sr/cli/consts.py").read_text(encoding="utf-8")
        self.assertEqual(
            set(
                re.findall(
                    r'"ghcr\.io/vllm-project/semantic-router/([a-z0-9-]+):', consts
                )
            ),
            set(CLI_RUNTIME_IMAGES),
        )

    def test_previous_release_is_explicit_compatible_and_not_head_parent(self):
        self.assertEqual(
            previous_release(
                "0.4.0", ["v0.2.0", "v0.3.0", "v0.4.0", "v0.4.0-rc1", "v1.0.0"]
            ),
            "v0.3.0",
        )
        with self.assertRaises(ValueError):
            previous_release("1.0.0", ["v0.3.0"])

    def test_release_performance_base_uses_a_compatible_vela_anchor(self):
        self.assertEqual(
            performance_base("0.4.0", ["v0.3.0"]),
            "12597be5ffae2319d856f230d61ca26248eb9b3b",
        )
        # v0.4.0 predates the model runtime; #5886 runs the current harness.
        self.assertEqual(
            performance_base("0.5.0", ["v0.3.0", "v0.4.0"]),
            "4952ca055d0f99adf746b67aa61cfb7223d0739b",
        )

    def test_later_cycles_compare_with_a_release_that_has_the_model_runtime(self):
        tags = ["v0.3.0", "v0.4.0", "v0.5.0", "v0.5.1"]
        self.assertEqual(performance_base("0.6.0", tags), "v0.5.1")
        self.assertEqual(performance_base("0.5.2", tags), "v0.5.1")
        with self.assertRaisesRegex(ValueError, "declare the 0.4.1 base"):
            performance_base("0.4.1", tags)
        with self.assertRaisesRegex(ValueError, "predates the model runtime"):
            performance_base("0.6.0", ["v0.3.0", "v0.4.0"])

    def test_shared_artifact_loaders_select_their_runtime_consumers(self):
        for path in (
            "tools/ci/image_artifacts.py",
            ".github/actions/load-ci-images/action.yml",
        ):
            self.assertTrue(
                {"local.cli", "operator", "e2e.envoy-ai-gateway"}
                <= set(classify([path]).selected_jobs)
            )

    def test_platform_output_lists_one_worker_per_contract(self):
        plan = make_plan(["tools/ci/run_model_tests.py"], source_sha=SHA)
        outputs = github_outputs(plan)
        batches = json.loads(outputs["platform"])
        self.assertEqual([row["verifications"][0]["id"] for row in batches], [MODELS])

    def test_runtime_combinations_are_qualified_rows_not_cartesian_product(self):
        records = verification_records(load_domain_registry())
        platform = [
            record for record in records.values() if record["executor"] == "platform"
        ]
        self.assertEqual(
            {(r["runtime"], r["device"], r["platform"]) for r in platform},
            {("model-runtime", "cpu", "linux/amd64")},
        )
        self.assertIn("cuda", load_catalog()["full_cpu"]["excluded"])


if __name__ == "__main__":
    unittest.main()
