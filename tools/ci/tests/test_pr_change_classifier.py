from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/ci"))
from ci_plan import github_outputs, make_plan, previous_release  # noqa: E402
from classify_pr_changes import classify, full_e2e_profiles  # noqa: E402
from domain_registry import load_domain_registry, profile_records  # noqa: E402
from run_model_tests import CLASSIFIER_TESTS, OWNED_OMNI_TESTS  # noqa: E402
from verification_catalog import (  # noqa: E402
    full_cpu_ids,
    load_catalog,
    profile_image_dependencies,
    verification_records,
)

SHA = "a" * 40
NATIVE = {"native.candle-cpu", "native.ort-cpu"}
IMAGE_CALIBRATION = "native.image-calibration-cpu"


class SelectionTests(unittest.TestCase):
    def test_image_calibration_inputs_select_the_native_verification(self):
        for path in (
            "config/fragments/signal/embedding/image-routing.yaml",
            "tools/calibration/image-routing/main.go",
            "e2e/profiles/multimodal-routing/profile_test.go",
            "e2e/testcases/testdata/image-fixtures/office.jpg",
            "website/static/img/example.png",
            "dashboard/frontend/public/example.png",
            "candle-binding/src/lib.rs",
            "src/semantic-router/pkg/classification/embedding.go",
            "src/semantic-router/pkg/config/registry.go",
            "src/semantic-router/pkg/modeldownload/revision_receipt_test.go",
            "src/semantic-router/tools/model-test-assets/multimodal.go",
            "tools/make/models.mk",
            "tools/make/common.mk",
            "tools/ci/image_calibration.py",
            "tools/ci/runtime_evidence.py",
            "tools/ci/workflow_evidence.py",
            ".github/workflows/build-native.yml",
            ".github/workflows/test-native.yml",
        ):
            with self.subTest(path=path):
                self.assertIn(IMAGE_CALIBRATION, classify([path]).selected_jobs)
        self.assertNotIn(
            IMAGE_CALIBRATION,
            classify(["src/semantic-router/pkg/extproc/processor.go"]).selected_jobs,
        )

    def test_prepared_image_dependencies_select_calibration_and_routing(self):
        expected = {IMAGE_CALIBRATION, "e2e.multimodal-routing"}
        for path in (
            "config/assets/image-routing/manifest.json",
            "tools/calibration/image-routing/prepare_assets.py",
            "tools/calibration/image-routing/testdata/prototype-protocol.json",
            "tools/models/vela_omni/export.py",
            "onnx-binding/src/model_architectures/embedding/omni/image.rs",
            "onnx-binding/src/core/session.rs",
            "onnx-binding/Cargo.lock",
            "src/semantic-router/pkg/embedding/embedding.go",
            "src/semantic-router/pkg/modelruntime/embedding_owned.go",
            "src/semantic-router/pkg/modelruntime/native/embedding.go",
            "src/semantic-router/pkg/modelruntime/native/embedding_omni.go",
            "src/semantic-router/pkg/modelruntime/native/ort_execution.go",
        ):
            with self.subTest(path=path):
                self.assertTrue(expected <= set(classify([path]).selected_jobs))

    def test_every_authored_calibration_asset_selects_its_consumer(self):
        path = ROOT / "tools/calibration/image-routing/testdata/calibration-set.json"
        manifest = json.loads(path.read_text())
        fixtures = {
            row["image_file"] for row in manifest["positives"] + manifest["excluded"]
        } | set(manifest["negatives"])
        self.assertTrue(fixtures)
        for fixture in sorted(fixtures):
            self.assertIn(IMAGE_CALIBRATION, classify([fixture]).selected_jobs, fixture)

    def test_manual_image_calibration_uses_the_same_plan_contract_and_build(self):
        plan = make_plan([], source_sha=SHA, requested=(IMAGE_CALIBRATION,))
        self.assertEqual(plan["expected_verification_ids"], [IMAGE_CALIBRATION])
        self.assertEqual(plan["images"], [])
        self.assertTrue(plan["native"])
        self.assertFalse(plan["publish_images"])
        record = plan["verifications"][0]
        self.assertEqual(record["source_sha"], SHA)
        self.assertEqual(record["platform_id"], "ort-cpu")
        self.assertEqual(record["workflow"], ".github/workflows/test-native.yml")
        self.assertEqual(record["reasons"], ["manual-selection"])
        self.assertIn(IMAGE_CALIBRATION, full_cpu_ids())
        self.assertIn("e2e.multimodal-routing", full_cpu_ids())
        for requested in (("unknown",), (IMAGE_CALIBRATION, IMAGE_CALIBRATION)):
            with self.assertRaises(ValueError):
                make_plan([], source_sha=SHA, requested=requested)
        for overrides in ({"full": True}, {"draft": True}, {"profile": "release"}):
            with self.assertRaises(ValueError):
                make_plan(
                    [], source_sha=SHA, requested=(IMAGE_CALIBRATION,), **overrides
                )
        with self.assertRaises(ValueError):
            make_plan(["README.md"], source_sha=SHA, requested=(IMAGE_CALIBRATION,))

    def test_manual_selection_is_generic_and_never_publishes(self):
        for name in ("native.ort-cpu", "local.cli", "cli-package"):
            with self.subTest(name=name):
                plan = make_plan([], source_sha=SHA, requested=(name,))
                self.assertEqual(plan["expected_verification_ids"], [name])
                self.assertEqual(plan["profile"], "pr")
                self.assertFalse(plan["publish_images"])
                self.assertFalse(plan["publish_python"])
                self.assertFalse(plan["publish_helm"])
                record = plan["verifications"][0]
                self.assertEqual(set(plan["images"]), set(record["images"]))
                self.assertEqual(plan["native"], record["native"])
        workflow = yaml.load(
            (ROOT / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader
        )
        self.assertEqual(
            workflow["on"]["workflow_dispatch"]["inputs"]["verification"]["type"],
            "string",
        )
        for job in (
            "image-router",
            "image-local",
            "image-fixtures",
            "image-distribution",
            "package",
        ):
            self.assertEqual(
                workflow["jobs"][job]["with"]["mode"],
                "${{ fromJSON(needs.plan.outputs.plan).profile }}",
            )
        self.assertIn("--results", workflow["jobs"]["gate"]["steps"][-1]["run"])

    def test_docs_and_paper_do_not_schedule_models_or_containers(self):
        for path in ("README.md", "website/docs/overview.md", "paper/main.tex"):
            selected = classify([path])
            self.assertFalse(
                any(
                    name.startswith(("native.", "e2e.", "local."))
                    for name in selected.selected_jobs
                )
            )
            self.assertEqual(selected.pr_images, ())
            self.assertNotIn("paper", selected.selected_jobs)

    def test_all_native_entrypoints_select_exact_runtime_consumers(self):
        fixtures = {
            "tools/make/openvino.mk": {"native.openvino-cpu"},
            "openvino-binding/openvino_binding_test.go": {"native.openvino-cpu"},
            "tools/make/models.mk": {*NATIVE, "native.openvino-cpu", "performance"},
            "tools/ci/run_model_tests.py": NATIVE,
            "tools/make/rust.mk": {*NATIVE, "core", "performance"},
            "tools/make/common.mk": {*NATIVE, "core", "performance"},
            "src/semantic-router/tools/model-test-assets/main.go": {
                *NATIVE,
                "native.openvino-cpu",
                "performance",
            },
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
                {*NATIVE, "native.openvino-cpu", "performance"}
                <= set(classify([path]).selected_jobs),
                path,
            )

    def test_owned_openvino_selects_shared_runtime_and_exact_provider_inputs(self):
        for path in (
            "src/semantic-router/pkg/modelruntime/embedding_api.go",
            "src/semantic-router/pkg/modelruntime/native/openvino_enabled.go",
            "src/semantic-router/pkg/modelruntime/native/openvino_integration_test.go",
            "src/semantic-router/pkg/classification/classifier_full_context_test.go",
        ):
            with self.subTest(path=path):
                self.assertTrue(
                    {*NATIVE, "native.openvino-cpu"}
                    <= set(classify([path]).selected_jobs)
                )
        for path in (
            "src/semantic-router/pkg/config/default_execution_openvino.go",
            "src/semantic-router/pkg/config/model_deployments.go",
            "src/semantic-router/pkg/config/model_deployments_openvino_test.go",
        ):
            self.assertIn("native.openvino-cpu", classify([path]).selected_jobs, path)
        selected = set(classify(["tools/ci/openvino_evidence.py"]).selected_jobs)
        self.assertTrue({"harness-tools", "native.openvino-cpu"} <= selected)
        self.assertFalse(selected & {*NATIVE, "recipe-conformance", IMAGE_CALIBRATION})
        self.assertNotIn(
            "native.openvino-cpu",
            classify(["tools/ci/run_model_tests.py"]).selected_jobs,
        )

    def test_test_names_cannot_downgrade_integration_boundaries(self):
        cases = {
            "e2e/testing/vllm-sr-cli/test_integration.py": {"local.cli"},
            "e2e/testing/memory_tests/test_retrieval.py": {"local.memory"},
            "src/semantic-router/pkg/cache/redis_exact_cache_integration_test.go": {
                "storage"
            },
            "src/semantic-router/pkg/classification/unified_classifier_integration_test.go": NATIVE,
            "e2e/testcases/istio_routes_test.go": {"e2e.istio"},
        }
        for path, expected in cases.items():
            self.assertTrue(classify([path]).test_only)
            self.assertTrue(expected <= set(classify([path]).selected_jobs), path)

    def test_openvino_shared_build_inputs_select_its_owned_runtime(self):
        for path in (
            "tools/ci/native_artifact.py",
            ".github/workflows/build-native.yml",
            ".github/actions/load-native-artifact/action.yml",
            "tools/make/rust.mk",
            "tools/make/common.mk",
            "tools/make/build-run-test.mk",
        ):
            with self.subTest(path=path):
                plan = make_plan([path], source_sha=SHA)
                self.assertIn("native.openvino-cpu", plan["expected_verification_ids"])
                self.assertTrue(plan["native"])
                self.assertFalse(plan["full_cpu"])
        for path in (
            "tools/ci/run_model_tests.py",
            "tools/make/dashboard.mk",
            "tools/make/soak.mk",
        ):
            with self.subTest(unrelated_path=path):
                self.assertNotIn("native.openvino-cpu", classify([path]).selected_jobs)

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
                    plan = make_plan([path], source_sha=SHA, profile=profile)
                    self.assertIn("local.cli", plan["expected_verification_ids"])
                    self.assertTrue({"vllm-sr", "dashboard"} <= set(plan["images"]))
        for path in (
            "src/vllm-sr/README.md",
            "src/vllm-sr/tests/test_container_start.py",
        ):
            with self.subTest(unrelated_path=path):
                plan = make_plan([path], source_sha=SHA)
                self.assertNotIn("local.cli", plan["expected_verification_ids"])
                self.assertEqual(plan["images"], [])
        for path in (
            "src/vllm-sr/cli/evaluation/runtime_factors.py",
            "src/vllm-sr/cli/commands/chat.py",
        ):
            with self.subTest(packaged_cli_path=path):
                plan = make_plan([path], source_sha=SHA)
                self.assertNotIn("local.cli", plan["expected_verification_ids"])
                self.assertEqual(plan["images"], ["decision-runtime-cpu"])

    def test_operator_request_helper_selects_its_real_deployment(self):
        path = "tools/ci/check_operator_request.py"
        self.assertTrue((ROOT / path).is_file())
        plan = make_plan([path], source_sha=SHA)
        self.assertIn("operator", plan["expected_verification_ids"])
        self.assertTrue(
            {"operator", "operator-bundle", "extproc", "provider-mocker"}
            <= set(plan["images"])
        )
        self.assertNotIn(
            "operator",
            classify(["tools/ci/tests/test_operator_request.py"]).selected_jobs,
        )

    def test_required_classifier_definitions_select_live_models(self):
        definitions = {}
        for path in (ROOT / "src/semantic-router/pkg/classification").glob("*_test.go"):
            for name in re.findall(r"^func (Test\w+)\(", path.read_text(), re.M):
                definitions.setdefault(name, []).append(
                    path.relative_to(ROOT).as_posix()
                )
        for name in CLASSIFIER_TESTS:
            self.assertEqual(len(definitions.get(name, [])), 1, name)
            self.assertTrue(
                set(classify(definitions[name]).selected_jobs) >= NATIVE, name
            )

    def test_owned_omni_definitions_select_prepared_artifact_lane(self):
        for package, tests in OWNED_OMNI_TESTS.items():
            paths = ROOT / "src/semantic-router/pkg" / package
            for name in tests:
                definitions = [
                    path.relative_to(ROOT).as_posix()
                    for path in paths.glob("*_test.go")
                    if re.search(rf"^func {re.escape(name)}\(", path.read_text(), re.M)
                ]
                self.assertEqual(len(definitions), 1, name)
                self.assertIn(
                    "native.ort-cpu",
                    classify(definitions).selected_jobs,
                    name,
                )

    def test_owning_workflow_edits_select_executor_contracts(self):
        cases = {
            "test-native.yml": {*NATIVE, "native.openvino-cpu"},
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
            "src/fleet-sim/tests/test_simulation.py": "fleet-sim",
            "src/training/tests/test_export.py": "training",
            "tools/ci/training-test-requirements.txt": "training",
            "tools/test/services/provider-mocker/tests/test_fixture_latency.py": "mock-provider",
            "bench/test_agentic_routing_experiment.py": "learning-tools",
            "bench/test_openai_fault_proxy.py": "soak-tools",
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

    def test_published_model_profiles_plan_their_backend_image(self):
        for profile in ("vela-omni", "vela-halu"):
            with self.subTest(profile=profile):
                identifier = f"e2e.{profile}"
                plan = make_plan([], source_sha=SHA, requested=(identifier,))
                self.assertEqual(plan["expected_verification_ids"], [identifier])
                self.assertEqual(
                    plan["verifications"][0]["images"], ["extproc", "provider-mocker"]
                )
                self.assertEqual(set(plan["images"]), {"extproc", "provider-mocker"})

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
        for profile in ("agentgateway", "aibrix", "istio", "llm-d", "streaming"):
            self.assertIn(profile, full_e2e_profiles())
        self.assertNotIn("e2e.dynamo", plans[0]["expected_verification_ids"])
        self.assertNotIn("e2e.router-replay", plans[0]["expected_verification_ids"])

    def test_full_cpu_never_removes_affected_pr_verifications(self):
        for path in (
            "e2e/profiles/route-action/profile.go",
            "tools/test/services/provider-mocker/provider_mocker/app.py",
        ):
            affected = make_plan([path], source_sha=SHA)
            for profile in ("pr", "nightly", "release"):
                with self.subTest(path=path, profile=profile):
                    complete = make_plan(
                        [path], source_sha=SHA, profile=profile, full=True
                    )
                    self.assertTrue(
                        set(affected["expected_verification_ids"])
                        <= set(complete["expected_verification_ids"])
                    )
                    self.assertTrue(set(affected["images"]) <= set(complete["images"]))
                    self.assertNotIn(
                        "e2e.response-api-redis", complete["expected_verification_ids"]
                    )

    def test_local_classifier_backend_is_a_required_cpu_product_contract(self):
        identity = "e2e.local-classifier-backend"
        self.assertIn(identity, full_cpu_ids())
        for profile in ("pr", "nightly", "release"):
            with self.subTest(profile=profile):
                plan = make_plan([], source_sha=SHA, profile=profile, full=True)
                record = next(
                    row for row in plan["verifications"] if row["id"] == identity
                )
                self.assertEqual(record["executor"], "e2e")
                self.assertEqual(record["boundary"], ["e2e"])
                self.assertEqual(record["runtime"], "candle")
                self.assertEqual(record["device"], "cpu")
                self.assertEqual(record["platform"], "linux/amd64")
                self.assertEqual(record["images"], ["extproc", "provider-mocker"])
        for path in (
            "e2e/profiles/local-classifier-backend/profile.go",
            "e2e/testcases/local_classifier_routing.go",
            "src/semantic-router/pkg/classification/generic_classifier_local.go",
            "src/semantic-router/pkg/classification/native_labels.go",
        ):
            with self.subTest(path=path):
                self.assertIn(identity, classify([path]).selected_jobs)

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

    def test_main_only_publishes_affected_inputs(self):
        docs = make_plan(["README.md"], source_sha=SHA, profile="main")
        self.assertFalse(docs["publish_python"])
        self.assertFalse(docs["publish_images"])
        cli = make_plan(["src/vllm-sr/cli/core.py"], source_sha=SHA, profile="main")
        self.assertTrue(cli["publish_python"])
        self.assertIn("decision-runtime-cpu", cli["publish_images"])
        self.assertIn("cli-package", cli["expected_verification_ids"])

    def test_main_decision_package_never_publishes_without_qualification(self):
        for path in (
            "src/vllm-sr/tests/test_decision_runtime_server.py",
            "config/catalog/README.md",
            "website/static/model-catalog/catalog.json",
        ):
            with self.subTest(path=path):
                plan = make_plan([path], source_sha=SHA, profile="main")
                self.assertFalse(plan["publish_python"])
        for path in (
            "config/catalog/manifest.yaml",
            "config/catalog/resources/models/single/llm-semantic-router.yaml",
            "config/catalog/schemas/catalog-resources-v1.schema.json",
        ):
            with self.subTest(path=path):
                plan = make_plan([path], source_sha=SHA, profile="main")
                self.assertTrue(plan["publish_python"])
                self.assertIn("decision-runtime-cpu", plan["publish_images"])
                self.assertIn("decision-runtime-cpu", plan["images"])

    def test_decision_cpu_enters_trusted_image_flow_without_scheduling_rocm(self):
        for path in (
            "src/vllm-sr/decision_runtime/image/Dockerfile",
            "src/vllm-sr/cli/serve.py",
            "src/vllm-sr/setup.py",
            "src/semantic-router/pkg/configschema/router-config-v0.3.schema.json",
        ):
            with self.subTest(path=path):
                pr = make_plan([path], source_sha=SHA)
                self.assertIn("decision-runtime-cpu", pr["images"])
                self.assertNotIn("decision-runtime-rocm", pr["images"])
                self.assertEqual(pr["publish_images"], [])
                self.assertIn(
                    "decision-runtime-cpu", pr["image_producers"]["image-distribution"]
                )
                main = make_plan([path], source_sha=SHA, profile="main")
                self.assertIn("decision-runtime-cpu", main["publish_images"])
        for profile in ("nightly", "release"):
            plan = make_plan([], source_sha=SHA, profile=profile)
            self.assertIn("decision-runtime-cpu", plan["publish_images"])
            self.assertNotIn("decision-runtime-rocm", plan["images"])

    def test_previous_release_is_explicit_compatible_and_not_head_parent(self):
        self.assertEqual(
            previous_release(
                "0.4.0", ["v0.2.0", "v0.3.0", "v0.4.0", "v0.4.0-rc1", "v1.0.0"]
            ),
            "v0.3.0",
        )
        with self.assertRaises(ValueError):
            previous_release("1.0.0", ["v0.3.0"])

    def test_shared_artifact_loaders_select_their_runtime_consumers(self):
        for path in (
            "tools/ci/native_artifact.py",
            ".github/actions/load-native-artifact/action.yml",
        ):
            self.assertTrue(
                {
                    "core",
                    "storage",
                    "dashboard",
                    "generated-contracts",
                    *NATIVE,
                    "performance",
                }
                <= set(classify([path]).selected_jobs)
            )
        for path in (
            "tools/ci/image_artifacts.py",
            ".github/actions/load-ci-images/action.yml",
        ):
            self.assertTrue(
                {"local.cli", "operator", "e2e.envoy-ai-gateway"}
                <= set(classify([path]).selected_jobs)
            )

    def test_build_native_output_is_distinct_from_native_matrix(self):
        plan = make_plan(["candle-binding/src/lib.rs"], source_sha=SHA)
        outputs = github_outputs(plan)
        self.assertEqual(outputs["build_native"], "true")
        self.assertIsInstance(json.loads(outputs["native-shared"]), list)
        self.assertGreater(len(json.loads(outputs["native-shared"])), 0)

    def test_riscv_is_an_emulated_native_contract_with_preserved_source_triggers(self):
        identity = "native.candle-riscv64-qemu"
        for path in (
            "candle-binding/src/lib.rs",
            "ml-binding/ml_binding.go",
            "nlp-binding/nlp_binding.go",
            "tools/make/rust.mk",
            "tools/docker/check-native-abi.sh",
            "tools/ci/riscv-qemu-router-smoke.sh",
            "tools/ci/riscv_evidence.py",
            "src/semantic-router/tools/model-test-assets/main.go",
            "src/semantic-router/pkg/config/registry.go",
            "src/semantic-router/pkg/modeldownload/revisions.go",
            "tools/ci/runtime_evidence.py",
            "tools/make/models.mk",
            "e2e/config/config.riscv-qemu.yaml",
            "src/semantic-router/pkg/classification/unified_classifier_cgo_candle.go",
            "src/semantic-router/pkg/cache/valkey_cache_unavailable.go",
            "src/semantic-router/pkg/cache/exact_cache_valkey.go",
            "src/semantic-router/pkg/memory/valkey_store_integration_test.go",
            "src/semantic-router/pkg/vectorstore/valkey_backend.go",
            "src/semantic-router/pkg/extproc/router_memory.go",
            "src/semantic-router/pkg/extproc/router_memory_valkey.go",
            "src/semantic-router/pkg/extproc/router_memory_valkey_unavailable.go",
            ".github/workflows/test-native.yml",
        ):
            with self.subTest(path=path):
                self.assertIn(identity, classify([path]).selected_jobs)
        for profile in ("pr", "main"):
            plan = make_plan(
                ["candle-binding/src/lib.rs"], source_sha=SHA, profile=profile
            )
            self.assertIn(identity, plan["expected_verification_ids"])
        plan = make_plan([], source_sha=SHA, requested=(identity,))
        self.assertFalse(plan["native"])
        self.assertEqual(plan["images"], [])
        record = plan["verifications"][0]
        self.assertEqual(record["executor"], "native")
        self.assertEqual(record["workflow"], ".github/workflows/test-native.yml")
        self.assertEqual(record["platform"], "linux/riscv64")
        self.assertEqual(
            record["execution"], {"mode": "qemu-user", "host_platform": "linux/amd64"}
        )
        self.assertIn(identity, full_cpu_ids())
        self.assertFalse((ROOT / ".github/workflows/riscv-qemu.yml").exists())

    def test_runtime_combinations_are_qualified_rows_not_cartesian_product(self):
        records = verification_records(load_domain_registry())
        native = [
            record for record in records.values() if record["executor"] == "native"
        ]
        self.assertEqual(
            {(r["runtime"], r["device"]) for r in native},
            {("candle", "cpu"), ("ort", "cpu"), ("openvino", "cpu")},
        )
        self.assertIn("cuda", load_catalog()["full_cpu"]["excluded"])


if __name__ == "__main__":
    unittest.main()
