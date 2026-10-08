from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/ci"))
from ci_plan import (  # noqa: E402
    github_outputs,
    make_plan,
    performance_base,
    previous_release,
)
from classify_pr_changes import classify, full_e2e_profiles  # noqa: E402
from domain_registry import load_domain_registry, profile_records  # noqa: E402
from verification_catalog import (  # noqa: E402
    full_cpu_ids,
    load_catalog,
    profile_image_dependencies,
    verification_records,
)

SHA = "a" * 40
IMAGE_CALIBRATION = "platform.image-calibration-cpu"
MODELS = "platform.models-cpu"


class SelectionTests(unittest.TestCase):
    def test_image_calibration_inputs_select_the_platform_verification(self):
        for path in (
            "config/fragments/signal/embedding/image-routing.yaml",
            "tools/calibration/image-routing/main.go",
            "e2e/profiles/multimodal-routing/profile_test.go",
            "e2e/testcases/testdata/image-fixtures/office.jpg",
            "website/static/img/example.png",
            "dashboard/frontend/public/example.png",
            "src/model-runtime/vllm_srun/families/multimodal_embedding/family.py",
            "src/model-runtime/vllm_srun/registry/tables/omni.py",
            "src/semantic-router/pkg/classification/embedding.go",
            "src/semantic-router/pkg/config/registry.go",
            "src/semantic-router/pkg/modeldownload/revision_receipt_test.go",
            "src/semantic-router/pkg/modelruntime/serving/runtime.go",
            "tools/make/models.mk",
            "tools/make/common.mk",
            "tools/ci/image_calibration.py",
            "tools/ci/runtime_evidence.py",
            "tools/ci/workflow_evidence.py",
            ".github/workflows/test-platform.yml",
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
            "src/model-runtime/vllm_srun/families/multimodal_embedding/family.py",
            "src/semantic-router/pkg/embedding/embedding.go",
            "src/semantic-router/pkg/modelruntime/embedding_owned.go",
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
        self.assertFalse(plan["publish_images"])
        record = plan["verifications"][0]
        self.assertEqual(record["source_sha"], SHA)
        self.assertEqual(record["platform_id"], "model-runtime-cpu")
        self.assertEqual(record["workflow"], ".github/workflows/test-platform.yml")
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
        self.assertIn("--results", workflow["jobs"]["gate"]["steps"][-1]["run"])

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
            "tools/make/models.mk": {IMAGE_CALIBRATION, "performance"},
            "tools/make/common.mk": {IMAGE_CALIBRATION, "performance"},
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
            "e2e/testcases/istio_routes_test.go": {"e2e.istio"},
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
                    plan = make_plan([path], source_sha=SHA, profile=profile)
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
            "test-platform.yml": {IMAGE_CALIBRATION, MODELS},
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
        for profile in ("vela-omni", "vela-halu", "vela-shield"):
            with self.subTest(profile=profile):
                identifier = f"e2e.{profile}"
                plan = make_plan([], source_sha=SHA, requested=(identifier,))
                self.assertEqual(plan["expected_verification_ids"], [identifier])
                self.assertEqual(
                    plan["verifications"][0]["images"], ["vllm-sr", "provider-mocker"]
                )
                self.assertEqual(set(plan["images"]), {"vllm-sr", "provider-mocker"})

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
                self.assertEqual(record["runtime"], "model-runtime")
                self.assertEqual(record["device"], "cpu")
                self.assertEqual(record["platform"], "linux/amd64")
                self.assertEqual(record["images"], ["vllm-sr", "provider-mocker"])
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
        self.assertIn("cli-package", cli["expected_verification_ids"])

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
        # v0.4.0 predates the model runtime; #4707 runs the current harness.
        self.assertEqual(
            performance_base("0.5.0", ["v0.3.0", "v0.4.0"]),
            "abae8ff99df2fdab372f0fb6d032b305907b9f44",
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
        plan = make_plan(["tools/calibration/image-routing/main.go"], source_sha=SHA)
        outputs = github_outputs(plan)
        batches = json.loads(outputs["platform"])
        self.assertEqual(
            [row["verifications"][0]["id"] for row in batches], [IMAGE_CALIBRATION]
        )

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
