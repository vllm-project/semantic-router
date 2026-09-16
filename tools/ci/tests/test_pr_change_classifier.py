from __future__ import annotations

import json
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

    def test_calibration_inputs_select_the_image_calibration_gate(self) -> None:
        """Every input that can move the calibrated image-routing thresholds
        must run the calibration job, which feeds the required PR Gate."""
        for path in (
            "config/fragments/signal/embedding/image-routing.yaml",
            "src/semantic-router/cmd/image-routing-calibration/main.go",
            "src/semantic-router/cmd/image-routing-calibration/testdata/calibration-set.json",
            "e2e/testcases/testdata/image-fixtures/code_screenshot.jpg",
            "e2e/profiles/multimodal-routing/crds/intelligentroute.yaml",
            # The CRD-mirror test runs only inside the gate, and e2e.mk defines
            # the make target the gate invokes for it.
            "e2e/profiles/multimodal-routing/profile_test.go",
            "tools/make/e2e.mk",
            "website/static/img/blog/new-screenshot.png",
            ".github/workflows/image-routing-calibration.yml",
            # Scoring implementation: the whole native binding and the whole
            # classification package, so a refactor cannot move scoring code
            # out from under a filename-level trigger.
            "candle-binding/src/model_architectures/embedding/multimodal_embedding.rs",
            "candle-binding/src/model_architectures/attention/chunked_sdpa.rs",
            "candle-binding/src/core/similarity.rs",
            "candle-binding/src/ffi/embedding.rs",
            "candle-binding/src/ffi/types.rs",
            "candle-binding/src/model_architectures/unified_interface.rs",
            "candle-binding/build.rs",
            "candle-binding/semantic-router.go",
            "candle-binding/go.mod",
            "candle-binding/Cargo.lock",
            "src/semantic-router/pkg/classification/embedding_classifier_scoring.go",
            "src/semantic-router/pkg/classification/prototype_bank.go",
            "src/semantic-router/pkg/classification/prototype_clustering.go",
            "src/semantic-router/pkg/classification/request_image_embedding_cache.go",
            "src/semantic-router/pkg/classification/classifier_signal_group_similarity.go",
            "src/semantic-router/pkg/classification/openvino_backend_stub.go",
            # Config contracts the tool decodes and the classifier consumes.
            "src/semantic-router/pkg/config/canonical_config.go",
            "src/semantic-router/pkg/config/embedding_config.go",
            "src/semantic-router/pkg/config/prototype_scoring_config.go",
            "src/semantic-router/pkg/config/signal_config.go",
            "src/semantic-router/pkg/config/canonical_defaults.go",
            # Dependency manifests and the native build recipe the gate runs.
            "src/semantic-router/go.mod",
            "src/semantic-router/go.sum",
            "Makefile",
            "tools/make/rust.mk",
            "tools/make/build-run-test.mk",
        ):
            with self.subTest(path=path):
                self.assertIn("image-calibration", classify([path]).selected_jobs)

    def test_every_calibration_fixture_is_covered_by_the_gate(self) -> None:
        """The domain is a directory list while the manifest may name any
        tracked image; a later change to a listed image outside the covered
        directories would move the calibrated thresholds without running the
        gate. Every positive, negative, and excluded path must select it."""
        manifest = json.loads(
            (
                REPO_ROOT
                / "src/semantic-router/cmd/image-routing-calibration/testdata/calibration-set.json"
            ).read_text()
        )
        paths = [entry["image_file"] for entry in manifest["positives"]]
        paths += manifest["negatives"]
        paths += [entry["image_file"] for entry in manifest.get("excluded", [])]
        self.assertGreater(len(paths), 0)
        uncovered = [
            path
            for path in paths
            if "image-calibration" not in classify([path]).selected_jobs
        ]
        self.assertEqual(
            uncovered, [], "manifest fixtures outside the image-calibration domain"
        )

    def test_unrelated_router_change_does_not_run_the_calibration_gate(self) -> None:
        result = classify(["src/semantic-router/pkg/extproc/processor.go"])

        self.assertNotIn("image-calibration", result.selected_jobs)

    def test_candle_binding_selects_riscv_qemu_smoke(self) -> None:
        self.assert_classification(
            "candle-binding/src/lib.rs",
            ("quality", "security", "core-tests", "riscv-qemu"),
        )

    def test_riscv_make_target_selects_qemu_smoke(self) -> None:
        self.assert_classification(
            "tools/make/rust.mk",
            ("quality", "security", "riscv-qemu"),
        )

    def test_riscv_router_smoke_inputs_select_qemu_job(self) -> None:
        fixtures = {
            "nlp-binding/nlp_binding.go": (
                "quality",
                "security",
                "core-tests",
                "riscv-qemu",
            ),
            "e2e/config/config.riscv-qemu.yaml": (
                "quality",
                "security",
                "riscv-qemu",
            ),
            "tools/ci/riscv-qemu-router-smoke.sh": (
                "quality",
                "security",
                "riscv-qemu",
            ),
            "tools/docker/check-native-abi.sh": (
                "quality",
                "security",
                "riscv-qemu",
            ),
        }
        for path, jobs in fixtures.items():
            with self.subTest(path=path):
                self.assert_classification(path, jobs)
        self.assert_classification(
            "ml-binding/ml_binding.go",
            ("quality", "security", "core-tests", "e2e", "riscv-qemu"),
            profiles=("ml-model-selection",),
        )

    def test_runtime_cli_surface_has_explicit_integration_escalation(self) -> None:
        self.assert_classification(
            "src/vllm-sr/cli/commands/runtime.py",
            ("quality", "security", "core-tests", "cli"),
        )

    def test_python_selector_changes_keep_native_parity_gate(self) -> None:
        for path in (
            "src/training/model_selection/ml_model_selection/models.py",
            "src/training/model_selection/ml_model_selection/tests/test_native_parity.py",
            "src/training/model_selection/ml_model_selection/requirements-parity.txt",
        ):
            with self.subTest(path=path):
                self.assertIn("core-tests", classify([path]).selected_jobs)
        self.assertNotIn(
            "core-tests",
            classify(["src/training/model_classifier/train.py"]).selected_jobs,
        )

    def test_workflow_only_change_runs_only_its_reusable_workflow(self) -> None:
        fixtures = {
            ".github/workflows/performance-test.yml": "performance",
            ".github/workflows/operator-ci.yml": "operator",
            ".github/workflows/integration-test-memory.yml": "memory",
            ".github/workflows/openvino-binding-ci.yml": "openvino",
            ".github/workflows/riscv-qemu.yml": "riscv-qemu",
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
        self.assertEqual(
            result.publish_images,
            ("extproc", "extproc-rocm", "vllm-sr", "vllm-sr-rocm"),
        )

    def test_rocm_images_publish_when_shared_build_inputs_change(self) -> None:
        for path in (
            "src/semantic-router/pkg/extproc/processor.go",
            "config/knowledge_bases/example.json",
            "candle-binding/src/lib.rs",
            "ml-binding/src/lib.rs",
            "nlp-binding/nlp_binding.go",
            "onnx-binding/instance/instance.go",
            "openvino-binding/semantic-router.go",
            "tools/docker/check-native-abi.sh",
            ".dockerignore",
        ):
            with self.subTest(path=path):
                result = classify([path])
                self.assertTrue(
                    {"extproc-rocm", "vllm-sr-rocm"}.issubset(result.publish_images)
                )
                self.assertEqual(result.pr_images, ())

    def test_rocm_publication_policy_changes_refresh_both_images(self) -> None:
        result = classify(["tools/agent/domains.yaml"])

        self.assertEqual(result.publish_images, ("extproc-rocm", "vllm-sr-rocm"))
        self.assertEqual(result.pr_images, ())

    def test_rocm_publication_distinguishes_runtime_specific_inputs(self) -> None:
        fixtures = {
            "src/vllm-sr/cli/builtin_recipes.py": {"vllm-sr-rocm"},
            "src/vllm-sr/start-router.sh": {"vllm-sr-rocm"},
            "tools/docker/entrypoint.sh": {"extproc-rocm"},
            "tools/make/rust.mk": {"extproc-rocm"},
            "Makefile": {"extproc-rocm"},
            "dashboard/frontend/src/App.tsx": set(),
        }
        for path, expected in fixtures.items():
            with self.subTest(path=path):
                result = classify([path])
                self.assertEqual(
                    set(result.publish_images) & {"extproc-rocm", "vllm-sr-rocm"},
                    expected,
                )

    def test_builtin_recipe_changes_publish_the_rocm_cli_runtime(self) -> None:
        result = classify(["config/recipes/built-in/mom-v1/manifest.yaml"])

        self.assertIn("vllm-sr-rocm", result.publish_images)
        self.assertEqual(result.pr_images, ())

    def test_generated_api_contracts_keep_their_drift_check(self) -> None:
        for path in (
            "website/docs/api/apiserver.md",
            "website/static/openapi/apiserver/apiserver.openapi.json",
            "tools/openapi-gen/main.go",
        ):
            with self.subTest(path=path):
                self.assertIn("core-tests", classify([path]).selected_jobs)

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
            (
                "quality",
                "security",
                "core-tests",
                "image-calibration",
                "e2e",
                "recipe-conformance",
            ),
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
