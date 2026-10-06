"""Model changes must select the worker that actually executes their contracts."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/ci"))

from classify_pr_changes import classify  # noqa: E402
from domain_registry import load_domain_registry  # noqa: E402
from verification_catalog import verification_records  # noqa: E402

IMAGE_CALIBRATION = "platform.image-calibration-cpu"


class ModelContractSelectionTests(unittest.TestCase):
    def test_halu_artifact_selects_its_deployed_contract(self):
        for path in (
            "tools/models/vela_halu/export.py",
            "src/model-runtime/vllm_srun/heads/grounded.py",
        ):
            with self.subTest(path=path):
                selected = set(classify([path]).selected_jobs)
                self.assertIn("e2e.vela-halu", selected)
                self.assertNotIn("e2e.vela-omni", selected)
                self.assertNotIn(IMAGE_CALIBRATION, selected)

    def test_omni_family_keeps_conformance_and_routing_coverage(self):
        path = "src/model-runtime/vllm_srun/families/multimodal_embedding/family.py"
        selected = set(classify([path]).selected_jobs)
        self.assertTrue(
            {IMAGE_CALIBRATION, "e2e.vela-omni", "e2e.multimodal-routing"} <= selected
        )
        self.assertNotIn("e2e.vela-halu", selected)

    def test_the_opt_in_omni_bundle_producer_runs_its_own_checks(self):
        selected = set(classify(["tools/models/vela_omni/export.py"]).selected_jobs)
        self.assertIn("core", selected)
        self.assertNotIn(IMAGE_CALIBRATION, selected)
        self.assertNotIn("e2e.vela-omni", selected)

    def test_store_embedding_consumers_run_in_core_instead_of_calibration(self):
        for path in (
            "src/semantic-router/pkg/cache/embedding_dimension_test.go",
            "src/semantic-router/pkg/memory/embedding_provider_test.go",
        ):
            with self.subTest(path=path):
                selected = set(classify([path]).selected_jobs)
                self.assertIn("core", selected)
                self.assertNotIn(IMAGE_CALIBRATION, selected)

    def test_deployment_contracts_declare_the_model_runtime(self):
        records = verification_records(load_domain_registry())
        for profile in ("vela-halu", "vela-shield", "vela-omni", "multimodal-routing"):
            with self.subTest(profile=profile):
                record = records[f"e2e.{profile}"]
                self.assertEqual(record["runtime"], "model-runtime")
                self.assertEqual(record["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
