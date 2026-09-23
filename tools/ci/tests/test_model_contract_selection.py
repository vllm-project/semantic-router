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


class ModelContractSelectionTests(unittest.TestCase):
    def test_halu_artifact_selects_its_real_model_and_deployed_contracts(self):
        selected = set(classify(["tools/models/vela_halu/export.py"]).selected_jobs)
        self.assertTrue({"native.candle-cpu", "e2e.vela-halu"} <= selected)
        self.assertNotIn("e2e.vela-omni", selected)
        self.assertNotIn("native.image-calibration-cpu", selected)

    def test_omni_artifact_keeps_runtime_conformance_and_routing_coverage(self):
        selected = set(classify(["tools/models/vela_omni/export.py"]).selected_jobs)
        self.assertTrue(
            {
                "native.ort-cpu",
                "native.image-calibration-cpu",
                "e2e.vela-omni",
                "e2e.multimodal-routing",
            }
            <= selected
        )
        self.assertNotIn("e2e.vela-halu", selected)

    def test_prepared_model_consumers_select_runtime_instead_of_calibration(self):
        for path in (
            "src/semantic-router/pkg/cache/omni_storage_integration_test.go",
            "src/semantic-router/pkg/cache/embedding_dimension_test.go",
            "src/semantic-router/pkg/memory/embedding_provider_test.go",
        ):
            with self.subTest(path=path):
                selected = set(classify([path]).selected_jobs)
                self.assertIn("native.ort-cpu", selected)
                self.assertNotIn("native.image-calibration-cpu", selected)

    def test_deployment_contracts_declare_their_actual_model_runtime(self):
        records = verification_records(load_domain_registry())
        for profile, runtime in (
            ("vela-halu", "candle"),
            ("vela-omni", "ort"),
            ("multimodal-routing", "ort"),
        ):
            with self.subTest(profile=profile):
                record = records[f"e2e.{profile}"]
                self.assertEqual(record["runtime"], runtime)
                self.assertEqual(record["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
