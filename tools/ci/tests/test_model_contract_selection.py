"""Model changes must select the worker that actually executes their contracts."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/ci"))

from classify_pr_changes import classify  # noqa: E402

MODELS = "platform.models-cpu"


class ModelContractSelectionTests(unittest.TestCase):
    def test_halu_artifact_runs_its_own_contract_checks(self):
        selected = set(classify(["tools/models/vela_halu/export.py"]).selected_jobs)
        self.assertIn("core", selected)
        self.assertNotIn(MODELS, selected)

    def test_omni_family_selects_the_contract_that_serves_omni_nano(self):
        family = "src/model-runtime/vllm_srun/families/multimodal_embedding/family.py"
        for path in (family, "src/model-runtime/vllm_srun/registry/tables/omni.py"):
            with self.subTest(path=path):
                self.assertIn(MODELS, classify([path]).selected_jobs)

    def test_the_opt_in_omni_bundle_producer_runs_its_own_checks(self):
        selected = set(classify(["tools/models/vela_omni/export.py"]).selected_jobs)
        self.assertIn("core", selected)
        self.assertNotIn(MODELS, selected)

    def test_store_embedding_consumers_run_in_core_instead_of_the_model_contract(self):
        for path in (
            "src/semantic-router/pkg/cache/embedding_dimension_test.go",
            "src/semantic-router/pkg/memory/embedding_provider_test.go",
        ):
            with self.subTest(path=path):
                selected = set(classify([path]).selected_jobs)
                self.assertIn("core", selected)
                self.assertNotIn(MODELS, selected)


if __name__ == "__main__":
    unittest.main()
