import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime_evidence
from ci_results import collection_errors

sys.path.insert(0, str(runtime_evidence.ROOT / "tools/calibration/recipe"))
from recipe_conformance import (
    cpu_inventory,
    discover_inventory,
    load_probe_manifest,
)
from recipe_conformance_sources import (
    discover_source_inventories,
    source_matrix_payload,
)


class RuntimeEvidenceTests(unittest.TestCase):
    def test_recipe_uses_authored_acceptance_without_hiding_observations(self):
        evaluation = {
            "results": [{"id": "stress", "matched": False, "http_status": 200}],
            "passed": True,
            "execution": {"complete": True},
        }
        cases = runtime_evidence.recipe_cases(evaluation, "recipe:")
        self.assertFalse(cases[0]["matched"])
        evidence = {
            "cases": cases,
            "expected_cases": ["recipe:request:stress", "recipe:acceptance"],
        }
        self.assertEqual(collection_errors(evidence, "test"), [])
        evaluation["passed"] = False
        evidence["cases"] = runtime_evidence.recipe_cases(evaluation, "recipe:")
        self.assertTrue(collection_errors(evidence, "test"))

    def test_recipe_cannot_pass_with_failed_or_missing_requests(self):
        evaluation = {
            "results": [{"id": "one", "matched": False, "http_status": 503}],
            "passed": True,
            "execution": {"complete": True},
        }
        errors = collection_errors(
            {
                "cases": runtime_evidence.recipe_cases(evaluation, "recipe:"),
                "expected_cases": [
                    "recipe:request:one",
                    "recipe:request:two",
                    "recipe:acceptance",
                ],
            },
            "test",
        )
        self.assertTrue(any("failed" in error for error in errors))
        self.assertTrue(any("missing" in error for error in errors))

    def write_planned_recipe_outputs(self, directory: Path) -> list[Path]:
        inventories = discover_source_inventories(
            runtime_evidence.ROOT / "config/recipes",
            lambda root: cpu_inventory(discover_inventory(root)),
        )
        receipts = []
        for job in source_matrix_payload(inventories, None, runtime_evidence.ROOT)[
            "include"
        ]:
            receipt = directory / f"image-{job['shard']}.json"
            receipt.write_text(
                json.dumps([{"id": "image:vllm-sr", "sha256": "a" * 64}])
            )
            receipts.append(receipt)
        for item in inventories:
            for recipe in item.recipes:
                _, probes = load_probe_manifest(
                    item.source.recipes_root / recipe.name / "probes.yaml"
                )
                report = directory / item.source.report_subdir / recipe.name
                report.mkdir(parents=True)
                (report / "eval-report.json").write_text(
                    json.dumps(
                        {
                            "evaluation": {
                                "results": [
                                    {
                                        "id": probe.probe_id,
                                        "matched": True,
                                        "http_status": 200,
                                    }
                                    for probe in probes
                                ],
                                "passed": True,
                                "execution": {"complete": True},
                            }
                        }
                    )
                )
        return receipts

    def test_recipe_evidence_reads_one_receipt_per_planned_preview_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            receipts = self.write_planned_recipe_outputs(path)
            self.assertIn(
                "image-built-in-latest-mom-v1.json", {r.name for r in receipts}
            )
            result = runtime_evidence.recipe_evidence(path)
            self.assertEqual(collection_errors(result, "test"), [])
            self.assertEqual(len(result["artifacts"]), len(receipts))

    def test_recipe_evidence_fails_closed_without_a_planned_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            self.write_planned_recipe_outputs(path)[0].unlink()
            with self.assertRaises(FileNotFoundError):
                runtime_evidence.recipe_evidence(path)


if __name__ == "__main__":
    unittest.main()
