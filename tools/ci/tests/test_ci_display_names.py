"""Human check names are separate from stable execution and artifact identities."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/ci"))
import verification_catalog as catalog  # noqa: E402
from ci_plan import make_plan  # noqa: E402
from domain_registry import load_domain_registry  # noqa: E402


def workflow(filename):
    return yaml.load(
        (ROOT / ".github/workflows" / filename).read_text(), Loader=yaml.BaseLoader
    )


class DisplayNameTests(unittest.TestCase):
    def test_component_matrix_has_distinct_human_names_and_stable_ids(self):
        plan = make_plan([], source_sha="a" * 40, full=True)
        rows = [row for row in plan["verifications"] if row["executor"] == "tools"]
        labels = {row["display_name"] for row in rows}
        self.assertEqual(len(labels), len(rows))
        self.assertTrue(
            {
                "CLI Unit Tests",
                "Fleet Simulation",
                "Training Contracts",
                "Provider Simulator",
                "Routing Tools",
                "CI Harness",
                "E2E Framework",
                "ONNX Artifacts",
                "Attention Graph Rewriter",
                "Soak Tools",
            }
            <= labels
        )
        by_id = {row["id"]: row for row in rows}
        self.assertEqual(by_id["learning-tools"]["target"], "test-learning-tools")
        self.assertEqual(by_id["soak-tools"]["target"], "soak-test")
        self.assertEqual(by_id["mock-provider"]["target"], "test-provider-simulator")
        self.assertEqual(
            workflow("test-tools.yml")["jobs"]["tests"]["name"],
            "${{ fromJSON(inputs.batch).display_name }}",
        )

    def test_every_selected_contract_has_a_readable_unique_name(self):
        registry = load_domain_registry()
        self.assertEqual(catalog.catalog_errors(registry), [])
        records = catalog.verification_records(registry)
        self.assertTrue(
            all(row["display_name"][0].isupper() for row in records.values())
        )
        self.assertEqual(
            len({row["display_name"] for row in records.values()}), len(records)
        )
        for name, profile in registry["profiles"].items():
            self.assertEqual(
                records["e2e." + name]["display_name"], profile["display_name"]
            )
        modified = copy.deepcopy(catalog.load_catalog())
        modified["verifications"]["core"].pop("display_name")
        with patch.object(catalog, "load_catalog", return_value=modified):
            self.assertTrue(
                any(
                    "display_name" in error
                    for error in catalog.catalog_errors(registry)
                )
            )

    def test_reusable_matrix_callers_name_their_human_dimension(self):
        # GitHub appends every object field to a static matrix caller name.
        # Referencing the display dimension keeps check names readable.
        jobs = workflow("ci.yml")["jobs"]
        for identity, job in jobs.items():
            matrix = job.get("strategy", {}).get("matrix", {})
            if "uses" not in job or not matrix:
                continue
            field = "batch" if "batch" in matrix else "verification"
            with self.subTest(job=identity):
                self.assertIn("matrix." + field + ".display_name", job["name"])

    def test_dynamic_leaf_jobs_use_display_labels(self):
        for filename, key in (
            ("test-native.yml", "inference"),
            ("test-local.yml", "integration"),
            ("integration-test-k8s.yml", "integration-test"),
        ):
            self.assertIn(".display_name", workflow(filename)["jobs"][key]["name"])
        self.assertEqual(
            workflow("build-native.yml")["jobs"]["build"]["name"], "CPU Libraries"
        )

    def test_product_workflow_leaves_do_not_fall_back_to_generic_ids(self):
        filenames = {
            Path(row["workflow"]).name
            for row in catalog.verification_records(load_domain_registry()).values()
        } | {"build-native.yml", "build-artifacts.yml", "ci-changes.yml"}
        for filename in filenames:
            for key, job in workflow(filename)["jobs"].items():
                if "uses" in job:
                    continue
                with self.subTest(workflow=filename, job=key):
                    self.assertTrue(
                        job.get("name"), "leaf must name its executed contract"
                    )
                    self.assertNotIn(
                        job["name"],
                        {
                            "tests",
                            "test",
                            "build",
                            "result",
                            "package",
                            "integration-test",
                        },
                    )


if __name__ == "__main__":
    unittest.main()
