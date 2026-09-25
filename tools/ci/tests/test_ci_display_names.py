"""Readable check paths do not define contracts, workers, or dependencies."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/ci"))
import verification_catalog as catalog  # noqa: E402
from ci_plan import github_outputs, make_plan, render_plan_summary  # noqa: E402
from domain_registry import load_domain_registry  # noqa: E402
from execution_batches import ALL_DISPATCH_JOBS, EXECUTOR_JOBS  # noqa: E402


def workflow(filename):
    return yaml.load(
        (ROOT / ".github/workflows" / filename).read_text(), Loader=yaml.BaseLoader
    )


class DisplayNameTests(unittest.TestCase):
    def test_categories_do_not_change_execution_or_verification_identity(self):
        records = catalog.verification_records(load_domain_registry())
        image = records["native.image-calibration-cpu"]
        self.assertEqual(image["category"], "conformance")
        self.assertEqual(image["executor"], "native")
        self.assertEqual(image["runtime"], "ort")
        self.assertEqual(records["recipe-conformance"]["category"], "conformance")
        self.assertEqual(records["e2e.multimodal-routing"]["category"], "e2e")
        self.assertEqual(catalog.catalog_errors(load_domain_registry()), [])
        modified = copy.deepcopy(catalog.load_catalog())
        modified["verifications"]["core"].pop("category")
        with patch.object(catalog, "load_catalog", return_value=modified):
            self.assertTrue(
                any(
                    "category" in error
                    for error in catalog.catalog_errors(load_domain_registry())
                )
            )

    def test_contract_labels_remain_readable_and_unique(self):
        records = catalog.verification_records(load_domain_registry())
        labels = [record["display_name"] for record in records.values()]
        self.assertEqual(len(labels), len(set(labels)))
        self.assertTrue(all(label and label[0].isupper() for label in labels))
        for name, profile in load_domain_registry()["profiles"].items():
            self.assertEqual(
                records["e2e." + name]["display_name"], profile["display_name"]
            )

    def test_optional_callers_are_static_and_matrix_axes_only_hold_worker_labels(self):
        jobs = workflow("ci.yml")["jobs"]
        self.assertEqual(set(jobs), {*ALL_DISPATCH_JOBS, "gate"})
        for identity, job in jobs.items():
            self.assertTrue(job["name"].strip())
            self.assertNotIn("${{", job["name"])
            self.assertIn(
                job["name"].split(" / ")[0],
                {"Plan", "Quality", "Artifacts", "Tests", "Gate"},
            )
            if "strategy" in job:
                self.assertEqual(
                    job["strategy"]["matrix"],
                    {
                        "label": "${{ fromJSON(needs.plan.outputs.worker_labels)['"
                        + identity
                        + "'] }}"
                    },
                )
        for identity in (
            "security",
            "core",
            "storage",
            "dashboard",
            "operator",
            "recipes",
            "performance",
            "package",
        ):
            self.assertNotIn("strategy", jobs[identity], "one contract is not a matrix")

    def test_dispatch_preserves_every_contract_exactly_once(self):
        for plan in (
            make_plan([], source_sha="a" * 40, full=True),
            make_plan([], source_sha="a" * 40, draft=True),
            make_plan(
                [],
                source_sha="a" * 40,
                requested=(
                    "native.ort-cpu",
                    "native.image-calibration-cpu",
                    "local.memory",
                    "cli-unit",
                    "e2e.vela-omni",
                ),
            ),
        ):
            before = copy.deepcopy(plan)
            output = {
                key: json.loads(value) for key, value in github_outputs(plan).items()
            }
            self.assertEqual(plan, before)
            self.assertEqual(output["plan"], before)
            self.assertEqual(set(output["dispatch"]), set(EXECUTOR_JOBS))
            actual = []
            for job, rows in output["dispatch"].items():
                self.assertEqual(list(rows), output["worker_labels"][job])
                for row in rows.values():
                    actual.extend(row.get("verifications", [row]))
            self.assertEqual(
                sorted(actual, key=lambda row: row["id"]),
                sorted(plan["verifications"], key=lambda row: row["id"]),
            )
            self.assertEqual(output["build_images"], plan["build_images"])
            self.assertEqual(output["publish_images"], plan["publish_images"])
            acquired = [
                image
                for row in output["image_producers"].values()
                for image in row["images"]
            ]
            self.assertEqual(sorted(acquired), plan["images"])
            rebuilt = [
                image
                for row in output["image_producers"].values()
                for image in row["build_images"]
            ]
            self.assertEqual(sorted(rebuilt), plan["build_images"])

    def test_local_contracts_use_environment_shards_and_native_feature_is_evidence(
        self,
    ):
        plan = make_plan([], source_sha="a" * 40, full=True)
        output = {key: json.loads(value) for key, value in github_outputs(plan).items()}
        self.assertEqual(output["worker_labels"]["local"], ["Shard 1", "Shard 2"])
        self.assertTrue(
            all(
                "Image" not in label
                for label in output["worker_labels"]["native-shared"]
            )
        )
        summary = render_plan_summary(plan)
        self.assertIn("Tests / Conformance | Image Routing Conformance", summary)
        self.assertIn("Tests / Conformance | Recipe Preview", summary)
        for filename, job in (
            ("test-native.yml", "inference"),
            ("test-local.yml", "integration"),
            ("test-tools.yml", "tests"),
        ):
            self.assertEqual(
                workflow(filename)["jobs"][job]["name"], "Execute Contracts"
            )

    def test_consumers_wait_only_for_actual_dependency_lanes(self):
        jobs = workflow("ci.yml")["jobs"]
        self.assertEqual(jobs["recipes"]["needs"], ["plan", "image-local"])
        self.assertEqual(
            jobs["local"]["needs"],
            ["plan", "image-local", "image-dashboard", "image-fixtures"],
        )
        self.assertEqual(jobs["e2e-router"]["needs"], ["plan", "image-router"])
        self.assertEqual(
            jobs["e2e-fixtures"]["needs"], ["plan", "image-router", "image-fixtures"]
        )
        self.assertEqual(jobs["native-independent"]["needs"], ["plan"])
        for name, job in jobs.items():
            if name != "gate":
                self.assertNotIn("image-distribution", job.get("needs", []))


if __name__ == "__main__":
    unittest.main()
