"""Human check names are separate from stable execution and artifact identities."""

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
from ci_plan import EXECUTORS, github_outputs, make_plan  # noqa: E402
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
        self.assertEqual(by_id["mock-provider"]["target"], "test-provider-mocker")
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

    def test_reusable_callers_keep_skipped_names_static_and_matrix_values_scalar(self):
        # Job-level if runs before matrix expansion. Static caller names also
        # need scalar axes so active jobs do not append entire contract objects.
        jobs = workflow("ci.yml")["jobs"]
        callers = set()
        for identity, job in jobs.items():
            matrix = job.get("strategy", {}).get("matrix", {})
            if "uses" not in job or not matrix:
                continue
            callers.add(identity)
            source = "component_batches" if identity == "tools" else identity
            argument = "batch" if identity == "tools" else "verification"
            with self.subTest(job=identity):
                self.assertTrue(job["name"].strip())
                self.assertNotIn("${{", job["name"])
                self.assertEqual(
                    matrix,
                    {
                        "label": "${{ fromJSON(needs.plan.outputs."
                        + source
                        + ").*.display_name }}"
                    },
                )
                self.assertEqual(
                    job["with"][argument],
                    "${{ toJSON(fromJSON(needs.plan.outputs.dispatch)."
                    + identity
                    + "[matrix.label]) }}",
                )
        self.assertEqual(callers, set(EXECUTORS) - {"quality", "generated"})

    def test_dispatch_round_trip_preserves_full_empty_and_subset_plans(self):
        plans = {
            "full": make_plan([], source_sha="a" * 40, full=True),
            "empty": make_plan([], source_sha="a" * 40, draft=True),
            "subset": make_plan(
                [],
                source_sha="a" * 40,
                requested=(
                    "native.candle-cpu",
                    "e2e.envoy-ai-gateway",
                    "ck-rewrite",
                    "cli-unit",
                ),
            ),
        }
        for scenario, plan in plans.items():
            before = copy.deepcopy(plan)
            outputs = github_outputs(plan)
            dispatch = json.loads(outputs["dispatch"])
            with self.subTest(plan=scenario):
                self.assertEqual(plan, before)
                self.assertEqual(json.loads(outputs["plan"]), before)
                self.assertEqual(
                    set(dispatch), set(EXECUTORS) - {"quality", "generated"}
                )
                unchanged = {
                    key: before[key]
                    for key in (
                        "images",
                        "build_images",
                        "publish_images",
                        "multiarch",
                        "publish_helm",
                        "publish_python",
                    )
                }
                source = before["image_sources"].get("provider-mocker")
                unchanged["published_images"] = (
                    [source] if source and source["source"] == "published" else []
                )
                unchanged.update(plan=before, build_native=before["native"])
                for executor in EXECUTORS:
                    rows = (
                        plan["component_batches"]
                        if executor == "tools"
                        else [
                            row
                            for row in plan["verifications"]
                            if row["executor"] == executor
                        ]
                    )
                    output = "component_batches" if executor == "tools" else executor
                    unchanged[output] = rows
                    if executor not in dispatch:
                        continue
                    labels = [row["display_name"] for row in rows]
                    self.assertEqual(set(dispatch[executor]), set(labels))
                    self.assertEqual(len(labels), len(set(labels)))
                    self.assertEqual(
                        [dispatch[executor][label] for label in labels], rows
                    )
                self.assertEqual(
                    {
                        key: json.loads(value)
                        for key, value in outputs.items()
                        if key != "dispatch"
                    },
                    unchanged,
                )

    def test_dispatch_rejects_missing_or_ambiguous_labels(self):
        full = make_plan([], source_sha="a" * 40, full=True)
        for family in ("native", "tools"):
            for invalid in (None, "", " ", "duplicate"):
                plan = copy.deepcopy(full)
                rows = (
                    plan["component_batches"]
                    if family == "tools"
                    else [
                        row
                        for row in plan["verifications"]
                        if row["executor"] == family
                    ]
                )
                rows[0]["display_name"] = (
                    rows[1]["display_name"] if invalid == "duplicate" else invalid
                )
                with (
                    self.subTest(family=family, invalid=invalid),
                    self.assertRaises(ValueError),
                ):
                    github_outputs(plan)

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
