from __future__ import annotations

import copy
import importlib.util
import io
import json
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

MODULE_PATH = Path(__file__).resolve().parents[1] / "audit_model_catalog.py"
SPEC = importlib.util.spec_from_file_location("audit_model_catalog", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


class ModelCatalogAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = {"defaults": {"intelligence_index": "test/index@1.0.0"}}
        self.resources = {
            "models": [
                {
                    "id": "acme/reasoner",
                    "publisher": "Acme",
                    "kind": "physical",
                    "reasoning_family": "acme",
                },
                {
                    "id": "other/chat",
                    "publisher": "Other",
                    "kind": "physical",
                },
                {
                    "id": "router/recipe",
                    "publisher": "Router",
                    "kind": "virtual",
                },
            ],
            "reasoning_families": [{"id": "acme", "levels": ["low", "high"]}],
            "providers": [
                {
                    "id": "acme-api",
                    "category": "model_api",
                    "support_tier": "native",
                    "models": [
                        {"id": "reasoner", "catalog": "acme/reasoner"},
                        {"id": "private"},
                    ],
                },
                {
                    "id": "empty-api",
                    "category": "model_api",
                    "support_tier": "compatible",
                    "models": [],
                },
            ],
            "indices": [
                {
                    "id": "test/index@1.0.0",
                    "components": [
                        {
                            "benchmark": "bench/a@1.0.0",
                            "benchmark_profile": "standard",
                            "metric": "accuracy",
                        },
                        {
                            "benchmark": "bench/b@1.0.0",
                            "benchmark_profile": "agent",
                            "metric": "resolved",
                        },
                    ],
                }
            ],
            "evaluations": [
                {
                    "model": "acme/reasoner",
                    "reasoning_effort": "high",
                    "benchmark": "bench/a@1.0.0",
                    "benchmark_profile": "standard",
                    "metrics": {"accuracy": 0.8, "secondary": 0.7},
                    "status": "available",
                    "evidence": {"provenance": "third_party"},
                },
                {
                    "model": "acme/reasoner",
                    "reasoning_effort": "high",
                    "benchmark": "bench/b@1.0.0",
                    "benchmark_profile": "wrong-profile",
                    "metrics": {"resolved": 0.6},
                    "status": "available",
                    "evidence": {"provenance": "third_party"},
                },
                {
                    "model": "other/chat",
                    "reasoning_effort": "unspecified",
                    "benchmark": "bench/a@1.0.0",
                    "benchmark_profile": "standard",
                    "metrics": {"accuracy": 0.5},
                    "status": "missing",
                    "evidence": {"provenance": "operator"},
                },
            ],
        }

    def test_reports_counts_distinct_benchmarks_and_exact_default_slots(self) -> None:
        report = audit.build_audit(self.manifest, self.resources)

        self.assertEqual(
            report["counts"],
            {
                "physical_models": 2,
                "virtual_models": 1,
                "providers": 2,
                "evaluations": 3,
                "available_evaluations": 2,
            },
        )
        self.assertEqual(
            report["selectable_effort_coverage"],
            {
                "minimum_available_benchmarks": 5,
                "total": 2,
                "complete": 0,
                "partial": 1,
                "unmeasured": 1,
            },
        )
        high = next(
            row
            for row in report["model_efforts"]
            if row["model"] == "acme/reasoner" and row["reasoning_effort"] == "high"
        )
        self.assertEqual(high["available_benchmark_count"], 2)
        self.assertEqual(high["default_slots_available_count"], 1)
        self.assertEqual(
            high["default_slots_available"],
            [
                {
                    "benchmark": "bench/a@1.0.0",
                    "benchmark_profiles": ["standard"],
                    "metric": "accuracy",
                }
            ],
        )
        self.assertEqual(
            high["default_slots_missing"],
            [
                {
                    "benchmark": "bench/b@1.0.0",
                    "benchmark_profiles": ["agent"],
                    "metric": "resolved",
                }
            ],
        )
        low = next(
            row
            for row in report["model_efforts"]
            if row["model"] == "acme/reasoner" and row["reasoning_effort"] == "low"
        )
        self.assertEqual(low["available_benchmark_count"], 0)
        self.assertEqual(
            report["models_without_evaluations"], ["other/chat", "router/recipe"]
        )
        self.assertEqual(report["physical_models_without_evaluations"], ["other/chat"])
        self.assertEqual(
            report["physical_models_below_five"],
            [
                {
                    "model": "acme/reasoner",
                    "best_evidence_bucket": {
                        "reasoning_effort": "high",
                        "provenance": "third_party",
                    },
                    "available_benchmark_count": 2,
                },
                {
                    "model": "other/chat",
                    "best_evidence_bucket": None,
                    "available_benchmark_count": 0,
                },
            ],
        )
        self.assertEqual(
            {
                (row["model"], row["reasoning_effort"])
                for row in report["model_efforts_below_five"]
            },
            {
                ("acme/reasoner", "low"),
                ("acme/reasoner", "high"),
                ("other/chat", "unspecified"),
            },
        )

    def test_filters_scope_models_evaluations_publishers_and_providers(self) -> None:
        report = audit.build_audit(self.manifest, self.resources, publishers={"acme"})

        self.assertEqual(report["counts"]["physical_models"], 1)
        self.assertEqual(report["counts"]["virtual_models"], 0)
        self.assertEqual(report["counts"]["evaluations"], 2)
        self.assertEqual(report["counts"]["providers"], 1)
        self.assertEqual([row["publisher"] for row in report["publishers"]], ["Acme"])
        self.assertEqual([row["provider"] for row in report["providers"]], ["acme-api"])

    def test_explicit_gates_distinguish_model_from_effort_coverage(
        self,
    ) -> None:
        resources = copy.deepcopy(self.resources)
        resources["evaluations"].append(
            {
                "model": "acme/reasoner",
                "reasoning_effort": "unspecified",
                "benchmark": "bench/a@1.0.0",
                "benchmark_profile": "standard",
                "metrics": {"accuracy": 0.4},
                "status": "available",
                "evidence": {"provenance": "third_party"},
            }
        )
        report = audit.build_audit(self.manifest, resources, models={"acme/reasoner"})

        self.assertEqual(
            [
                row["reasoning_effort"]
                for row in audit.gate_failures(report, 2, scope="effort")
            ],
            ["low", "unspecified"],
        )
        self.assertEqual(
            [
                row["reasoning_effort"]
                for row in audit.gate_failures(report, 2, scope="selectable-effort")
            ],
            ["low"],
        )
        self.assertEqual(audit.gate_failures(report, 1, scope="model"), [])
        self.assertEqual(audit.gate_failures(report, 0, scope="effort"), [])
        self.assertEqual(audit.gate_failures(report, 0, scope="selectable-effort"), [])
        with self.assertRaisesRegex(ValueError, "unsupported gate scope"):
            audit.gate_failures(report, 1, scope="unknown")

    def test_model_gate_does_not_pool_distinct_provenance_buckets(self) -> None:
        resources = copy.deepcopy(self.resources)
        resources["evaluations"].extend(
            [
                {
                    "model": "acme/reasoner",
                    "reasoning_effort": "high",
                    "benchmark": f"bench/vendor-{index}@1.0.0",
                    "benchmark_profile": "standard",
                    "metrics": {"accuracy": 0.6},
                    "status": "available",
                    "evidence": {"provenance": "vendor_claimed"},
                }
                for index in range(2)
            ]
        )

        report = audit.build_audit(self.manifest, resources, models={"acme/reasoner"})

        high = next(
            row for row in report["model_efforts"] if row["reasoning_effort"] == "high"
        )
        self.assertEqual(high["available_benchmark_count"], 4)
        self.assertEqual(
            {
                (row["reasoning_effort"], row["provenance"]): row[
                    "available_benchmark_count"
                ]
                for row in report["model_evidence_buckets"]
            },
            {("high", "third_party"): 2, ("high", "vendor_claimed"): 2},
        )
        failures = audit.gate_failures(report, 3, scope="model")
        self.assertEqual(len(failures), 1)
        self.assertEqual(
            failures[0]["best_evidence_bucket"],
            {"reasoning_effort": "high", "provenance": "third_party"},
        )
        self.assertEqual(failures[0]["best_evidence_bucket_benchmark_count"], 2)

    def test_selectable_effort_cli_gate_ignores_evidence_only_buckets(self) -> None:
        resources = copy.deepcopy(self.resources)
        resources["evaluations"].append(
            {
                "model": "acme/reasoner",
                "reasoning_effort": "unspecified",
                "benchmark": "bench/a@1.0.0",
                "benchmark_profile": "standard",
                "metrics": {"accuracy": 0.4},
                "status": "available",
                "evidence": {"provenance": "third_party"},
            }
        )
        output = io.StringIO()
        errors = io.StringIO()
        with (
            mock.patch.object(
                audit,
                "load_and_validate",
                return_value=(self.manifest, resources, {}),
            ),
            redirect_stdout(output),
            redirect_stderr(errors),
        ):
            result = audit.main(
                [
                    "--models",
                    "acme/reasoner",
                    "--require-min-evaluations-per-selectable-effort",
                    "2",
                ]
            )

        self.assertEqual(result, 1)
        self.assertIn("1 physical selectable-effort rows", errors.getvalue())
        self.assertIn("complete=0 partial=1 unmeasured=1", output.getvalue())

    def test_json_output_includes_selectable_effort_coverage(self) -> None:
        output = io.StringIO()
        with (
            mock.patch.object(
                audit,
                "load_and_validate",
                return_value=(self.manifest, self.resources, {}),
            ),
            redirect_stdout(output),
        ):
            result = audit.main(["--json"])

        self.assertEqual(result, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(
            payload["selectable_effort_coverage"],
            {
                "minimum_available_benchmarks": 5,
                "total": 2,
                "complete": 0,
                "partial": 1,
                "unmeasured": 1,
            },
        )

    def test_unknown_filters_are_rejected(self) -> None:
        with self.assertRaisesRegex(audit.CatalogBuildError, "unknown catalog model"):
            audit.build_audit(self.manifest, self.resources, models={"missing/model"})
        with self.assertRaisesRegex(
            audit.CatalogBuildError, "unknown catalog publisher"
        ):
            audit.build_audit(self.manifest, self.resources, publishers={"Missing"})
        with self.assertRaisesRegex(audit.CatalogBuildError, "select no models"):
            audit.build_audit(
                self.manifest,
                self.resources,
                publishers={"Acme"},
                models={"other/chat"},
            )


if __name__ == "__main__":
    unittest.main()
