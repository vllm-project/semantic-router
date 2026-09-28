from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "generate_model_catalog.py"
SPEC = importlib.util.spec_from_file_location("generate_model_catalog", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
catalog = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = catalog
SPEC.loader.exec_module(catalog)

import catalog_evaluations as evaluations  # noqa: E402


class CatalogIndexEvaluationTests(unittest.TestCase):
    def test_nested_index_preserves_domain_score_and_record_lineage(self) -> None:
        resources = {
            "models": [{"id": "example/model", "kind": "physical"}],
            "reasoning_families": [],
            "benchmarks": [
                {
                    "id": "example/bench@1.0.0",
                    "domain": "reasoning",
                    "default_profile": "standard",
                    "profiles": [{"id": "standard"}],
                    "metrics": [{"id": "score"}],
                }
            ],
            "evaluations": [
                {
                    "id": "example/run@1.0.0",
                    "model": "example/model",
                    "benchmark": "example/bench@1.0.0",
                    "benchmark_profile": "standard",
                    "reasoning_effort": "default",
                    "status": "available",
                    "metrics": {"score": 0.8},
                    "evidence": {"provenance": "operator"},
                }
            ],
            "indices": [
                {
                    "id": "example/domain@1.0.0",
                    "scale": [0, 100],
                    "missing": {"policy": "require_all"},
                    "components": [
                        {
                            "benchmark": "example/bench@1.0.0",
                            "metric": "score",
                            "benchmark_profile": "standard",
                            "weight": 1.0,
                            "normalization": {"type": "identity"},
                        }
                    ],
                },
                {
                    "id": "example/composite@1.0.0",
                    "scale": [0, 100],
                    "missing": {"policy": "require_all"},
                    "components": [
                        {
                            "index": "example/domain@1.0.0",
                            "weight": 1.0,
                            "normalization": {"type": "identity"},
                        }
                    ],
                },
            ],
        }

        materialized = catalog._index_results(resources)
        results = {result["index"]: result for result in materialized}
        self.assertEqual(results["example/domain@1.0.0"]["score"], 80.0)
        self.assertEqual(
            results["example/domain@1.0.0"]["domains"], {"reasoning": 80.0}
        )
        self.assertEqual(
            results["example/composite@1.0.0"]["provenance"],
            ["example/run@1.0.0"],
        )
        self.assertEqual(
            evaluations.index_leaf_components(
                resources["indices"], "example/composite@1.0.0"
            ),
            resources["indices"][0]["components"],
        )
        self.assertEqual(
            evaluations.evaluation_coverage(
                resources,
                "example/composite@1.0.0",
                materialized,
            ),
            [
                {
                    "model": "example/model",
                    "reasoning_effort": "default",
                    "benchmark": "example/bench@1.0.0",
                    "benchmark_profiles": ["standard"],
                    "benchmark_profile": "standard",
                    "metric": "score",
                    "status": "available",
                    "value": 0.8,
                    "evaluation": "example/run@1.0.0",
                }
            ],
        )

    def test_extended_normalizations_are_supported(self) -> None:
        self.assertAlmostEqual(
            catalog._normalize_component(
                5.0,
                {
                    "type": "piecewise_linear",
                    "points": [
                        {"input": 0.0, "output": 0.0},
                        {"input": 10.0, "output": 1.0},
                    ],
                },
            ),
            0.5,
        )
        self.assertEqual(
            catalog._normalize_component(
                2.0, {"type": "lookup", "values": {"2": 0.75}}
            ),
            0.75,
        )

    def test_ordered_profiles_prefer_the_first_available_exact_match(self) -> None:
        resources = {
            "models": [{"id": "example/model", "kind": "physical"}],
            "reasoning_families": [],
            "benchmarks": [
                {
                    "id": "example/bench@1.0.0",
                    "domain": "reasoning",
                    "default_profile": "published",
                    "profiles": [{"id": "independent"}, {"id": "published"}],
                    "metrics": [{"id": "score"}],
                }
            ],
            "evaluations": [
                {
                    "id": "example/published@1.0.0",
                    "model": "example/model",
                    "benchmark": "example/bench@1.0.0",
                    "benchmark_profile": "published",
                    "reasoning_effort": "default",
                    "status": "available",
                    "metrics": {"score": 0.9},
                    "evidence": {"provenance": "vendor_claimed"},
                },
                {
                    "id": "example/independent@1.0.0",
                    "model": "example/model",
                    "benchmark": "example/bench@1.0.0",
                    "benchmark_profile": "independent",
                    "reasoning_effort": "default",
                    "status": "available",
                    "metrics": {"score": 0.7},
                    "evidence": {"provenance": "third_party"},
                },
            ],
            "indices": [
                {
                    "id": "example/index@1.0.0",
                    "scale": [0, 100],
                    "missing": {"policy": "require_all"},
                    "components": [
                        {
                            "benchmark": "example/bench@1.0.0",
                            "metric": "score",
                            "benchmark_profiles": ["independent", "published"],
                            "weight": 1.0,
                            "normalization": {"type": "identity"},
                        }
                    ],
                }
            ],
        }

        result = catalog._index_results(resources)[0]
        self.assertEqual(result["score"], 70.0)
        self.assertEqual(result["provenance"], ["example/independent@1.0.0"])
        self.assertEqual(result["components"][0]["benchmark_profile"], "independent")
        self.assertEqual(
            result["components"][0]["benchmark_profiles"],
            ["independent", "published"],
        )

    def test_incomplete_evidence_is_partial_without_a_score(self) -> None:
        resources = {
            "models": [{"id": "example/model", "kind": "physical"}],
            "reasoning_families": [],
            "benchmarks": [
                {
                    "id": "example/one@1.0.0",
                    "domain": "reasoning",
                    "default_profile": "standard",
                    "profiles": [{"id": "standard"}],
                    "metrics": [{"id": "score"}],
                },
                {
                    "id": "example/two@1.0.0",
                    "domain": "coding",
                    "default_profile": "standard",
                    "profiles": [{"id": "standard"}],
                    "metrics": [{"id": "score"}],
                },
            ],
            "evaluations": [
                {
                    "id": "example/run@1.0.0",
                    "model": "example/model",
                    "benchmark": "example/one@1.0.0",
                    "benchmark_profile": "standard",
                    "reasoning_effort": "default",
                    "status": "available",
                    "metrics": {"score": 0.8},
                    "evidence": {"provenance": "operator"},
                }
            ],
            "indices": [
                {
                    "id": "example/index@1.0.0",
                    "scale": [0, 100],
                    "missing": {"policy": "require_all"},
                    "components": [
                        {
                            "benchmark": "example/one@1.0.0",
                            "metric": "score",
                            "benchmark_profile": "standard",
                            "weight": 0.5,
                            "normalization": {"type": "identity"},
                        },
                        {
                            "benchmark": "example/two@1.0.0",
                            "metric": "score",
                            "benchmark_profile": "standard",
                            "weight": 0.5,
                            "normalization": {"type": "identity"},
                        },
                    ],
                }
            ],
        }

        result = catalog._index_results(resources)[0]
        self.assertEqual(result["status"], "partial")
        self.assertIsNone(result["score"])
        self.assertEqual(result["coverage"], 0.5)
        self.assertEqual(result["provenance"], ["example/run@1.0.0"])

    def test_nested_index_rejects_stray_benchmark_profile_fields(self) -> None:
        with self.assertRaisesRegex(
            catalog.CatalogBuildError,
            "must reference exactly one metric or index",
        ):
            evaluations._validate_index_component(
                {
                    "index": "example/base@1.0.0",
                    "benchmark_profile": "",
                    "weight": 1.0,
                },
                "indices[0].components[0]",
                {"example/base@1.0.0"},
                {},
            )

    def test_stale_projection_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "catalog.json"
            output.write_bytes(b"stale")
            self.assertEqual(catalog.check({output: b"current"}), 1)


if __name__ == "__main__":
    unittest.main()
