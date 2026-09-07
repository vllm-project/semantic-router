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

        results = {
            result["index"]: result for result in catalog._index_results(resources)
        }
        self.assertEqual(results["example/domain@1.0.0"]["score"], 80.0)
        self.assertEqual(
            results["example/domain@1.0.0"]["domains"], {"reasoning": 80.0}
        )
        self.assertEqual(
            results["example/composite@1.0.0"]["provenance"],
            ["example/run@1.0.0"],
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

    def test_stale_projection_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "catalog.json"
            output.write_bytes(b"stale")
            self.assertEqual(catalog.check({output: b"current"}), 1)


if __name__ == "__main__":
    unittest.main()
