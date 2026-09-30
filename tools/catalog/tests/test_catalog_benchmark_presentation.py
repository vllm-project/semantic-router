from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "generate_model_catalog.py"
SPEC = importlib.util.spec_from_file_location("generate_model_catalog", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
catalog = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = catalog
SPEC.loader.exec_module(catalog)


class CatalogBenchmarkPresentationTests(unittest.TestCase):
    def test_benchmark_tags_and_metric_normalization_are_validated(self) -> None:
        benchmark = {
            "id": "example/bench@1.0.0",
            "display_name": "Example",
            "domain": "reasoning",
            "tags": ["core"],
            "default_profile": "standard",
            "profiles": [
                {
                    "id": "standard",
                    "display_name": "Standard",
                    "description": "Published standard profile.",
                }
            ],
            "metrics": [
                {
                    "id": "elo",
                    "unit": "elo",
                    "direction": "higher_is_better",
                    "range": [0, 3000],
                    "normalization": {
                        "type": "linear_clamp",
                        "min": 500,
                        "max": 2500,
                    },
                }
            ],
        }
        catalog._metric_catalog([benchmark])

        benchmark["tags"] = ["core", "core"]
        with self.assertRaisesRegex(catalog.CatalogBuildError, "unique slug tags"):
            catalog._metric_catalog([benchmark])

        benchmark["tags"] = []
        with self.assertRaisesRegex(catalog.CatalogBuildError, "cannot be empty"):
            catalog._metric_catalog([benchmark])

        benchmark["tags"] = ["core"]
        benchmark["metrics"][0]["normalization"] = {
            "type": "linear_clamp",
            "min": 2500,
            "max": 500,
        }
        with self.assertRaisesRegex(catalog.CatalogBuildError, "bounds are invalid"):
            catalog._metric_catalog([benchmark])


if __name__ == "__main__":
    unittest.main()
