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


class ModelCatalogCompilerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocols = [
            {
                "id": "openai/chat-completions@1",
                "operations": [
                    {"id": "create", "method": "POST", "path": "/v1/chat/completions"},
                    {"id": "list_models", "method": "GET", "path": "/v1/models"},
                ],
            }
        ]

    def test_repository_catalog_validates_and_renders_every_projection(self) -> None:
        outputs = catalog.render_outputs()
        self.assertEqual(
            set(outputs),
            {
                catalog.RECIPE_MANIFEST,
                catalog.CLI_MANIFEST,
                catalog.GO_OUTPUT,
                catalog.DASHBOARD_OUTPUT,
                catalog.WEBSITE_OUTPUT,
            },
        )
        self.assertEqual(catalog.check(outputs), 0)

    def test_default_intelligence_index_pins_published_v4_1_1_methodology(self) -> None:
        _, resources, _ = catalog.load_and_validate()
        index = next(
            item
            for item in resources["indices"]
            if item["id"] == "vllm-sr/intelligence@1.0.0"
        )
        self.assertEqual(index["missing"], {"policy": "require_all"})
        self.assertEqual(
            index["domains"],
            {
                "agents": 0.34,
                "coding": 0.24,
                "general": 0.18,
                "scientific_reasoning": 0.24,
            },
        )
        self.assertEqual(
            {
                component["metric"]: component["weight"]
                for component in index["components"]
            },
            {
                "artificial-analysis/gdpval-aa@2.0.0#elo": 0.20,
                "artificial-analysis/tau3-banking@1.0.0#pass_at_1": 0.14,
                "terminal-bench/terminal-bench@2.1.0#pass_at_1": 0.16,
                "scicode/scicode@1.0.0#pass_at_1": 0.08,
                "artificial-analysis/lcr@1.0.0#pass_at_1": 0.06,
                "artificial-analysis/omniscience@1.0.0#accuracy": 0.08,
                "artificial-analysis/omniscience@1.0.0#non_hallucination_rate": 0.04,
                "cais/humanitys-last-exam@1.0.0#pass_at_1": 0.12,
                "idavidrein/gpqa-diamond@1.0.0#pass_at_1": 0.06,
                "critpt/critpt@1.0.0#pass_at_1": 0.06,
            },
        )
        gdpval = next(
            component["normalization"]
            for component in index["components"]
            if component["metric"] == "artificial-analysis/gdpval-aa@2.0.0#elo"
        )
        self.assertEqual(gdpval, {"type": "linear_clamp", "min": 500, "max": 2500})
        self.assertEqual(catalog._normalize_component(500, gdpval), 0)
        self.assertEqual(catalog._normalize_component(1500, gdpval), 0.5)
        self.assertEqual(catalog._normalize_component(2500, gdpval), 1)

    def test_schema_failure_reports_the_resource_path(self) -> None:
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "required": ["id"],
            "properties": {"id": {"type": "string", "minLength": 1}},
        }
        with self.assertRaisesRegex(catalog.CatalogBuildError, r"provider\.id"):
            catalog._validate_schema({"id": ""}, schema, "provider")

    def test_security_validation_rejects_secret_fields_and_literals(self) -> None:
        with self.assertRaisesRegex(catalog.CatalogBuildError, "secret-like field"):
            catalog._validate_security({"provider": {"api_key": "not-published"}})
        with self.assertRaisesRegex(
            catalog.CatalogBuildError, "credential-like literal"
        ):
            catalog._validate_security({"source": "Bearer abcdefghijklmnop"})

    def test_provider_default_headers_cannot_store_credentials(self) -> None:
        provider = {
            "id": "example",
            "category": "model_api",
            "support_tier": "compatible",
            "protocols": ["openai/chat-completions@1"],
            "default_protocol": "openai/chat-completions@1",
            "supported_operations": ["openai/chat-completions@1#create"],
            "auth": {
                "strategy": "bearer",
                "header": "Authorization",
                "prefix": "Bearer",
            },
            "presentation": {"logo": "monogram", "monogram": "E", "monochrome": True},
            "conformance": {"status": "unverified"},
            "default_headers": {"Authorization": "not-a-secret"},
        }
        with self.assertRaisesRegex(catalog.CatalogBuildError, "credential headers"):
            catalog._validate_providers([provider], self.protocols)

    def test_provider_reasoning_transport_is_a_known_semantic_adapter(self) -> None:
        provider = {
            "id": "example",
            "category": "model_api",
            "support_tier": "compatible",
            "protocols": ["openai/chat-completions@1"],
            "default_protocol": "openai/chat-completions@1",
            "supported_operations": ["openai/chat-completions@1#create"],
            "reasoning_transport": "hostname_switch",
            "auth": {
                "strategy": "bearer",
                "header": "Authorization",
                "prefix": "Bearer",
            },
            "presentation": {"logo": "monogram", "monogram": "E", "monochrome": True},
            "conformance": {"status": "unverified"},
        }
        with self.assertRaisesRegex(
            catalog.CatalogBuildError, "reasoning_transport is unsupported"
        ):
            catalog._validate_providers([provider], self.protocols)

    def test_provider_operations_are_explicit_protocol_subsets(self) -> None:
        provider = {
            "id": "example",
            "category": "model_api",
            "support_tier": "compatible",
            "protocols": ["openai/chat-completions@1"],
            "default_protocol": "openai/chat-completions@1",
            "supported_operations": ["openai/chat-completions@1#delete_model"],
            "auth": {
                "strategy": "bearer",
                "header": "Authorization",
                "prefix": "Bearer",
            },
            "presentation": {"logo": "monogram", "monogram": "E", "monochrome": True},
            "conformance": {"status": "unverified"},
        }
        with self.assertRaisesRegex(
            catalog.CatalogBuildError, "unknown or duplicate operation"
        ):
            catalog._validate_providers([provider], self.protocols)

    def test_offering_protocol_must_be_supported_end_to_end(self) -> None:
        providers = {
            "example": {
                "protocols": ["openai/chat-completions@1"],
                "supported_operations": ["openai/chat-completions@1#create"],
            }
        }
        models = {"example/model": {"protocols": ["anthropic/messages@1"]}}
        offering = {
            "id": "example/model@1",
            "provider": "example",
            "model": "example/model",
            "protocols": ["openai/chat-completions@1"],
        }
        with self.assertRaisesRegex(
            catalog.CatalogBuildError, "not supported by its model"
        ):
            catalog._validate_offerings(
                [offering], providers, models, {"openai/chat-completions@1"}
            )

    def test_missing_index_components_remain_unavailable(self) -> None:
        resources = {
            "models": [{"id": "example/model", "kind": "physical"}],
            "benchmarks": [
                {
                    "id": "example/bench@1.0.0",
                    "domain": "reasoning",
                    "metrics": [{"id": "score"}],
                }
            ],
            "evaluations": [],
            "indices": [
                {
                    "id": "example/index@1.0.0",
                    "scale": [0, 100],
                    "missing": {"policy": "require_all"},
                    "components": [
                        {
                            "metric": "example/bench@1.0.0#score",
                            "weight": 1.0,
                            "normalization": {"type": "identity"},
                        }
                    ],
                }
            ],
        }
        self.assertEqual(
            catalog._index_results(resources),
            [
                {
                    "model": "example/model",
                    "index": "example/index@1.0.0",
                    "status": "missing",
                    "score": None,
                    "coverage": 0.0,
                    "components": [
                        {
                            "metric": "example/bench@1.0.0#score",
                            "weight": 1.0,
                            "status": "missing",
                            "value": None,
                            "normalized": None,
                        }
                    ],
                    "provenance": [],
                }
            ],
        )

    def test_nested_index_preserves_domain_score_and_record_lineage(self) -> None:
        resources = {
            "models": [{"id": "example/model", "kind": "physical"}],
            "benchmarks": [
                {
                    "id": "example/bench@1.0.0",
                    "domain": "reasoning",
                    "metrics": [{"id": "score"}],
                }
            ],
            "evaluations": [
                {
                    "id": "example/run@1.0.0",
                    "model": "example/model",
                    "status": "available",
                    "metrics": {"example/bench@1.0.0#score": 0.8},
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
                            "metric": "example/bench@1.0.0#score",
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
