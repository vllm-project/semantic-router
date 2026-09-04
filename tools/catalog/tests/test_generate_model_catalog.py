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

    def test_default_intelligence_index_is_public_and_coverage_aware(self) -> None:
        _, resources, _ = catalog.load_and_validate()
        index = next(
            item
            for item in resources["indices"]
            if item["id"] == "vllm-sr/intelligence@1.0.0"
        )
        self.assertEqual(
            index["missing"], {"policy": "require_coverage", "minimum": 0.6}
        )
        self.assertEqual(
            index["domains"],
            {
                "general_reasoning": 0.20,
                "scientific_reasoning": 0.20,
                "frontier_reasoning": 0.20,
                "software_engineering": 0.20,
                "agentic_systems": 0.20,
            },
        )
        self.assertEqual(
            {
                component["metric"]: component["weight"]
                for component in index["components"]
            },
            {
                "tiger-ai-lab/mmlu-pro@1.0.0#accuracy": 0.20,
                "idavidrein/gpqa-diamond@1.0.0#accuracy": 0.20,
                "cais/humanitys-last-exam@1.0.0#accuracy": 0.20,
                "swe-bench/verified@1.0.0#resolved": 0.20,
                "harbor/terminal-bench@2.1.0#resolved": 0.20,
            },
        )
        self.assertTrue(
            all(
                component["normalization"] == {"type": "identity"}
                for component in index["components"]
            )
        )

    def test_mainstream_model_inventory_covers_current_and_previous_generations(
        self,
    ) -> None:
        _, resources, _ = catalog.load_and_validate()
        physical_models = {
            item["id"]: item
            for item in resources["models"]
            if item["kind"] == "physical"
        }
        offerings = resources["offerings"]

        self.assertGreaterEqual(len(physical_models), 365)
        self.assertGreaterEqual(len(offerings), 395)
        expected_by_publisher = {
            "OpenAI": {
                "openai/gpt-4.1",
                "openai/gpt-5.5",
                "openai/gpt-5.6-sol",
            },
            "Anthropic": {
                "anthropic/claude-3.5-sonnet",
                "anthropic/claude-fable-5",
                "anthropic/claude-mythos-5",
            },
            "Google": {
                "google/gemini-2.5-pro",
                "google/gemini-3.8-flash",
                "google/gemma-3-27b-it",
            },
            "Meta": {
                "meta/llama-2-70b-chat",
                "meta/llama-3.3-70b-instruct",
                "meta/llama-4-maverick-17b-128e-instruct",
            },
            "Alibaba Cloud": {
                "qwen/qwen2.5-72b-instruct",
                "qwen/qwen3-235b-a22b",
                "qwen/qwen3.8-max",
            },
            "DeepSeek": {
                "deepseek/deepseek-v2-chat",
                "deepseek/deepseek-r1",
                "deepseek/deepseek-v4-pro",
            },
            "Z.ai": {
                "zai/glm-4.5",
                "zai/glm-5.3",
            },
            "Moonshot AI": {
                "moonshot/kimi-k2-instruct",
                "moonshot/kimi-k3",
            },
            "NVIDIA": {
                "nvidia/llama-3.1-nemotron-70b-instruct-hf",
                "nvidia/nemotron-3-ultra",
            },
            "Mistral AI": {
                "mistral/mixtral-8x7b-instruct-v0.1",
                "mistral/mistral-small-4",
            },
            "MiniMax": {
                "minimax/minimax-m1-80k",
                "minimax/minimax-m3",
            },
            "Cohere": {
                "cohere/command-a-plus-05-2026",
                "cohere/north-mini-code-1.0",
            },
            "Celeris": {"celeris/celeris-1"},
            "Inception": {
                "inception/mercury-2",
                "inception/mercury-2.5-preview",
            },
        }
        for publisher, expected_ids in expected_by_publisher.items():
            self.assertTrue(expected_ids.issubset(physical_models))
            self.assertTrue(
                all(
                    physical_models[model_id]["publisher"] == publisher
                    for model_id in expected_ids
                )
            )

        offered_models = {item["model"] for item in offerings}
        self.assertTrue(set(physical_models).issubset(offered_models))

    def test_virtual_model_pool_can_reference_operator_defined_models(self) -> None:
        catalog._validate_virtual_model_role(
            {
                "name": "private",
                "required": True,
                "minimum_candidates": 1,
                "traits": ["local_only"],
                "recommended_pool": ["operator/private-model"],
            },
            "models[0].roles[0]",
        )

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

    def test_physical_model_requires_an_offering(self) -> None:
        with self.assertRaisesRegex(
            catalog.CatalogBuildError, "require at least one provider offering"
        ):
            catalog._validate_offerings(
                [],
                {},
                {
                    "example/model": {
                        "kind": "physical",
                        "lifecycle": "active",
                        "protocols": ["openai/chat-completions@1"],
                    }
                },
                {"openai/chat-completions@1"},
            )

    def test_available_evaluation_metric_is_unambiguous(self) -> None:
        records = [
            {
                "id": f"example/run-{index}@1.0.0",
                "model": "example/model",
                "subject": {},
                "metrics": {"example/bench@1.0.0#score": value},
                "status": "available",
                "evidence": {
                    "provenance": "operator",
                    "verification": "claimed",
                    "redistributable": True,
                },
            }
            for index, value in enumerate((0.7, 0.8), start=1)
        ]
        metrics = {
            "example/bench@1.0.0#score": {
                "range": [0, 1],
                "direction": "higher_is_better",
            }
        }
        with self.assertRaisesRegex(
            catalog.CatalogBuildError, "one available value is allowed"
        ):
            catalog._validate_evaluations(records, {"example/model"}, metrics)

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
