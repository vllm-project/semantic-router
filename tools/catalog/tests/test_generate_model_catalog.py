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

DEFAULT_INDEX_COMPONENT_COUNT = 5


def assert_expected_models(
    test_case: unittest.TestCase,
    physical_models: dict[str, dict[str, object]],
    expected_by_publisher: dict[str, set[str]],
) -> None:
    for publisher, expected_ids in expected_by_publisher.items():
        test_case.assertTrue(expected_ids.issubset(physical_models))
        test_case.assertTrue(
            all(
                physical_models[model_id]["publisher"] == publisher
                for model_id in expected_ids
            )
        )


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
                f"{component['benchmark']}#{component['metric']}": component["weight"]
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
                "google/diffusiongemma-26b-a4b-it",
            },
            "Meta": {
                "meta/llama-2-70b-chat",
                "meta/llama-3.3-70b-instruct",
                "meta/llama-4-maverick-17b-128e-instruct",
                "meta/muse-spark-1.3",
                "meta/muse-glimmer-30b",
            },
            "Alibaba Cloud": {
                "qwen/qwen2.5-72b-instruct",
                "qwen/qwen3-235b-a22b",
                "qwen/qwen3.7-flash",
                "qwen/qwen3.8-max",
                "qwen/qwen3-max",
            },
            "DeepSeek": {
                "deepseek/deepseek-v2-chat",
                "deepseek/deepseek-r1",
                "deepseek/deepseek-v4-pro",
                "deepseek/deepseek-v4-flash-vision-exp",
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
                "nvidia/openreasoning-nemotron-32b",
                "nvidia/nemotron-terminal-32b",
            },
            "Mistral AI": {
                "mistral/mixtral-8x7b-instruct-v0.1",
                "mistral/mistral-small-4",
            },
        }
        assert_expected_models(self, physical_models, expected_by_publisher)

    def test_mainstream_model_inventory_covers_specialist_frontier_publishers(
        self,
    ) -> None:
        _, resources, _ = catalog.load_and_validate()
        physical_models = {
            item["id"]: item
            for item in resources["models"]
            if item["kind"] == "physical"
        }
        expected_by_publisher = {
            "MiniMax": {
                "minimax/minimax-m1-80k",
                "minimax/minimax-m3",
            },
            "Cohere": {
                "cohere/command-a-plus-05-2026",
                "cohere/north-mini-code-1.0",
                "cohere/tiny-aya-global",
                "cohere/tiny-aya-water",
            },
            "xAI": {
                "xai/grok-4.6",
                "xai/grok-build-0.1",
            },
            "Celeris": {"celeris/celeris-1"},
            "Inception": {
                "inception/mercury-2",
                "inception/mercury-2.5-preview",
            },
            "ByteDance Seed": {
                "bytedance/seed-oss-36b-instruct",
                "bytedance/seed-2.1-pro",
                "bytedance/seed-2.1-turbo",
            },
            "Tencent": {
                "tencent/hy3",
                "tencent/hy4-preview",
                "tencent/hy-mt2-30b-a3b",
            },
            "Xiaomi": {
                "xiaomi/mimo-7b-rl",
                "xiaomi/mimo-v2-pro",
                "xiaomi/mimo-v2.5-pro",
                "xiaomi/mimo-v2-omni",
            },
            "Thinking Machines Lab": {
                "thinking-machines/inkling",
                "thinking-machines/inkling-small",
            },
            "LG AI Research": {
                "lg/k-exaone-2.0-750b-a37b",
                "lg/exaone-4.5-33b",
            },
            "Microsoft AI": {"microsoft/mai-thinking-1"},
            "Institute of Foundation Models": {
                "mbzuai/k2-horizon-375b-a23b",
                "mbzuai/k2-think-v2",
            },
            "SK Telecom": {"skt/ax-k2"},
            "Dots Studio": {"dots/dots3-note-preview"},
            "AI9Stars": {
                "ai9stars/g9v3-39b-a5b",
                "ai9stars/g9v3-3b",
            },
            "Poolside": {
                "poolside/laguna-xs-2.1",
                "poolside/laguna-s-2.1",
            },
            "Arcee AI": {"arcee/trinity-large-thinking"},
            "Kuaishou KwaiPilot": {"kwaipilot/kat-coder-v2.5-dev"},
            "Sakana AI": {"sakana/fugu-ultra"},
            "Aion Labs": {"aion/aion-3.0"},
            "Writer": {"writer/palmyra-x5"},
            "Perceptron": {"perceptron/perceptron-mk1"},
            "Upstage": {"upstage/solar-open2-250b"},
            "Motif Technologies": {
                "motif/motif-2-12.7b-reasoning",
                "motif/motif-3",
            },
            "Meituan LongCat": {
                "meituan/longcat-flash-lite",
                "meituan/longcat-2.0",
            },
            "Liquid AI": {
                "liquid/lfm2-24b-a2b",
                "liquid/lfm2-8b-a1b",
                "liquid/lfm2.5-1.2b-thinking",
                "liquid/lfm2.5-2.6b",
                "liquid/lfm2.5-8b-a1b",
                "liquid/lfm2.5-vl-1.6b",
            },
        }
        assert_expected_models(self, physical_models, expected_by_publisher)

    def test_mainstream_model_inventory_covers_regional_and_open_publishers(
        self,
    ) -> None:
        _, resources, _ = catalog.load_and_validate()
        physical_models = {
            item["id"]: item
            for item in resources["models"]
            if item["kind"] == "physical"
        }
        expected_by_publisher = {
            "Nex AGI": {
                "nex-agi/nex-n2-mini",
                "nex-agi/nex-n2-pro",
            },
            "Baidu": {
                "baidu/ernie-4.5-300b-a47b",
                "baidu/ernie-5.0",
                "baidu/ernie-5.0-thinking-preview",
            },
            "IBM": {
                "ibm/granite-4.1-30b",
                "ibm/granite-4.2-30b",
            },
            "NAVER": {
                "naver/hyperclovax-seed-think-14b",
                "naver/hyperclovax-seed-think-32b",
            },
            "Multiverse Computing": {
                "multiverse/quasar-438b",
                "multiverse/hypernova-60b-2605",
            },
            "Sarvam AI": {
                "sarvam/sarvam-30b",
                "sarvam/sarvam-105b",
            },
            "Prime Intellect": {"prime-intellect/intellect-3"},
            "ServiceNow": {"servicenow/apriel-1.6-15b-thinker"},
            "Deep Cogito": {"deepcogito/cogito-671b-v2.1"},
            "Agnes AI": {
                "agnes/agnes-2.5-pro-alpha",
                "agnes/agnes-2.5-pro-beta",
            },
            "Nous Research": {
                "nousresearch/hermes-3-llama-3.1-70b",
                "nousresearch/deephermes-3-llama-3-8b-preview",
                "nousresearch/hermes-4-70b",
                "nousresearch/hermes-4-405b",
            },
            "Apodex AI": {
                "apodex/apodex-1.1",
                "apodex/apodex-1.1-mini",
            },
            "Perplexity": {
                "perplexity/sonar",
                "perplexity/sonar-deep-research",
            },
            "Swiss AI Initiative": {
                "swiss-ai/apertus-v1.5-8b",
                "swiss-ai/apertus-v1.5-70b",
            },
            "Nanbeige LLM Lab": {
                "nanbeige/nanbeige4.1-3b",
                "nanbeige/nanbeige4.2-3b",
            },
        }
        assert_expected_models(self, physical_models, expected_by_publisher)

    def test_catalog_inventory_is_bound_and_includes_provider_baseline(self) -> None:
        _, resources, _ = catalog.load_and_validate()
        physical_models = {
            item["id"]: item
            for item in resources["models"]
            if item["kind"] == "physical"
        }
        provider_bindings = [
            {**binding, "provider": provider["id"]}
            for provider in resources["providers"]
            for binding in provider.get("models", [])
        ]
        self.assertGreaterEqual(len(physical_models), 504)
        self.assertGreaterEqual(len(provider_bindings), 762)

        bound_models = {item["catalog"] for item in provider_bindings}
        self.assertTrue(set(physical_models).issubset(bound_models))

        provider_ids = {provider["id"] for provider in resources["providers"]}
        self.assertTrue(
            {
                "agnes",
                "apodex",
                "baidu-qianfan",
                "compactifai",
                "perplexity",
                "sarvam",
                "vllm",
                "sglang",
            }.issubset(provider_ids)
        )

    def test_provider_owns_its_model_bindings(self) -> None:
        resources_root = catalog.SOURCE_ROOT / "resources"
        self.assertTrue((resources_root / "providers").is_dir())
        self.assertFalse((resources_root / "offerings").exists())
        self.assertFalse((resources_root / "provider-models").exists())

        _, resources, _ = catalog.load_and_validate()
        self.assertTrue(
            all(
                {"catalog", "id", "protocols"}.issubset(binding)
                for provider in resources["providers"]
                for binding in provider.get("models", [])
            )
        )

        providers = {provider["id"]: provider for provider in resources["providers"]}
        self.assertEqual(
            providers["perplexity"]["path_overrides"][
                "openai/chat-completions@1#create"
            ],
            "/v1/sonar",
        )
        self.assertEqual(
            next(
                binding
                for binding in providers["dashscope"]["models"]
                if binding["catalog"] == "qwen/qwen3-max"
            )["reasoning_transport"],
            "top_level_boolean",
        )
        for runtime in ("vllm", "sglang"):
            self.assertEqual(
                next(
                    binding
                    for binding in providers[runtime]["models"]
                    if binding["catalog"] == "sarvam/sarvam-105b"
                )["reasoning_transport"],
                "top_level_effort",
            )

    def test_every_model_reasoning_mode_has_five_default_evaluation_slots(
        self,
    ) -> None:
        manifest, resources, _ = catalog.load_and_validate()
        coverage = catalog._evaluation_coverage(
            resources, manifest["defaults"]["intelligence_index"]
        )
        slots: dict[tuple[str, str], list[dict[str, object]]] = {}
        for row in coverage:
            slots.setdefault((row["model"], row["reasoning_effort"]), []).append(row)
        self.assertTrue(slots)
        self.assertTrue(
            all(len(rows) == DEFAULT_INDEX_COMPONENT_COUNT for rows in slots.values())
        )
        self.assertTrue(
            all(
                len(
                    {
                        (row["benchmark"], row["benchmark_profile"], row["metric"])
                        for row in rows
                    }
                )
                == DEFAULT_INDEX_COMPONENT_COUNT
                for rows in slots.values()
            )
        )
        self.assertEqual(
            {model["id"] for model in resources["models"]},
            {model for model, _ in slots},
        )

    def test_frontier_evaluation_coverage_keeps_published_values_and_gaps(self) -> None:
        manifest, resources, _ = catalog.load_and_validate()
        coverage = catalog._evaluation_coverage(
            resources, manifest["defaults"]["intelligence_index"]
        )
        available: dict[tuple[str, str], set[str]] = {}
        for row in coverage:
            if row["status"] == "available":
                available.setdefault(
                    (row["model"], row["reasoning_effort"]), set()
                ).add(row["benchmark"])

        expected_counts = {
            ("deepseek/deepseek-v4-flash", "max"): 5,
            ("deepseek/deepseek-v4-pro", "max"): 5,
            ("zai/glm-5.3-flash", "max"): 2,
            ("qwen/qwen3.8-27b", "xhigh"): 3,
            ("tencent/hy3", "high"): 4,
            ("thinking-machines/inkling", "xhigh"): 4,
            ("microsoft/mai-thinking-1", "published"): 3,
            ("xiaomi/mimo-7b-rl", "default"): 2,
            ("arcee/trinity-large-thinking", "default"): 3,
            ("kwaipilot/kat-coder-v2.5-dev", "enabled"): 2,
            ("nvidia/openreasoning-nemotron-32b", "default"): 3,
            ("skt/ax-k2", "enabled"): 3,
            ("mbzuai/k2-think-v2", "high"): 2,
            ("agnes/agnes-2.5-pro-alpha", "published"): 3,
            ("multiverse/hypernova-60b-2605", "high"): 3,
            ("sarvam/sarvam-105b", "published"): 3,
            ("ibm/granite-4.1-30b", "default"): 2,
            ("nousresearch/hermes-4-70b", "enabled"): 2,
            ("nousresearch/hermes-4-70b", "disabled"): 2,
            ("nousresearch/hermes-4-405b", "enabled"): 2,
            ("nousresearch/hermes-4-405b", "disabled"): 2,
            ("apodex/apodex-1.1", "published"): 2,
            ("cohere/tiny-aya-global", "published"): 1,
            ("google/diffusiongemma-26b-a4b-it", "published"): 3,
            ("nanbeige/nanbeige4.1-3b", "default"): 2,
            ("nanbeige/nanbeige4.2-3b", "enabled"): 3,
        }
        self.assertEqual(
            {key: len(available.get(key, set())) for key in expected_counts},
            expected_counts,
        )
        self.assertNotIn(
            "cais/humanitys-last-exam@1.0.0",
            available[("zai/glm-5.3-flash", "max")],
        )
        for effort in ("none", "low", "medium"):
            self.assertNotIn(("qwen/qwen3.8-27b", effort), available)

    def test_third_party_measurements_are_anchored_to_official_republications(
        self,
    ) -> None:
        _, resources, _ = catalog.load_and_validate()
        third_party = [
            record
            for record in resources["evaluations"]
            if record["evidence"]["provenance"] == "third_party"
        ]
        self.assertTrue(third_party)
        self.assertTrue(
            all(
                "artificialanalysis.ai" not in record["evidence"]["source"]
                for record in third_party
            )
        )
        self.assertTrue(
            all(
                record["subject"]["source_kind"]
                in {
                    "official_vendor_republication",
                    "official_cross-vendor_comparison",
                    "official_model_card",
                }
                for record in third_party
            )
        )

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

        provider["reasoning_transport"] = "reasoning_object"
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

    def test_provider_catalog_model_protocol_must_be_supported(self) -> None:
        providers = {
            "example": {
                "protocols": ["openai/chat-completions@1"],
                "supported_operations": ["openai/chat-completions@1#create"],
                "models": [
                    {
                        "catalog": "example/model",
                        "id": "native-model",
                        "protocols": ["openai/chat-completions@1"],
                    }
                ],
            }
        }
        models = {"example/model": {"kind": "physical", "lifecycle": "active"}}
        catalog._validate_provider_bindings(
            providers, models, {"openai/chat-completions@1"}
        )

    def test_physical_model_requires_a_provider_binding(self) -> None:
        with self.assertRaisesRegex(
            catalog.CatalogBuildError, "require at least one provider binding"
        ):
            catalog._validate_provider_bindings(
                {},
                {
                    "example/model": {
                        "kind": "physical",
                        "lifecycle": "active",
                    }
                },
                {"openai/chat-completions@1"},
            )

    def test_available_evaluation_metric_is_unambiguous(self) -> None:
        records = [
            {
                "id": f"example/run-{index}@1.0.0",
                "model": "example/model",
                "benchmark": "example/bench@1.0.0",
                "benchmark_profile": "standard",
                "reasoning_effort": "default",
                "subject": {},
                "metrics": {"score": value},
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
                "profiles": {"standard"},
            }
        }
        with self.assertRaisesRegex(
            catalog.CatalogBuildError, "one available value is allowed"
        ):
            catalog._validate_evaluations(
                records,
                {"example/model": {"id": "example/model"}},
                {},
                metrics,
            )

    def test_missing_index_components_remain_unavailable(self) -> None:
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
            "evaluations": [],
            "indices": [
                {
                    "id": "example/index@1.0.0",
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
                }
            ],
        }
        self.assertEqual(
            catalog._index_results(resources),
            [
                {
                    "model": "example/model",
                    "reasoning_effort": "default",
                    "index": "example/index@1.0.0",
                    "status": "missing",
                    "score": None,
                    "coverage": 0.0,
                    "components": [
                        {
                            "benchmark": "example/bench@1.0.0",
                            "metric": "score",
                            "benchmark_profile": "standard",
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


if __name__ == "__main__":
    unittest.main()
