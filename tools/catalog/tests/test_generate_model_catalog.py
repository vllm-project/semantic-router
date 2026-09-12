from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from typing import Any

import yaml

MODULE_PATH = Path(__file__).resolve().parents[1] / "generate_model_catalog.py"
SPEC = importlib.util.spec_from_file_location("generate_model_catalog", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
catalog = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = catalog
SPEC.loader.exec_module(catalog)

from catalog_evaluations import evaluation_coverage  # noqa: E402

DEFAULT_INDEX_COMPONENT_COUNT = 6
DEFAULT_INDEX_ID = "vllm-sr/intelligence@1.0.0"
MINIMUM_RANKABLE_MODEL_COUNT = 21
OPENAI_LONG_CONTEXT_TOKENS = 1_050_000


def _evaluation_benchmarks_by_bucket(
    resources: dict[str, Any], model_ids: set[str]
) -> dict[tuple[str, str, str], set[str]]:
    buckets: dict[tuple[str, str, str], set[str]] = {}
    for evaluation in resources["evaluations"]:
        if evaluation["model"] not in model_ids:
            continue
        key = (
            evaluation["model"],
            evaluation["reasoning_effort"],
            evaluation["evidence"]["provenance"],
        )
        buckets.setdefault(key, set()).add(evaluation["benchmark"])
    return buckets


class ModelCatalogCompilerTests(unittest.TestCase):
    def test_dashboard_image_context_includes_the_shared_public_snapshot(self) -> None:
        dockerignore = (catalog.REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")
        self.assertIn(
            "!website/static/model-catalog/catalog.json",
            dockerignore,
        )

    def test_repository_catalog_validates_and_renders_every_projection(self) -> None:
        outputs = catalog.render_outputs()
        self.assertEqual(
            set(outputs),
            {
                catalog.RECIPE_MANIFEST,
                catalog.GO_OUTPUT,
                catalog.WEBSITE_OUTPUT,
            },
        )
        self.assertEqual(catalog.check(outputs), 0)
        public_snapshot = json.loads(outputs[catalog.WEBSITE_OUTPUT])
        self.assertNotIn("inventory", public_snapshot)
        self.assertNotIn("evaluation_coverage", public_snapshot)
        statuses = {result["status"] for result in public_snapshot["index_results"]}
        self.assertEqual(statuses, {"available", "partial", "missing"})
        self.assertTrue(
            all(
                (result["score"] is not None) == (result["status"] == "available")
                for result in public_snapshot["index_results"]
            )
        )
        runtime_manifest = yaml.safe_load(outputs[catalog.RECIPE_MANIFEST])
        self.assertNotIn("inventory", runtime_manifest)
        self.assertNotIn("evaluation_coverage", runtime_manifest)
        self.assertTrue(
            all(
                result["status"] == "available"
                for result in runtime_manifest["index_results"]
            )
        )
        self.assertLess(
            len(runtime_manifest["index_results"]),
            len(public_snapshot["index_results"]),
        )

    def test_default_intelligence_index_is_public_and_complete_case(self) -> None:
        _, resources, _ = catalog.load_and_validate()
        indices = {item["id"]: item for item in resources["indices"]}
        index = indices[DEFAULT_INDEX_ID]
        self.assertEqual(index["missing"], {"policy": "require_all"})
        self.assertEqual(
            index["domains"],
            {
                "general": 0.20,
                "reasoning": 0.40,
                "coding": 0.20,
                "agentic": 0.20,
            },
        )
        self.assertEqual(
            {
                component["index"]: component["weight"]
                for component in index["components"]
            },
            {
                "vllm-sr/general@1.0.0": 0.20,
                "vllm-sr/reasoning@1.0.0": 0.40,
                "vllm-sr/coding@1.0.0": 0.20,
                "vllm-sr/agentic@1.0.0": 0.20,
            },
        )
        expected_capabilities = {
            "vllm-sr/general@1.0.0": {
                "tiger-ai-lab/mmlu-pro@1.0.0#accuracy": 1.0,
            },
            "vllm-sr/reasoning@1.0.0": {
                "idavidrein/gpqa-diamond@1.0.0#accuracy": 0.5,
                "cais/humanitys-last-exam@1.0.0#accuracy": 0.5,
            },
            "vllm-sr/coding@1.0.0": {
                "livecodebench/livecodebench@6.0.0#pass_at_1": 0.5,
                "scicode-bench/scicode@1.0.0#score": 0.5,
            },
            "vllm-sr/agentic@1.0.0": {
                "harbor/terminal-bench@2.1.0#resolved": 1.0,
            },
        }
        for index_id, expected_components in expected_capabilities.items():
            capability = indices[index_id]
            self.assertEqual(capability["missing"], {"policy": "require_all"})
            self.assertEqual(
                {
                    f"{component['benchmark']}#{component['metric']}": component[
                        "weight"
                    ]
                    for component in capability["components"]
                },
                expected_components,
            )
        self.assertTrue(
            all(
                component["normalization"] == {"type": "identity"}
                for definition in [index]
                + [indices[index_id] for index_id in expected_capabilities]
                for component in definition["components"]
            )
        )

    def test_default_intelligence_index_has_a_broad_unique_model_cohort(self) -> None:
        outputs = catalog.render_outputs()
        snapshot = json.loads(outputs[catalog.WEBSITE_OUTPUT])
        rankable_models = {
            result["model"]
            for result in snapshot["index_results"]
            if result["index"] == DEFAULT_INDEX_ID and result["status"] == "available"
        }

        self.assertGreaterEqual(
            len(rankable_models),
            MINIMUM_RANKABLE_MODEL_COUNT,
            "the core index must retain more than 20 uniquely rankable models",
        )
        self.assertTrue(
            {
                "deepseek/deepseek-v3.1-terminus",
                "google/gemma-3-27b-it",
                "meta/llama-4-maverick-17b-128e-instruct",
                "minimax/minimax-m2.7",
                "mistral/mistral-small-3.2-24b-instruct",
                "moonshot/kimi-k2.6",
                "nvidia/nemotron-3-nano-30b-a3b",
                "openai/gpt-oss-20b",
                "qwen/qwen3.5-397b-a17b",
                "zai/glm-5.1",
            }.issubset(rankable_models)
        )

    def test_benchmark_display_contract_is_percentage_ready_and_core_is_curated(
        self,
    ) -> None:
        _, resources, _ = catalog.load_and_validate()
        benchmarks = {item["id"]: item for item in resources["benchmarks"]}
        self.assertEqual(
            {
                benchmark_id
                for benchmark_id, benchmark in benchmarks.items()
                if "core" in benchmark.get("tags", [])
            },
            {
                "tiger-ai-lab/mmlu-pro@1.0.0",
                "idavidrein/gpqa-diamond@1.0.0",
                "cais/humanitys-last-exam@1.0.0",
                "harbor/terminal-bench@2.1.0",
                "livecodebench/livecodebench@6.0.0",
                "scicode-bench/scicode@1.0.0",
            },
        )
        for benchmark in benchmarks.values():
            for metric in benchmark["metrics"]:
                if metric["unit"] in {"proportion", "fraction"}:
                    continue
                self.assertIn(
                    "normalization",
                    metric,
                    f"{benchmark['id']}#{metric['id']} needs a percentage display scale",
                )

        for benchmark_id in (
            "artificial-analysis/gdpval-aa@2.0.0",
            "artificial-analysis/briefcase@1.0.0",
        ):
            elo = next(
                metric
                for metric in benchmarks[benchmark_id]["metrics"]
                if metric["id"] == "elo"
            )
            self.assertEqual(
                elo["normalization"],
                {"type": "linear_clamp", "min": 500, "max": 2500},
            )

    def test_physical_inventory_is_the_curated_mainstream_creator_set(self) -> None:
        manifest, resources, _ = catalog.load_and_validate()
        physical_models = {
            item["id"]: item
            for item in resources["models"]
            if item["kind"] == "physical"
        }
        physical_policy = manifest["inventory"]["physical"]
        default_minimum = physical_policy["default_min_representatives"]
        minimum_by_creator = {
            creator["publisher"]: creator.get("min_representatives", default_minimum)
            for creator in physical_policy["creators"]
        }
        representatives_by_creator = {
            creator["publisher"]: creator["representative_models"]
            for creator in physical_policy["creators"]
        }
        self.assertGreaterEqual(len(minimum_by_creator), 20)
        self.assertGreaterEqual(
            len(physical_models),
            sum(minimum_by_creator.values()),
            "the curated inventory may grow without changing a snapshot count",
        )
        self.assertEqual(
            {model["publisher"] for model in physical_models.values()},
            set(minimum_by_creator),
        )
        for creator, minimum in minimum_by_creator.items():
            self.assertGreaterEqual(
                len(representatives_by_creator[creator]),
                minimum,
                f"{creator} must retain its current representative model depth",
            )
            for model_id in representatives_by_creator[creator]:
                self.assertEqual(physical_models[model_id]["publisher"], creator)
                self.assertIn(
                    physical_models[model_id]["lifecycle"], {"active", "experimental"}
                )
        self.assertIn("openai/gpt-6-astra", physical_models)

    def test_gpt_6_astra_day_zero_contract_is_complete(self) -> None:
        manifest, resources, _ = catalog.load_and_validate()
        models = {model["id"]: model for model in resources["models"]}
        astra = models["openai/gpt-6-astra"]
        self.assertEqual(
            astra["limits"],
            {
                "context_window_size": 1_050_000,
                "max_output_tokens": 128_000,
            },
        )
        self.assertEqual(astra["knowledge_cutoff"], "2026-04-30")
        self.assertEqual(astra["reasoning_family"], "gpt-6-astra")
        self.assertTrue(
            {"chat", "reasoning", "tools", "structured_output", "vision"}.issubset(
                astra["capabilities"]
            )
        )

        families = {family["id"]: family for family in resources["reasoning_families"]}
        reasoning = families["gpt-6-astra"]
        self.assertEqual(reasoning["levels"], ["low", "medium", "high", "xhigh", "max"])
        self.assertEqual(reasoning["modes"], ["enabled"])
        self.assertNotIn("disabled", reasoning)
        self.assertNotIn(
            "default",
            reasoning,
            "the official Astra model page does not publish a default effort",
        )

        providers = {provider["id"]: provider for provider in resources["providers"]}
        binding = next(
            item
            for item in providers["openai"]["models"]
            if item["catalog"] == "openai/gpt-6-astra"
        )
        self.assertEqual(binding["id"], "gpt-6-astra")
        self.assertEqual(
            binding["protocols"],
            ["openai/chat-completions@1", "openai/responses@1"],
        )
        self.assertEqual(binding["reasoning_modes"], ["enabled"])
        self.assertEqual(binding["reasoning_efforts"], reasoning["levels"])
        self.assertEqual(
            binding["reasoning_efforts_by_protocol"],
            {
                "openai/chat-completions@1": [
                    "low",
                    "medium",
                    "high",
                    "xhigh",
                ]
            },
        )
        self.assertEqual(
            binding["pricing"],
            {
                "currency": "USD",
                "prompt_per_1m": 10.0,
                "cached_input_per_1m": 1.0,
                "cache_write_per_1m": 12.5,
                "completion_per_1m": 50.0,
            },
        )
        self.assertEqual(
            binding["restrictions"],
            {
                "tools_protocols": ["openai/responses@1"],
                "long_context_pricing": {
                    "input_threshold_tokens": 272000,
                    "prompt_multiplier": 2.0,
                    "cached_input_multiplier": 2.0,
                    "cache_write_multiplier": 2.0,
                    "completion_multiplier": 1.5,
                },
                "unsupported_request_fields": {
                    "openai/chat-completions@1": [
                        "temperature",
                        "top_p",
                        "top_logprobs",
                        "logprobs",
                    ],
                    "openai/responses@1": [
                        "temperature",
                        "top_p",
                        "top_logprobs",
                    ],
                },
                "unsupported_include_values": ["message.output_text.logprobs"],
            },
        )

        evaluations = [
            item
            for item in resources["evaluations"]
            if item["model"] == "openai/gpt-6-astra"
        ]
        launch_evaluations = [
            item
            for item in evaluations
            if item["id"].startswith("openai/gpt-6-astra-launch-")
        ]
        self.assertEqual(len(launch_evaluations), 5)
        self.assertEqual(
            {item["benchmark"] for item in launch_evaluations},
            {
                "idavidrein/gpqa-diamond@1.0.0",
                "cais/humanitys-last-exam@1.0.0",
                "datacurve/deep-swe@1.1.0",
                "arc-prize/arc-agi-1@1.0.0",
                "arc-prize/arc-agi-2@1.0.0",
            },
        )
        self.assertTrue(
            all(
                item["reasoning_effort"] == "unspecified" for item in launch_evaluations
            )
        )
        self.assertTrue(
            all(
                item["subject"]["result_selection"]
                == "maximum_across_supported_efforts"
                for item in launch_evaluations
            )
        )
        self.assertTrue(
            all(
                item["evidence"]["provenance"] == "vendor_claimed"
                for item in launch_evaluations
            )
        )

        independent_evaluations = [
            item
            for item in evaluations
            if item["id"].startswith("independent/gpt-6-astra-")
        ]
        independent_efforts = {"low", "medium", "high", "xhigh", "max"}
        independent_benchmarks = {
            "idavidrein/gpqa-diamond@1.0.0",
            "cais/humanitys-last-exam@1.0.0",
            "harbor/terminal-bench@2.1.0",
            "scicode-bench/scicode@1.0.0",
        }
        self.assertEqual(len(independent_evaluations), 20)
        self.assertEqual(
            {
                (item["reasoning_effort"], item["benchmark"])
                for item in independent_evaluations
            },
            {
                (effort, benchmark)
                for effort in independent_efforts
                for benchmark in independent_benchmarks
            },
        )
        self.assertTrue(
            all(
                item["evidence"]["provenance"] == "third_party"
                and item["evidence"]["verification"] == "imported"
                and item["subject"]["run_kind"] == "independent"
                for item in independent_evaluations
            )
        )

        openai_inventory = next(
            creator
            for creator in manifest["inventory"]["physical"]["creators"]
            if creator["publisher"] == "OpenAI"
        )
        self.assertEqual(
            openai_inventory["representative_models"],
            ["openai/gpt-6-astra", "openai/gpt-5.6-sol", "openai/gpt-5.5"],
        )

    def test_baidu_creator_has_exact_evidence_buckets_and_real_bindings(self) -> None:
        manifest, resources, _ = catalog.load_and_validate()
        baidu_models = {
            "baidu/ernie-5.1",
            "baidu/ernie-5.0",
            "baidu/ernie-4.5-300b-a47b",
        }
        creators = {
            creator["publisher"]: creator["representative_models"]
            for creator in manifest["inventory"]["physical"]["creators"]
        }
        self.assertEqual(set(creators["Baidu"]), baidu_models)

        models = {model["id"]: model for model in resources["models"]}
        self.assertEqual(
            models["baidu/ernie-4.5-300b-a47b"]["released_at"],
            "2025-06-30",
        )

        buckets = _evaluation_benchmarks_by_bucket(resources, baidu_models)
        self.assertGreaterEqual(
            len(buckets[("baidu/ernie-5.1", "unspecified", "vendor_claimed")]),
            5,
        )
        self.assertGreaterEqual(
            len(buckets[("baidu/ernie-5.0", "unspecified", "vendor_claimed")]),
            5,
        )
        self.assertGreaterEqual(
            len(buckets[("baidu/ernie-4.5-300b-a47b", "disabled", "third_party")]),
            5,
        )

        providers = {provider["id"]: provider for provider in resources["providers"]}
        ai_studio = providers["baidu-ai-studio"]
        self.assertEqual(
            ai_studio["default_base_url"],
            "https://aistudio.baidu.com/llm/lmapi/v3",
        )
        self.assertEqual(ai_studio["auth"]["strategy"], "bearer")
        self.assertEqual(
            ai_studio["path_overrides"]["openai/chat-completions@1#create"],
            "/chat/completions",
        )
        self.assertNotIn("featured", ai_studio["presentation"])
        expected_bindings = {
            ("baidu-ai-studio", "baidu/ernie-5.1", "ernie-5.1", "first_party"),
            ("baidu-qianfan", "baidu/ernie-5.0", "ernie-5.0", "first_party"),
            (
                "vllm",
                "baidu/ernie-4.5-300b-a47b",
                "baidu/ERNIE-4.5-300B-A47B-PT",
                "self_hosted",
            ),
            (
                "sglang",
                "baidu/ernie-4.5-300b-a47b",
                "baidu/ERNIE-4.5-300B-A47B-PT",
                "self_hosted",
            ),
        }
        actual_bindings = {
            (
                provider["id"],
                binding["catalog"],
                binding["id"],
                binding["relationship"],
            )
            for provider in resources["providers"]
            for binding in provider.get("models", [])
            if binding["catalog"] in baidu_models
        }
        self.assertTrue(expected_bindings.issubset(actual_bindings))

        evaluations = {
            evaluation["id"]: evaluation for evaluation in resources["evaluations"]
        }
        self.assertEqual(
            evaluations["baidu/ernie-5.1-release-deepsearchqa@1.0.0"]["metrics"],
            {"f1": 0.773},
        )
        self.assertEqual(
            evaluations["baidu/ernie-5.0-paper-simpleqa@1.0.0"]["metrics"],
            {"accuracy": 0.7401},
        )
        self.assertEqual(
            evaluations["independent/ernie-4.5-300b-a47b-critpt@1.0.0"]["metrics"],
            {"score": 0},
        )

    def test_stepfun_creator_has_exact_efforts_and_real_bindings(self) -> None:
        manifest, resources, _ = catalog.load_and_validate()
        stepfun_models = {
            "stepfun/step-3.7-flash",
            "stepfun/step-3.5-flash",
            "stepfun/step3-vl-10b",
        }
        creators = {
            creator["publisher"]: creator["representative_models"]
            for creator in manifest["inventory"]["physical"]["creators"]
        }
        self.assertEqual(set(creators["StepFun"]), stepfun_models)

        buckets = _evaluation_benchmarks_by_bucket(resources, stepfun_models)
        self.assertGreaterEqual(
            len(buckets[("stepfun/step-3.7-flash", "high", "third_party")]),
            5,
        )
        self.assertGreaterEqual(
            len(buckets[("stepfun/step-3.5-flash", "enabled", "third_party")]),
            5,
        )
        self.assertGreaterEqual(
            len(buckets[("stepfun/step3-vl-10b", "enabled", "third_party")]),
            5,
        )

        reasoning_families = {
            family["id"]: family for family in resources["reasoning_families"]
        }
        self.assertEqual(
            reasoning_families["step-3.7"]["levels"],
            ["low", "medium", "high"],
        )
        providers = {provider["id"]: provider for provider in resources["providers"]}
        self.assertEqual(
            providers["stepfun"]["default_base_url"],
            "https://api.stepfun.ai/v1",
        )
        self.assertNotIn("featured", providers["stepfun"]["presentation"])
        vllm_bindings = {
            binding["catalog"]: binding
            for binding in providers["vllm"]["models"]
            if "catalog" in binding
        }
        self.assertEqual(
            vllm_bindings["stepfun/step3-vl-10b"]["restrictions"],
            {
                "minimum_vllm_version": "0.14.0rc2.dev143+gc0a350ca7",
                "trust_remote_code": True,
            },
        )
        expected_bindings = {
            (
                "stepfun",
                "stepfun/step-3.7-flash",
                "step-3.7-flash",
                "first_party",
            ),
            (
                "stepfun",
                "stepfun/step-3.5-flash",
                "step-3.5-flash",
                "first_party",
            ),
            (
                "vllm",
                "stepfun/step3-vl-10b",
                "stepfun-ai/Step3-VL-10B",
                "self_hosted",
            ),
            (
                "sglang",
                "stepfun/step3-vl-10b",
                "stepfun-ai/Step3-VL-10B",
                "self_hosted",
            ),
        }
        actual_bindings = {
            (
                provider["id"],
                binding["catalog"],
                binding["id"],
                binding["relationship"],
            )
            for provider in resources["providers"]
            for binding in provider.get("models", [])
            if binding["catalog"] in stepfun_models
        }
        self.assertTrue(expected_bindings.issubset(actual_bindings))

        evaluations = {
            evaluation["id"]: evaluation for evaluation in resources["evaluations"]
        }
        self.assertEqual(
            evaluations["independent/step3-vl-10b-enabled-lcr@1.1.0"]["metrics"],
            {"score": 0},
        )

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
        self.assertGreaterEqual(len(provider_bindings), len(physical_models))

        bound_models = {item["catalog"] for item in provider_bindings}
        self.assertTrue(set(physical_models).issubset(bound_models))

        provider_ids = {provider["id"] for provider in resources["providers"]}
        self.assertTrue(
            {
                "agnes",
                "apodex",
                "baidu-ai-studio",
                "baidu-qianfan",
                "compactifai",
                "perplexity",
                "sarvam",
                "stepfun",
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
        physical_model_ids = {
            model["id"] for model in resources["models"] if model["kind"] == "physical"
        }
        self.assertTrue(
            all(
                {"catalog", "relationship", "id", "protocols"}.issubset(binding)
                for provider in resources["providers"]
                for binding in provider.get("models", [])
            )
        )
        relationships = {
            provider["id"]: {
                binding["relationship"] for binding in provider.get("models", [])
            }
            for provider in resources["providers"]
            if provider.get("models")
        }
        self.assertEqual(relationships["bedrock"], {"first_party"})
        self.assertEqual(
            relationships["baidu-qianfan"], {"first_party", "managed_cloud"}
        )
        self.assertEqual(relationships["baidu-ai-studio"], {"first_party"})
        for provider_id in ("openrouter", "compactifai", "deepinfra", "novita"):
            self.assertEqual(relationships[provider_id], {"gateway"})
        for provider_id in ("vllm", "sglang"):
            self.assertEqual(relationships[provider_id], {"self_hosted"})

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
                if binding["catalog"] == "qwen/qwen3.7-max"
            )["reasoning_transport"],
            "top_level_boolean",
        )
        for runtime in ("vllm", "sglang"):
            self.assertTrue(
                all(
                    binding["catalog"] in physical_model_ids
                    for binding in providers[runtime]["models"]
                )
            )

    def test_cloudflare_workers_ai_provider_contract_is_complete(self) -> None:
        _, resources, _ = catalog.load_and_validate()
        providers = {provider["id"]: provider for provider in resources["providers"]}
        provider = providers["cloudflare-workers-ai"]

        self.assertEqual(provider["display_name"], "Cloudflare Workers AI")
        self.assertEqual(provider["category"], "model_api")
        self.assertEqual(provider["support_tier"], "compatible")
        self.assertNotIn("default_base_url", provider)
        self.assertEqual(provider["protocols"], ["openai/chat-completions@1"])
        self.assertEqual(provider["default_protocol"], "openai/chat-completions@1")
        self.assertEqual(
            provider["supported_operations"],
            ["openai/chat-completions@1#create"],
        )
        self.assertEqual(
            provider["auth"],
            {
                "strategy": "bearer",
                "header": "Authorization",
                "prefix": "Bearer",
            },
        )
        self.assertEqual(
            provider["presentation"],
            {"logo": "monogram", "monogram": "Cf", "monochrome": False},
        )
        self.assertEqual(provider["conformance"], {"status": "unverified"})
        self.assertNotIn("models", provider)

    def test_core_reasoning_families_match_native_control_surfaces(self) -> None:
        _, resources, _ = catalog.load_and_validate()
        families = {item["id"]: item for item in resources["reasoning_families"]}
        self.assertEqual(
            families["grok-4.6"]["levels"], ["low", "medium", "high", "xhigh"]
        )
        self.assertEqual(families["grok-4.6"]["default"], "high")
        self.assertEqual(families["grok-4.6"]["modes"], ["enabled"])
        self.assertEqual(families["claude-effort-always-on"]["modes"], ["adaptive"])
        self.assertEqual(
            families["claude-effort-adaptive"]["modes"],
            ["adaptive", "disabled"],
        )
        self.assertEqual(families["claude-effort-opt-in"]["default_mode"], "disabled")
        self.assertEqual(families["kimi-k3"]["levels"], ["low", "high", "max"])
        self.assertEqual(families["kimi-k3"]["default"], "max")
        self.assertEqual(families["hunyuan-hy4"]["levels"], ["high"])
        self.assertEqual(families["hunyuan-hy4"]["disabled"], "no_think")
        self.assertEqual(families["qwen3.8"]["activation_parameter"], "enable_thinking")
        self.assertEqual(families["qwen3.8"]["levels"], ["low", "medium", "xhigh"])
        self.assertEqual(families["qwen3.8"]["modes"], ["enabled", "disabled"])
        self.assertEqual(families["qwen3.8-always-on"]["modes"], ["enabled"])
        self.assertEqual(
            families["inkling"]["levels"],
            ["minimal", "low", "medium", "high", "xhigh", "max"],
        )
        self.assertEqual(families["glm"]["modes"], ["enabled", "disabled"])
        self.assertEqual(families["glm"]["default_mode"], "enabled")
        self.assertEqual(
            families["glm-5.2"],
            {
                "id": "glm-5.2",
                "type": "reasoning_effort",
                "parameter": "reasoning_effort",
                "activation_parameter": "enable_thinking",
                "levels": ["high", "max"],
                "default": "max",
                "modes": ["enabled", "disabled"],
                "default_mode": "enabled",
            },
        )
        self.assertEqual(
            families["minimax-m3"]["modes"],
            ["disabled", "adaptive", "enabled"],
        )
        self.assertEqual(families["minimax-m3"]["default_mode"], "adaptive")
        self.assertEqual(
            families["nemotron-super"]["effort_flags"], {"low": "low_effort"}
        )
        self.assertEqual(
            families["nemotron-ultra"]["effort_flags"],
            {"medium": "medium_effort"},
        )
        self.assertEqual(families["kimi-k2-always-on"]["modes"], ["enabled"])
        self.assertEqual(
            families["muse-glimmer"]["levels"],
            ["low", "medium", "high", "xhigh"],
        )
        self.assertEqual(families["mistral-none-high"]["levels"], ["high"])
        self.assertEqual(families["mistral-none-high"]["disabled"], "none")

    def test_core_reasoning_model_and_provider_bindings(self) -> None:
        _, resources, _ = catalog.load_and_validate()
        models = {item["id"]: item for item in resources["models"]}
        providers = {item["id"]: item for item in resources["providers"]}

        expected_models = {
            "xai/grok-4.6": "grok-4.6",
            "tencent/hy4-preview": "hunyuan-hy4",
            "moonshot/kimi-k3": "kimi-k3",
            "moonshot/kimi-k2.7-code": "kimi-k2-always-on",
            "minimax/minimax-m3": "minimax-m3",
            "meta/muse-glimmer-30b": "muse-glimmer",
            "mistral/mistral-medium-3.5": "mistral-none-high",
            "mistral/mistral-small-4": "mistral-none-high",
            "qwen/qwen3.8-27b": "qwen3.8",
            "qwen/qwen3.8-2.4t-a95b": "qwen3.8-always-on",
            "zai/glm-5.2": "glm-5.2",
            "anthropic/claude-fable-5": "claude-effort-always-on",
            "anthropic/claude-fable-5.1": "claude-effort-always-on",
            "anthropic/claude-sonnet-5": "claude-effort-adaptive",
            "anthropic/claude-opus-4.8": "claude-effort-opt-in",
            "anthropic/claude-opus-5": "claude-effort-adaptive",
            "nvidia/nemotron-3.5-lightning": "nemotron-thinking-toggle",
            "nvidia/nemotron-3-super": "nemotron-super",
            "nvidia/nemotron-3-ultra": "nemotron-ultra",
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning": "nemotron-thinking-toggle",
            "nvidia/nemotron-cascade-2-30b-a3b": "nemotron-thinking-toggle",
        }
        self.assertEqual(
            {
                model_id: models[model_id].get("reasoning_family")
                for model_id in expected_models
            },
            expected_models,
        )
        self.assertEqual(
            providers["anthropic"]["reasoning_transport"], "output_config_effort"
        )
        self.assertEqual(providers["xai"]["reasoning_transport"], "top_level_effort")
        zai_bindings = {
            binding["catalog"]: binding for binding in providers["zai"]["models"]
        }
        for model_id in ("zai/glm-5.2", "zai/glm-5.3", "zai/glm-5.3-flash"):
            self.assertEqual(
                zai_bindings[model_id]["reasoning_transport"],
                "thinking_object_effort",
            )
        moonshot_bindings = {
            binding["catalog"]: binding for binding in providers["moonshot"]["models"]
        }
        self.assertEqual(
            moonshot_bindings["moonshot/kimi-k2.7-code"]["reasoning_transport"],
            "thinking_object",
        )
        minimax_binding = next(
            binding
            for binding in providers["minimax"]["models"]
            if binding["catalog"] == "minimax/minimax-m3"
        )
        self.assertEqual(
            minimax_binding["reasoning_modes"], ["disabled", "adaptive", "enabled"]
        )
        for runtime in ("vllm", "sglang"):
            qwen_binding = next(
                binding
                for binding in providers[runtime]["models"]
                if binding["catalog"] == "qwen/qwen3.8-27b"
            )
            self.assertEqual(
                qwen_binding["reasoning_transport"],
                "top_level_effort_template_switch",
            )
        dashscope_qwen = next(
            binding
            for binding in providers["dashscope"]["models"]
            if binding["catalog"] == "qwen/qwen3.8-max"
        )
        self.assertEqual(
            dashscope_qwen["reasoning_transport"],
            "top_level_effort_boolean_switch",
        )
        self.assertIn("openai/responses@1", dashscope_qwen["protocols"])

    def test_frontier_context_limits_keep_exact_published_units(self) -> None:
        _, resources, _ = catalog.load_and_validate()
        models = {item["id"]: item for item in resources["models"]}

        openai_1050k = {
            model_id
            for model_id, model in models.items()
            if model.get("limits", {}).get("context_window_size")
            == OPENAI_LONG_CONTEXT_TOKENS
        }
        self.assertEqual(
            openai_1050k,
            {
                "openai/gpt-5.4",
                "openai/gpt-5.5",
                "openai/gpt-6-astra",
                "openai/gpt-5.6-luna",
                "openai/gpt-5.6-sol",
                "openai/gpt-5.6-terra",
            },
            "1.05M is an exact published limit, not a rounded 1M display value",
        )
        self.assertEqual(
            models["zai/glm-5.1"]["limits"]["context_window_size"], 200_000
        )
        self.assertEqual(
            models["nvidia/nemotron-3-super"]["limits"]["context_window_size"],
            1_048_576,
        )
        self.assertEqual(
            models["nvidia/nemotron-3-ultra"]["limits"]["context_window_size"],
            1_048_576,
        )
        for model_id in ("zai/glm-5.2", "zai/glm-5.3", "zai/glm-5.3-flash"):
            self.assertEqual(
                models[model_id]["limits"]["context_window_size"],
                1_048_576,
            )
        for model_id in ("qwen/qwen3.8-max", "qwen/qwen3.7-max"):
            self.assertEqual(
                models[model_id]["limits"]["context_window_size"],
                1_000_000,
            )
        self.assertEqual(
            models["amazon/nova-premier-v1"]["limits"],
            {"context_window_size": 1_000_000, "max_output_tokens": 10_000},
        )
        self.assertEqual(
            models["amazon/nova-pro-v1"]["limits"],
            {"context_window_size": 300_000, "max_output_tokens": 10_000},
        )

    def test_every_model_reasoning_mode_materializes_default_slots_including_missing(
        self,
    ) -> None:
        manifest, resources, _ = catalog.load_and_validate()
        coverage = evaluation_coverage(
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
                        (
                            row["benchmark"],
                            tuple(row["benchmark_profiles"]),
                            row["metric"],
                        )
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
        qwen_low = slots[("qwen/qwen3.8-27b", "low")]
        self.assertEqual(len(qwen_low), DEFAULT_INDEX_COMPONENT_COUNT)
        self.assertEqual(
            {row["status"] for row in qwen_low},
            {"available", "missing"},
        )
        self.assertTrue(
            any(row["status"] == "available" for rows in slots.values() for row in rows)
        )

    def test_frontier_evaluation_coverage_keeps_unspecified_values_and_gaps(
        self,
    ) -> None:
        manifest, resources, _ = catalog.load_and_validate()
        coverage = evaluation_coverage(
            resources, manifest["defaults"]["intelligence_index"]
        )
        available: dict[tuple[str, str], set[str]] = {}
        for row in coverage:
            if row["status"] == "available":
                available.setdefault(
                    (row["model"], row["reasoning_effort"]), set()
                ).add(row["benchmark"])

        expected_counts = {
            ("deepseek/deepseek-v4-flash", "max"): 6,
            ("deepseek/deepseek-v4-pro", "max"): 6,
            ("zai/glm-5.3-flash", "max"): 4,
            ("qwen/qwen3.8-27b", "xhigh"): 5,
            ("tencent/hy3", "high"): 3,
            ("thinking-machines/inkling", "max"): 4,
            ("microsoft/mai-thinking-1", "unspecified"): 3,
            ("cohere/tiny-aya-global", "unspecified"): 1,
        }
        self.assertEqual(
            {key: len(available.get(key, set())) for key in expected_counts},
            expected_counts,
        )
        self.assertNotIn(
            "tiger-ai-lab/mmlu-pro@1.0.0",
            available[("zai/glm-5.3-flash", "max")],
        )
        self.assertNotIn(("qwen/qwen3.8-27b", "none"), available)
        for effort in ("low", "medium", "xhigh"):
            self.assertNotIn(
                "tiger-ai-lab/mmlu-pro@1.0.0",
                available[("qwen/qwen3.8-27b", effort)],
            )

    def test_glm_5_2_independent_max_runs_keep_their_exact_effort(self) -> None:
        _, resources, _ = catalog.load_and_validate()
        max_runs = [
            record
            for record in resources["evaluations"]
            if record["model"] == "zai/glm-5.2"
            and record.get("subject", {}).get("source_model") == "GLM-5.2 (max)"
        ]
        disabled_runs = [
            record
            for record in resources["evaluations"]
            if record["model"] == "zai/glm-5.2"
            and record.get("subject", {}).get("source_model")
            == "GLM-5.2 (Non-reasoning)"
        ]

        self.assertEqual(len(max_runs), 9)
        self.assertEqual(
            {record["reasoning_effort"] for record in max_runs},
            {"max"},
        )
        self.assertEqual(len(disabled_runs), 8)
        self.assertEqual(
            {record["reasoning_effort"] for record in disabled_runs},
            {"disabled"},
        )

    def test_third_party_measurements_have_an_auditable_run_identity(
        self,
    ) -> None:
        _, resources, _ = catalog.load_and_validate()
        third_party = [
            record
            for record in resources["evaluations"]
            if record["evidence"]["provenance"] == "third_party"
        ]
        self.assertTrue(third_party)
        official_source_kinds = {
            "official_vendor_republication",
            "official_cross-vendor_comparison",
            "official_model_card",
        }
        for record in third_party:
            self.assertTrue(record["evidence"]["source"].startswith("https://"))
            subject = record["subject"]
            if subject.get("source_kind") in official_source_kinds:
                continue
            self.assertEqual(subject.get("run_kind"), "independent")
            self.assertTrue(subject.get("source_model"))
            self.assertRegex(subject.get("source_model_slug", ""), catalog.SLUG)

    def test_fireworks_serverless_model_ids_are_not_stale(self) -> None:
        _, resources, _ = catalog.load_and_validate()
        providers = {provider["id"]: provider for provider in resources["providers"]}
        fireworks = providers["fireworks"]
        expected_mappings = {
            "meta/muse-glimmer-30b": "accounts/fireworks/models/muse-glimmer-30b",
            "moonshot/kimi-k3": "accounts/fireworks/models/kimi-k3",
            "thinking-machines/inkling": "accounts/fireworks/models/inkling",
        }
        actual_mappings = {
            binding["catalog"]: binding["id"] for binding in fireworks["models"]
        }
        self.assertEqual(actual_mappings, expected_mappings)

    def test_fireworks_serverless_mappings_have_no_duplicates(self) -> None:
        _, resources, _ = catalog.load_and_validate()
        providers = {provider["id"]: provider for provider in resources["providers"]}
        fireworks = providers["fireworks"]
        bindings = fireworks["models"]
        native_ids = [binding["id"] for binding in bindings]
        self.assertEqual(len(native_ids), len(set(native_ids)))
        model_protocol_pairs = [
            (binding["catalog"], protocol)
            for binding in bindings
            for protocol in binding["protocols"]
        ]
        self.assertEqual(len(model_protocol_pairs), len(set(model_protocol_pairs)))

    def test_fireworks_models_use_supported_protocols(self) -> None:
        _, resources, _ = catalog.load_and_validate()
        providers = {provider["id"]: provider for provider in resources["providers"]}
        fireworks = providers["fireworks"]
        expected_protocols = {"openai/chat-completions@1"}
        self.assertEqual(expected_protocols, set(fireworks["protocols"]))
        for binding in fireworks["models"]:
            self.assertEqual(set(binding["protocols"]), expected_protocols)


if __name__ == "__main__":
    unittest.main()
