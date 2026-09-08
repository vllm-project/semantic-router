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

DEFAULT_INDEX_COMPONENT_COUNT = 5
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
        self.assertTrue(
            all(
                result["status"] == "available"
                for result in public_snapshot["index_results"]
            )
        )
        runtime_manifest = yaml.safe_load(outputs[catalog.RECIPE_MANIFEST])
        self.assertNotIn("inventory", runtime_manifest)
        self.assertNotIn("evaluation_coverage", runtime_manifest)

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
                "swe-bench/verified@1.0.0",
                "harbor/terminal-bench@2.1.0",
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
        self.assertNotIn("openai/gpt-6-astra", physical_models)

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
        missing_qwen_low = slots[("qwen/qwen3.8-27b", "low")]
        self.assertEqual(len(missing_qwen_low), DEFAULT_INDEX_COMPONENT_COUNT)
        self.assertEqual(
            {row["status"] for row in missing_qwen_low},
            {"missing"},
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
            ("deepseek/deepseek-v4-flash", "max"): 5,
            ("deepseek/deepseek-v4-pro", "max"): 5,
            ("zai/glm-5.3-flash", "max"): 2,
            ("qwen/qwen3.8-27b", "xhigh"): 3,
            ("tencent/hy3", "high"): 4,
            ("thinking-machines/inkling", "max"): 4,
            ("microsoft/mai-thinking-1", "unspecified"): 3,
            ("cohere/tiny-aya-global", "unspecified"): 1,
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


if __name__ == "__main__":
    unittest.main()
