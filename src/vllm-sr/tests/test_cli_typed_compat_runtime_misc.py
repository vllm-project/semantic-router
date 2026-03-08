# ruff: noqa: E402

import sys
import tempfile
import unittest
from pathlib import Path

import yaml

CLI_ROOT = Path(__file__).resolve().parents[1]
if str(CLI_ROOT) not in sys.path:
    sys.path.insert(0, str(CLI_ROOT))

from cli.compat_blocks import dump_typed_compat_block, get_typed_compat_blocks
from cli.defaults import load_embedded_defaults
from cli.merger import merge_configs
from cli.parser import ConfigParseError, parse_user_config


def _base_user_config() -> dict:
    return {
        "version": "v0.1",
        "listeners": [{"name": "http", "address": "0.0.0.0", "port": 8899}],
        "signals": {
            "keywords": [
                {
                    "name": "billing_keywords",
                    "operator": "contains",
                    "keywords": ["invoice"],
                }
            ]
        },
        "providers": {
            "models": [
                {
                    "name": "qwen3-4b",
                    "endpoints": [
                        {"name": "primary", "weight": 100, "endpoint": "localhost:8000"}
                    ],
                }
            ],
            "default_model": "qwen3-4b",
        },
        "decisions": [
            {
                "name": "billing-route",
                "description": "Route billing traffic",
                "priority": 100,
                "rules": {
                    "operator": "AND",
                    "conditions": [{"type": "keyword", "name": "billing_keywords"}],
                },
                "modelRefs": [{"model": "qwen3-4b"}],
            }
        ],
    }


def _parse_config_dict(config_data: dict):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(config_data, f, sort_keys=False)
        config_path = f.name

    try:
        return parse_user_config(config_path)
    finally:
        Path(config_path).unlink(missing_ok=True)


class TestCLITypedRuntimeMiscCompat(unittest.TestCase):
    def test_root_provider_defaults_normalize_into_nested_providers(self):
        defaults = load_embedded_defaults()
        config_data = _base_user_config()
        config_data["providers"].pop("default_model", None)
        config_data.update(
            {
                "default_model": "qwen3-4b",
                "default_reasoning_effort": "medium",
                "reasoning_families": {
                    "qwen3": {"type": "effort", "parameter": "reasoning_effort"}
                },
            }
        )

        user_config = _parse_config_dict(config_data)

        self.assertEqual(
            "qwen3-4b",
            user_config.providers.default_model,
        )
        self.assertEqual(
            "medium",
            user_config.providers.default_reasoning_effort,
        )
        self.assertEqual(
            {"qwen3": {"type": "effort", "parameter": "reasoning_effort"}},
            {
                name: family.model_dump()
                for name, family in user_config.providers.reasoning_families.items()
            },
        )
        extra = getattr(user_config, "model_extra", {}) or {}
        self.assertNotIn("default_model", extra)
        self.assertNotIn("default_reasoning_effort", extra)
        self.assertNotIn("reasoning_families", extra)

        merged = merge_configs(user_config, defaults)
        self.assertEqual("qwen3-4b", merged["default_model"])
        self.assertEqual("medium", merged["default_reasoning_effort"])
        self.assertEqual(
            {"qwen3": {"type": "effort", "parameter": "reasoning_effort"}},
            merged["reasoning_families"],
        )

    def test_runtime_top_level_keys_use_typed_compat_path(self):
        defaults = load_embedded_defaults()
        config_data = _base_user_config()
        config_data.update(
            {
                "config_source": "file",
                "mom_registry": {
                    "models/mom-domain-classifier": "LLM-Semantic-Router/lora_intent_classifier"
                },
                "strategy": "priority",
            }
        )

        user_config = _parse_config_dict(config_data)
        compat_blocks = get_typed_compat_blocks(user_config)

        self.assertIsNotNone(compat_blocks.runtime_top_level)
        self.assertEqual(
            {
                "config_source": "file",
                "mom_registry": {
                    "models/mom-domain-classifier": "LLM-Semantic-Router/lora_intent_classifier"
                },
                "strategy": "priority",
            },
            dump_typed_compat_block(compat_blocks.runtime_top_level),
        )
        extra = getattr(user_config, "model_extra", {}) or {}
        self.assertNotIn("config_source", extra)
        self.assertNotIn("mom_registry", extra)
        self.assertNotIn("strategy", extra)

        merged = merge_configs(user_config, defaults)
        self.assertEqual("file", merged["config_source"])
        self.assertEqual("priority", merged["strategy"])
        self.assertEqual(
            {
                "models/mom-domain-classifier": "LLM-Semantic-Router/lora_intent_classifier"
            },
            merged["mom_registry"],
        )

    def test_vector_store_uses_typed_compat_path(self):
        defaults = load_embedded_defaults()
        config_data = _base_user_config()
        config_data["vector_store"] = {
            "enabled": True,
            "backend_type": "llama_stack",
            "file_storage_dir": "/tmp/vsr-data",
            "embedding_model": "mmbert",
            "embedding_dimension": 768,
            "ingestion_workers": 2,
            "supported_formats": [".txt", ".md"],
            "llama_stack": {
                "endpoint": "http://localhost:8321",
                "embedding_model": "all-MiniLM-L6-v2",
                "request_timeout_seconds": 30,
                "search_type": "hybrid",
            },
        }

        user_config = _parse_config_dict(config_data)
        compat_blocks = get_typed_compat_blocks(user_config)

        self.assertIsNotNone(compat_blocks.vector_store)
        self.assertEqual(
            config_data["vector_store"],
            dump_typed_compat_block(compat_blocks.vector_store),
        )
        self.assertNotIn("vector_store", getattr(user_config, "model_extra", {}) or {})

        merged = merge_configs(user_config, defaults)
        self.assertEqual(config_data["vector_store"], merged["vector_store"])

    def test_parse_rejects_unknown_vector_store_field(self):
        config_data = _base_user_config()
        config_data["vector_store"] = {
            "enabled": True,
            "backend_type": "memory",
            "totally_unknown_field": True,
        }

        with self.assertRaises(ConfigParseError) as ctx:
            _parse_config_dict(config_data)

        self.assertIn("vector_store", str(ctx.exception))
        self.assertIn("totally_unknown_field", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
