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
from cli.compat_blocks_model_selection import ModelSelectionCompatConfig
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


class TestCLITopLevelCompatibility(unittest.TestCase):
    def test_parse_rejects_unknown_top_level_block(self):
        config_data = _base_user_config()
        config_data["totally_unknown_block"] = {"enabled": True}

        with self.assertRaises(ConfigParseError) as ctx:
            _parse_config_dict(config_data)

        self.assertIn("unsupported top-level config keys", str(ctx.exception))
        self.assertIn("totally_unknown_block", str(ctx.exception))

    def test_model_selection_uses_typed_compat_path(self):
        defaults = load_embedded_defaults()
        config_data = _base_user_config()
        config_data["model_selection"] = {
            "enabled": True,
            "method": "elo",
            "elo": {
                "k_factor": 24,
                "category_weighted": True,
                "storage_path": "/tmp/elo.json",
            },
            "ml": {
                "models_path": "models/model_selection",
                "embedding_dim": 1024,
                "knn": {
                    "k": 5,
                    "pretrained_path": "models/model_selection/knn_model.json",
                },
            },
        }

        user_config = _parse_config_dict(config_data)
        compat_blocks = get_typed_compat_blocks(user_config)
        merged = merge_configs(user_config, defaults)
        typed_model_selection = dump_typed_compat_block(compat_blocks.model_selection)

        self.assertIsNotNone(compat_blocks.model_selection)
        self.assertTrue(typed_model_selection["enabled"])
        self.assertEqual("elo", typed_model_selection["method"])
        self.assertEqual(24.0, typed_model_selection["elo"]["k_factor"])
        self.assertTrue(typed_model_selection["elo"]["category_weighted"])
        self.assertEqual("/tmp/elo.json", typed_model_selection["elo"]["storage_path"])
        self.assertEqual(
            config_data["model_selection"]["ml"], typed_model_selection["ml"]
        )
        self.assertNotIn(
            "model_selection", getattr(user_config, "model_extra", {}) or {}
        )
        self.assertEqual(typed_model_selection, merged["model_selection"])

    def test_embedded_defaults_use_runtime_model_selection_shape(self):
        defaults = load_embedded_defaults()
        model_selection = defaults["model_selection"]

        self.assertNotIn("default_algorithm", model_selection)
        self.assertNotIn("llm_candidates_path", model_selection)
        self.assertNotIn("training_data_path", model_selection)
        self.assertNotIn("custom_training", model_selection)
        self.assertEqual("knn", model_selection["method"])
        self.assertEqual(
            "models/model_selection/knn_model.json",
            model_selection["ml"]["knn"]["pretrained_path"],
        )

        typed_model_selection = ModelSelectionCompatConfig(**model_selection)
        self.assertEqual(
            model_selection,
            dump_typed_compat_block(typed_model_selection),
        )

    def test_parse_rejects_invalid_legacy_provider_runtime_key(self):
        config_data = _base_user_config()
        config_data["model_config"] = "not-a-dict"

        with self.assertRaises(ConfigParseError) as ctx:
            _parse_config_dict(config_data)

        self.assertIn("model_config", str(ctx.exception))

    def test_parse_normalizes_legacy_signal_runtime_blocks(self):
        defaults = load_embedded_defaults()
        config_data = _base_user_config()
        config_data.update(
            {
                "categories": [
                    {
                        "name": "billing",
                        "description": "Billing domain",
                        "mmlu_categories": ["business"],
                    }
                ],
                "language_rules": [
                    {
                        "name": "english",
                        "description": "Match English prompts",
                    }
                ],
                "context_rules": [
                    {
                        "name": "long_context",
                        "min_tokens": "8K",
                        "max_tokens": "128K",
                        "description": "Long context requests",
                    }
                ],
                "role_bindings": [
                    {
                        "name": "admin_binding",
                        "role": "admin",
                        "subjects": [{"kind": "User", "name": "alice"}],
                        "description": "Bind alice to admin",
                    }
                ],
            }
        )

        user_config = _parse_config_dict(config_data)
        extra = getattr(user_config, "model_extra", {}) or {}

        for key in ("categories", "language_rules", "context_rules", "role_bindings"):
            self.assertNotIn(key, extra)

        self.assertEqual("billing", user_config.signals.domains[0].name)
        self.assertEqual("english", user_config.signals.language[0].name)
        self.assertEqual("long_context", user_config.signals.context[0].name)
        self.assertEqual("admin", user_config.signals.role_bindings[0].role)

        merged = merge_configs(user_config, defaults)
        self.assertEqual(config_data["categories"], merged["categories"])
        self.assertEqual(config_data["language_rules"], merged["language_rules"])
        self.assertEqual(config_data["context_rules"], merged["context_rules"])
        self.assertEqual(config_data["role_bindings"], merged["role_bindings"])


if __name__ == "__main__":
    unittest.main()
