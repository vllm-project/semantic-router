# ruff: noqa: E402

import sys
import tempfile
import unittest
from pathlib import Path

import yaml

CLI_ROOT = Path(__file__).resolve().parents[1]
if str(CLI_ROOT) not in sys.path:
    sys.path.insert(0, str(CLI_ROOT))

from cli.compat_blocks import get_typed_compat_blocks
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

    def test_merge_preserves_named_legacy_top_level_block(self):
        defaults = load_embedded_defaults()
        config_data = _base_user_config()
        config_data["classifier"] = defaults["classifier"]

        user_config = _parse_config_dict(config_data)
        merged = merge_configs(user_config, defaults)

        self.assertEqual(defaults["classifier"], merged["classifier"])

    def test_prompt_guard_uses_typed_compat_path(self):
        defaults = load_embedded_defaults()
        config_data = _base_user_config()
        config_data["prompt_guard"] = defaults["prompt_guard"]

        user_config = _parse_config_dict(config_data)
        compat_blocks = get_typed_compat_blocks(user_config)

        self.assertIsNotNone(compat_blocks.prompt_guard)
        self.assertEqual(
            defaults["prompt_guard"],
            compat_blocks.prompt_guard.model_dump(exclude_none=True),
        )
        self.assertNotIn("prompt_guard", getattr(user_config, "model_extra", {}) or {})

        merged = merge_configs(user_config, defaults)
        self.assertEqual(defaults["prompt_guard"], merged["prompt_guard"])

    def test_parse_rejects_unknown_prompt_guard_field(self):
        config_data = _base_user_config()
        config_data["prompt_guard"] = {"enabled": True, "totally_unknown_field": True}

        with self.assertRaises(ConfigParseError) as ctx:
            _parse_config_dict(config_data)

        self.assertIn("prompt_guard", str(ctx.exception))
        self.assertIn("totally_unknown_field", str(ctx.exception))

    def test_looper_uses_typed_compat_path(self):
        defaults = load_embedded_defaults()
        config_data = _base_user_config()
        config_data["looper"] = {
            **defaults["looper"],
            "retry_count": 2,
            "model_endpoints": {
                "qwen3-4b": "http://localhost:8899/v1/chat/completions",
            },
        }

        user_config = _parse_config_dict(config_data)
        compat_blocks = get_typed_compat_blocks(user_config)

        self.assertIsNotNone(compat_blocks.looper)
        self.assertEqual(
            config_data["looper"],
            compat_blocks.looper.model_dump(exclude_none=True),
        )
        self.assertNotIn("looper", getattr(user_config, "model_extra", {}) or {})

        merged = merge_configs(user_config, defaults)
        self.assertEqual(config_data["looper"], merged["looper"])

    def test_parse_rejects_unknown_looper_field(self):
        config_data = _base_user_config()
        config_data["looper"] = {
            "endpoint": "http://localhost:8899/v1/chat/completions",
            "totally_unknown_field": True,
        }

        with self.assertRaises(ConfigParseError) as ctx:
            _parse_config_dict(config_data)

        self.assertIn("looper", str(ctx.exception))
        self.assertIn("totally_unknown_field", str(ctx.exception))

    def test_observability_uses_typed_compat_path(self):
        defaults = load_embedded_defaults()
        config_data = _base_user_config()
        config_data["observability"] = defaults["observability"]

        user_config = _parse_config_dict(config_data)
        compat_blocks = get_typed_compat_blocks(user_config)

        self.assertIsNotNone(compat_blocks.observability)
        self.assertEqual(
            config_data["observability"],
            compat_blocks.observability.model_dump(exclude_none=True),
        )
        self.assertNotIn(
            "observability",
            getattr(user_config, "model_extra", {}) or {},
        )

        merged = merge_configs(user_config, defaults)
        self.assertEqual(config_data["observability"], merged["observability"])

    def test_parse_rejects_unknown_observability_field(self):
        config_data = _base_user_config()
        config_data["observability"] = {
            "metrics": {
                "enabled": True,
                "totally_unknown_field": True,
            }
        }

        with self.assertRaises(ConfigParseError) as ctx:
            _parse_config_dict(config_data)

        self.assertIn("observability", str(ctx.exception))
        self.assertIn("totally_unknown_field", str(ctx.exception))

    def test_router_replay_uses_typed_compat_path(self):
        defaults = load_embedded_defaults()
        config_data = _base_user_config()
        config_data["router_replay"] = {
            **defaults["router_replay"],
            "redis": {
                "address": "redis.internal:6379",
                "db": 2,
                "key_prefix": "router-replay:",
            },
        }

        user_config = _parse_config_dict(config_data)
        compat_blocks = get_typed_compat_blocks(user_config)

        self.assertIsNotNone(compat_blocks.router_replay)
        self.assertEqual(
            config_data["router_replay"],
            compat_blocks.router_replay.model_dump(exclude_none=True),
        )
        self.assertNotIn(
            "router_replay",
            getattr(user_config, "model_extra", {}) or {},
        )

        merged = merge_configs(user_config, defaults)
        self.assertEqual(config_data["router_replay"], merged["router_replay"])

    def test_parse_rejects_unknown_router_replay_field(self):
        config_data = _base_user_config()
        config_data["router_replay"] = {
            "store_backend": "memory",
            "totally_unknown_field": True,
        }

        with self.assertRaises(ConfigParseError) as ctx:
            _parse_config_dict(config_data)

        self.assertIn("router_replay", str(ctx.exception))
        self.assertIn("totally_unknown_field", str(ctx.exception))

    def test_tools_uses_typed_compat_path(self):
        defaults = load_embedded_defaults()
        config_data = _base_user_config()
        config_data["tools"] = {
            **defaults["tools"],
            "advanced_filtering": {
                "enabled": True,
                "candidate_pool_size": 12,
                "min_combined_score": 0.35,
                "weights": {
                    "embed": 0.6,
                    "lexical": 0.2,
                    "tag": 0.1,
                    "name": 0.05,
                    "category": 0.05,
                },
                "allow_tools": ["lookup_invoice"],
                "block_tools": ["send_email"],
            },
        }

        user_config = _parse_config_dict(config_data)
        compat_blocks = get_typed_compat_blocks(user_config)

        self.assertIsNotNone(compat_blocks.tools)
        self.assertEqual(
            config_data["tools"],
            compat_blocks.tools.model_dump(exclude_none=True),
        )
        self.assertNotIn("tools", getattr(user_config, "model_extra", {}) or {})

        merged = merge_configs(user_config, defaults)
        self.assertEqual(config_data["tools"], merged["tools"])

    def test_parse_rejects_unknown_tools_field(self):
        config_data = _base_user_config()
        config_data["tools"] = {"enabled": True, "totally_unknown_field": True}

        with self.assertRaises(ConfigParseError) as ctx:
            _parse_config_dict(config_data)

        self.assertIn("tools", str(ctx.exception))
        self.assertIn("totally_unknown_field", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
