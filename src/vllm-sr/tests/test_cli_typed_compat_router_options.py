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
from cli.parser import parse_user_config


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


class TestCLITypedRouterOptionsCompat(unittest.TestCase):
    def test_router_options_use_typed_compat_path(self):
        defaults = load_embedded_defaults()
        config_data = _base_user_config()
        config_data.update(
            {
                "auto_model_name": "MoMRouter",
                "clear_route_cache": defaults["clear_route_cache"],
                "include_config_models_in_list": True,
                "streamed_body_mode": True,
                "max_streamed_body_bytes": 131072,
                "streamed_body_timeout_sec": 15,
            }
        )

        user_config = _parse_config_dict(config_data)
        compat_blocks = get_typed_compat_blocks(user_config)

        self.assertIsNotNone(compat_blocks.router_options)
        expected_router_options = {
            "auto_model_name": "MoMRouter",
            "clear_route_cache": defaults["clear_route_cache"],
            "include_config_models_in_list": True,
            "streamed_body_mode": True,
            "max_streamed_body_bytes": 131072,
            "streamed_body_timeout_sec": 15,
        }
        self.assertEqual(
            expected_router_options,
            dump_typed_compat_block(compat_blocks.router_options),
        )

        extra = getattr(user_config, "model_extra", {}) or {}
        self.assertNotIn("auto_model_name", extra)
        self.assertNotIn("clear_route_cache", extra)
        self.assertNotIn("include_config_models_in_list", extra)
        self.assertNotIn("streamed_body_mode", extra)
        self.assertNotIn("max_streamed_body_bytes", extra)
        self.assertNotIn("streamed_body_timeout_sec", extra)

        merged = merge_configs(user_config, defaults)
        for key, value in expected_router_options.items():
            self.assertEqual(value, merged[key])


if __name__ == "__main__":
    unittest.main()
