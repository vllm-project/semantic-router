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


class TestCLITypedAPICompat(unittest.TestCase):
    def test_api_uses_typed_compat_path(self):
        defaults = load_embedded_defaults()
        config_data = _base_user_config()
        config_data["api"] = {
            "batch_classification": {
                "max_batch_size": 32,
                "concurrency_threshold": 4,
                "max_concurrency": 8,
                "metrics": {
                    "enabled": True,
                    "detailed_goroutine_tracking": False,
                    "high_resolution_timing": True,
                    "sample_rate": 0.5,
                    "batch_size_ranges": [
                        {"min": 1, "max": 8, "label": "small"},
                        {"min": 9, "max": 32, "label": "medium"},
                    ],
                    "duration_buckets": [0.01, 0.1, 1.0],
                    "size_buckets": [8, 16, 32],
                },
            }
        }

        user_config = _parse_config_dict(config_data)
        compat_blocks = get_typed_compat_blocks(user_config)

        self.assertIsNotNone(compat_blocks.api)
        self.assertEqual(config_data["api"], dump_typed_compat_block(compat_blocks.api))
        self.assertNotIn("api", getattr(user_config, "model_extra", {}) or {})

        merged = merge_configs(user_config, defaults)
        self.assertEqual(config_data["api"], merged["api"])

    def test_parse_rejects_unknown_api_field(self):
        config_data = _base_user_config()
        config_data["api"] = {
            "batch_classification": {
                "max_batch_size": 32,
                "unknown_limit": 9,
            }
        }

        with self.assertRaises(ConfigParseError) as ctx:
            _parse_config_dict(config_data)

        self.assertIn("api", str(ctx.exception))
        self.assertIn("unknown_limit", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
