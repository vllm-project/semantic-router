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


class TestCLITypedSemanticCacheCompat(unittest.TestCase):
    def test_semantic_cache_uses_typed_compat_path(self):
        defaults = load_embedded_defaults()
        config_data = _base_user_config()
        config_data["semantic_cache"] = {
            **defaults["semantic_cache"],
            "backend_type": "redis",
            "embedding_model": "mmbert",
            "redis": {
                "connection": {
                    "host": "localhost",
                    "port": 6379,
                    "database": 0,
                    "timeout": 30,
                    "tls": {"enabled": False},
                },
                "index": {
                    "name": "semantic_cache_idx",
                    "prefix": "doc:",
                    "vector_field": {
                        "name": "embedding",
                        "dimension": 384,
                        "metric_type": "COSINE",
                    },
                    "index_type": "HNSW",
                    "params": {"M": 16, "efConstruction": 64},
                },
                "search": {"topk": 1},
                "logging": {"level": "info", "enable_metrics": True},
                "development": {"auto_create_index": True},
            },
        }

        user_config = _parse_config_dict(config_data)
        compat_blocks = get_typed_compat_blocks(user_config)

        self.assertIsNotNone(compat_blocks.semantic_cache)
        self.assertEqual(
            config_data["semantic_cache"],
            dump_typed_compat_block(compat_blocks.semantic_cache),
        )
        self.assertNotIn(
            "semantic_cache",
            getattr(user_config, "model_extra", {}) or {},
        )

        merged = merge_configs(user_config, defaults)
        self.assertEqual(config_data["semantic_cache"], merged["semantic_cache"])

    def test_parse_rejects_unknown_semantic_cache_field(self):
        config_data = _base_user_config()
        config_data["semantic_cache"] = {
            "enabled": True,
            "redis": {
                "connection": {
                    "host": "localhost",
                    "totally_unknown_field": True,
                }
            },
        }

        with self.assertRaises(ConfigParseError) as ctx:
            _parse_config_dict(config_data)

        self.assertIn("semantic_cache", str(ctx.exception))
        self.assertIn("totally_unknown_field", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
