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


class TestCLITypedBackendModelCompat(unittest.TestCase):
    def test_image_gen_backends_use_typed_compat_path(self):
        defaults = load_embedded_defaults()
        config_data = _base_user_config()
        config_data["image_gen_backends"] = {
            "vllm_omni_local": {
                "type": "vllm_omni",
                "base_url": "http://localhost:8001",
                "model": "Qwen/Qwen-Image",
                "num_inference_steps": 4,
                "cfg_scale": 0.0,
                "default_width": 1024,
                "default_height": 1024,
                "timeout_seconds": 120,
            }
        }

        user_config = _parse_config_dict(config_data)
        compat_blocks = get_typed_compat_blocks(user_config)

        self.assertIsNotNone(compat_blocks.image_gen_backends)
        self.assertEqual(
            config_data["image_gen_backends"],
            dump_typed_compat_block(compat_blocks.image_gen_backends),
        )
        self.assertNotIn(
            "image_gen_backends",
            getattr(user_config, "model_extra", {}) or {},
        )

        merged = merge_configs(user_config, defaults)
        self.assertEqual(
            config_data["image_gen_backends"],
            merged["image_gen_backends"],
        )

    def test_provider_profiles_use_typed_compat_path(self):
        defaults = load_embedded_defaults()
        config_data = _base_user_config()
        config_data["provider_profiles"] = {
            "openai_prod": {
                "type": "openai",
                "base_url": "https://api.openai.com/v1",
                "auth_header": "Authorization",
                "auth_prefix": "Bearer",
                "extra_headers": {"x-tenant": "prod"},
                "api_version": "2024-10-01",
                "chat_path": "/chat/completions",
            }
        }

        user_config = _parse_config_dict(config_data)
        compat_blocks = get_typed_compat_blocks(user_config)

        self.assertIsNotNone(compat_blocks.provider_profiles)
        self.assertEqual(
            config_data["provider_profiles"],
            dump_typed_compat_block(compat_blocks.provider_profiles),
        )
        self.assertNotIn(
            "provider_profiles",
            getattr(user_config, "model_extra", {}) or {},
        )

        merged = merge_configs(user_config, defaults)
        self.assertEqual(
            config_data["provider_profiles"],
            merged["provider_profiles"],
        )

    def test_parse_rejects_unknown_image_gen_backend_field(self):
        config_data = _base_user_config()
        config_data["image_gen_backends"] = {
            "vllm_omni_local": {
                "type": "vllm_omni",
                "totally_unknown_field": True,
            }
        }

        with self.assertRaises(ConfigParseError) as ctx:
            _parse_config_dict(config_data)

        self.assertIn("image_gen_backends", str(ctx.exception))
        self.assertIn("totally_unknown_field", str(ctx.exception))

    def test_parse_rejects_unknown_provider_profile_field(self):
        config_data = _base_user_config()
        config_data["provider_profiles"] = {
            "openai_prod": {
                "type": "openai",
                "totally_unknown_field": True,
            }
        }

        with self.assertRaises(ConfigParseError) as ctx:
            _parse_config_dict(config_data)

        self.assertIn("provider_profiles", str(ctx.exception))
        self.assertIn("totally_unknown_field", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
