# ruff: noqa: E402

import sys
import tempfile
import unittest
from pathlib import Path

import yaml

CLI_ROOT = Path(__file__).resolve().parents[1]
if str(CLI_ROOT) not in sys.path:
    sys.path.insert(0, str(CLI_ROOT))

from cli.defaults import load_embedded_defaults
from cli.merger import merge_configs
from cli.parser import parse_user_config
from cli.validator import validate_user_config


def _base_user_config() -> dict:
    return {
        "version": "v0.1",
        "listeners": [
            {
                "name": "main",
                "address": "0.0.0.0",
                "port": 8080,
            }
        ],
        "signals": {},
        "providers": {
            "models": [
                {
                    "name": "router-model",
                    "endpoints": [
                        {
                            "name": "primary",
                            "endpoint": "127.0.0.1:8000",
                            "protocol": "http",
                            "weight": 1,
                        }
                    ],
                }
            ],
            "default_model": "router-model",
        },
        "decisions": [
            {
                "name": "default-route",
                "description": "Default route",
                "priority": 10,
                "rules": {"operator": "AND", "conditions": []},
                "modelRefs": [{"model": "router-model", "use_reasoning": False}],
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


class TestCLISignalMergePaths(unittest.TestCase):
    def test_modality_signal_merges_into_runtime_rules(self):
        config_data = _base_user_config()
        config_data["signals"]["modality"] = [
            {
                "name": "DIFFUSION",
                "description": "Image generation via diffusion model",
            }
        ]
        config_data["decisions"][0]["rules"] = {
            "operator": "AND",
            "conditions": [{"type": "modality", "name": "DIFFUSION"}],
        }

        user_config = _parse_config_dict(config_data)

        self.assertEqual([], validate_user_config(user_config))

        merged = merge_configs(user_config, load_embedded_defaults())

        self.assertEqual(
            [
                {
                    "name": "DIFFUSION",
                    "description": "Image generation via diffusion model",
                }
            ],
            merged["modality_rules"],
        )

    def test_role_bindings_validate_against_role_and_merge(self):
        config_data = _base_user_config()
        config_data["signals"]["role_bindings"] = [
            {
                "name": "premium-binding",
                "role": "premium_tier",
                "subjects": [{"kind": "Group", "name": "premium"}],
                "description": "Premium users",
            }
        ]
        config_data["decisions"][0]["rules"] = {
            "operator": "AND",
            "conditions": [{"type": "authz", "name": "premium_tier"}],
        }

        user_config = _parse_config_dict(config_data)

        self.assertEqual([], validate_user_config(user_config))

        merged = merge_configs(user_config, load_embedded_defaults())

        self.assertEqual(
            [
                {
                    "name": "premium-binding",
                    "role": "premium_tier",
                    "subjects": [{"kind": "Group", "name": "premium"}],
                    "description": "Premium users",
                }
            ],
            merged["role_bindings"],
        )


if __name__ == "__main__":
    unittest.main()
