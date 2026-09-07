import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.config_migration import migrate_config_data  # noqa: E402
from cli.parser import ConfigParseError, parse_user_config  # noqa: E402


def _legacy_model_catalog_config() -> dict:
    return {
        "version": "v0.1",
        "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        "default_model": "qwen3-32b",
        "reasoning_families": {
            "qwen3": {"type": "chat_template_kwargs", "parameter": "enable_thinking"}
        },
        "keyword_rules": [
            {
                "name": "code-keywords",
                "operator": "OR",
                "keywords": ["debug", "algorithm"],
            }
        ],
        "provider_profiles": {
            "openai-prod": {
                "type": "openai",
                "base_url": "https://api.openai.com/v1",
                "auth_header": "Authorization",
                "auth_prefix": "Bearer",
                "chat_path": "/chat/completions",
            }
        },
        "vllm_endpoints": [
            {
                "name": "local-primary",
                "address": "127.0.0.1",
                "port": 8000,
                "protocol": "http",
                "weight": 80,
            },
            {
                "name": "openai",
                "provider_profile": "openai-prod",
                "weight": 20,
            },
        ],
        "model_config": {
            "qwen3-32b": {
                "preferred_endpoints": ["local-primary", "openai"],
                "reasoning_family": "qwen3",
                "description": "Premium reasoning tier",
                "capabilities": ["chat", "reasoning"],
                "loras": [
                    {
                        "name": "computer-science-expert",
                        "description": "Adapter for advanced computer science prompts",
                    }
                ],
                "api_format": "openai",
                "pricing": {
                    "currency": "USD",
                    "prompt_per_1m": 1.2,
                    "cached_input_per_1m": 0.3,
                    "completion_per_1m": 3.4,
                },
                "external_model_ids": {"openai": "qwen3-32b"},
                "access_key": "sk-test-openai",
            }
        },
        "decisions": [
            {
                "name": "cs-route",
                "description": "Route computer science prompts",
                "priority": 100,
                "rules": {
                    "operator": "AND",
                    "conditions": [{"type": "domain", "name": "computer science"}],
                },
                "modelRefs": [
                    {"model": "qwen3-32b", "lora_name": "computer-science-expert"}
                ],
            }
        ],
    }


def test_migrate_config_data_promotes_legacy_provider_defaults_and_signals():
    migrated = migrate_config_data(_legacy_model_catalog_config())

    assert migrated["providers"]["defaults"]["model"] == "qwen3-32b"
    assert migrated["routing"]["signals"]["keywords"][0]["name"] == "code-keywords"


def test_migrate_config_data_promotes_legacy_lora_catalog_and_backend_refs():
    migrated = migrate_config_data(_legacy_model_catalog_config())

    assert migrated["routing"]["modelCards"] == [
        {
            "name": "qwen3-32b",
            "description": "Premium reasoning tier",
            "capabilities": ["chat", "reasoning"],
            "loras": [
                {
                    "name": "computer-science-expert",
                    "description": "Adapter for advanced computer science prompts",
                }
            ],
        }
    ]
    assert migrated["providers"]["models"] == [
        {
            "name": "qwen3-32b",
            "reasoning": {
                "type": "chat_template_kwargs",
                "parameter": "enable_thinking",
            },
            "pricing": {
                "currency": "USD",
                "prompt_per_1m": 1.2,
                "cached_input_per_1m": 0.3,
                "completion_per_1m": 3.4,
            },
            "api_format": "openai",
            "external_model_ids": {"openai": "qwen3-32b"},
            "backend_refs": [
                {
                    "name": "local-primary",
                    "endpoint": "127.0.0.1:8000",
                    "protocol": "http",
                    "weight": 80,
                    "api_key": "sk-test-openai",
                    "provider": "vllm",
                },
                {
                    "name": "openai",
                    "base_url": "https://api.openai.com/v1",
                    "provider": "openai",
                    "auth_header": "Authorization",
                    "auth_prefix": "Bearer",
                    "chat_path": "/chat/completions",
                    "weight": 20,
                    "api_key": "sk-test-openai",
                },
            ],
        }
    ]


def test_migrate_config_data_preserves_decision_lora_reference():
    migrated = migrate_config_data(_legacy_model_catalog_config())

    assert (
        migrated["routing"]["decisions"][0]["modelRefs"][0]["lora_name"]
        == "computer-science-expert"
    )


def test_parse_user_config_rejects_deprecated_provider_model_loras(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "listeners": [
                    {"name": "http-8899", "address": "0.0.0.0", "port": 8899}
                ],
                "providers": {
                    "defaults": {"default_model": "qwen3-32b"},
                    "models": [
                        {
                            "name": "qwen3-32b",
                            "loras": [{"name": "computer-science-expert"}],
                            "backend_refs": [
                                {
                                    "endpoint": "host.docker.internal:8000",
                                    "protocol": "http",
                                }
                            ],
                        }
                    ],
                },
                "routing": {
                    "modelCards": [{"name": "qwen3-32b"}],
                    "decisions": [
                        {
                            "name": "default-route",
                            "description": "fallback",
                            "priority": 100,
                            "rules": {"operator": "AND", "conditions": []},
                            "modelRefs": [{"model": "qwen3-32b"}],
                        }
                    ],
                },
            },
            sort_keys=False,
        )
    )

    try:
        parse_user_config(str(config_path))
    except ConfigParseError as exc:
        assert "providers.models[0].loras" in str(exc)
    else:
        raise AssertionError("expected ConfigParseError")


def test_migrate_config_data_preserves_hallucination_endpoint_backend():
    legacy = {
        "version": "v0.3",
        "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        "providers": {"defaults": {"default_model": "gpt-4o-mini"}},
        "routing": {
            "modelCards": [{"name": "gpt-4o-mini"}],
            "decisions": [
                {
                    "name": "default-route",
                    "description": "fallback",
                    "priority": 100,
                    "rules": {"operator": "AND", "conditions": []},
                    "modelRefs": [{"model": "gpt-4o-mini"}],
                }
            ],
        },
        "global": {
            "hallucination_mitigation": {
                "enabled": True,
                "hallucination_model": {
                    "backend": "endpoint",
                    "endpoint": "http://127.0.0.1:8077/v1",
                    "include_explanation": True,
                    "model_id": "KRLabsOrg/lettucedect-v2-qwen-2b",
                },
            }
        },
    }

    migrated = migrate_config_data(legacy)

    detector = migrated["global"]["model_catalog"]["modules"][
        "hallucination_mitigation"
    ]["detector"]
    assert detector["backend"] == "endpoint"
    assert detector["endpoint"] == "http://127.0.0.1:8077/v1"
    assert detector["include_explanation"] is True
    assert detector["model_id"] == "KRLabsOrg/lettucedect-v2-qwen-2b"
    # Legacy detectors are given a stable model_ref during migration.
    assert detector["model_ref"] == "hallucination_detector"
