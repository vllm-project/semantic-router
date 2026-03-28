import sys
from pathlib import Path

import yaml
from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.config_migration import migrate_config_data  # noqa: E402
from cli.main import main  # noqa: E402
from cli.parser import ConfigParseError, parse_user_config  # noqa: E402


def test_migrate_config_data_splits_legacy_provider_models():
    legacy = {
        "version": "v0.1",
        "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        "signals": {
            "keywords": [
                {"name": "math_terms", "operator": "OR", "keywords": ["algebra"]}
            ]
        },
        "decisions": [
            {
                "name": "default-route",
                "description": "fallback",
                "priority": 100,
                "rules": {"operator": "AND", "conditions": []},
                "modelRefs": [{"model": "gpt-4o"}],
            }
        ],
        "providers": {
            "default_model": "gpt-4o",
            "reasoning_families": {
                "openai": {"type": "reasoning_effort", "parameter": "reasoning_effort"}
            },
            "default_reasoning_effort": "high",
            "models": [
                {
                    "name": "gpt-4o",
                    "endpoints": [
                        {
                            "name": "primary",
                            "endpoint": "api.openai.com:443",
                            "protocol": "https",
                            "weight": 100,
                        }
                    ],
                    "access_key": "sk-test",
                    "reasoning_family": "openai",
                    "description": "General reasoning model",
                    "capabilities": ["general", "reasoning"],
                    "modality": "text",
                    "quality_score": 0.95,
                }
            ],
        },
        "memory": {
            "enabled": True,
            "default_retrieval_limit": 3,
        },
    }

    migrated = migrate_config_data(legacy)

    assert migrated["version"] == "v0.3"
    assert migrated["routing"]["signals"]["keywords"][0]["name"] == "math_terms"
    assert migrated["routing"]["decisions"][0]["name"] == "default-route"
    assert migrated["routing"]["modelCards"] == [
        {
            "name": "gpt-4o",
            "description": "General reasoning model",
            "capabilities": ["general", "reasoning"],
            "quality_score": 0.95,
            "modality": "text",
        }
    ]
    assert migrated["providers"]["models"] == [
        {
            "name": "gpt-4o",
            "reasoning_family": "openai",
            "backend_refs": [
                {
                    "name": "primary",
                    "endpoint": "api.openai.com:443",
                    "protocol": "https",
                    "weight": 100,
                    "api_key": "sk-test",
                }
            ],
        }
    ]
    assert migrated["providers"]["defaults"]["default_model"] == "gpt-4o"
    assert migrated["global"]["stores"]["memory"]["enabled"] is True


def test_cli_config_migrate_writes_canonical_yaml(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.1",
                "listeners": [
                    {"name": "http-8899", "address": "0.0.0.0", "port": 8899}
                ],
                "providers": {
                    "default_model": "gpt-4o-mini",
                    "models": [
                        {
                            "name": "gpt-4o-mini",
                            "endpoints": [
                                {
                                    "name": "primary",
                                    "endpoint": "host.docker.internal:8000",
                                    "protocol": "http",
                                    "weight": 100,
                                }
                            ],
                        }
                    ],
                },
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
            sort_keys=False,
        )
    )

    runner = CliRunner()
    result = runner.invoke(main, ["config", "migrate", "--config", str(config_path)])

    assert result.exit_code == 0

    migrated_path = tmp_path / "config.migrated.yaml"
    migrated = yaml.safe_load(migrated_path.read_text())

    assert migrated["version"] == "v0.3"
    assert migrated["providers"]["defaults"]["default_model"] == "gpt-4o-mini"
    assert migrated["routing"]["modelCards"][0]["name"] == "gpt-4o-mini"
    assert migrated["providers"]["models"][0]["backend_refs"] == [
        {
            "name": "primary",
            "endpoint": "host.docker.internal:8000",
            "protocol": "http",
            "weight": 100,
        }
    ]


def test_migrate_config_data_moves_global_modules_under_model_catalog():
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
            "model_catalog": {
                "system": {"prompt_guard": "models/mom-jailbreak-classifier"}
            },
            "modules": {"prompt_guard": {"enabled": True, "model_ref": "prompt_guard"}},
        },
    }

    migrated = migrate_config_data(legacy)

    assert "modules" not in migrated["global"]
    assert migrated["global"]["model_catalog"]["modules"]["prompt_guard"] == {
        "enabled": True,
        "model_ref": "prompt_guard",
    }


def test_parse_user_config_rejects_deprecated_global_modules(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "listeners": [
                    {"name": "http-8899", "address": "0.0.0.0", "port": 8899}
                ],
                "providers": {
                    "defaults": {"default_model": "gpt-4o-mini"},
                    "models": [
                        {
                            "name": "gpt-4o-mini",
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
                "global": {"modules": {"prompt_guard": {"model_ref": "prompt_guard"}}},
            },
            sort_keys=False,
        )
    )

    try:
        parse_user_config(str(config_path))
    except ConfigParseError as exc:
        assert "global.modules" in str(exc)
    else:
        raise AssertionError("expected ConfigParseError")


def test_parse_user_config_rejects_legacy_flat_signal_blocks(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "listeners": [
                    {"name": "http-8899", "address": "0.0.0.0", "port": 8899}
                ],
                "providers": {
                    "defaults": {"default_model": "gpt-4o-mini"},
                    "models": [
                        {
                            "name": "gpt-4o-mini",
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
                "keyword_rules": [
                    {
                        "name": "legacy-keywords",
                        "operator": "OR",
                        "keywords": ["hello"],
                    }
                ],
            },
            sort_keys=False,
        )
    )

    try:
        parse_user_config(str(config_path))
    except ConfigParseError as exc:
        assert "keyword_rules" in str(exc)
    else:
        raise AssertionError("expected ConfigParseError")


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

    assert migrated["providers"]["defaults"]["default_model"] == "qwen3-32b"
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
            "reasoning_family": "qwen3",
            "pricing": {
                "currency": "USD",
                "prompt_per_1m": 1.2,
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
