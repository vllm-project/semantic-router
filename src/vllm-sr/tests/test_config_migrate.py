import sys
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.config_generator import generate_envoy_config_from_user_config  # noqa: E402
from cli.config_migration import migrate_config_data  # noqa: E402
from cli.main import main  # noqa: E402
from cli.models import UserConfig  # noqa: E402
from cli.parser import ConfigParseError, parse_user_config  # noqa: E402
from cli.validator import validate_user_config  # noqa: E402


def test_migrate_preserves_recipes_entrypoints_and_explicit_empty_auto_aliases():
    source = {
        "version": "v0.3",
        "providers": {"defaults": {}, "models": []},
        "routing": {"modelCards": []},
        "entrypoints": [
            {"model_names": ["amd/rocm-v1-balanced"], "recipe": "balanced"}
        ],
        "recipes": [
            {"name": "balanced", "routing": {"decisions": []}},
        ],
        "global": {"router": {"auto_model_names": []}},
    }

    migrated = migrate_config_data(source)

    assert migrated["entrypoints"] == source["entrypoints"]
    assert migrated["recipes"] == source["recipes"]
    assert migrated["global"]["router"]["auto_model_names"] == []


def test_migrate_is_idempotent_for_catalog_and_custom_provider_models():
    source = {
        "version": "v0.3",
        "listeners": [],
        "providers": {
            "defaults": {"model": "frontier", "reasoning_effort": "high"},
            "models": [
                {
                    "name": "frontier",
                    "catalog": "openai/gpt-5.4",
                    "reliability": {
                        "lb_policy": "least_request",
                        "retry_count": 2,
                    },
                    "backend_refs": [
                        {
                            "provider": "openai",
                            "api_key_env": "OPENAI_API_KEY",
                        }
                    ],
                },
                {
                    "name": "lab-model",
                    "reasoning": {
                        "type": "chat_template_kwargs",
                        "parameter": "enable_thinking",
                        "levels": ["disabled", "enabled"],
                        "default": "enabled",
                        "disabled": "disabled",
                    },
                    "reliability": {"retry_count": 1},
                    "backend_refs": [
                        {"provider": "vllm", "endpoint": "127.0.0.1:8000"}
                    ],
                },
            ],
        },
        "routing": {
            "modelCards": [
                {
                    "name": "openai/gpt-5.4",
                    "description": "Approved production overlay",
                },
                {
                    "name": "lab-model",
                    "publisher": "Example Lab",
                    "presentation": {
                        "logo": "monogram",
                        "monogram": "EL",
                        "monochrome": False,
                    },
                    "distribution": {
                        "type": "open_weights",
                        "source": "https://example.com/models/lab-model",
                    },
                },
            ]
        },
    }

    migrated = migrate_config_data(source)
    migrated_again = migrate_config_data(migrated)

    assert migrated_again == migrated
    by_name = {model["name"]: model for model in migrated["providers"]["models"]}
    assert by_name["frontier"]["catalog"] == "openai/gpt-5.4"
    assert by_name["frontier"]["reliability"] == {
        "lb_policy": "least_request",
        "retry_count": 2,
    }
    assert by_name["lab-model"]["reasoning"]["disabled"] == "disabled"
    assert by_name["lab-model"]["reliability"] == {"retry_count": 1}
    UserConfig.model_validate(migrated)


def test_migrate_materializes_legacy_router_owned_anthropic_backend(tmp_path):
    source = {
        "version": "v0.3",
        "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        "providers": {
            "models": [
                {
                    "name": "claude-legacy",
                    "api_format": "anthropic",
                }
            ]
        },
        "routing": {},
    }

    migrated = migrate_config_data(source)

    assert migrated["providers"]["models"][0]["backend_refs"] == [
        {"provider": "anthropic"}
    ]
    parsed = UserConfig.model_validate(migrated)
    assert validate_user_config(parsed, log_summary=False) == []
    output = tmp_path / "envoy.yaml"
    generate_envoy_config_from_user_config(parsed, str(output))
    rendered = yaml.safe_load(output.read_text(encoding="utf-8"))
    anthropic_cluster = next(
        cluster
        for cluster in rendered["static_resources"]["clusters"]
        if cluster["name"] == "model_claude_2dlegacy_cluster"
    )
    endpoint = anthropic_cluster["load_assignment"]["endpoints"][0]["lb_endpoints"][0][
        "endpoint"
    ]["address"]["socket_address"]
    assert endpoint == {"address": "api.anthropic.com", "port_value": 443}


def test_migrate_keeps_external_gateway_anthropic_metadata_backendless():
    source = {
        "version": "v0.3",
        "listeners": [],
        "providers": {
            "models": [
                {
                    "name": "claude-metadata-only",
                    "api_format": "anthropic",
                }
            ]
        },
        "routing": {},
    }

    migrated = migrate_config_data(source)

    assert "backend_refs" not in migrated["providers"]["models"][0]
    assert migrate_config_data(migrated) == migrated


def test_migrate_coalesces_non_conflicting_alias_cards_by_catalog_identity():
    source = {
        "version": "v0.3",
        "listeners": [],
        "providers": {
            "models": [
                {
                    "name": "prod",
                    "catalog": "openai/gpt-5.4",
                    "backend_refs": [{"provider": "openai"}],
                },
                {
                    "name": "canary",
                    "catalog": "openai/gpt-5.4",
                    "backend_refs": [{"provider": "openai"}],
                },
            ]
        },
        "routing": {
            "modelCards": [
                {"name": "prod", "description": "Approved production profile"},
                {"name": "canary", "tags": ["canary"]},
            ]
        },
    }

    migrated = migrate_config_data(source)

    assert migrated["routing"]["modelCards"] == [
        {
            "name": "openai/gpt-5.4",
            "description": "Approved production profile",
            "tags": ["canary"],
        }
    ]
    assert migrate_config_data(migrated) == migrated
    UserConfig.model_validate(migrated)


def test_migrate_coalesces_existing_canonical_card_with_alias_metadata():
    source = {
        "version": "v0.3",
        "providers": {
            "models": [
                {
                    "name": "prod",
                    "catalog": "openai/gpt-5.4",
                    "backend_refs": [{"provider": "openai"}],
                }
            ]
        },
        "routing": {
            "modelCards": [
                {"name": "openai/gpt-5.4", "description": "Shared"},
                {"name": "prod", "tags": ["approved"]},
            ]
        },
    }

    migrated = migrate_config_data(source)

    assert migrated["routing"]["modelCards"] == [
        {
            "name": "openai/gpt-5.4",
            "description": "Shared",
            "tags": ["approved"],
        }
    ]


def test_migrate_rejects_conflicting_alias_cards_by_catalog_identity():
    source = {
        "version": "v0.3",
        "providers": {
            "models": [
                {
                    "name": "prod",
                    "catalog": "openai/gpt-5.4",
                    "backend_refs": [{"provider": "openai"}],
                },
                {
                    "name": "canary",
                    "catalog": "openai/gpt-5.4",
                    "backend_refs": [{"provider": "openai"}],
                },
            ]
        },
        "routing": {
            "modelCards": [
                {"name": "prod", "description": "Production"},
                {"name": "canary", "description": "Canary"},
            ]
        },
    }

    with pytest.raises(
        ValueError,
        match=r"aliases collapse to 'openai/gpt-5.4'.*description",
    ):
        migrate_config_data(source)


def test_migrate_relocates_legacy_flat_empty_auto_aliases():
    migrated = migrate_config_data(
        {
            "global": {
                "auto_model_names": [],
            },
        }
    )

    assert migrated["global"]["router"]["auto_model_names"] == []
    assert "auto_model_names" not in migrated["global"]


def test_migrate_prefers_canonical_auto_aliases_over_legacy_flat_value():
    migrated = migrate_config_data(
        {
            "global": {
                "auto_model_names": [],
                "router": {
                    "auto_model_names": ["router/canonical"],
                },
            },
        }
    )

    assert migrated["global"]["router"]["auto_model_names"] == ["router/canonical"]
    assert "auto_model_names" not in migrated["global"]


def test_migrate_config_preserves_entrypoints_and_recipes():
    config = {
        "version": "v0.3",
        "routing": {},
        "entrypoints": [{"model_names": ["vllm-sr/private"], "recipe": "private"}],
        "recipes": [
            {
                "name": "private",
                "routing": {
                    "signals": {
                        "metadata": [
                            {
                                "name": "private",
                                "key": "cohort",
                                "predicate": {"equals": "private"},
                            }
                        ]
                    }
                },
            }
        ],
    }

    migrated = migrate_config_data(config)

    assert migrated["entrypoints"] == config["entrypoints"]
    assert migrated["recipes"] == config["recipes"]


def _legacy_provider_models_config() -> dict:
    return {
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
                "openai": {
                    "type": "reasoning_effort",
                    "parameter": "reasoning_effort",
                    "activation_parameter": "reasoning_enabled",
                    "levels": ["none", "high"],
                    "default": "high",
                    "disabled": "none",
                }
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


def test_migrate_config_data_splits_legacy_provider_models():
    legacy = _legacy_provider_models_config()

    migrated = migrate_config_data(legacy)

    assert migrated["version"] == "v0.3"
    assert migrated["routing"]["signals"]["keywords"][0]["name"] == "math_terms"
    assert migrated["routing"]["decisions"][0]["name"] == "default-route"
    assert migrated["routing"]["modelCards"] == [
        {
            "name": "gpt-4o",
            "description": "General reasoning model",
            "capabilities": ["general", "reasoning"],
            "modality": "text",
        }
    ]
    assert migrated["evaluation"]["records"] == [
        {
            "model": "gpt-4o",
            "benchmark": "vllm-sr/operator-rating@1.0.0",
            "metrics": {"score": 0.95},
        }
    ]
    assert migrated["providers"]["models"] == [
        {
            "name": "gpt-4o",
            "reasoning": {
                "type": "reasoning_effort",
                "parameter": "reasoning_effort",
                "activation_parameter": "reasoning_enabled",
                "levels": ["none", "high"],
                "default": "high",
                "disabled": "none",
            },
            "backend_refs": [
                {
                    "name": "primary",
                    "endpoint": "api.openai.com:443",
                    "protocol": "https",
                    "weight": 100,
                    "api_key": "sk-test",
                    "provider": "vllm",
                }
            ],
        }
    ]
    assert migrated["providers"]["defaults"]["model"] == "gpt-4o"
    assert "reasoning_families" not in migrated["providers"]["defaults"]
    assert migrated["global"]["stores"]["memory"]["enabled"] is True
    UserConfig.model_validate(migrated)


def test_migrate_config_data_unifies_evaluation_definitions_and_records():
    migrated = migrate_config_data(
        {
            "version": "v0.3",
            "evaluation_catalog": {"benchmarks": []},
            "providers": {
                "models": [{"name": "frontier", "catalog": "vendor/frontier"}]
            },
            "routing": {
                "modelCards": [
                    {
                        "name": "frontier",
                        "evaluations": [
                            {
                                "benchmark": "acme/reasoning@1.0.0",
                                "metrics": {"accuracy": 0.82},
                            }
                        ],
                    }
                ]
            },
        }
    )

    assert "evaluation_catalog" not in migrated
    assert migrated["evaluation"]["records"] == [
        {
            "model": "vendor/frontier",
            "benchmark": "acme/reasoning@1.0.0",
            "metrics": {"accuracy": 0.82},
        }
    ]
    assert migrated["routing"].get("modelCards") is None
    UserConfig.model_validate(migrated)


def test_migrate_removes_reasoning_registry_when_it_was_the_only_default():
    migrated = migrate_config_data(
        {
            "version": "v0.3",
            "providers": {
                "defaults": {
                    "reasoning_families": {
                        "private": {
                            "type": "chat_template_kwargs",
                            "parameter": "enable_thinking",
                        }
                    }
                },
                "models": [
                    {
                        "name": "private",
                        "reasoning_family": "private",
                    },
                    {
                        "name": "built-in-family",
                        "reasoning_family": "qwen3",
                    },
                ],
            },
            "routing": {},
        }
    )

    assert "defaults" not in migrated["providers"]
    assert migrated["providers"]["models"][0]["reasoning"] == {
        "type": "chat_template_kwargs",
        "parameter": "enable_thinking",
    }
    assert migrated["providers"]["models"][1]["reasoning"] == {"family": "qwen3"}


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
    assert result.stderr == ""
    assert "✓ Configuration migrated" in result.stdout
    assert "Files" in result.stdout
    assert f"Source  {config_path}" in result.stdout

    migrated_path = tmp_path / "config.migrated.yaml"
    assert f"Output  {migrated_path}" in result.stdout
    migrated = yaml.safe_load(migrated_path.read_text())

    assert migrated["version"] == "v0.3"
    assert migrated["providers"]["defaults"]["model"] == "gpt-4o-mini"
    assert "modelCards" not in migrated["routing"]
    assert migrated["providers"]["models"][0]["backend_refs"] == [
        {
            "name": "primary",
            "endpoint": "host.docker.internal:8000",
            "protocol": "http",
            "weight": 100,
            "provider": "vllm",
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
