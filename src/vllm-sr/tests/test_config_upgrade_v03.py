from __future__ import annotations

# ruff: noqa: E402 -- tests add the source tree before importing the CLI package.
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.config_upgrade_v03 import migrate_v03_config_data
from cli.config_upgrade_v03_support import ConfigMigrationError
from cli.main import main
from cli.models import UserConfig
from cli.parser import ConfigParseError, parse_user_config
from cli.validator import validate_user_config


def valid_v03_config() -> dict:
    return {
        "version": "v0.3",
        "listeners": [
            {
                "name": "http-8899",
                "address": "0.0.0.0",
                "port": 8899,
                "timeout": "300s",
            }
        ],
        "providers": {
            "defaults": {
                "default_model": "local/fast",
                "default_reasoning_effort": "medium",
                "reasoning_families": {
                    "qwen": {
                        "type": "chat_template_kwargs",
                        "parameter": "enable_thinking",
                    }
                },
            },
            "models": [
                {
                    "name": "local/fast",
                    "reasoning_family": "qwen",
                    "provider_model_id": "qwen-fast",
                    "api_format": "openai",
                    "reliability": {"retry_count": 2},
                    "pricing": {
                        "currency": "USD",
                        "prompt_per_1m": 0.1,
                        "cached_input_per_1m": 0.025,
                        "completion_per_1m": 0.4,
                    },
                    "backend_refs": [
                        {
                            "name": "primary",
                            "endpoint": "model.internal:8000/v1",
                            "protocol": "http",
                            "type": "vllm",
                            "weight": 1,
                            "api_key_env": "FAST_API_KEY",
                        }
                    ],
                }
            ],
        },
        "routing": {
            "strategy": "priority",
            "modelCards": [
                {
                    "name": "local/fast",
                    "description": "Fast model",
                    "capabilities": ["chat", "tools", "reasoning"],
                    "loras": [{"name": "code"}],
                    "modality": "text",
                }
            ],
            "decisions": [
                {
                    "name": "Simple",
                    "description": "Handle ordinary work.",
                    "priority": 100,
                    "rules": {"operator": "AND", "conditions": []},
                    "modelRefs": [
                        {
                            "model": "local/fast",
                            "lora_name": "code",
                            "weight": 2,
                            "use_reasoning": True,
                            "reasoning_effort": "medium",
                        }
                    ],
                }
            ],
        },
        "global": {
            "router": {
                "config_source": "file",
                "auto_model_names": ["vllm-sr/blend", "blend"],
            }
        },
    }


def test_upgrade_splits_models_recipes_and_entrypoint_assignments() -> None:
    result = migrate_v03_config_data(valid_v03_config())
    document = result.document

    assert document["version"] == "v0.4"
    assert document["global"]["billing"]["currency"] == "USD"
    assert result.summary.models == 1
    assert result.summary.recipes == 1
    assert result.summary.entrypoints == 1
    model = document["models"][0]
    assert model == {
        "name": "local/fast",
        "card": {
            "description": "Fast model",
            "capabilities": ["chat", "tools", "reasoning"],
            "modality": "text",
            "loras": ["code"],
            "reasoning": {
                "type": "chat_template_kwargs",
                "efforts": ["medium"],
            },
        },
        "connections": [
            {
                "provider": "vllm",
                "interface": "chat",
                "model": "qwen-fast",
                "endpoint": "http://model.internal:8000/v1",
                "credential": "local-fast-primary-credential",
            }
        ],
        "runtime": {"max_retries": 2},
        "pricing": {
            "input_cost_per_million_tokens": "0.1",
            "output_cost_per_million_tokens": "0.4",
            "cache_read_cost_per_million_tokens": "0.025",
        },
    }
    decision = document["recipes"][0]["document"]["decisions"][0]
    assert "modelRefs" not in decision
    assert document["entrypoints"] == [
        {
            "name": "vllm-sr/blend",
            "aliases": ["blend"],
            "recipe": "default",
            "assignments": {
                "Simple": {
                    "models": [
                        {
                            "model": "local/fast",
                            "weight": "2",
                            "lora": "code",
                            "reasoning": {"enabled": True, "effort": "medium"},
                        }
                    ]
                }
            },
        }
    ]
    assert document["global"]["services"]["backend_credentials"] == {
        "local-fast-primary-credential": {
            "credential_adapter_id": "bearer",
            "secret_env": "FAST_API_KEY",
        }
    }
    parsed = UserConfig.model_validate(document)
    assert validate_user_config(parsed, log_summary=False) == []


def test_upgrade_preserves_named_recipe_and_all_entrypoint_aliases() -> None:
    source = valid_v03_config()
    source["routing"]["decisions"] = []
    source["global"]["router"].pop("auto_model_names")
    source["recipes"] = [
        {
            "name": "private",
            "description": "Private route",
            "routing": {
                "decisions": [
                    {
                        "name": "Private",
                        "description": "Keep requests local.",
                        "priority": 100,
                        "rules": {"operator": "AND", "conditions": []},
                        "modelRefs": [{"model": "local/fast"}],
                    }
                ]
            },
        }
    ]
    source["entrypoints"] = [
        {
            "model_names": ["vllm-sr/private", "private", "private"],
            "recipe": "private",
        }
    ]

    result = migrate_v03_config_data(source)

    private = next(
        recipe for recipe in result.document["recipes"] if recipe["name"] == "private"
    )
    assert private["description"] == "Private route"
    entrypoint = next(
        item for item in result.document["entrypoints"] if item["recipe"] == "private"
    )
    assert entrypoint["name"] == "vllm-sr/private"
    assert entrypoint["aliases"] == ["private"]


@pytest.mark.parametrize("version", ["v0.2", "v0.4", None])
def test_upgrade_accepts_exactly_v03(version: str | None) -> None:
    source = valid_v03_config()
    source["version"] = version

    with pytest.raises(ConfigMigrationError) as raised:
        migrate_v03_config_data(source)

    assert raised.value.issues[0].code == "unsupported_source_version"


def test_upgrade_rejects_plaintext_secrets_with_source_path() -> None:
    source = valid_v03_config()
    backend = source["providers"]["models"][0]["backend_refs"][0]
    backend.pop("api_key_env")
    backend["api_key"] = "sk-plaintext-must-not-migrate"

    with pytest.raises(ConfigMigrationError) as raised:
        migrate_v03_config_data(source)

    issue = next(
        issue for issue in raised.value.issues if issue.code == "plaintext_secret"
    )
    assert issue.path == "providers.models[0].backend_refs[0].api_key"
    assert "sk-plaintext" not in str(raised.value)


def test_upgrade_rejects_plaintext_authorization_header() -> None:
    source = valid_v03_config()
    backend = source["providers"]["models"][0]["backend_refs"][0]
    backend["extra_headers"] = {"Authorization": "Bearer private-token"}

    with pytest.raises(ConfigMigrationError) as raised:
        migrate_v03_config_data(source)

    assert any(issue.code == "plaintext_secret" for issue in raised.value.issues)
    assert "private-token" not in str(raised.value)


def test_cli_never_echoes_secret_from_malformed_yaml() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        source_path = Path("malformed.yaml")
        source_path.write_text(
            "version: v0.3\napi_key: private-value\n  invalid: yaml\n",
            encoding="utf-8",
        )

        result = runner.invoke(
            main, ["config", "migrate", "--config", str(source_path)]
        )

        assert result.exit_code == 1
        assert "invalid_source_yaml" in result.output
        assert "private-value" not in result.output


def test_upgrade_rejects_lossy_transport_and_algorithm_configuration() -> None:
    source = valid_v03_config()
    source["providers"]["models"][0]["reliability"]["health_check_path"] = "/health"
    source["routing"]["decisions"][0]["algorithm"] = {
        "type": "fusion",
        "fusion": {"model": "local/fast"},
    }

    with pytest.raises(ConfigMigrationError) as raised:
        migrate_v03_config_data(source)

    codes = {issue.code for issue in raised.value.issues}
    assert "unsupported_reliability_policy" in codes
    assert "embedded_algorithm_model" in codes


def test_upgrade_rejects_global_model_selection_instead_of_dropping_it() -> None:
    source = valid_v03_config()
    source["global"]["router"]["model_selection"] = {
        "method": "hybrid",
        "enabled": True,
    }

    with pytest.raises(ConfigMigrationError) as raised:
        migrate_v03_config_data(source)

    assert any(issue.code == "global_model_selection" for issue in raised.value.issues)


def test_runtime_parser_remains_strict_v04() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        path = Path("config-v0.3.yaml")
        path.write_text(
            yaml.safe_dump(valid_v03_config(), sort_keys=False), encoding="utf-8"
        )
        with pytest.raises(ConfigParseError):
            parse_user_config(str(path), log_summary=False)


def test_cli_migrate_writes_distinct_validated_output_and_never_source() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        source_path = Path("config.yaml")
        original = yaml.safe_dump(valid_v03_config(), sort_keys=False)
        source_path.write_text(original, encoding="utf-8")

        result = runner.invoke(
            main, ["config", "migrate", "--config", str(source_path)]
        )

        assert result.exit_code == 0, result.output
        assert "Configuration migrated" in result.output
        output_path = Path("config.v0.4.yaml")
        assert output_path.is_file()
        assert source_path.read_text(encoding="utf-8") == original
        output = yaml.safe_load(output_path.read_text(encoding="utf-8"))
        assert output["version"] == "v0.4"
        assert "sk-" not in output_path.read_text(encoding="utf-8")


def test_cli_migrate_requires_force_only_for_existing_output() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        source_path = Path("source.yaml")
        output_path = Path("next.yaml")
        source_path.write_text(
            yaml.safe_dump(valid_v03_config(), sort_keys=False), encoding="utf-8"
        )
        output_path.write_text("sentinel\n", encoding="utf-8")

        blocked = runner.invoke(
            main,
            [
                "config",
                "migrate",
                "--config",
                str(source_path),
                "--output",
                str(output_path),
            ],
        )
        assert blocked.exit_code == 1
        assert "output_exists" in blocked.output
        assert output_path.read_text(encoding="utf-8") == "sentinel\n"

        replaced = runner.invoke(
            main,
            [
                "config",
                "migrate",
                "--config",
                str(source_path),
                "--output",
                str(output_path),
                "--force",
            ],
        )
        assert replaced.exit_code == 0, replaced.output
        assert yaml.safe_load(output_path.read_text(encoding="utf-8"))["version"] == (
            "v0.4"
        )


def test_cli_migrate_blocks_source_overwrite_even_with_force() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        source_path = Path("source.yaml")
        original = yaml.safe_dump(valid_v03_config(), sort_keys=False)
        source_path.write_text(original, encoding="utf-8")

        result = runner.invoke(
            main,
            [
                "config",
                "migrate",
                "--config",
                str(source_path),
                "--output",
                str(source_path),
                "--force",
            ],
        )

        assert result.exit_code == 1
        assert "source_overwrite_forbidden" in result.output
        assert source_path.read_text(encoding="utf-8") == original


def test_cli_config_help_exposes_offline_upgrade_contract() -> None:
    result = CliRunner().invoke(main, ["config", "migrate", "--help"])

    assert result.exit_code == 0
    assert "canonical v0.3" in result.output
    assert "never" in result.output
    assert "overwritten" in result.output


def test_input_fixture_is_not_mutated() -> None:
    source = valid_v03_config()
    before = deepcopy(source)

    migrate_v03_config_data(source)

    assert source == before
