from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.config_upgrade_v03 import (  # noqa: E402
    ConfigMigrationError,
    migrate_v03_config_data,
)
from cli.main import main  # noqa: E402
from cli.models import UserConfig  # noqa: E402
from cli.parser import ConfigParseError, parse_user_config  # noqa: E402
from cli.validator import validate_user_config  # noqa: E402


def previous_v03_config() -> dict:
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
                    "reliability": {
                        "lb_policy": "least_request",
                        "retry_count": 2,
                        "retry_on": "connect-failure,request-timeout",
                        "health_check_path": "/health",
                        "health_check_interval": "10s",
                        "health_check_timeout": "2s",
                        "consecutive_5xx": 5,
                        "base_ejection_time": "30s",
                        "max_ejection_percent": 50,
                    },
                    "pricing": {
                        "currency": "USD",
                        "prompt_per_1m": 0.1,
                        "cached_input_per_1m": 0.025,
                        "cache_write_per_1m": 0.05,
                        "completion_per_1m": 0.4,
                    },
                    "backend_refs": [
                        {
                            "name": "primary",
                            "endpoint": "model.internal:8000",
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
                }
            ],
            "decisions": [
                {
                    "name": "Simple",
                    "description": "Handle ordinary work.",
                    "priority": 100,
                    "rules": {"operator": "AND", "conditions": []},
                    "modelRefs": [{"model": "local/fast"}],
                }
            ],
        },
        "global": {
            "router": {
                "auto_model_names": ["vllm-sr/blend", "blend"],
                "config_source": "file",
                "skip_processing": {"enabled": False},
            }
        },
    }


def test_rewrite_applies_the_approved_v03_breaks() -> None:
    result = migrate_v03_config_data(previous_v03_config())
    document = result.document

    assert document["version"] == "v0.3"
    assert "models" not in document
    assert (
        document["providers"]["models"][0]["backend_refs"][0]["api_key_env"]
        == "FAST_API_KEY"
    )
    assert document["routing"]["modelCards"][0]["name"] == "local/fast"
    assert document["routing"]["decisions"][0]["modelRefs"] == [{"model": "local/fast"}]
    assert document["global"]["router"]["auto_model_names"] == [
        "vllm-sr/blend",
        "blend",
    ]
    assert "config_source" not in document["global"]["router"]
    assert "skip_processing" not in document["global"]["router"]

    model = document["providers"]["models"][0]
    assert "reliability" not in model
    assert model["control"] == {
        "retry": {"count": 2, "on": ["unavailable", "timeout"]},
    }
    assert model["pricing"] == {
        "input_cost_per_million_tokens": "0.1",
        "output_cost_per_million_tokens": "0.4",
        "cache_read_cost_per_million_tokens": "0.025",
        "cache_write_cost_per_million_tokens": "0.05",
    }
    assert document["global"]["billing"] == {"currency": "USD"}
    assert result.summary.models == 1
    assert result.summary.pricing_blocks == 1
    assert result.summary.control_blocks == 1
    assert result.summary.removed_noop_fields == 9

    parsed = UserConfig.model_validate(document)
    assert validate_user_config(parsed, log_summary=False) == []


def test_rewrite_rejects_status_only_retry_evidence() -> None:
    source = previous_v03_config()
    source["providers"]["models"][0]["reliability"]["retry_on"] = "5xx"

    with pytest.raises(ConfigMigrationError) as raised:
        migrate_v03_config_data(source)

    assert any(issue.code == "unsafe_retry_trigger" for issue in raised.value.issues)


def test_rewrite_preserves_recipes_entrypoints_and_model_refs() -> None:
    source = previous_v03_config()
    source["routing"]["decisions"] = []
    source["recipes"] = [
        {
            "name": "private",
            "description": "Private route",
            "routing": {
                "decisions": [
                    {
                        "name": "Private",
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
            "model_names": ["vllm-sr/private", "private"],
            "recipe": "private",
        }
    ]

    document = migrate_v03_config_data(source).document

    assert document["recipes"] == source["recipes"]
    assert document["entrypoints"] == source["entrypoints"]


@pytest.mark.parametrize("version", ["v0.2", " v0.3 ", None])
def test_rewrite_accepts_exactly_v03(version: str | None) -> None:
    source = previous_v03_config()
    source["version"] = version

    with pytest.raises(ConfigMigrationError) as raised:
        migrate_v03_config_data(source)

    assert raised.value.issues[0].code == "unsupported_source_version"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("listeners", [{"api_keys": ["consumer-secret"]}]),
        ("authz", {"default": "allow"}),
        ("ratelimit", {"rpm": 12}),
    ],
)
def test_rewrite_fails_closed_for_nonempty_static_consumer_access(
    path: str, value: object
) -> None:
    source = previous_v03_config()
    if path == "listeners":
        source["listeners"] = value
    else:
        source["global"][path] = value

    with pytest.raises(ConfigMigrationError) as raised:
        migrate_v03_config_data(source)

    assert any(issue.code.startswith("removed_static") for issue in raised.value.issues)


def test_rewrite_removes_empty_static_consumer_access_fields() -> None:
    source = previous_v03_config()
    source["listeners"][0]["api_keys"] = []
    source["global"]["authz"] = {}

    document = migrate_v03_config_data(source).document

    assert "api_keys" not in document["listeners"][0]
    assert "authz" not in document["global"]


@pytest.mark.parametrize(
    ("field_name", "value", "issue_code"),
    [
        (
            "config_source",
            "kubernetes",
            "removed_kubernetes_config_source",
        ),
        (
            "skip_processing",
            {"enabled": True},
            "removed_enabled_skip_processing",
        ),
    ],
)
def test_rewrite_rejects_removed_behavior_bearing_router_fields(
    field_name: str, value: object, issue_code: str
) -> None:
    source = previous_v03_config()
    source["global"]["router"][field_name] = value

    with pytest.raises(ConfigMigrationError) as raised:
        migrate_v03_config_data(source)

    assert any(issue.code == issue_code for issue in raised.value.issues)


@pytest.mark.parametrize(
    "value",
    ["yes", {"enabled": "false"}, {"enabled": False, "header": "x-bypass"}],
)
def test_rewrite_rejects_ambiguous_skip_processing(value: object) -> None:
    source = previous_v03_config()
    source["global"]["router"]["skip_processing"] = value

    with pytest.raises(ConfigMigrationError) as raised:
        migrate_v03_config_data(source)

    assert any(issue.code == "invalid_skip_processing" for issue in raised.value.issues)


def test_rewrite_preserves_supported_literal_provider_api_key() -> None:
    source = previous_v03_config()
    backend = source["providers"]["models"][0]["backend_refs"][0]
    backend.pop("api_key_env")
    backend["api_key"] = "provider-secret"

    document = migrate_v03_config_data(source).document

    assert (
        document["providers"]["models"][0]["backend_refs"][0]["api_key"]
        == "provider-secret"
    )


def test_rewrite_rejects_plaintext_backend_secret_headers() -> None:
    source = previous_v03_config()
    source["providers"]["models"][0]["backend_refs"][0]["extra_headers"] = {
        "Authorization": "Bearer provider-secret"
    }

    with pytest.raises(ConfigMigrationError) as raised:
        migrate_v03_config_data(source)

    assert any(issue.code == "plaintext_secret" for issue in raised.value.issues)


def test_rewrite_preserves_management_token_environment_references() -> None:
    source = previous_v03_config()
    source["global"]["services"] = {
        "management_api": {
            "auth": {
                "mode": "bearer",
                "tokens": [{"env": "MGMT_TOKEN", "role": "admin"}],
            }
        }
    }

    document = migrate_v03_config_data(source).document

    assert document["global"]["services"]["management_api"]["auth"]["tokens"] == [
        {"env": "MGMT_TOKEN", "role": "admin"}
    ]


def test_runtime_parser_accepts_current_v03_and_rejects_previous_fields(
    tmp_path,
) -> None:
    previous_path = tmp_path / "previous.yaml"
    previous_path.write_text(
        yaml.safe_dump(previous_v03_config(), sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(ConfigParseError, match="reliability"):
        parse_user_config(str(previous_path), log_summary=False)

    current = migrate_v03_config_data(previous_v03_config()).document
    current_path = tmp_path / "current.yaml"
    current_path.write_text(yaml.safe_dump(current, sort_keys=False), encoding="utf-8")
    parsed = parse_user_config(str(current_path), log_summary=False)
    assert parsed.version == "v0.3"


def test_cli_migrate_writes_distinct_current_v03_output() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        source_path = Path("config.yaml")
        original = yaml.safe_dump(previous_v03_config(), sort_keys=False)
        source_path.write_text(original, encoding="utf-8")

        result = runner.invoke(
            main, ["config", "migrate", "--config", str(source_path)]
        )

        assert result.exit_code == 0, result.output
        output_path = Path("config.migrated.yaml")
        assert output_path.is_file()
        assert source_path.read_text(encoding="utf-8") == original
        assert (
            yaml.safe_load(output_path.read_text(encoding="utf-8"))["version"] == "v0.3"
        )


def test_cli_migrate_requires_force_only_for_existing_output() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        source_path = Path("source.yaml")
        output_path = Path("next.yaml")
        source_path.write_text(
            yaml.safe_dump(previous_v03_config(), sort_keys=False),
            encoding="utf-8",
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
        assert (
            yaml.safe_load(output_path.read_text(encoding="utf-8"))["version"] == "v0.3"
        )


def test_cli_migrate_blocks_source_overwrite_even_with_force() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        source_path = Path("source.yaml")
        original = yaml.safe_dump(previous_v03_config(), sort_keys=False)
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
        assert source_path.read_text(encoding="utf-8") == original


def test_input_fixture_is_not_mutated() -> None:
    source = previous_v03_config()
    before = deepcopy(source)

    migrate_v03_config_data(source)

    assert source == before
