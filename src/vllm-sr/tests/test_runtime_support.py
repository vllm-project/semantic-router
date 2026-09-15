import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from cli.commands.runtime_support import (
    append_passthrough_env_vars,
    apply_runtime_mode_env_vars,
    config_env_references,
    configure_recipe_env_bindings,
    required_config_env_references,
    sensitive_env_names,
    validate_config_recipe_env_bindings,
)


def test_runtime_support_import_does_not_load_optional_cli_dependencies():
    script = """
import sys

import cli.commands.runtime_support

assert "cli.commands.config" not in sys.modules
assert "cli.commands.model" not in sys.modules
assert "jinja2" not in sys.modules
assert "requests" not in sys.modules
"""

    subprocess.run([sys.executable, "-c", script], check=True)


def test_apply_runtime_mode_env_vars_sets_dashboard_readonly_when_requested():
    env_vars: dict[str, str] = {}

    apply_runtime_mode_env_vars(
        env_vars=env_vars,
        minimal=False,
        readonly=True,
        setup_mode=False,
        platform=None,
    )

    assert env_vars["DASHBOARD_READONLY"] == "true"


def test_apply_runtime_mode_env_vars_skips_dashboard_readonly_in_minimal_mode():
    env_vars: dict[str, str] = {}

    apply_runtime_mode_env_vars(
        env_vars=env_vars,
        minimal=True,
        readonly=True,
        setup_mode=False,
        platform=None,
    )

    assert env_vars["DISABLE_DASHBOARD"] == "true"
    assert "DASHBOARD_READONLY" not in env_vars


def test_apply_runtime_mode_env_vars_sets_router_log_level_when_requested():
    env_vars: dict[str, str] = {}

    apply_runtime_mode_env_vars(
        env_vars=env_vars,
        minimal=False,
        readonly=False,
        setup_mode=False,
        platform=None,
        log_level="DEBUG",
    )

    assert env_vars["SR_LOG_LEVEL"] == "debug"


def test_append_passthrough_env_vars_includes_router_logging_settings(monkeypatch):
    monkeypatch.setenv("SR_LOG_LEVEL", "debug")
    monkeypatch.setenv("SR_LOG_ENCODING", "console")

    env_vars: dict[str, str] = {}
    append_passthrough_env_vars(env_vars)

    assert env_vars["SR_LOG_LEVEL"] == "debug"
    assert env_vars["SR_LOG_ENCODING"] == "console"


def test_append_passthrough_env_vars_includes_envoy_log_level_without_changing_router_level(
    monkeypatch,
):
    monkeypatch.setenv("SR_LOG_LEVEL", "debug")
    monkeypatch.setenv("VLLM_SR_ENVOY_LOG_LEVEL", "WARNING")

    env_vars: dict[str, str] = {}
    append_passthrough_env_vars(env_vars)

    assert env_vars["SR_LOG_LEVEL"] == "debug"
    assert env_vars["VLLM_SR_ENVOY_LOG_LEVEL"] == "WARNING"


def test_append_passthrough_env_vars_forwards_keys_named_by_trusted_source_config(
    monkeypatch, tmp_path
):
    """Operator-selected source config retains its established passthrough behavior."""
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "providers": {
                    "models": [{"name": "gemini", "api_key_env": "GEMINI_API_KEY"}]
                },
                "global": {
                    "stores": {"vector_store": {"password": "${VALKEY_PASSWORD}"}}
                },
            }
        )
    )
    monkeypatch.setenv("GEMINI_API_KEY", "gk-test")
    monkeypatch.setenv("VALKEY_PASSWORD", "vp-test")
    monkeypatch.delenv("UNREFERENCED_API_KEY", raising=False)

    env_vars: dict[str, str] = {}
    append_passthrough_env_vars(env_vars, config)

    assert env_vars["GEMINI_API_KEY"] == "gk-test"
    assert env_vars["VALKEY_PASSWORD"] == "vp-test"
    assert "UNREFERENCED_API_KEY" not in env_vars


def test_append_passthrough_env_vars_masks_management_credential(
    monkeypatch, tmp_path, caplog
):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "global": {
                    "services": {
                        "management_api": {
                            "auth": {
                                "tokens": [
                                    {"env": "CUSTOM_MGMT_TOKEN", "role": "viewer"}
                                ]
                            }
                        }
                    }
                }
            }
        )
    )
    monkeypatch.setenv("CUSTOM_MGMT_TOKEN", "never-print-this-value")

    env_vars: dict[str, str] = {}
    with caplog.at_level("INFO", logger="cli.commands.runtime_support"):
        append_passthrough_env_vars(env_vars, config)

    assert env_vars["CUSTOM_MGMT_TOKEN"] == "never-print-this-value"
    assert "CUSTOM_MGMT_TOKEN=***" in caplog.text
    assert "never-print-this-value" not in caplog.text


@pytest.mark.parametrize(
    "credential_config",
    [
        {"api_key_env": "SR_LOG_LEVEL"},
        {"api_key": "${SR_LOG_LEVEL}"},
    ],
)
def test_discovered_credential_overrides_static_unmasked_passthrough_rule(
    monkeypatch, tmp_path, caplog, credential_config
):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump({"providers": {"models": [credential_config]}}),
        encoding="utf-8",
    )
    canary = "credential-log-canary"
    monkeypatch.setenv("SR_LOG_LEVEL", canary)

    env_vars: dict[str, str] = {}
    with caplog.at_level("INFO", logger="cli.commands.runtime_support"):
        append_passthrough_env_vars(env_vars, config)

    assert env_vars["SR_LOG_LEVEL"] == canary
    assert "SR_LOG_LEVEL=***" in caplog.text
    assert canary not in caplog.text


def test_config_env_references_reads_api_key_env_and_interpolations(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "providers": {"models": [{"api_key_env": "MISTRAL_API_KEY"}]},
                "embedding_models": {"endpoint": {"api_key_env": "EMBEDDING_API_KEY"}},
                "note": "uses ${REDIS_AUTH_TOKEN} at runtime",
                "bare": "postgres://$DATABASE_PASSWORD@db",
                "fallback": "${OPTIONAL_TOKEN:-development}",
            }
        )
    )

    assert config_env_references(config) == {
        "MISTRAL_API_KEY",
        "EMBEDDING_API_KEY",
        "REDIS_AUTH_TOKEN",
        "OPTIONAL_TOKEN",
    }
    assert config_env_references(None) == set()
    assert config_env_references(tmp_path / "missing.yaml") == set()


def test_required_config_env_references_includes_every_router_consulted_name(
    tmp_path: Path,
):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "provider": {"api_key_env": "PROVIDER_API_KEY"},
                "database": "postgres://$DATABASE_PASSWORD@db",
                "cache": "${CACHE_PASSWORD}",
                "optional": "${OPTIONAL_TOKEN:-development}",
                "unset_only": "${UNSET_ONLY_TOKEN-development}",
                "lowercase": "$lowercase_token",
                "escaped": "$$NOT_CONSULTED",
            }
        )
    )

    assert required_config_env_references(config) == {
        "PROVIDER_API_KEY",
        "DATABASE_PASSWORD",
        "CACHE_PASSWORD",
        "OPTIONAL_TOKEN",
        "UNSET_ONLY_TOKEN",
        "lowercase_token",
    }


def test_required_package_env_references_fail_closed_on_denied_host_names(
    tmp_path: Path,
):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "provider": {"api_key_env": "HOME"},
                "path": "${PATH}",
            }
        )
    )

    assert required_config_env_references(config) == {"HOME", "PATH"}
    with pytest.raises(ValueError, match="invalid Recipe environment binding"):
        validate_config_recipe_env_bindings(config, [])


@pytest.mark.parametrize(
    "reference",
    [
        "${lowercase_token}",
        "${lowercase-token}",
        "${PATH:-/usr/bin}",
        "${VLLM_SR_RECIPE_STORE_DIR}",
        "${VLLM_SR_MANAGED_STORAGE_BACKENDS-default}",
        "$VLLM_SR_STACK_NAME",
        "${VLLM_SR_ACTIVE_RECIPE_DIR}",
    ],
)
def test_required_package_env_references_reject_invalid_or_reserved_names(
    tmp_path: Path, reference: str
):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"value": reference}))

    with pytest.raises(ValueError, match="invalid Recipe environment binding"):
        validate_config_recipe_env_bindings(config, [])


def test_config_env_references_excludes_process_identity_vars(tmp_path):
    """A config referencing ${PATH}/${HOME} must not pull host process state into the container."""
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "note": "installed under ${HOME}/.cache, resolved via ${PATH}",
                "providers": {"models": [{"api_key_env": "MISTRAL_API_KEY"}]},
            }
        )
    )

    refs = config_env_references(config)
    assert refs == {"MISTRAL_API_KEY"}


def test_config_env_references_excludes_cli_controlled_override_vars(tmp_path):
    """A config that happens to mention a CLI override var must not preempt the CLI's own default."""
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump({"note": "see ${DISABLE_DASHBOARD} and ${DASHBOARD_PLATFORM}"})
    )

    assert config_env_references(config) == set()


def test_append_passthrough_env_vars_does_not_forward_process_identity_vars(
    monkeypatch, tmp_path
):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"note": "runs from ${PATH}"}))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    env_vars: dict[str, str] = {}
    append_passthrough_env_vars(env_vars, config)

    assert "PATH" not in env_vars


def test_sensitive_env_names_covers_config_named_credentials(tmp_path):
    """A key the config names must be treated as a secret, not inlined into a manifest."""
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump({"providers": {"models": [{"api_key_env": "GEMINI_API_KEY"}]}})
    )

    assert "GEMINI_API_KEY" in sensitive_env_names(config)
    assert "HF_TOKEN" in sensitive_env_names(config)
    assert "HF_ENDPOINT" not in sensitive_env_names(config)
    assert "GEMINI_API_KEY" not in sensitive_env_names(None)


def test_configure_recipe_env_bindings_requires_explicit_names_and_masks_values(
    monkeypatch, caplog
):
    monkeypatch.setenv("GEMINI_API_KEY", "gk-test")
    monkeypatch.setenv("VALKEY_PASSWORD", "vp-test")
    env_vars: dict[str, str] = {}

    names = configure_recipe_env_bindings(
        env_vars, ["VALKEY_PASSWORD", "GEMINI_API_KEY", "GEMINI_API_KEY"]
    )

    assert names == ("GEMINI_API_KEY", "VALKEY_PASSWORD")
    assert env_vars["VLLM_SR_RECIPE_ENV_ALLOWLIST"] == (
        "GEMINI_API_KEY,VALKEY_PASSWORD"
    )
    assert env_vars["GEMINI_API_KEY"] == "gk-test"
    assert env_vars["VALKEY_PASSWORD"] == "vp-test"
    assert "gk-test" not in caplog.text
    assert "vp-test" not in caplog.text


def test_configure_recipe_env_bindings_rejects_values_and_missing_host_env(
    monkeypatch,
):
    monkeypatch.delenv("MISSING_API_KEY", raising=False)
    with pytest.raises(ValueError, match="without NAME=value"):
        configure_recipe_env_bindings({}, ["API_KEY=secret"])
    with pytest.raises(ValueError, match="no non-empty host value"):
        configure_recipe_env_bindings({}, ["MISSING_API_KEY"])


def test_configure_recipe_env_bindings_accepts_names_only_env_allowlist(monkeypatch):
    monkeypatch.setenv("VLLM_SR_RECIPE_ENV_ALLOWLIST", "SECOND_API_KEY,FIRST_API_KEY")
    monkeypatch.setenv("FIRST_API_KEY", "first")
    monkeypatch.setenv("SECOND_API_KEY", "second")
    env_vars: dict[str, str] = {}

    names = configure_recipe_env_bindings(env_vars, [])

    assert names == ("FIRST_API_KEY", "SECOND_API_KEY")
    assert env_vars["VLLM_SR_RECIPE_ENV_ALLOWLIST"] == ("FIRST_API_KEY,SECOND_API_KEY")


def test_validate_config_recipe_env_bindings_fails_closed(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "providers": {
                    "models": [{"name": "gemini", "api_key_env": "GEMINI_API_KEY"}]
                },
                "database": "postgres://$DATABASE_PASSWORD@db",
                "optional": "${OPTIONAL_TOKEN:-development}",
            }
        )
    )

    with pytest.raises(ValueError, match="--recipe-env GEMINI_API_KEY"):
        validate_config_recipe_env_bindings(config, ["DATABASE_PASSWORD"])

    with pytest.raises(ValueError, match="--recipe-env OPTIONAL_TOKEN"):
        validate_config_recipe_env_bindings(
            config, ["DATABASE_PASSWORD", "GEMINI_API_KEY"]
        )

    validate_config_recipe_env_bindings(
        config, ["DATABASE_PASSWORD", "GEMINI_API_KEY", "OPTIONAL_TOKEN"]
    )


@pytest.mark.parametrize(
    "reference", ["${OPTIONAL_TOKEN:-development}", "${OPTIONAL_TOKEN-development}"]
)
def test_package_restart_fallback_reference_requires_explicit_allowlist(
    tmp_path: Path, reference: str
):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"optional": reference}))

    with pytest.raises(ValueError, match="--recipe-env OPTIONAL_TOKEN"):
        validate_config_recipe_env_bindings(config, [])

    validate_config_recipe_env_bindings(config, ["OPTIONAL_TOKEN"])
