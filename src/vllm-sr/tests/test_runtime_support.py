import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from cli.commands.runtime_management_credentials import (
    management_credential_env_names,
)
from cli.commands.runtime_support import (
    append_passthrough_env_vars,
    apply_runtime_mode_env_vars,
    config_env_references,
    sensitive_env_names,
)
from cli.storage_secrets import POSTGRES_PASSWORD_ENV, REDIS_PASSWORD_ENV


def test_runtime_support_import_does_not_load_optional_cli_dependencies():
    script = """
import sys

import cli.commands.runtime_support

assert "cli.commands.config" not in sys.modules
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
        platform=None,
    )

    assert env_vars["DASHBOARD_READONLY"] == "true"


def test_apply_runtime_mode_env_vars_skips_dashboard_readonly_in_minimal_mode():
    env_vars: dict[str, str] = {}

    apply_runtime_mode_env_vars(
        env_vars=env_vars,
        minimal=True,
        readonly=True,
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


def test_append_passthrough_env_vars_forwards_keys_named_by_trusted_source_config(
    monkeypatch, tmp_path
):
    """Operator-selected source config retains its established passthrough behavior."""
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "global": {
                    "services": {
                        "backend_credentials": {
                            "gemini": {
                                "credential_adapter_id": "bearer",
                                "secret_env": "GEMINI_API_KEY",
                            }
                        }
                    },
                    "stores": {"vector_store": {"password": "${VALKEY_PASSWORD}"}},
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


def test_management_credential_schema_is_sensitive(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "global": {
                    "services": {
                        "management_api": {
                            "auth": {"tokens": [{"env": "CUSTOM_MANAGEMENT_TOKEN"}]}
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    assert management_credential_env_names(config) == {"CUSTOM_MANAGEMENT_TOKEN"}
    assert "CUSTOM_MANAGEMENT_TOKEN" in sensitive_env_names(config)


@pytest.mark.parametrize(
    "environment_name", ["lowercase_token", "HOME", "PATH", "VLLM_SR_PLATFORM"]
)
def test_management_credential_schema_rejects_invalid_env_name(
    tmp_path: Path, environment_name: str
):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "global": {
                    "services": {
                        "management_api": {
                            "auth": {"tokens": [{"env": environment_name}]}
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="env name is invalid"):
        management_credential_env_names(config)


def test_discovered_credential_overrides_static_unmasked_passthrough_rule(
    monkeypatch, tmp_path, caplog
):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "global": {
                    "services": {
                        "backend_credentials": {
                            "router_log": {
                                "credential_adapter_id": "bearer",
                                "secret_env": "SR_LOG_LEVEL",
                            }
                        }
                    }
                }
            }
        ),
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


def test_config_env_references_reads_named_credentials_and_store_refs(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "global": {
                    "services": {
                        "backend_credentials": {
                            "mistral": {
                                "credential_adapter_id": "bearer",
                                "secret_env": "MISTRAL_API_KEY",
                            },
                            "embedding": {
                                "credential_adapter_id": "bearer",
                                "secret_env": "EMBEDDING_API_KEY",
                            },
                        }
                    },
                    "stores": {
                        "access": {
                            "postgres": {"dsn_env": "DATABASE_PASSWORD"},
                            "valkey": {"password_env": "REDIS_AUTH_TOKEN"},
                        }
                    },
                }
            }
        )
    )

    assert config_env_references(config) == {
        "MISTRAL_API_KEY",
        "EMBEDDING_API_KEY",
        "REDIS_AUTH_TOKEN",
        "DATABASE_PASSWORD",
    }
    assert config_env_references(None) == set()
    assert config_env_references(tmp_path / "missing.yaml") == set()


def test_config_env_references_reads_durable_bootstrap_env_fields(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "global": {
                    "stores": {
                        "management": {"postgres": {"dsn_env": "ROUTER_POSTGRES_DSN"}}
                    },
                    "services": {
                        "routing_security": {"hmac_keyring_env": "ROUTER_CONTROL_HMAC"},
                    },
                }
            }
        )
    )

    assert config_env_references(config) == {
        "ROUTER_CONTROL_HMAC",
        "ROUTER_POSTGRES_DSN",
    }
    assert {
        "ROUTER_CONTROL_HMAC",
        "ROUTER_POSTGRES_DSN",
    } <= sensitive_env_names(config)


def test_append_passthrough_env_vars_masks_durable_bootstrap_env_fields(
    monkeypatch, tmp_path, caplog
):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "global": {
                    "services": {
                        "routing_security": {"hmac_keyring_env": "ROUTER_CONTROL_HMAC"}
                    }
                }
            }
        )
    )
    monkeypatch.setenv("ROUTER_CONTROL_HMAC", "never-print-managed-root")

    env_vars: dict[str, str] = {}
    with caplog.at_level("INFO", logger="cli.commands.runtime_support"):
        append_passthrough_env_vars(env_vars, config)

    assert env_vars["ROUTER_CONTROL_HMAC"] == "never-print-managed-root"
    assert "ROUTER_CONTROL_HMAC=***" in caplog.text
    assert "never-print-managed-root" not in caplog.text


def test_config_env_references_excludes_process_identity_vars(tmp_path):
    """A config referencing ${PATH}/${HOME} must not pull host process state into the container."""
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "providers": {
                    "models": [
                        {
                            "name": "mistral",
                            "provider_model_id": "mistral-large",
                            "backend_refs": [
                                {
                                    "provider": "mistral",
                                    "base_url": "https://api.mistral.ai/v1",
                                }
                            ],
                        }
                    ]
                },
                "routing": {
                    "modelCards": [
                        {
                            "name": "mistral",
                            "description": "Installed under ${HOME}, resolved via ${PATH}.",
                        }
                    ]
                },
                "global": {
                    "services": {
                        "backend_credentials": {
                            "mistral": {
                                "credential_adapter_id": "bearer",
                                "secret_env": "MISTRAL_API_KEY",
                            }
                        }
                    }
                },
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
        yaml.safe_dump(
            {
                "global": {
                    "services": {
                        "backend_credentials": {
                            "gemini": {
                                "credential_adapter_id": "bearer",
                                "secret_env": "GEMINI_API_KEY",
                            }
                        }
                    }
                }
            }
        )
    )

    assert "GEMINI_API_KEY" in sensitive_env_names(config)
    assert "HF_TOKEN" in sensitive_env_names(config)
    assert "HF_ENDPOINT" not in sensitive_env_names(config)
    assert "GEMINI_API_KEY" not in sensitive_env_names(None)
    assert {
        POSTGRES_PASSWORD_ENV,
        REDIS_PASSWORD_ENV,
    } <= sensitive_env_names(None)
