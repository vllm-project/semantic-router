from __future__ import annotations

import importlib

import pytest
import yaml
from cli.bootstrap import BootstrapResult
from cli.runtime_management_config import (
    _configured_management_port,
    _configured_management_readiness_token_env,
)
from click.testing import CliRunner

runtime_commands = importlib.import_module("cli.commands.runtime")
main = importlib.import_module("cli.main").main


def _management_config(*, mode: str, tokens: object = None) -> dict:
    auth = {"mode": mode}
    if tokens is not None:
        auth["tokens"] = tokens
    return {
        "global": {
            "services": {
                "management_api": {"port": 9090, "auth": auth},
            }
        }
    }


def test_management_config_defaults_and_explicit_port():
    assert _configured_management_port({}) == 8080
    assert _configured_management_port(_management_config(mode="disabled")) == 9090


def test_management_readiness_omits_auth_when_disabled():
    config = _management_config(mode="disabled")

    assert _configured_management_readiness_token_env(config, {}) is None


def test_management_readiness_selects_first_available_bearer_env():
    config = _management_config(
        mode="bearer",
        tokens=[
            {"env": "MISSING", "role": "viewer"},
            {"env": "CATALOG_MANAGEMENT_TOKEN", "role": "operator"},
        ],
    )

    assert (
        _configured_management_readiness_token_env(
            config, {"CATALOG_MANAGEMENT_TOKEN": "secret-value"}
        )
        == "CATALOG_MANAGEMENT_TOKEN"
    )


def test_management_readiness_rejects_missing_bearer_value():
    config = _management_config(
        mode="bearer", tokens=[{"env": "MISSING", "role": "viewer"}]
    )

    with pytest.raises(ValueError, match=r"available token with ready\.read"):
        _configured_management_readiness_token_env(config, {})


@pytest.mark.parametrize(
    "environment_name", ["lowercase_token", "HOME", "PATH", "VLLM_SR_PLATFORM"]
)
def test_management_readiness_rejects_noncanonical_or_reserved_env_name(
    environment_name: str,
):
    config = _management_config(
        mode="bearer", tokens=[{"env": environment_name, "role": "viewer"}]
    )

    with pytest.raises(ValueError, match="env name is invalid"):
        _configured_management_readiness_token_env(
            config, {environment_name: "secret-value"}
        )


def test_management_readiness_validates_non_readiness_token_env_names():
    config = _management_config(
        mode="bearer",
        tokens=[
            {"env": "lowercase_writer", "role": "writer"},
            {"env": "READY", "role": "readiness"},
        ],
    )
    config["global"]["services"]["management_api"]["auth"]["roles"] = {
        "writer": ["config.write"],
        "readiness": ["ready.read"],
    }

    with pytest.raises(ValueError, match="env name is invalid"):
        _configured_management_readiness_token_env(config, {"READY": "ready-value"})


@pytest.mark.parametrize("value", ["line-one\nline-two", "line-one\rline-two"])
def test_management_readiness_rejects_multiline_bearer(value: str):
    config = _management_config(
        mode="bearer", tokens=[{"env": "TOKEN", "role": "viewer"}]
    )

    with pytest.raises(ValueError, match="single line"):
        _configured_management_readiness_token_env(config, {"TOKEN": value})


def test_management_readiness_skips_token_without_ready_permission():
    config = _management_config(
        mode="bearer",
        tokens=[
            {"env": "WRITE_ONLY", "role": "writer"},
            {"env": "READY", "role": "readiness"},
        ],
    )
    config["global"]["services"]["management_api"]["auth"]["roles"] = {
        "writer": ["config.write"],
        "readiness": ["ready.read"],
    }

    assert (
        _configured_management_readiness_token_env(
            config, {"WRITE_ONLY": "wrong-token", "READY": "right-token"}
        )
        == "READY"
    )


def test_management_readiness_rejects_token_without_ready_permission():
    config = _management_config(
        mode="bearer", tokens=[{"env": "TOKEN", "role": "writer"}]
    )
    config["global"]["services"]["management_api"]["auth"]["roles"] = {
        "writer": ["config.write"]
    }

    with pytest.raises(ValueError, match=r"available token with ready\.read"):
        _configured_management_readiness_token_env(config, {"TOKEN": "secret-value"})


@pytest.mark.parametrize("permission", [" ready.read ", " * "])
def test_management_readiness_matches_router_permission_exactly(permission: str):
    config = _management_config(
        mode="bearer", tokens=[{"env": "TOKEN", "role": "readiness"}]
    )
    config["global"]["services"]["management_api"]["auth"]["roles"] = {
        "readiness": [permission]
    }

    with pytest.raises(ValueError, match=r"available token with ready\.read"):
        _configured_management_readiness_token_env(config, {"TOKEN": "secret-value"})


def test_management_readiness_matches_router_auth_mode_exactly():
    config = _management_config(
        mode=" bearer ", tokens=[{"env": "TOKEN", "role": "viewer"}]
    )

    with pytest.raises(ValueError, match="auth mode must be disabled or bearer"):
        _configured_management_readiness_token_env(config, {"TOKEN": "secret-value"})


@pytest.mark.parametrize("mode", [False, 0, [], {}])
def test_management_readiness_rejects_falsey_non_string_auth_mode(mode: object):
    config = _management_config(mode=mode, tokens=[{"env": "TOKEN", "role": "viewer"}])

    with pytest.raises(ValueError, match="auth mode must be disabled or bearer"):
        _configured_management_readiness_token_env(config, {"TOKEN": "secret-value"})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("tokens", {}, "auth tokens must be a list"),
        ("roles", [], "auth roles must be a mapping"),
    ],
)
def test_management_readiness_rejects_falsey_invalid_auth_shapes(
    field: str, value: object, message: str
):
    config = _management_config(
        mode="bearer", tokens=[{"env": "TOKEN", "role": "viewer"}]
    )
    config["global"]["services"]["management_api"]["auth"][field] = value

    with pytest.raises(ValueError, match=message):
        _configured_management_readiness_token_env(config, {"TOKEN": "secret-value"})


def test_management_readiness_uses_router_last_role_for_duplicate_token_value():
    config = _management_config(
        mode="bearer",
        tokens=[
            {"env": "READY", "role": "readiness"},
            {"env": "WRITER", "role": "writer"},
        ],
    )
    config["global"]["services"]["management_api"]["auth"]["roles"] = {
        "readiness": ["ready.read"],
        "writer": ["config.write"],
    }

    with pytest.raises(ValueError, match=r"available token with ready\.read"):
        _configured_management_readiness_token_env(
            config, {"READY": "same-value", "WRITER": "same-value"}
        )


def test_management_readiness_accepts_duplicate_value_when_last_role_can_read():
    config = _management_config(
        mode="bearer",
        tokens=[
            {"env": "WRITER", "role": "writer"},
            {"env": "READY", "role": "readiness"},
        ],
    )
    config["global"]["services"]["management_api"]["auth"]["roles"] = {
        "readiness": ["ready.read"],
        "writer": ["config.write"],
    }

    assert (
        _configured_management_readiness_token_env(
            config, {"READY": "same-value", "WRITER": "same-value"}
        )
        == "READY"
    )


def test_custom_serve_forwards_management_readiness_credential(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "listeners": [
                    {"name": "http-8899", "address": "0.0.0.0", "port": 8899}
                ],
                "global": {
                    "services": {
                        "management_api": {
                            "auth": {
                                "mode": "bearer",
                                "tokens": [
                                    {"env": "CUSTOM_MGMT_TOKEN", "role": "viewer"}
                                ],
                            }
                        }
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    class _StubBackend:
        def deploy(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("CUSTOM_MGMT_TOKEN", "never-print-this-value")
    monkeypatch.setattr(
        runtime_commands,
        "ensure_bootstrap_workspace",
        lambda _path: BootstrapResult(
            config_path=config_path,
            output_dir=tmp_path / ".vllm-sr",
            setup_mode=False,
        ),
    )
    monkeypatch.setattr(
        runtime_commands, "_build_backend", lambda *_args, **_kwargs: _StubBackend()
    )

    result = CliRunner().invoke(main, ["serve", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert captured["env_vars"]["CUSTOM_MGMT_TOKEN"] == "never-print-this-value"
    assert "never-print-this-value" not in result.output
