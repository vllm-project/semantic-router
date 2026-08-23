"""Where this stack's storage credentials may and may not appear.

Split out of ``test_split_runtime_stack``: every assertion here is a negative
one about exposure -- the values must reach the Router child process
environment and nothing else, never an argv list, never the Dashboard container
that holds the runtime socket -- so the tests carry their own command-and-env
capture rather than the plain command capture the sibling module uses.
"""

from types import SimpleNamespace

import pytest
from cli import container_cli, container_start, storage_secrets
from cli.runtime_stack import resolve_runtime_stack
from cli.storage_secrets import (
    POSTGRES_PASSWORD_ENV,
    REDIS_PASSWORD_ENV,
    STORAGE_SECRET_ENV_NAMES,
)


@pytest.fixture(autouse=True)
def _split_runtime_topology(monkeypatch):
    monkeypatch.setenv("VLLM_SR_TOPOLOGY", "split")


def _stub_valid_container_cli(monkeypatch, tmp_path):
    docker_bin = tmp_path / "docker"
    docker_bin.write_text("")
    monkeypatch.setattr(
        container_start,
        "resolve_container_cli_path",
        lambda preferred_path=None: str(docker_bin),
    )
    return docker_bin


def _minimal_stack_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.4\nlisteners:\n  - name: http-8899\n"
        "    address: 0.0.0.0\n    port: 8899\n"
        "global:\n  services:\n    backend_egress:\n"
        "      policy_file: /app/config/backend-egress-policy.yaml\n"
        "  runtime_refs:\n"
        "    provider_env: PROVIDER_API_KEY\n"
        "    management_env: MANAGEMENT_TOKEN\n"
    )
    return config_path


def _stub_runtime_images(monkeypatch):
    monkeypatch.setattr(container_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_start,
        "get_runtime_images",
        lambda **kwargs: {
            "router": "test-image",
            "envoy": "test-image",
            "dashboard": "test-image",
        },
    )
    monkeypatch.setattr(
        container_start, "_render_split_envoy_config", lambda *args, **kwargs: None
    )


def _capture_run_commands_with_env(monkeypatch):
    captured = []

    def fake_run(cmd, capture_output, text, check, env=None):
        captured.append((cmd, env))
        return SimpleNamespace(stdout="container-id\n", stderr="")

    monkeypatch.setattr(container_start.subprocess, "run", fake_run)
    return captured


def _commands_by_container(captured):
    return {cmd[cmd.index("--name") + 1]: (cmd, env) for cmd, env in captured}


def test_container_start_vllm_sr_gives_storage_credentials_to_router_alone(
    tmp_path, monkeypatch
):
    config_path = _minimal_stack_config(tmp_path)
    _stub_runtime_images(monkeypatch)
    _stub_valid_container_cli(monkeypatch, tmp_path)
    secrets = storage_secrets.ensure_storage_secrets(
        tmp_path,
        stack_layout=resolve_runtime_stack(),
        volumes=storage_secrets.StorageVolumes(postgres="pg-data", redis="redis-data"),
    )
    provider_secret = "provider-secret-canary"
    management_secret = "management-secret-canary"
    dashboard_secret = "dashboard-secret-canary"
    monkeypatch.setenv("DASHBOARD_JWT_SECRET", dashboard_secret)
    captured = _capture_run_commands_with_env(monkeypatch)

    rc, _, _ = container_cli.container_start_vllm_sr(
        str(config_path),
        {
            "PROVIDER_API_KEY": provider_secret,
            "MANAGEMENT_TOKEN": management_secret,
        },
        [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        state_root_dir=str(tmp_path),
        minimal=False,
        router_child_env=storage_secrets.storage_secret_env(secrets),
    )

    assert rc == 0
    commands = _commands_by_container(captured)
    router_cmd, router_env = commands["vllm-sr-router-container"]
    dashboard_cmd, dashboard_env = commands["vllm-sr-dashboard-container"]
    envoy_cmd, envoy_env = commands["vllm-sr-envoy-container"]

    for name in STORAGE_SECRET_ENV_NAMES:
        # Inherited form: the name alone, never `NAME=value`.
        assert name in router_cmd
        assert not any(str(item).startswith(f"{name}=") for item in router_cmd)
        assert name not in dashboard_cmd
        assert name not in envoy_cmd

    assert router_env[POSTGRES_PASSWORD_ENV] == secrets.postgres.password
    assert router_env[REDIS_PASSWORD_ENV] == secrets.redis.password
    assert router_env["PROVIDER_API_KEY"] == provider_secret
    assert router_env["MANAGEMENT_TOKEN"] == management_secret
    assert "DASHBOARD_JWT_SECRET" not in router_env

    assert "DASHBOARD_JWT_SECRET" in dashboard_cmd
    assert dashboard_env["DASHBOARD_JWT_SECRET"] == dashboard_secret
    assert "PROVIDER_API_KEY" not in dashboard_cmd
    assert "MANAGEMENT_TOKEN" not in dashboard_cmd
    assert POSTGRES_PASSWORD_ENV not in dashboard_cmd
    assert REDIS_PASSWORD_ENV not in dashboard_cmd
    assert "PROVIDER_API_KEY" not in dashboard_env
    assert "MANAGEMENT_TOKEN" not in dashboard_env
    assert POSTGRES_PASSWORD_ENV not in dashboard_env
    assert REDIS_PASSWORD_ENV not in dashboard_env

    for name in (
        "PROVIDER_API_KEY",
        "MANAGEMENT_TOKEN",
        "DASHBOARD_JWT_SECRET",
        *STORAGE_SECRET_ENV_NAMES,
    ):
        assert name not in envoy_cmd
        assert name not in envoy_env
    for cmd, _ in captured:
        assert secrets.postgres.password not in cmd
        assert secrets.redis.password not in cmd
        assert provider_secret not in cmd
        assert management_secret not in cmd
        assert dashboard_secret not in cmd


def test_container_start_vllm_sr_omits_storage_credentials_without_state(
    tmp_path, monkeypatch
):
    config_path = _minimal_stack_config(tmp_path)
    _stub_runtime_images(monkeypatch)
    _stub_valid_container_cli(monkeypatch, tmp_path)
    captured = _capture_run_commands_with_env(monkeypatch)

    rc, _, _ = container_cli.container_start_vllm_sr(
        str(config_path),
        {},
        [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        state_root_dir=str(tmp_path),
        minimal=False,
    )

    assert rc == 0
    for cmd, env in captured:
        assert env is None
        for name in STORAGE_SECRET_ENV_NAMES:
            assert name not in cmd
