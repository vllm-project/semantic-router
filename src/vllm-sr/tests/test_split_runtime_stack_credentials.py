"""Where this stack's storage credentials may and may not appear.

Split out of ``test_split_runtime_stack``: every assertion here is a negative
one about exposure -- the values must reach the Router child process
environment and nothing else, never an argv list, never the Dashboard container
that holds the runtime socket -- so the tests carry their own command-and-env
capture rather than the plain command capture the sibling module uses.
"""

import subprocess
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
        "version: v0.1\nlisteners:\n  - name: http-8899\n"
        "    address: 0.0.0.0\n    port: 8899\n"
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

    monkeypatch.setattr(subprocess, "run", fake_run)
    return captured


def _commands_by_container(captured):
    """Index the creating command of each container.

    Router also needs a `network connect` and a `start`, neither of which names
    a container with `--name`; they carry no environment of their own and are
    not what these assertions are about.
    """
    return {
        cmd[cmd.index("--name") + 1]: (cmd, env)
        for cmd, env in captured
        if "--name" in cmd
    }


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
    captured = _capture_run_commands_with_env(monkeypatch)

    rc, _, _ = container_cli.container_start_vllm_sr(
        str(config_path),
        {},
        [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        state_root_dir=str(tmp_path),
        minimal=False,
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
    # Every other container inherits this process's environment untouched.
    assert dashboard_env is None
    assert envoy_env is None
    for cmd, _ in captured:
        assert secrets.postgres.password not in cmd
        assert secrets.redis.password not in cmd


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
