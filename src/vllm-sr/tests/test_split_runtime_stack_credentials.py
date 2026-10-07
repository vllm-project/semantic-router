"""Where this stack's credentials may and may not appear.

Split out of ``test_split_runtime_stack``: every assertion here is a negative
one about exposure -- the storage credentials must reach the Router child
process environment and nothing else, the management credential the Router's
and the Dashboard's, and neither an argv list or a log record -- so the tests
carry their own command-and-env capture rather than the plain command capture
the sibling module uses.
"""

import logging
import os
import subprocess
from types import SimpleNamespace

import pytest
from cli import container_cli, container_start, storage_secrets
from cli.commands.runtime_paths import DASHBOARD_STATE_GID
from cli.management_credential import (
    management_credential_path,
    stack_management_credential,
)
from cli.recipe_topology_contract import MANAGEMENT_CREDENTIAL_ENV
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
    return docker_bin


def _minimal_stack_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.3\nlisteners:\n  - name: http-8899\n"
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
    # Dashboard gets only the independent benchmark service token.
    assert "SR_BENCH_TOKEN" in dashboard_env
    assert not set(STORAGE_SECRET_ENV_NAMES) & set(dashboard_env)
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
        if env is not None:
            assert not set(STORAGE_SECRET_ENV_NAMES) & set(env)
        for name in STORAGE_SECRET_ENV_NAMES:
            assert name not in cmd


def _bearer_stack_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.3\nlisteners:\n  - name: http-8899\n"
        "    address: 0.0.0.0\n    port: 8899\n"
        "global:\n  services:\n    management_api:\n"
        "      bind_address: 0.0.0.0\n      port: 8080\n      auth:\n"
        "        mode: bearer\n        tokens:\n"
        f"          - env: {MANAGEMENT_CREDENTIAL_ENV}\n"
        "            role: dashboard_control_plane\n"
        "        roles:\n          dashboard_control_plane:\n"
        "            - config.read\n"
    )
    return config_path


def _start_stack(config_path, tmp_path, *, minimal=False):
    return container_cli.container_start_vllm_sr(
        str(config_path),
        {},
        [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        state_root_dir=str(tmp_path),
        minimal=minimal,
    )


def _passes_by_name(cmd, name):
    """The inherited form: the name alone, never `NAME=value`."""

    return name in cmd and not any(str(item).startswith(f"{name}=") for item in cmd)


def test_the_management_credential_reaches_the_dashboard_and_a_router_that_binds_it(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.delenv(MANAGEMENT_CREDENTIAL_ENV, raising=False)
    config_path = _bearer_stack_config(tmp_path)
    _stub_runtime_images(monkeypatch)
    captured = _capture_run_commands_with_env(monkeypatch)
    caplog.set_level(logging.DEBUG)

    rc, _, _ = _start_stack(config_path, tmp_path)

    assert rc == 0
    token = stack_management_credential(tmp_path, stack_layout=resolve_runtime_stack())
    commands = _commands_by_container(captured)
    for container in ("vllm-sr-router-container", "vllm-sr-dashboard-container"):
        cmd, env = commands[container]
        assert _passes_by_name(cmd, MANAGEMENT_CREDENTIAL_ENV)
        assert env[MANAGEMENT_CREDENTIAL_ENV] == token
    envoy_cmd, envoy_env = commands["vllm-sr-envoy-container"]
    assert MANAGEMENT_CREDENTIAL_ENV not in envoy_cmd
    assert envoy_env is None
    for cmd, _ in captured:
        assert token not in " ".join(cmd)
    assert token not in caplog.text
    for path in (tmp_path / ".vllm-sr").rglob("*"):
        if path.is_file() and path != management_credential_path(
            tmp_path, stack_layout=resolve_runtime_stack()
        ):
            assert token not in path.read_text(errors="replace"), path


def test_a_router_that_binds_no_credential_gets_none(tmp_path, monkeypatch):
    monkeypatch.delenv(MANAGEMENT_CREDENTIAL_ENV, raising=False)
    config_path = _minimal_stack_config(tmp_path)
    _stub_runtime_images(monkeypatch)
    captured = _capture_run_commands_with_env(monkeypatch)

    rc, _, _ = _start_stack(config_path, tmp_path)

    assert rc == 0
    commands = _commands_by_container(captured)
    router_cmd, router_env = commands["vllm-sr-router-container"]
    assert MANAGEMENT_CREDENTIAL_ENV not in router_cmd
    assert router_env is None
    dashboard_cmd, dashboard_env = commands["vllm-sr-dashboard-container"]
    assert _passes_by_name(dashboard_cmd, MANAGEMENT_CREDENTIAL_ENV)
    assert len(dashboard_env[MANAGEMENT_CREDENTIAL_ENV]) == 64


def test_the_operator_credential_reaches_both_containers_by_name(tmp_path, monkeypatch):
    operator_token = "f" * 64
    monkeypatch.setenv(MANAGEMENT_CREDENTIAL_ENV, operator_token)
    config_path = _bearer_stack_config(tmp_path)
    _stub_runtime_images(monkeypatch)
    captured = _capture_run_commands_with_env(monkeypatch)

    rc, _, _ = _start_stack(config_path, tmp_path)

    assert rc == 0
    commands = _commands_by_container(captured)
    for container in ("vllm-sr-router-container", "vllm-sr-dashboard-container"):
        cmd, env = commands[container]
        assert _passes_by_name(cmd, MANAGEMENT_CREDENTIAL_ENV)
        assert env[MANAGEMENT_CREDENTIAL_ENV] == operator_token
    assert not (tmp_path / ".vllm-sr" / "management-credential").exists()


def test_a_stack_without_a_consumer_stores_no_management_credential(
    tmp_path, monkeypatch
):
    monkeypatch.delenv(MANAGEMENT_CREDENTIAL_ENV, raising=False)
    config_path = _minimal_stack_config(tmp_path)
    _stub_runtime_images(monkeypatch)
    captured = _capture_run_commands_with_env(monkeypatch)

    rc, _, _ = _start_stack(config_path, tmp_path, minimal=True)

    assert rc == 0
    assert not any(MANAGEMENT_CREDENTIAL_ENV in cmd for cmd, _ in captured)
    assert not (tmp_path / ".vllm-sr" / "management-credential").exists()


def test_the_dashboard_shares_the_recipe_store_with_the_cli_users_group(
    tmp_path, monkeypatch
):
    config_path = _minimal_stack_config(tmp_path)
    _stub_runtime_images(monkeypatch)
    captured = _capture_run_commands_with_env(monkeypatch)

    rc, _, _ = _start_stack(config_path, tmp_path)

    assert rc == 0
    dashboard_cmd, _ = _commands_by_container(captured)["vllm-sr-dashboard-container"]
    expected = DASHBOARD_STATE_GID if os.getgid() == 0 else os.getgid()
    assert f"VLLM_SR_RECIPE_STORE_GID={expected}" in dashboard_cmd
