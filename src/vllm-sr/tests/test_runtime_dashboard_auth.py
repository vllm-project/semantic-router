"""Dashboard signing keys survive local replacement without data-plane exposure."""

import subprocess
from types import SimpleNamespace

import pytest
from cli import container_cli, container_start
from cli.commands import runtime_paths
from cli.commands.runtime_support import (
    append_passthrough_env_vars,
    normalize_recipe_env_names,
)


@pytest.mark.parametrize("secret", [None, "", "  ", "stable-test-signing-key"])
def test_local_stack_scopes_jwt_secret_and_hides_its_value(
    tmp_path, monkeypatch, caplog, secret
):
    if secret is None:
        monkeypatch.delenv("DASHBOARD_JWT_SECRET", raising=False)
    else:
        monkeypatch.setenv("DASHBOARD_JWT_SECRET", secret)
    monkeypatch.setenv("VLLM_SR_TOPOLOGY", "split")
    monkeypatch.setattr(runtime_paths, "_current_posix_user_id", lambda: None)
    monkeypatch.setattr(container_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_start,
        "get_runtime_images",
        lambda **_kwargs: {
            "router": "router-image",
            "envoy": "envoy-image",
            "dashboard": "dashboard-image",
        },
    )
    monkeypatch.setattr(
        container_start, "_render_split_envoy_config", lambda *args, **kwargs: None
    )
    docker_bin = tmp_path / "docker"
    docker_bin.write_text("")
    monkeypatch.setattr(
        container_start,
        "resolve_container_cli_path",
        lambda preferred_path=None: str(docker_bin),
    )
    config = tmp_path / "config.yaml"
    config.write_text("version: v0.3\nlisteners: []\n")
    commands = []

    def fake_run(cmd, **_kwargs):
        commands.append(cmd)
        return SimpleNamespace(stdout="container-id\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    # Even an incorrectly supplied common value must not leak to any container.
    env = {"DASHBOARD_JWT_SECRET": "untrusted-common-value"}
    for _ in range(2):
        rc, _, _ = container_cli.container_start_vllm_sr(
            str(config), env, [], network_name="vllm-sr-network", minimal=False
        )
        assert rc == 0
    creation_commands = [cmd for cmd in commands if "--name" in cmd]
    assert len(creation_commands) == 8
    for cmd in creation_commands:
        name = cmd[cmd.index("--name") + 1]
        values = [cmd[index + 1] for index, item in enumerate(cmd[:-1]) if item == "-e"]
        if name == "vllm-sr-dashboard-container" and secret and secret.strip():
            assert "DASHBOARD_JWT_SECRET" in values
        else:
            assert not any(value.startswith("DASHBOARD_JWT_SECRET") for value in values)
    rendered = repr(commands) + caplog.text
    assert "untrusted-common-value" not in rendered
    assert "stable-test-signing-key" not in rendered


def test_dashboard_jwt_secret_cannot_become_recipe_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHBOARD_JWT_SECRET", "stable-test-signing-key")
    config = tmp_path / "config.yaml"
    config.write_text("value: ${DASHBOARD_JWT_SECRET}\n")
    env = {}
    append_passthrough_env_vars(env, config)
    assert "DASHBOARD_JWT_SECRET" not in env
    with pytest.raises(ValueError, match="Invalid Recipe environment binding"):
        normalize_recipe_env_names(["DASHBOARD_JWT_SECRET"])
