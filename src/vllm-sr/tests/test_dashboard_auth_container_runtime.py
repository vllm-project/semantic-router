from types import SimpleNamespace

import pytest
from cli import container_cli, container_start
from cli.k8s_backend import K8sBackend


@pytest.fixture(autouse=True)
def _split_runtime(monkeypatch):
    monkeypatch.setenv("VLLM_SR_TOPOLOGY", "split")


def _write_config(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "version: v0.1\nlisteners:\n"
        "  - name: http-8899\n"
        "    address: 0.0.0.0\n"
        "    port: 8899\n"
    )
    return config


def _stub_runtime(monkeypatch, tmp_path, *, dashboard=True):
    images = {"router": "router-image", "envoy": "envoy-image"}
    if dashboard:
        images["dashboard"] = "dashboard-image"
    monkeypatch.setattr(container_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(container_start, "get_runtime_images", lambda **_kwargs: images)
    monkeypatch.setattr(
        container_start, "_render_split_envoy_config", lambda *_args, **_kwargs: None
    )
    docker_bin = tmp_path / "docker"
    docker_bin.write_text("")
    monkeypatch.setattr(
        container_start,
        "resolve_container_cli_path",
        lambda preferred_path=None: str(docker_bin),
    )


def _capture_runs(monkeypatch):
    captured = []

    def fake_run(cmd, **kwargs):
        captured.append((cmd, kwargs.get("env")))
        return SimpleNamespace(stdout="container-id\n", stderr="")

    monkeypatch.setattr(container_start.subprocess, "run", fake_run)
    return captured


def _service_run(captured, service):
    expected_name = f"vllm-sr-{service}-container"
    for command, process_env in captured:
        if (
            "--name" in command
            and command[command.index("--name") + 1] == expected_name
        ):
            return command, process_env
    raise AssertionError(f"missing {service} command")


def test_dashboard_auth_is_name_inherited_isolated_and_readonly_mounted(
    tmp_path, monkeypatch, caplog
):
    config = _write_config(tmp_path)
    blocklist = tmp_path / "passwords.txt"
    blocklist.write_text("known-compromised-password\n")
    jwt_secret = "jwt-secret-value"
    admin_password = "admin-password-value"
    _stub_runtime(monkeypatch, tmp_path)
    captured = _capture_runs(monkeypatch)

    with caplog.at_level("DEBUG"):
        result = container_cli.container_start_vllm_sr(
            str(config),
            {
                "DASHBOARD_JWT_SECRET": jwt_secret,
                "DASHBOARD_JWT_EXPIRY_HOURS": "12",
                "DASHBOARD_ADMIN_EMAIL": "admin@example.com",
                "DASHBOARD_ADMIN_PASSWORD": admin_password,
                "DASHBOARD_ADMIN_NAME": "Local Admin",
                "DASHBOARD_ALLOW_OPEN_BOOTSTRAP": "false",
                "DASHBOARD_PASSWORD_BLOCKLIST_PATH": str(blocklist),
            },
            [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
            minimal=False,
        )

    assert result[0] == 0
    router_cmd, router_env = _service_run(captured, "router")
    envoy_cmd, envoy_env = _service_run(captured, "envoy")
    dashboard_cmd, dashboard_env = _service_run(captured, "dashboard")
    for command, process_env in ((router_cmd, router_env), (envoy_cmd, envoy_env)):
        joined = " ".join(command)
        assert "DASHBOARD_JWT_SECRET" not in joined
        assert "DASHBOARD_ADMIN_PASSWORD" not in joined
        assert "DASHBOARD_ADMIN_EMAIL" not in joined
        assert "DASHBOARD_JWT_SECRET" not in process_env
        assert "DASHBOARD_ADMIN_PASSWORD" not in process_env

    assert "DASHBOARD_JWT_SECRET" in dashboard_cmd
    assert f"DASHBOARD_JWT_SECRET={jwt_secret}" not in dashboard_cmd
    assert "DASHBOARD_ADMIN_PASSWORD" in dashboard_cmd
    assert f"DASHBOARD_ADMIN_PASSWORD={admin_password}" not in dashboard_cmd
    assert "DASHBOARD_ADMIN_EMAIL=admin@example.com" in dashboard_cmd
    assert "DASHBOARD_JWT_EXPIRY_HOURS=12" in dashboard_cmd
    assert (
        f"{blocklist.resolve()}:/etc/vllm-sr/dashboard-auth/"
        "password-blocklist.txt:ro,z"
    ) in dashboard_cmd
    assert (
        "DASHBOARD_PASSWORD_BLOCKLIST_PATH="
        "/etc/vllm-sr/dashboard-auth/password-blocklist.txt"
    ) in dashboard_cmd
    assert dashboard_env["DASHBOARD_JWT_SECRET"] == jwt_secret
    assert dashboard_env["DASHBOARD_ADMIN_PASSWORD"] == admin_password
    assert jwt_secret not in caplog.text
    assert admin_password not in caplog.text


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_direct_start_rejects_blocklist_before_path_mutation(
    tmp_path, monkeypatch, kind
):
    config = _write_config(tmp_path)
    blocklist = tmp_path / kind
    if kind == "directory":
        blocklist.mkdir()
    state_root = tmp_path / "state"
    monkeypatch.setattr(container_start, "get_container_runtime", lambda: "docker")

    with pytest.raises(ValueError, match="DASHBOARD_PASSWORD_BLOCKLIST_PATH"):
        container_cli.container_start_vllm_sr(
            str(config),
            {"DASHBOARD_PASSWORD_BLOCKLIST_PATH": str(blocklist)},
            [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
            state_root_dir=str(state_root),
            minimal=False,
        )

    assert not state_root.exists()


def test_minimal_mode_ignores_dashboard_auth_and_invalid_blocklist(
    tmp_path, monkeypatch
):
    config = _write_config(tmp_path)
    _stub_runtime(monkeypatch, tmp_path, dashboard=False)
    captured = _capture_runs(monkeypatch)

    result = container_cli.container_start_vllm_sr(
        str(config),
        {
            "DASHBOARD_PASSWORD_BLOCKLIST_PATH": str(tmp_path / "missing"),
            "DASHBOARD_ADMIN_PASSWORD": "must-not-propagate",
        },
        [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        minimal=True,
    )

    assert result[0] == 0
    _service_run(captured, "router")
    _service_run(captured, "envoy")
    with pytest.raises(AssertionError, match="missing dashboard"):
        _service_run(captured, "dashboard")


def test_k8s_deploy_drops_local_dashboard_auth_before_secret_planning(monkeypatch):
    backend = K8sBackend(
        namespace="test",
        context=None,
        release_name="test",
        chart_dir="/unused",
    )
    captured = {}
    monkeypatch.setattr(backend, "_require_tool", lambda _name: None)
    monkeypatch.setattr("cli.k8s_backend.print_vllm_logo", lambda: None)

    def capture_and_stop(env_vars):
        captured.update(env_vars)
        raise RuntimeError("stop after sanitized planning input")

    monkeypatch.setattr(backend, "_plan_env_secret", capture_and_stop)

    with pytest.raises(RuntimeError, match="sanitized planning input"):
        backend.deploy(
            "/unused/config.yaml",
            env_vars={
                "DASHBOARD_JWT_SECRET": "jwt-secret-value",
                "DASHBOARD_ADMIN_PASSWORD": "admin-password-value",
                "SR_LOG_LEVEL": "info",
            },
        )

    assert captured == {"SR_LOG_LEVEL": "info"}
