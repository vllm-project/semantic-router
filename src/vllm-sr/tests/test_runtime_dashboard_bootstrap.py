import pytest

from cli import container_start
from cli.commands.runtime_support import append_passthrough_env_vars
from cli.container_start import _build_dashboard_runtime_env
from cli.runtime_stack import resolve_runtime_stack


def test_dashboard_open_bootstrap_defaults_true_without_admin(monkeypatch):
    monkeypatch.delenv("DASHBOARD_ALLOW_OPEN_BOOTSTRAP", raising=False)
    monkeypatch.delenv("DASHBOARD_ADMIN_EMAIL", raising=False)
    monkeypatch.delenv("DASHBOARD_ADMIN_PASSWORD", raising=False)

    dashboard_env = _build_dashboard_runtime_env(
        common_env={},
        listener_port=8899,
        stack_layout=resolve_runtime_stack(stack_name="test", port_offset=100),
    )

    assert dashboard_env["DASHBOARD_ALLOW_OPEN_BOOTSTRAP"] == "true"
    assert dashboard_env["OPENCLAW_ENABLED"] == "false"
    assert dashboard_env["ML_PIPELINE_ENABLED"] == "false"


def test_dashboard_bootstrap_admin_is_scoped_to_dashboard(monkeypatch):
    monkeypatch.delenv("DASHBOARD_ALLOW_OPEN_BOOTSTRAP", raising=False)
    monkeypatch.setenv("DASHBOARD_ADMIN_EMAIL", "core@vllm-sr.ai")
    monkeypatch.setenv("DASHBOARD_ADMIN_PASSWORD", "core")
    monkeypatch.setenv("DASHBOARD_ADMIN_NAME", "Core")

    env_vars: dict[str, str] = {}
    append_passthrough_env_vars(env_vars)

    assert "DASHBOARD_ADMIN_EMAIL" not in env_vars
    assert "DASHBOARD_ADMIN_PASSWORD" not in env_vars
    assert "DASHBOARD_ADMIN_NAME" not in env_vars

    dashboard_env = _build_dashboard_runtime_env(
        common_env=env_vars,
        listener_port=8899,
        stack_layout=resolve_runtime_stack(stack_name="test", port_offset=100),
    )
    assert dashboard_env["DASHBOARD_ADMIN_EMAIL"] == "core@vllm-sr.ai"
    assert dashboard_env["DASHBOARD_ADMIN_PASSWORD"] == "core"
    assert dashboard_env["DASHBOARD_ADMIN_NAME"] == "Core"
    assert "DASHBOARD_ALLOW_OPEN_BOOTSTRAP" not in dashboard_env


def test_dashboard_open_bootstrap_respects_explicit_true(monkeypatch):
    monkeypatch.setenv("DASHBOARD_ALLOW_OPEN_BOOTSTRAP", "true")
    monkeypatch.setenv("DASHBOARD_ADMIN_EMAIL", "admin@example.com")
    monkeypatch.setenv("DASHBOARD_ADMIN_PASSWORD", "secret")

    dashboard_env = _build_dashboard_runtime_env(
        common_env={},
        listener_port=8899,
        stack_layout=resolve_runtime_stack(stack_name="test", port_offset=100),
    )

    assert dashboard_env["DASHBOARD_ALLOW_OPEN_BOOTSTRAP"] == "true"


def test_dashboard_open_bootstrap_respects_explicit_false(monkeypatch):
    monkeypatch.setenv("DASHBOARD_ALLOW_OPEN_BOOTSTRAP", "false")
    monkeypatch.delenv("DASHBOARD_ADMIN_EMAIL", raising=False)
    monkeypatch.delenv("DASHBOARD_ADMIN_PASSWORD", raising=False)

    dashboard_env = _build_dashboard_runtime_env(
        common_env={},
        listener_port=8899,
        stack_layout=resolve_runtime_stack(stack_name="test", port_offset=100),
    )

    assert dashboard_env["DASHBOARD_ALLOW_OPEN_BOOTSTRAP"] == "false"


@pytest.mark.parametrize(
    ("admin_env_name", "admin_env_value"),
    [
        ("DASHBOARD_ADMIN_EMAIL", "admin@example.com"),
        ("DASHBOARD_ADMIN_PASSWORD", "secret"),
    ],
)
def test_dashboard_open_bootstrap_defaults_true_with_partial_admin(
    monkeypatch, admin_env_name: str, admin_env_value: str
):
    monkeypatch.delenv("DASHBOARD_ALLOW_OPEN_BOOTSTRAP", raising=False)
    monkeypatch.delenv("DASHBOARD_ADMIN_EMAIL", raising=False)
    monkeypatch.delenv("DASHBOARD_ADMIN_PASSWORD", raising=False)
    monkeypatch.setenv(admin_env_name, admin_env_value)

    dashboard_env = _build_dashboard_runtime_env(
        common_env={},
        listener_port=8899,
        stack_layout=resolve_runtime_stack(stack_name="test", port_offset=100),
    )

    assert dashboard_env["DASHBOARD_ALLOW_OPEN_BOOTSTRAP"] == "true"


@pytest.mark.parametrize(
    ("host_bind", "expected"),
    [(None, "0.0.0.0"), ("127.0.0.1", "127.0.0.1")],
)
def test_dashboard_docker_published_address(monkeypatch, host_bind, expected):
    if host_bind is None:
        monkeypatch.delenv("VLLM_SR_DASHBOARD_HOST_BIND", raising=False)
    else:
        monkeypatch.setenv("VLLM_SR_DASHBOARD_HOST_BIND", host_bind)
    monkeypatch.setattr(container_start, "_runtime_mount_specs", lambda *a, **kw: [])
    monkeypatch.setattr(container_start, "_active_recipe_mount_specs", lambda *a: [])
    monkeypatch.setattr(
        container_start,
        "_build_dashboard_runtime_env",
        lambda **kw: {"OPENCLAW_ENABLED": "false"},
    )
    monkeypatch.setattr(
        container_start, "_build_service_run_command", lambda **kw: kw
    )
    stack = resolve_runtime_stack(stack_name="dashboard-bind-test", port_offset=100)
    spec = container_start._build_dashboard_runtime_command(
        runtime="docker",
        dashboard_image="test-dashboard",
        nofile_limit=1024,
        runtime_network_name="test-network",
        common_env={},
        config_dir="/tmp/test-config",
        listener_port=8899,
        openclaw_network_name=None,
        runtime_paths={
            "log_spool_dashboard_mount": "/tmp/dashboard-log:/app/logs",
            "log_spool_root": "/tmp/logs",
            "active_recipe_root": "",
            "runtime_container_config": "/app/config.yaml",
            "container_recipe_store_dir": "/app/recipes",
            "log_spool_gid": "1000",
        },
        stack_layout=stack,
        inherited_sensitive_env=set(),
        management_listener={"port": 8080},
    )
    assert spec["port_mappings"] == [(expected, stack.dashboard_port, 8700)]


def test_dashboard_docker_rejects_invalid_published_address(monkeypatch):
    monkeypatch.setenv("VLLM_SR_DASHBOARD_HOST_BIND", "192.0.2.1")
    with pytest.raises(ValueError, match="VLLM_SR_DASHBOARD_HOST_BIND"):
        container_start._dashboard_host_bind_address()
