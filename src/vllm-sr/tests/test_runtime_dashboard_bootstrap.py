import pytest
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


def test_dashboard_router_public_url_defaults_to_host_envoy_listener(monkeypatch):
    monkeypatch.delenv("DASHBOARD_ROUTER_PUBLIC_URL", raising=False)
    layout = resolve_runtime_stack(stack_name="test", port_offset=100)

    dashboard_env = _build_dashboard_runtime_env(
        common_env={},
        listener_port=8899,
        stack_layout=layout,
    )

    assert dashboard_env["DASHBOARD_ROUTER_PUBLIC_URL"] == "http://localhost:8999"
    assert dashboard_env["TARGET_ENVOY_URL"] == (
        "http://test-vllm-sr-envoy-container:8899"
    )


def test_dashboard_router_public_url_preserves_explicit_origin(monkeypatch):
    monkeypatch.setenv("DASHBOARD_ROUTER_PUBLIC_URL", "https://router.example.test")

    dashboard_env = _build_dashboard_runtime_env(
        common_env={},
        listener_port=8899,
        stack_layout=resolve_runtime_stack(stack_name="test", port_offset=100),
    )

    assert dashboard_env["DASHBOARD_ROUTER_PUBLIC_URL"] == "https://router.example.test"


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


def test_router_managed_bootstrap_stays_interactive_with_legacy_admin_env(monkeypatch):
    monkeypatch.delenv("DASHBOARD_ALLOW_OPEN_BOOTSTRAP", raising=False)
    monkeypatch.setenv("DASHBOARD_ADMIN_EMAIL", "core@vllm-sr.ai")
    monkeypatch.setenv("DASHBOARD_ADMIN_PASSWORD", "not-used-for-managed-bootstrap")
    monkeypatch.setenv(
        "DASHBOARD_ROUTER_BOOTSTRAP_TOKEN_FILE", "/run/secrets/bootstrap/router-token"
    )

    dashboard_env = _build_dashboard_runtime_env(
        common_env={},
        listener_port=8899,
        stack_layout=resolve_runtime_stack(stack_name="test", port_offset=100),
    )

    assert dashboard_env["DASHBOARD_ALLOW_OPEN_BOOTSTRAP"] == "true"


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
