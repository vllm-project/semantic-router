"""Tests for per-stack Grafana admin credentials and the Grafana container wiring."""

import stat
from pathlib import Path

from cli import container_support_services, runtime_lifecycle
from cli import grafana_credentials as gc
from cli.main import main
from cli.runtime_stack import resolve_runtime_stack
from click.testing import CliRunner


def _file_mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _monkeypatch_grafana_container(monkeypatch, captured):
    monkeypatch.setattr(
        container_support_services, "get_container_runtime", lambda: "docker"
    )
    monkeypatch.setattr(
        container_support_services, "_replace_existing_container", lambda _name: None
    )
    monkeypatch.setattr(
        container_support_services, "_render_template_copy", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        container_support_services,
        "_run_service_start",
        lambda cmd, _label: captured.update(cmd=cmd) or (0, "", ""),
    )
    # Each test owns the env: some need the explicit value, others need it unset.


def test_fresh_stack_generates_and_reuses_a_private_password_file(
    monkeypatch, tmp_path: Path
):
    monkeypatch.delenv(gc.GRAFANA_ADMIN_PASSWORD_ENV, raising=False)
    layout = resolve_runtime_stack()
    path = gc.grafana_password_path(tmp_path, stack_layout=layout)

    first = gc.resolve_grafana_admin_password(tmp_path, stack_layout=layout)
    second = gc.resolve_grafana_admin_password(tmp_path, stack_layout=layout)

    assert len(first) >= 40
    assert first == second, "a restart must keep the same credential"
    assert path.exists()
    assert _file_mode(path.parent) == 0o700
    assert _file_mode(path) == 0o600
    assert path.read_text(encoding="utf-8") == first
    assert not path.read_bytes().endswith(b"\n")


def test_explicit_env_password_is_materialized_for_the_container(
    monkeypatch, tmp_path: Path
):
    explicit = "operator-provided-password"
    monkeypatch.setenv(gc.GRAFANA_ADMIN_PASSWORD_ENV, explicit)
    layout = resolve_runtime_stack()

    active_path = gc.ensure_grafana_admin_password_file(tmp_path, stack_layout=layout)

    assert gc.resolve_grafana_admin_password(tmp_path, stack_layout=layout) == explicit
    # Explicit value must be on disk so bind-mounting never hits a missing source.
    assert active_path == gc.grafana_explicit_password_path(
        tmp_path, stack_layout=layout
    )
    assert active_path.read_text(encoding="utf-8") == explicit
    assert _file_mode(active_path) == 0o600
    assert not active_path.read_bytes().endswith(b"\n")
    assert not gc.grafana_password_path(tmp_path, stack_layout=layout).exists()


def test_explicit_env_password_survives_an_idempotent_restart(
    monkeypatch, tmp_path: Path
):
    explicit = "stable-operator-password"
    monkeypatch.setenv(gc.GRAFANA_ADMIN_PASSWORD_ENV, explicit)
    layout = resolve_runtime_stack()

    first = gc.resolve_grafana_admin_password(tmp_path, stack_layout=layout)
    second = gc.resolve_grafana_admin_password(tmp_path, stack_layout=layout)

    assert first == explicit
    assert second == explicit


def test_unsetting_explicit_env_falls_back_to_a_new_generated_secret(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv(gc.GRAFANA_ADMIN_PASSWORD_ENV, "operator-password")
    layout = resolve_runtime_stack()

    explicit_path = gc.ensure_grafana_admin_password_file(tmp_path, stack_layout=layout)
    explicit_value = gc.resolve_grafana_admin_password(tmp_path, stack_layout=layout)

    # Removal: unset the env; the next serve falls back to a persisted file.
    monkeypatch.delenv(gc.GRAFANA_ADMIN_PASSWORD_ENV, raising=False)
    persisted_path = gc.ensure_grafana_admin_password_file(
        tmp_path, stack_layout=layout
    )
    persisted_value = gc.resolve_grafana_admin_password(tmp_path, stack_layout=layout)

    assert persisted_path == gc.grafana_password_path(tmp_path, stack_layout=layout)
    assert persisted_path.exists()
    assert persisted_value != explicit_value
    assert explicit_path.exists()  # stale explicit file is ignored, not deleted.


def test_each_stack_gets_its_own_password_file(monkeypatch, tmp_path: Path):
    monkeypatch.delenv(gc.GRAFANA_ADMIN_PASSWORD_ENV, raising=False)
    default_layout = resolve_runtime_stack()
    custom_layout = resolve_runtime_stack(stack_name="team-b", port_offset=100)

    default_path = gc.grafana_password_path(tmp_path, stack_layout=default_layout)
    custom_path = gc.grafana_password_path(tmp_path, stack_layout=custom_layout)

    assert default_path != custom_path
    default_pw = gc.resolve_grafana_admin_password(
        tmp_path, stack_layout=default_layout
    )
    custom_pw = gc.resolve_grafana_admin_password(tmp_path, stack_layout=custom_layout)
    assert default_pw != custom_pw


def test_grafana_container_reads_password_from_secret_file_never_argv(
    monkeypatch, tmp_path: Path
):
    monkeypatch.delenv(gc.GRAFANA_ADMIN_PASSWORD_ENV, raising=False)
    captured: dict[str, object] = {}
    _monkeypatch_grafana_container(monkeypatch, captured)
    layout = resolve_runtime_stack()

    container_support_services.container_start_grafana(
        "test-network", str(tmp_path), stack_layout=layout
    )

    command = list(captured["cmd"])
    password = gc.resolve_grafana_admin_password(tmp_path, stack_layout=layout)

    assert (
        f"{gc.GRAFANA_ADMIN_PASSWORD_FILE_ENV}=" f"{gc.CONTAINER_GRAFANA_PASSWORD_PATH}"
    ) in command
    assert (
        f"{tmp_path}/.vllm-sr/grafana-credentials/admin-password:"
        f"{gc.CONTAINER_GRAFANA_PASSWORD_PATH}:ro,z"
    ) in command
    assert "GF_SECURITY_ADMIN_PASSWORD=admin" not in command
    assert password not in command
    assert not any(arg.startswith("GF_SECURITY_ADMIN_PASSWORD=") for arg in command)


def test_grafana_container_mounts_the_explicit_password_file(monkeypatch, tmp_path):
    # Regression: override path must mount a real file, value kept out of argv.
    explicit = "operator-supplied-password"
    monkeypatch.setenv(gc.GRAFANA_ADMIN_PASSWORD_ENV, explicit)
    captured: dict[str, object] = {}
    _monkeypatch_grafana_container(monkeypatch, captured)
    layout = resolve_runtime_stack()

    container_support_services.container_start_grafana(
        "test-network", str(tmp_path), stack_layout=layout
    )

    command = list(captured["cmd"])
    explicit_path = gc.grafana_explicit_password_path(tmp_path, stack_layout=layout)
    assert explicit_path.is_file()
    assert explicit_path.read_text(encoding="utf-8") == explicit
    assert (
        f"{gc.GRAFANA_ADMIN_PASSWORD_FILE_ENV}=" f"{gc.CONTAINER_GRAFANA_PASSWORD_PATH}"
    ) in command
    assert f"{explicit_path}:{gc.CONTAINER_GRAFANA_PASSWORD_PATH}:ro,z" in command
    assert "GF_SECURITY_ADMIN_PASSWORD=admin" not in command
    assert explicit not in command
    assert not any(arg.startswith("GF_SECURITY_ADMIN_PASSWORD=") for arg in command)


def test_runtime_summary_no_longer_advertises_a_fixed_admin_password(capsys):
    stack_layout = resolve_runtime_stack(stack_name="terminal-test", port_offset=200)

    runtime_lifecycle.log_runtime_summary(
        [{"name": "http-8899", "port": 8899}],
        stack_layout,
        dashboard_disabled=False,
        enable_observability=True,
        started_backends={"postgres", "redis"},
    )

    captured = capsys.readouterr()
    assert stack_layout.grafana_url in captured.out
    assert "(admin/admin)" not in captured.out


def test_main_cli_help_does_not_mention_admin_password(monkeypatch):
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "admin/admin" not in result.output
