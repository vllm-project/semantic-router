"""Tests for per-stack Grafana admin credentials and the Grafana container wiring."""

import configparser
import os
import stat
from pathlib import Path

from cli import container_support_services, runtime_lifecycle
from cli import grafana_credentials as gc
from cli.main import main
from cli.runtime_stack import resolve_runtime_stack
from click.testing import CliRunner

GRAFANA_SERVE_INI_TEMPLATE = (
    Path(gc.__file__).resolve().parent / "templates" / "grafana.serve.ini"
)
CONTAINER_GRAFANA_INI_PATH = "/etc/grafana/grafana.ini"


def _file_mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _password(tmp_path: Path, layout) -> str:
    return gc.ensure_grafana_admin_password_file(
        tmp_path, stack_layout=layout
    ).read_text(encoding="utf-8")


def _parse_grafana_ini(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    read = parser.read(path)
    assert read, f"failed to parse Grafana ini at {path}"
    return parser


def _assert_ini_denies_anonymous_admin(path: Path) -> None:
    """Anonymous users may view dashboards; they must not be org admins."""
    ini = _parse_grafana_ini(path)
    assert ini.getboolean("auth.anonymous", "enabled") is True
    assert ini.get("auth.anonymous", "org_role") == "Viewer"
    assert ini.getboolean("auth.basic", "enabled") is True
    assert ini.getboolean("auth", "disable_login_form") is False
    assert ini.get("security", "cookie_samesite") == "lax"
    assert ini.getboolean("security", "cookie_secure") is False


def _monkeypatch_grafana_container(monkeypatch, captured, *, render_templates=False):
    monkeypatch.setattr(
        container_support_services, "get_container_runtime", lambda: "docker"
    )
    monkeypatch.setattr(
        container_support_services, "_replace_existing_container", lambda _name: None
    )
    if not render_templates:
        monkeypatch.setattr(
            container_support_services, "_render_template_copy", lambda *_a, **_k: None
        )
    monkeypatch.setattr(
        container_support_services,
        "_run_service_start",
        lambda cmd, _label: captured.update(cmd=cmd) or (0, "", ""),
    )


def test_fresh_stack_generates_and_reuses_a_container_readable_password_file(
    monkeypatch, tmp_path: Path
):
    monkeypatch.delenv(gc.GRAFANA_ADMIN_PASSWORD_ENV, raising=False)
    layout = resolve_runtime_stack()
    path = gc.grafana_password_path(tmp_path, stack_layout=layout)

    first = _password(tmp_path, layout)
    second = _password(tmp_path, layout)

    assert len(first) >= 40
    assert first == second, "a restart must keep the same credential"
    assert path.exists()
    assert _file_mode(path.parent) == 0o700
    assert _file_mode(path) == 0o644
    assert path.read_text(encoding="utf-8") == first
    assert not path.read_bytes().endswith(b"\n")


def test_explicit_env_password_is_materialized_for_the_container(
    monkeypatch, tmp_path: Path
):
    explicit = "operator-provided-password"
    monkeypatch.setenv(gc.GRAFANA_ADMIN_PASSWORD_ENV, explicit)
    layout = resolve_runtime_stack()

    path = gc.ensure_grafana_admin_password_file(tmp_path, stack_layout=layout)

    assert path == gc.grafana_password_path(tmp_path, stack_layout=layout)
    assert path.read_text(encoding="utf-8") == explicit
    assert _password(tmp_path, layout) == explicit
    assert _file_mode(path.parent) == 0o700
    assert _file_mode(path) == 0o644
    assert not path.read_bytes().endswith(b"\n")


def test_each_stack_gets_its_own_password_file(monkeypatch, tmp_path: Path):
    monkeypatch.delenv(gc.GRAFANA_ADMIN_PASSWORD_ENV, raising=False)
    default_layout = resolve_runtime_stack()
    custom_layout = resolve_runtime_stack(stack_name="team-b", port_offset=100)

    default_path = gc.grafana_password_path(tmp_path, stack_layout=default_layout)
    custom_path = gc.grafana_password_path(tmp_path, stack_layout=custom_layout)

    assert default_path != custom_path
    assert _password(tmp_path, default_layout) != _password(tmp_path, custom_layout)


def test_grafana_container_reads_password_from_secret_file_never_argv(
    monkeypatch, tmp_path: Path
):
    monkeypatch.delenv(gc.GRAFANA_ADMIN_PASSWORD_ENV, raising=False)
    captured: dict[str, object] = {}
    _monkeypatch_grafana_container(monkeypatch, captured)
    layout = resolve_runtime_stack()
    (tmp_path / ".vllm-sr").mkdir(mode=0o700)

    container_support_services.container_start_grafana(
        "test-network", str(tmp_path), stack_layout=layout
    )

    command = list(captured["cmd"])
    password = _password(tmp_path, layout)
    mounted = gc.grafana_password_path(tmp_path, stack_layout=layout)
    assert _file_mode(mounted) == 0o644
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
    assert not any(arg.startswith("GF_SECURITY_ADMIN_USER=") for arg in command)


def test_grafana_container_mounts_the_explicit_password_file(monkeypatch, tmp_path):
    explicit = "operator-supplied-password"
    monkeypatch.setenv(gc.GRAFANA_ADMIN_PASSWORD_ENV, explicit)
    captured: dict[str, object] = {}
    _monkeypatch_grafana_container(monkeypatch, captured)
    layout = resolve_runtime_stack()
    (tmp_path / ".vllm-sr").mkdir(mode=0o700)

    container_support_services.container_start_grafana(
        "test-network", str(tmp_path), stack_layout=layout
    )

    command = list(captured["cmd"])
    path = gc.grafana_password_path(tmp_path, stack_layout=layout)
    assert path.is_file()
    assert path.read_text(encoding="utf-8") == explicit
    assert _file_mode(path) == 0o644
    assert (
        f"{gc.GRAFANA_ADMIN_PASSWORD_FILE_ENV}=" f"{gc.CONTAINER_GRAFANA_PASSWORD_PATH}"
    ) in command
    assert f"{path}:{gc.CONTAINER_GRAFANA_PASSWORD_PATH}:ro,z" in command
    assert explicit not in command
    assert not any(arg.startswith("GF_SECURITY_ADMIN_PASSWORD=") for arg in command)


def test_shipped_grafana_ini_restricts_anonymous_users_to_viewer():
    _assert_ini_denies_anonymous_admin(GRAFANA_SERVE_INI_TEMPLATE)


def test_grafana_container_renders_and_mounts_ini_without_anonymous_admin(
    monkeypatch, tmp_path: Path
):
    monkeypatch.delenv(gc.GRAFANA_ADMIN_PASSWORD_ENV, raising=False)
    captured: dict[str, object] = {}
    _monkeypatch_grafana_container(monkeypatch, captured, render_templates=True)
    layout = resolve_runtime_stack()
    (tmp_path / ".vllm-sr").mkdir(mode=0o700)

    container_support_services.container_start_grafana(
        "test-network", str(tmp_path), stack_layout=layout
    )

    rendered = tmp_path / ".vllm-sr" / "grafana" / "grafana.serve.ini"
    _assert_ini_denies_anonymous_admin(rendered)
    command = list(captured["cmd"])
    assert f"{os.path.abspath(rendered)}:{CONTAINER_GRAFANA_INI_PATH}:ro" in command
    assert f"127.0.0.1:{layout.grafana_port}:3000" in command


def test_runtime_summary_prints_the_admin_password_file_path_not_the_secret(
    monkeypatch, tmp_path: Path, capsys
):
    explicit = "operator-supplied-password"
    monkeypatch.setenv(gc.GRAFANA_ADMIN_PASSWORD_ENV, explicit)
    stack_layout = resolve_runtime_stack(stack_name="terminal-test", port_offset=200)
    password_file = gc.grafana_password_path(tmp_path, stack_layout=stack_layout)

    runtime_lifecycle.log_runtime_summary(
        [{"name": "http-8899", "port": 8899}],
        stack_layout,
        dashboard_disabled=False,
        enable_observability=True,
        started_backends={"postgres", "redis"},
        state_root_dir=str(tmp_path),
    )

    captured = capsys.readouterr()
    compact = "".join(captured.out.split())
    assert stack_layout.grafana_url in captured.out
    assert "Grafana admin password file" in captured.out
    assert "grafana-credentials" in compact
    assert password_file.name in captured.out
    assert "(admin/admin)" not in captured.out
    assert "generated admin password" not in captured.out
    assert explicit not in captured.out


def test_main_cli_help_does_not_mention_admin_password(monkeypatch):
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "admin/admin" not in result.output
