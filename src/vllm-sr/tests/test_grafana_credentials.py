import os
import stat
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli import container_support_services  # noqa: E402
from cli.commands.runtime_paths import CONTAINER_READABLE_STATE_FILE_MODE  # noqa: E402
from cli.grafana_credentials import (  # noqa: E402
    CONTAINER_GRAFANA_PASSWORD_PATH,
    GRAFANA_ADMIN_PASSWORD_ENV,
    GRAFANA_ADMIN_PASSWORD_FILE_ENV,
    ensure_grafana_admin_password_file,
)
from cli.runtime_stack import resolve_runtime_stack  # noqa: E402


def test_generated_password_is_reused_and_not_world_readable(tmp_path, monkeypatch):
    monkeypatch.delenv(GRAFANA_ADMIN_PASSWORD_ENV, raising=False)
    layout = resolve_runtime_stack()

    first = ensure_grafana_admin_password_file(tmp_path, stack_layout=layout)
    second = ensure_grafana_admin_password_file(tmp_path, stack_layout=layout)

    stored = first.read_bytes()
    assert first == second
    assert stored == second.read_bytes()
    assert stored
    assert b"\n" not in stored
    assert stat.S_IMODE(first.stat().st_mode) == CONTAINER_READABLE_STATE_FILE_MODE
    assert oct(first.parent.stat().st_mode & 0o777) == oct(0o700)


def test_explicit_password_overrides_the_stored_file(tmp_path, monkeypatch):
    layout = resolve_runtime_stack(stack_name="grafana-b", port_offset=20)
    monkeypatch.setenv(GRAFANA_ADMIN_PASSWORD_ENV, "stack-secret")

    path = ensure_grafana_admin_password_file(tmp_path, stack_layout=layout)

    assert path.name == "admin-password.grafana-b"
    assert path.read_text(encoding="utf-8") == "stack-secret"


def test_multiline_explicit_password_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv(GRAFANA_ADMIN_PASSWORD_ENV, "one\ntwo")
    with pytest.raises(ValueError, match="single-line"):
        ensure_grafana_admin_password_file(tmp_path)


def test_grafana_start_mounts_the_password_file_and_rekeys(monkeypatch, tmp_path):
    monkeypatch.delenv(GRAFANA_ADMIN_PASSWORD_ENV, raising=False)
    commands = []
    rekeys = []
    monkeypatch.setattr(
        container_support_services, "get_container_runtime", lambda: "docker"
    )
    monkeypatch.setattr(
        container_support_services, "_replace_existing_container", lambda _name: None
    )
    monkeypatch.setattr(
        container_support_services,
        "_run_service_start",
        lambda cmd, service: commands.append((cmd, service)) or (0, "", ""),
    )
    monkeypatch.setattr(
        container_support_services,
        "rekey_grafana_admin",
        lambda name, runtime=None: rekeys.append((name, runtime)) or (0, "", ""),
    )
    layout = resolve_runtime_stack()

    status = container_support_services.container_start_grafana(
        config_dir=str(tmp_path), stack_layout=layout
    )

    assert status == (0, "", "")
    cmd, service = commands[0]
    assert service == "Grafana"
    joined = " ".join(cmd)
    assert "GF_SECURITY_ADMIN_PASSWORD=admin" not in joined
    assert "GF_SECURITY_ADMIN_USER=admin" not in joined
    assert f"{GRAFANA_ADMIN_PASSWORD_FILE_ENV}={CONTAINER_GRAFANA_PASSWORD_PATH}" in cmd
    mount = next(
        item for item in cmd if item.endswith(f":{CONTAINER_GRAFANA_PASSWORD_PATH}:ro")
    )
    host_path = mount.split(":", 1)[0]
    assert Path(host_path).is_file()
    assert rekeys == [(layout.grafana_container_name, "docker")]


def test_failed_rekey_stops_the_container(monkeypatch, tmp_path):
    stopped = []
    monkeypatch.setattr(
        container_support_services, "get_container_runtime", lambda: "docker"
    )
    monkeypatch.setattr(
        container_support_services, "_replace_existing_container", lambda _name: None
    )
    monkeypatch.setattr(
        container_support_services,
        "_run_service_start",
        lambda _cmd, _service: (0, "", ""),
    )
    monkeypatch.setattr(
        container_support_services,
        "rekey_grafana_admin",
        lambda *_args, **_kwargs: (1, "", "reset failed"),
    )
    monkeypatch.setattr(
        container_support_services,
        "container_stop_container",
        stopped.append,
    )
    layout = resolve_runtime_stack()

    status = container_support_services.container_start_grafana(
        config_dir=str(tmp_path), stack_layout=layout
    )

    assert status[0] == 1
    assert "reset failed" in status[2]
    assert stopped == [layout.grafana_container_name]
    assert os.environ.get(GRAFANA_ADMIN_PASSWORD_ENV, "admin") != status[2]
