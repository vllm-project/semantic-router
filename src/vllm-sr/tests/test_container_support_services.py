import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli import container_support_services  # noqa: E402
from cli.runtime_stack import resolve_runtime_stack  # noqa: E402


def _stub_container_start(monkeypatch):
    commands = []
    replaced = []
    monkeypatch.setattr(
        container_support_services, "get_container_runtime", lambda: "docker"
    )
    monkeypatch.setattr(
        container_support_services,
        "_replace_existing_container",
        replaced.append,
    )
    monkeypatch.setattr(
        container_support_services,
        "_run_service_start",
        lambda cmd, service: commands.append((cmd, service)) or (0, "", ""),
    )
    return commands, replaced


def test_container_start_jaeger_uses_pinned_image(monkeypatch):
    commands = []
    monkeypatch.setattr(
        container_support_services, "get_container_runtime", lambda: "docker"
    )
    monkeypatch.setattr(
        container_support_services, "_replace_existing_container", lambda name: None
    )
    monkeypatch.setattr(
        container_support_services,
        "_run_service_start",
        lambda cmd, service: commands.append(cmd) or True,
    )

    container_support_services.container_start_jaeger(
        stack_layout=resolve_runtime_stack()
    )

    (cmd,) = commands
    assert "docker.io/jaegertracing/all-in-one:1.76.0" in cmd


def test_container_start_grafana_renders_live_allowed_origins(monkeypatch, tmp_path):
    monkeypatch.setenv(
        container_support_services.GRAFANA_LIVE_ALLOWED_ORIGINS_ENV,
        " https://dashboard.example.com, https://*.example.net ",
    )
    commands, replaced = _stub_container_start(monkeypatch)
    stack_layout = resolve_runtime_stack()

    container_support_services.container_start_grafana(
        config_dir=str(tmp_path), stack_layout=stack_layout
    )

    config = (tmp_path / ".vllm-sr" / "grafana" / "grafana.serve.ini").read_text(
        encoding="utf-8"
    )
    assert (
        "allowed_origins = https://dashboard.example.com, https://*.example.net"
        in config
    )
    assert "__GF_LIVE_ALLOWED_ORIGINS__" not in config
    assert replaced == [stack_layout.grafana_container_name]
    assert commands[0][1] == "Grafana"


def test_container_start_grafana_keeps_default_live_origin_behavior(
    monkeypatch, tmp_path
):
    monkeypatch.delenv(
        container_support_services.GRAFANA_LIVE_ALLOWED_ORIGINS_ENV, raising=False
    )
    _stub_container_start(monkeypatch)

    container_support_services.container_start_grafana(config_dir=str(tmp_path))

    config = (tmp_path / ".vllm-sr" / "grafana" / "grafana.serve.ini").read_text(
        encoding="utf-8"
    )
    assert "allowed_origins = \n" in config
    assert "__GF_LIVE_ALLOWED_ORIGINS__" not in config


def test_container_start_grafana_rejects_multiline_live_origins_before_replacement(
    monkeypatch, tmp_path
):
    monkeypatch.setenv(
        container_support_services.GRAFANA_LIVE_ALLOWED_ORIGINS_ENV,
        "https://dashboard.example.com\n[security]\nallow_embedding = false",
    )
    commands, replaced = _stub_container_start(monkeypatch)

    with pytest.raises(ValueError, match="must be a comma-separated list on one line"):
        container_support_services.container_start_grafana(config_dir=str(tmp_path))

    assert replaced == []
    assert commands == []


def test_support_service_images_never_float():
    source = Path(container_support_services.__file__).read_text(encoding="utf-8")
    assert ":latest" not in source
