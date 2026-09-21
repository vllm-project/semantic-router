import os
import stat
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli import container_support_services  # noqa: E402
from cli.container_observability import render_observability_template  # noqa: E402
from cli.runtime_stack import resolve_runtime_stack  # noqa: E402


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
    assert "SPAN_STORAGE_TYPE=badger" in cmd
    assert "BADGER_EPHEMERAL=false" in cmd
    assert "BADGER_SPAN_STORE_TTL=168h" in cmd
    assert "BADGER_DIRECTORY_KEY=/tmp/badger/keys" in cmd
    assert "BADGER_DIRECTORY_VALUE=/tmp/badger/values" in cmd
    assert "--user" not in cmd  # Preserve the image's non-root UID10001.


def test_jaeger_volume_is_stable_and_isolated_by_stack(monkeypatch):
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
        lambda cmd, _: commands.append(cmd),
    )
    first = resolve_runtime_stack(stack_name="observability-a", port_offset=100)
    second = resolve_runtime_stack(stack_name="observability-b", port_offset=200)
    for layout in (first, first, second):
        container_support_services.container_start_jaeger(stack_layout=layout)
    volumes = [command[command.index("-v") + 1] for command in commands]
    assert volumes[0] == volumes[1] == f"{first.jaeger_container_name}-data:/tmp"
    assert volumes[2] == f"{second.jaeger_container_name}-data:/tmp"
    assert volumes[0] != volumes[2]


def test_support_service_images_never_float():
    source = Path(container_support_services.__file__).read_text(encoding="utf-8")
    assert ":latest" not in source


def test_grafana_state_volume_is_stable_and_stack_scoped(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(
        container_support_services, "get_container_runtime", lambda: "docker"
    )
    monkeypatch.setattr(
        container_support_services, "_replace_existing_container", lambda _: None
    )
    monkeypatch.setattr(
        container_support_services,
        "_run_service_start",
        lambda command, _: commands.append(command),
    )
    first = resolve_runtime_stack(stack_name="grafana-a", port_offset=100)
    second = resolve_runtime_stack(stack_name="grafana-b", port_offset=200)
    for layout in (first, first, second):
        container_support_services.container_start_grafana(
            config_dir=str(tmp_path), stack_layout=layout
        )
    volumes = [
        [v for v in command if v.endswith(":/var/lib/grafana")] for command in commands
    ]
    assert (
        volumes[0]
        == volumes[1]
        == [f"{first.grafana_container_name}-data:/var/lib/grafana"]
    )
    assert volumes[2] == [f"{second.grafana_container_name}-data:/var/lib/grafana"]
    assert all("--user" not in command for command in commands)
    assert all("docker.io/grafana/grafana:11.5.1" in command for command in commands)


def test_collector_scrape_stays_on_the_selected_stack_network():
    layout = resolve_runtime_stack(stack_name="collector-fixture", port_offset=400)
    template = PROJECT_ROOT / "cli/templates/prometheus.serve.yaml"
    rendered = render_observability_template(template.read_text(), layout)
    assert f"{layout.jaeger_container_name}:14269" in rendered
    assert f"{layout.router_container_name}:9190" in rendered
    assert "localhost:14269" not in rendered
    assert "job_name: 'jaeger'" in rendered


@pytest.mark.parametrize("service", ["prometheus", "grafana"])
def test_generated_sidecar_configs_are_readable_under_private_umask(
    tmp_path, monkeypatch, service
):
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
        lambda cmd, name: commands.append(cmd) or (0, "", ""),
    )
    private_config = tmp_path / "config.yaml"
    private_config.write_text("private operator configuration")
    private_config.chmod(0o600)
    previous_umask = os.umask(0o077)
    try:
        getattr(container_support_services, f"container_start_{service}")(
            config_dir=str(tmp_path), stack_layout=resolve_runtime_stack()
        )
    finally:
        os.umask(previous_umask)

    (command,) = commands
    mounted_configs = [
        Path(argument.split(":", 1)[0])
        for argument in command
        if argument.endswith(":ro")
    ]
    assert mounted_configs
    for config in mounted_configs:
        assert config.read_text()
        assert stat.S_IMODE(config.stat().st_mode) == 0o644
    assert stat.S_IMODE(private_config.stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / ".vllm-sr").stat().st_mode) == 0o700
