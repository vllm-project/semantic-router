import os
import stat
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli import container_support_services  # noqa: E402
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


def test_support_service_images_never_float():
    source = Path(container_support_services.__file__).read_text(encoding="utf-8")
    assert ":latest" not in source


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
