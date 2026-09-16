import sys
from pathlib import Path

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
