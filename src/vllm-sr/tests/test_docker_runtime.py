import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

docker_runtime = importlib.import_module("cli.docker_runtime")


def _log_recorder():
    entries: list[tuple[str, str]] = []

    def record(level: str):
        return lambda message="": entries.append((level, str(message)))

    return entries, SimpleNamespace(
        info=record("info"),
        warning=record("warning"),
        error=record("error"),
    )


def _clear_runtime_cache() -> None:
    docker_runtime._detect_container_runtime.cache_clear()


def test_get_container_runtime_detects_docker(monkeypatch):
    _clear_runtime_cache()
    logs, recorder = _log_recorder()
    monkeypatch.setattr(docker_runtime, "log", recorder)
    monkeypatch.setattr(
        docker_runtime.shutil,
        "which",
        lambda name: "/usr/bin/docker" if name == "docker" else None,
    )

    assert docker_runtime.get_container_runtime() == "docker"
    assert ("info", "Detected container runtime: docker") in logs


def test_get_container_runtime_ignores_container_runtime_env(monkeypatch):
    _clear_runtime_cache()
    logs, recorder = _log_recorder()
    monkeypatch.setattr(docker_runtime, "log", recorder)
    monkeypatch.setenv("CONTAINER_RUNTIME", "podman")
    monkeypatch.setattr(
        docker_runtime.shutil,
        "which",
        lambda name: "/usr/bin/docker" if name == "docker" else None,
    )

    assert docker_runtime.get_container_runtime() == "docker"
    assert all("podman" not in message.lower() for _, message in logs)


def test_get_container_runtime_fails_with_docker_only_message(monkeypatch):
    _clear_runtime_cache()
    logs, recorder = _log_recorder()
    monkeypatch.setattr(docker_runtime, "log", recorder)
    monkeypatch.setattr(docker_runtime.shutil, "which", lambda name: None)

    with pytest.raises(SystemExit):
        docker_runtime.get_container_runtime()

    rendered_logs = "\n".join(message for _, message in logs)
    assert "Docker not found in PATH" in rendered_logs
    assert "Please install Docker to use this tool" in rendered_logs
    assert "Podman" not in rendered_logs
