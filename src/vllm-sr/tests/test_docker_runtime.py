import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli import docker_runtime  # noqa: E402


@pytest.fixture(autouse=True)
def clear_runtime_detection_cache(monkeypatch):
    monkeypatch.setattr(docker_runtime.sys, "platform", "linux")
    docker_runtime._detect_container_runtime.cache_clear()
    yield
    docker_runtime._detect_container_runtime.cache_clear()


class _Result:
    def __init__(self, stdout, stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _stub_docker_version(monkeypatch, *, podman_daemon: bool = False, daemon_ok: bool = True):
    """Make subprocess.run return a Docker-shaped daemon response."""
    def fake_run(cmd, *args, **kwargs):
        if cmd[1] == "--version":
            return _Result("Docker version 27.1.1, build deadbeef")
        if cmd[1] == "version":
            if not daemon_ok:
                return _Result("", "Cannot connect to the Docker daemon", 1)
            component = (
                {"Name": "Podman Engine", "Version": "5.4.2"}
                if podman_daemon
                else {"Name": "Engine", "Version": "27.1.1"}
            )
            return _Result(
                json.dumps(
                    {
                        "Client": {"Context": "default"},
                        "Server": {"Components": [component]},
                    }
                )
            )
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(docker_runtime.subprocess, "run", fake_run)


def _stub_podman_info(monkeypatch, *, ok: bool = True):
    def fake_run(cmd, *args, **kwargs):
        if cmd[1] == "info":
            if ok:
                return _Result('"linux"\n')
            return _Result("", "Cannot connect to Podman socket", 1)
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(docker_runtime.subprocess, "run", fake_run)


def test_detect_container_runtime_accepts_real_docker(monkeypatch):
    monkeypatch.delenv("CONTAINER_RUNTIME", raising=False)
    monkeypatch.setattr(
        docker_runtime.shutil,
        "which",
        lambda name: "/usr/local/bin/docker" if name == "docker" else None,
    )
    monkeypatch.setattr(docker_runtime.os.path, "realpath", lambda path: path)
    _stub_docker_version(monkeypatch)

    assert docker_runtime.get_container_runtime() == "docker"


def test_detect_container_runtime_accepts_podman_env_override(monkeypatch):
    monkeypatch.setenv("CONTAINER_RUNTIME", "podman")
    monkeypatch.setattr(
        docker_runtime.shutil,
        "which",
        lambda name: "/usr/bin/podman" if name == "podman" else None,
    )
    _stub_podman_info(monkeypatch)

    assert docker_runtime.get_container_runtime() == "podman"


def test_detect_container_runtime_accepts_docker_env_override(monkeypatch):
    monkeypatch.setenv("CONTAINER_RUNTIME", "docker")
    monkeypatch.setattr(
        docker_runtime.shutil,
        "which",
        lambda name: "/usr/local/bin/docker" if name == "docker" else None,
    )
    monkeypatch.setattr(docker_runtime.os.path, "realpath", lambda path: path)
    _stub_docker_version(monkeypatch)

    assert docker_runtime.get_container_runtime() == "docker"


def test_detect_container_runtime_rejects_unknown_env_runtime(monkeypatch):
    monkeypatch.setenv("CONTAINER_RUNTIME", "lxc")

    with pytest.raises(SystemExit):
        docker_runtime.get_container_runtime()


def test_detect_container_runtime_rejects_native_windows(monkeypatch):
    monkeypatch.setattr(docker_runtime.sys, "platform", "win32")

    with pytest.raises(SystemExit):
        docker_runtime.get_container_runtime()


def test_detect_container_runtime_falls_back_to_podman_when_only_podman_exists(monkeypatch):
    monkeypatch.delenv("CONTAINER_RUNTIME", raising=False)
    monkeypatch.setattr(
        docker_runtime.shutil,
        "which",
        lambda name: "/usr/bin/podman" if name == "podman" else None,
    )
    _stub_podman_info(monkeypatch)

    assert docker_runtime.get_container_runtime() == "podman"


def test_detect_container_runtime_falls_back_to_podman_when_docker_is_podman_symlink(
    monkeypatch,
):
    monkeypatch.delenv("CONTAINER_RUNTIME", raising=False)

    def fake_which(name):
        if name == "docker":
            return "/usr/local/bin/docker"
        if name == "podman":
            return "/opt/homebrew/bin/podman"
        return None

    monkeypatch.setattr(docker_runtime.shutil, "which", fake_which)
    monkeypatch.setattr(
        docker_runtime.os.path,
        "realpath",
        lambda path: (
            "/opt/homebrew/bin/podman" if path == "/usr/local/bin/docker" else path
        ),
    )
    _stub_podman_info(monkeypatch)

    assert docker_runtime.get_container_runtime() == "podman"


def test_detect_container_runtime_warns_but_accepts_podman_docker_daemon(monkeypatch):
    """A docker CLI talking to a Podman-compatible daemon is now tolerated."""
    monkeypatch.delenv("CONTAINER_RUNTIME", raising=False)
    monkeypatch.setattr(
        docker_runtime.shutil,
        "which",
        lambda name: "/usr/local/bin/docker" if name == "docker" else None,
    )
    monkeypatch.setattr(docker_runtime.os.path, "realpath", lambda path: path)
    _stub_docker_version(monkeypatch, podman_daemon=True)

    assert docker_runtime.get_container_runtime() == "docker"


def test_detect_container_runtime_rejects_when_docker_daemon_unavailable(monkeypatch):
    monkeypatch.delenv("CONTAINER_RUNTIME", raising=False)
    monkeypatch.setattr(
        docker_runtime.shutil,
        "which",
        lambda name: "/usr/local/bin/docker" if name == "docker" else None,
    )
    monkeypatch.setattr(docker_runtime.os.path, "realpath", lambda path: path)
    _stub_docker_version(monkeypatch, daemon_ok=False)

    with pytest.raises(SystemExit):
        docker_runtime.get_container_runtime()


def test_detect_container_runtime_rejects_when_podman_unavailable(monkeypatch):
    monkeypatch.setenv("CONTAINER_RUNTIME", "podman")
    monkeypatch.setattr(
        docker_runtime.shutil,
        "which",
        lambda name: "/usr/bin/podman" if name == "podman" else None,
    )
    _stub_podman_info(monkeypatch, ok=False)

    with pytest.raises(SystemExit):
        docker_runtime.get_container_runtime()


def test_resolve_docker_cli_path_returns_none_for_podman_shim(monkeypatch):
    """OpenClaw needs a real Docker CLI; podman shims should not satisfy it."""
    monkeypatch.setattr(docker_runtime.os.path, "realpath", lambda path: path)

    class Result:
        stdout = "Emulate Docker CLI using podman. podman version 5.0.0"
        stderr = ""

    monkeypatch.setattr(
        docker_runtime.subprocess, "run", lambda *args, **kwargs: Result()
    )

    assert docker_runtime.resolve_docker_cli_path("/usr/local/bin/docker") is None
