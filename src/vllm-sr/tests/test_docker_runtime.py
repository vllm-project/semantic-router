import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli import docker_runtime  # noqa: E402


@pytest.fixture(autouse=True)
def clear_runtime_detection_cache():
    docker_runtime._detect_container_runtime.cache_clear()
    yield
    docker_runtime._detect_container_runtime.cache_clear()


def test_detect_container_runtime_accepts_real_docker(monkeypatch):
    monkeypatch.delenv("CONTAINER_RUNTIME", raising=False)
    monkeypatch.setattr(
        docker_runtime.shutil,
        "which",
        lambda name: "/usr/local/bin/docker" if name == "docker" else None,
    )
    monkeypatch.setattr(docker_runtime.os.path, "realpath", lambda path: path)

    class Result:
        def __init__(self, stdout, stderr="", returncode=0):
            self.stdout = stdout
            self.stderr = stderr
            self.returncode = returncode

    def fake_run(cmd, *args, **kwargs):
        if cmd[1] == "--version":
            return Result("Docker version 27.1.1, build deadbeef")
        if cmd[1] == "version":
            return Result(
                json.dumps(
                    {
                        "Client": {"Context": "default"},
                        "Server": {
                            "Components": [{"Name": "Engine", "Version": "27.1.1"}]
                        },
                    }
                )
            )
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(docker_runtime.subprocess, "run", fake_run)

    assert docker_runtime.get_container_runtime() == "docker"


def test_detect_container_runtime_rejects_podman_env_override(monkeypatch):
    monkeypatch.setenv("CONTAINER_RUNTIME", "podman")

    with pytest.raises(SystemExit):
        docker_runtime.get_container_runtime()


def test_detect_container_runtime_rejects_podman_docker_symlink(monkeypatch):
    monkeypatch.delenv("CONTAINER_RUNTIME", raising=False)
    monkeypatch.setattr(
        docker_runtime.shutil,
        "which",
        lambda name: "/usr/local/bin/docker" if name == "docker" else None,
    )
    monkeypatch.setattr(
        docker_runtime.os.path,
        "realpath",
        lambda path: (
            "/opt/homebrew/bin/podman" if path == "/usr/local/bin/docker" else path
        ),
    )

    with pytest.raises(SystemExit):
        docker_runtime.get_container_runtime()


def test_detect_container_runtime_rejects_when_only_podman_exists(monkeypatch):
    monkeypatch.delenv("CONTAINER_RUNTIME", raising=False)
    monkeypatch.setattr(
        docker_runtime.shutil,
        "which",
        lambda name: "/usr/bin/podman" if name == "podman" else None,
    )

    with pytest.raises(SystemExit):
        docker_runtime.get_container_runtime()


def test_detect_container_runtime_rejects_podman_docker_daemon(monkeypatch):
    monkeypatch.delenv("CONTAINER_RUNTIME", raising=False)
    monkeypatch.setattr(
        docker_runtime.shutil,
        "which",
        lambda name: "/usr/local/bin/docker" if name == "docker" else None,
    )
    monkeypatch.setattr(docker_runtime.os.path, "realpath", lambda path: path)

    class Result:
        def __init__(self, stdout, stderr="", returncode=0):
            self.stdout = stdout
            self.stderr = stderr
            self.returncode = returncode

    def fake_run(cmd, *args, **kwargs):
        if cmd[1] == "--version":
            return Result("Docker version 29.3.0, build deadbeef")
        if cmd[1] == "version":
            return Result(
                json.dumps(
                    {
                        "Client": {"Context": "default"},
                        "Server": {
                            "Components": [
                                {"Name": "Podman Engine", "Version": "5.4.2"}
                            ]
                        },
                    }
                )
            )
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(docker_runtime.subprocess, "run", fake_run)

    with pytest.raises(SystemExit):
        docker_runtime.get_container_runtime()


def test_detect_container_runtime_rejects_when_docker_daemon_unavailable(monkeypatch):
    monkeypatch.delenv("CONTAINER_RUNTIME", raising=False)
    monkeypatch.setattr(
        docker_runtime.shutil,
        "which",
        lambda name: "/usr/local/bin/docker" if name == "docker" else None,
    )
    monkeypatch.setattr(docker_runtime.os.path, "realpath", lambda path: path)

    class Result:
        def __init__(self, stdout, stderr="", returncode=0):
            self.stdout = stdout
            self.stderr = stderr
            self.returncode = returncode

    def fake_run(cmd, *args, **kwargs):
        if cmd[1] == "--version":
            return Result("Docker version 29.3.0, build deadbeef")
        if cmd[1] == "version":
            return Result(
                "",
                "Cannot connect to the Docker daemon at unix:///Users/test/.docker/run/docker.sock",
                1,
            )
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(docker_runtime.subprocess, "run", fake_run)

    with pytest.raises(SystemExit):
        docker_runtime.get_container_runtime()


def test_resolve_docker_cli_path_rejects_podman_version_output(monkeypatch):
    monkeypatch.setattr(docker_runtime.os.path, "realpath", lambda path: path)

    class Result:
        stdout = "Emulate Docker CLI using podman. podman version 5.0.0"
        stderr = ""

    monkeypatch.setattr(
        docker_runtime.subprocess, "run", lambda *args, **kwargs: Result()
    )

    with pytest.raises(SystemExit):
        docker_runtime.resolve_docker_cli_path("/usr/local/bin/docker")
