import json
import logging
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli import container_runtime  # noqa: E402


@pytest.fixture(autouse=True)
def clear_runtime_detection_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(container_runtime.sys, "platform", "linux")
    # Isolate the install.sh-persisted runtime.env per test so detection never
    # reads a real ~/.local/share/vllm-sr from the developer machine.
    monkeypatch.setenv("VLLM_SR_INSTALL_ROOT", str(tmp_path))
    container_runtime._detect_container_runtime.cache_clear()
    yield
    container_runtime._detect_container_runtime.cache_clear()


def _write_persisted_runtime_env(tmp_path, value):
    (tmp_path / "runtime.env").write_text(
        f"CONTAINER_RUNTIME={value}\n", encoding="utf-8"
    )


class _Result:
    def __init__(self, stdout, stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _stub_docker_version(
    monkeypatch, *, podman_daemon: bool = False, daemon_ok: bool = True
):
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

    monkeypatch.setattr(container_runtime.subprocess, "run", fake_run)


def _stub_podman_info(monkeypatch, *, ok: bool = True):
    def fake_run(cmd, *args, **kwargs):
        if cmd[1] == "info":
            if ok:
                return _Result('"linux"\n')
            return _Result("", "Cannot connect to Podman socket", 1)
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(container_runtime.subprocess, "run", fake_run)


def test_detect_container_runtime_accepts_real_docker(monkeypatch):
    monkeypatch.delenv("CONTAINER_RUNTIME", raising=False)
    monkeypatch.setattr(
        container_runtime.shutil,
        "which",
        lambda name: "/usr/local/bin/docker" if name == "docker" else None,
    )
    monkeypatch.setattr(container_runtime.os.path, "realpath", lambda path: path)
    _stub_docker_version(monkeypatch)

    assert container_runtime.get_container_runtime() == "docker"


def test_detect_container_runtime_accepts_podman_env_override(monkeypatch):
    monkeypatch.setenv("CONTAINER_RUNTIME", "podman")
    monkeypatch.setattr(
        container_runtime.shutil,
        "which",
        lambda name: "/usr/bin/podman" if name == "podman" else None,
    )
    _stub_podman_info(monkeypatch)

    assert container_runtime.get_container_runtime() == "podman"


def test_detect_container_runtime_accepts_docker_env_override(monkeypatch):
    monkeypatch.setenv("CONTAINER_RUNTIME", "docker")
    monkeypatch.setattr(
        container_runtime.shutil,
        "which",
        lambda name: "/usr/local/bin/docker" if name == "docker" else None,
    )
    monkeypatch.setattr(container_runtime.os.path, "realpath", lambda path: path)
    _stub_docker_version(monkeypatch)

    assert container_runtime.get_container_runtime() == "docker"


def test_detect_container_runtime_rejects_unknown_env_runtime(monkeypatch):
    monkeypatch.setenv("CONTAINER_RUNTIME", "lxc")

    with pytest.raises(SystemExit):
        container_runtime.get_container_runtime()


def test_detect_container_runtime_honors_persisted_runtime_env_over_autodetect(
    monkeypatch, tmp_path
):
    """#3370: runtime.env persisted by install.sh must beat auto-detection.

    A Podman-only install persists `CONTAINER_RUNTIME=podman`; when Docker
    shows up later, auto-detection would prefer Docker and silently switch
    the runtime away from the documented persisted choice.
    """
    _write_persisted_runtime_env(tmp_path, "podman")
    monkeypatch.delenv("CONTAINER_RUNTIME", raising=False)

    def fake_which(name):
        if name == "docker":
            return "/usr/local/bin/docker"
        if name == "podman":
            return "/usr/bin/podman"
        return None

    monkeypatch.setattr(container_runtime.shutil, "which", fake_which)
    monkeypatch.setattr(container_runtime.os.path, "realpath", lambda path: path)
    _stub_docker_version(monkeypatch)
    _stub_podman_info(monkeypatch)

    assert container_runtime.get_container_runtime() == "podman"


def test_detect_container_runtime_env_var_overrides_persisted_file(
    monkeypatch, tmp_path
):
    """An explicit session env var is the stronger contract; the persisted
    file must not win over it."""
    _write_persisted_runtime_env(tmp_path, "podman")
    monkeypatch.setenv("CONTAINER_RUNTIME", "docker")
    monkeypatch.setattr(
        container_runtime.shutil,
        "which",
        lambda name: "/usr/local/bin/docker" if name == "docker" else None,
    )
    monkeypatch.setattr(container_runtime.os.path, "realpath", lambda path: path)
    _stub_docker_version(monkeypatch)

    assert container_runtime.get_container_runtime() == "docker"


def test_detect_container_runtime_ignores_unsupported_persisted_value(
    monkeypatch, tmp_path, caplog
):
    """A stale or hand-edited runtime.env must not abort; fall back to
    auto-detection with a warning."""
    _write_persisted_runtime_env(tmp_path, "lxc")
    monkeypatch.delenv("CONTAINER_RUNTIME", raising=False)
    monkeypatch.setattr(
        container_runtime.shutil,
        "which",
        lambda name: "/usr/local/bin/docker" if name == "docker" else None,
    )
    monkeypatch.setattr(container_runtime.os.path, "realpath", lambda path: path)
    _stub_docker_version(monkeypatch)

    with caplog.at_level(logging.WARNING, logger="cli.container_runtime"):
        assert container_runtime.get_container_runtime() == "docker"

    assert any(
        "Ignoring unsupported CONTAINER_RUNTIME=lxc" in record.message
        for record in caplog.records
    )


def test_detect_container_runtime_falls_back_when_persisted_runtime_missing_from_path(
    monkeypatch, tmp_path, caplog
):
    """A persisted runtime whose binary is gone is a stale hint, not an
    error: warn and auto-detect."""
    _write_persisted_runtime_env(tmp_path, "podman")
    monkeypatch.delenv("CONTAINER_RUNTIME", raising=False)
    monkeypatch.setattr(
        container_runtime.shutil,
        "which",
        lambda name: "/usr/local/bin/docker" if name == "docker" else None,
    )
    monkeypatch.setattr(container_runtime.os.path, "realpath", lambda path: path)
    _stub_docker_version(monkeypatch)

    with caplog.at_level(logging.WARNING, logger="cli.container_runtime"):
        assert container_runtime.get_container_runtime() == "docker"

    assert any(
        "CONTAINER_RUNTIME=podman" in record.message
        and "was not found in PATH" in record.message
        for record in caplog.records
    )


def test_detect_container_runtime_ignores_malformed_persisted_file(
    monkeypatch, tmp_path
):
    """A runtime.env without a CONTAINER_RUNTIME line reads as absent."""
    (tmp_path / "runtime.env").write_text(
        "# a comment\nOTHER_KEY=value\n\n", encoding="utf-8"
    )
    monkeypatch.delenv("CONTAINER_RUNTIME", raising=False)
    monkeypatch.setattr(
        container_runtime.shutil,
        "which",
        lambda name: "/usr/local/bin/docker" if name == "docker" else None,
    )
    monkeypatch.setattr(container_runtime.os.path, "realpath", lambda path: path)
    _stub_docker_version(monkeypatch)

    assert container_runtime.get_container_runtime() == "docker"


def test_detect_container_runtime_warns_when_persisted_file_unreadable(
    monkeypatch, tmp_path, caplog
):
    """An existing but unreadable runtime.env warns (so the ignored persisted
    selection stays diagnosable) and falls back to auto-detection."""
    env_path = tmp_path / "runtime.env"
    env_path.write_text("CONTAINER_RUNTIME=podman\n", encoding="utf-8")

    def unreadable_open(path, *args, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.delenv("CONTAINER_RUNTIME", raising=False)
    monkeypatch.setattr(
        container_runtime.shutil,
        "which",
        lambda name: "/usr/local/bin/docker" if name == "docker" else None,
    )
    monkeypatch.setattr(container_runtime.os.path, "realpath", lambda path: path)
    _stub_docker_version(monkeypatch)
    monkeypatch.setattr("builtins.open", unreadable_open)

    with caplog.at_level(logging.WARNING, logger="cli.container_runtime"):
        assert container_runtime.get_container_runtime() == "docker"

    assert any(
        "Could not read persisted CONTAINER_RUNTIME" in record.message
        for record in caplog.records
    )


def test_detect_container_runtime_warns_on_invalid_utf8_persisted_file(
    monkeypatch, tmp_path, caplog
):
    """A runtime.env containing invalid UTF-8 bytes warns (so the corrupted
    persisted selection stays diagnosable) and falls back to auto-detection."""
    env_path = tmp_path / "runtime.env"
    # Write raw invalid UTF-8 bytes that decode() cannot handle.
    env_path.write_bytes(b"\x80\x81CONTAINER_RUNTIME=podman\n")

    monkeypatch.delenv("CONTAINER_RUNTIME", raising=False)
    monkeypatch.setattr(
        container_runtime.shutil,
        "which",
        lambda name: "/usr/local/bin/docker" if name == "docker" else None,
    )
    monkeypatch.setattr(container_runtime.os.path, "realpath", lambda path: path)
    _stub_docker_version(monkeypatch)

    with caplog.at_level(logging.WARNING, logger="cli.container_runtime"):
        assert container_runtime.get_container_runtime() == "docker"

    assert any(
        "Could not read persisted CONTAINER_RUNTIME" in record.message
        for record in caplog.records
    )
    monkeypatch.setattr(container_runtime.sys, "platform", "win32")

    with pytest.raises(SystemExit):
        container_runtime.get_container_runtime()


def test_detect_container_runtime_falls_back_to_podman_when_only_podman_exists(
    monkeypatch,
):
    monkeypatch.delenv("CONTAINER_RUNTIME", raising=False)
    monkeypatch.setattr(
        container_runtime.shutil,
        "which",
        lambda name: "/usr/bin/podman" if name == "podman" else None,
    )
    _stub_podman_info(monkeypatch)

    assert container_runtime.get_container_runtime() == "podman"


def test_detect_container_runtime_falls_back_to_podman_on_macos_when_only_podman_exists(
    monkeypatch,
):
    """Reproduces #2954: macOS with only Podman installed (podman machine running)."""
    monkeypatch.setattr(container_runtime.sys, "platform", "darwin")
    monkeypatch.delenv("CONTAINER_RUNTIME", raising=False)
    monkeypatch.setattr(
        container_runtime.shutil,
        "which",
        lambda name: "/opt/homebrew/bin/podman" if name == "podman" else None,
    )
    _stub_podman_info(monkeypatch)

    assert container_runtime.get_container_runtime() == "podman"


def test_detect_container_runtime_reports_unreachable_when_macos_podman_machine_stopped(
    monkeypatch,
):
    """A stopped `podman machine` should fail as unreachable, not unsupported."""
    monkeypatch.setattr(container_runtime.sys, "platform", "darwin")
    monkeypatch.delenv("CONTAINER_RUNTIME", raising=False)
    monkeypatch.setattr(
        container_runtime.shutil,
        "which",
        lambda name: "/opt/homebrew/bin/podman" if name == "podman" else None,
    )
    _stub_podman_info(monkeypatch, ok=False)

    with pytest.raises(SystemExit):
        container_runtime.get_container_runtime()


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

    monkeypatch.setattr(container_runtime.shutil, "which", fake_which)
    monkeypatch.setattr(
        container_runtime.os.path,
        "realpath",
        lambda path: (
            "/opt/homebrew/bin/podman" if path == "/usr/local/bin/docker" else path
        ),
    )
    _stub_podman_info(monkeypatch)

    assert container_runtime.get_container_runtime() == "podman"


def test_detect_container_runtime_warns_but_accepts_podman_docker_daemon(monkeypatch):
    """A docker CLI talking to a Podman-compatible daemon is now tolerated."""
    monkeypatch.delenv("CONTAINER_RUNTIME", raising=False)
    monkeypatch.setattr(
        container_runtime.shutil,
        "which",
        lambda name: "/usr/local/bin/docker" if name == "docker" else None,
    )
    monkeypatch.setattr(container_runtime.os.path, "realpath", lambda path: path)
    _stub_docker_version(monkeypatch, podman_daemon=True)

    assert container_runtime.get_container_runtime() == "docker"


def test_detect_container_runtime_rejects_when_docker_daemon_unavailable(monkeypatch):
    monkeypatch.delenv("CONTAINER_RUNTIME", raising=False)
    monkeypatch.setattr(
        container_runtime.shutil,
        "which",
        lambda name: "/usr/local/bin/docker" if name == "docker" else None,
    )
    monkeypatch.setattr(container_runtime.os.path, "realpath", lambda path: path)
    _stub_docker_version(monkeypatch, daemon_ok=False)

    with pytest.raises(SystemExit):
        container_runtime.get_container_runtime()


def test_detect_container_runtime_rejects_when_podman_unavailable(monkeypatch):
    monkeypatch.setenv("CONTAINER_RUNTIME", "podman")
    monkeypatch.setattr(
        container_runtime.shutil,
        "which",
        lambda name: "/usr/bin/podman" if name == "podman" else None,
    )
    _stub_podman_info(monkeypatch, ok=False)

    with pytest.raises(SystemExit):
        container_runtime.get_container_runtime()


def test_resolve_container_cli_path_returns_none_for_podman_shim(monkeypatch):
    """OpenClaw needs a real Docker CLI; podman shims should not satisfy it."""
    monkeypatch.setattr(container_runtime.os.path, "realpath", lambda path: path)

    class Result:
        stdout = "Emulate Docker CLI using podman. podman version 5.0.0"
        stderr = ""

    monkeypatch.setattr(
        container_runtime.subprocess, "run", lambda *args, **kwargs: Result()
    )

    assert container_runtime.resolve_container_cli_path("/usr/local/bin/docker") is None


def test_container_image_exists_accepts_digest_pinned_images(monkeypatch):
    """#3277: `images -q` does not resolve repo@sha256 references, so a
    digest-pinned image present locally read as missing and
    `--image-pull-policy never` refused to start it. The presence check must
    use `image inspect`, which resolves digests, and must report presence
    through the exit status rather than stdout."""
    digest_ref = "ghcr.io/vllm-project/semantic-router@sha256:" + "a" * 64
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd

        class Result:
            # `image inspect` writes the manifest to stdout on success; the old
            # implementation keyed off stdout from `images -q`, which is empty
            # for a digest reference even when the image is present.
            returncode = 0
            stdout = '[{"Id":"sha256:deadbeef"}]'
            stderr = ""

        return Result()

    monkeypatch.setattr(container_runtime, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(container_runtime.subprocess, "run", fake_run)

    assert container_runtime.container_image_exists(digest_ref) is True
    assert seen["cmd"] == ["docker", "image", "inspect", digest_ref]


def test_container_image_exists_reports_missing_on_nonzero_exit(monkeypatch):
    """A genuinely absent image must still read as absent, including when the
    runtime writes diagnostics to stdout."""

    def fake_run(cmd, **kwargs):
        class Result:
            returncode = 1
            stdout = "[]\n"
            stderr = "Error: No such image"

        return Result()

    monkeypatch.setattr(container_runtime, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(container_runtime.subprocess, "run", fake_run)

    assert container_runtime.container_image_exists("missing:latest") is False
