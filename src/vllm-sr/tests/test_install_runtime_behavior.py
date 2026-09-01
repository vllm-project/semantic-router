"""Behavior tests for install.sh runtime selection.

These tests drive the real install.sh through a stubbed shell harness so we
can verify Docker/Podman precedence, explicit --runtime docker, --runtime
skip, and the exact runtime.env contents -- not just that the right strings
appear in the script.
"""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
HARNESS = Path(__file__).parent / "install_runtime_harness.sh"


def _run_harness(scenario: str) -> str:
    """Run the harness for one scenario and return its stdout.

    Any install.sh `info`/`done_step` chatter lands on stdout too; callers
    assert on substrings, so that is fine.
    """
    result = subprocess.run(
        ["bash", str(HARNESS), scenario, str(REPO_ROOT)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"harness failed for {scenario}: rc={result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result.stdout


def test_auto_prefers_docker_when_both_ready() -> None:
    """Docker and Podman both available under --runtime auto picks Docker."""
    out = _run_harness("auto-both-ready")

    assert "SELECTED_RUNTIME=docker" in out
    assert "RUNTIME_ENV_FILE=present" in out
    assert "CONTAINER_RUNTIME=docker" in out


def test_auto_falls_back_to_podman_when_docker_absent() -> None:
    """No Docker under --runtime auto falls back to a ready Podman."""
    out = _run_harness("auto-podman-only")

    assert "SELECTED_RUNTIME=podman" in out
    assert "RUNTIME_ENV_FILE=present" in out
    assert "CONTAINER_RUNTIME=podman" in out


def test_explicit_docker_does_not_drift_to_podman() -> None:
    """--runtime docker must not trigger the Podman fallback even when
    Podman is also available."""
    out = _run_harness("explicit-docker-both-ready")

    assert "SELECTED_RUNTIME=docker" in out
    assert "RUNTIME_ENV_FILE=present" in out
    assert "CONTAINER_RUNTIME=docker" in out


def test_skip_writes_no_runtime_env() -> None:
    """--runtime skip clears the selection and must not persist a file."""
    out = _run_harness("skip")

    assert "SELECTED_RUNTIME=" in out
    # `skip` should not leave a stale CONTAINER_RUNTIME behind.
    assert "RUNTIME_ENV_FILE=absent" in out
    assert "CONTAINER_RUNTIME=" not in out
