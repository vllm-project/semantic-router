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
        check=False,
    )
    assert result.returncode == 0, (
        f"harness failed for {scenario}: rc={result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result.stdout


def _trace_entries(line: str) -> list[str]:
    """Split a CALLS=/CALLS_TOTAL= line into the probe names it recorded."""
    return [entry for entry in line.split("=", 1)[1].split(",") if entry]


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


def test_explicit_podman_skips_docker_detection() -> None:
    """--runtime podman must select Podman even when Docker is also
    available, skipping Docker detection entirely."""
    out = _run_harness("explicit-podman-both-ready")

    assert "SELECTED_RUNTIME=podman" in out
    assert "RUNTIME_ENV_FILE=present" in out
    assert "CONTAINER_RUNTIME=podman" in out
    # The docker stub must not have been invoked at all.
    assert "CALLS=podman" in out
    assert "docker" not in out.split("CALLS=")[1].split("\n")[0]


def test_skip_writes_no_runtime_env() -> None:
    """--runtime skip clears the selection and must not persist a file."""
    out = _run_harness("skip")

    assert "SELECTED_RUNTIME=" in out
    # `skip` should not leave a stale CONTAINER_RUNTIME behind.
    assert "RUNTIME_ENV_FILE=absent" in out
    assert "CONTAINER_RUNTIME=" not in out


def test_printed_commands_include_runtime_flag() -> None:
    """When --runtime podman is selected, the printed restart/start commands
    must carry `--runtime podman` so users copy-paste the right invocation.

    print_install_plan() is the only production caller of
    detect_existing_runtime(), and the harness takes its CALLS= snapshot
    before that path runs. The accumulated CALLS_TOTAL= trace is therefore
    what proves Docker is never probed across the print path (#3441)."""
    out = _run_harness("print-command-podman")

    assert "SELECTED_RUNTIME=podman" in out

    early_calls = _trace_entries(
        next(line for line in out.splitlines() if line.startswith("CALLS="))
    )
    total_line = next(
        line for line in out.splitlines() if line.startswith("CALLS_TOTAL=")
    )
    total_calls = _trace_entries(total_line)

    # Keeps the assertion below from passing on a trace that never observed
    # the print path -- the blind spot this coverage exists to close.
    assert len(total_calls) > len(early_calls), (
        f"print path recorded no extra runtime probe: early={early_calls} "
        f"total={total_calls}"
    )
    assert (
        "docker" not in total_calls
    ), f"Docker was probed despite --runtime podman: {total_line}"

    restart_section = out.split("[PRINT_RESTART_COMMAND]")[1].split(
        "[PRINT_NEXT_STEPS]"
    )[0]
    assert "--runtime podman" in restart_section

    next_steps_section = out.split("[PRINT_NEXT_STEPS]")[1]
    assert "--runtime podman" in next_steps_section
