"""Behavior tests for install.sh runtime selection.

These tests drive the real install.sh through a stubbed shell harness so we
can verify Docker/Podman precedence, explicit --runtime docker, --runtime
skip, and the exact runtime.env contents -- not just that the right strings
appear in the script.
"""

import os
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
    """A fresh --runtime skip install writes no runtime.env of its own.

    An existing preference is a different case, covered by the seeded
    scenarios below: skip preserves it rather than deleting it (#4548).
    """
    out = _run_harness("skip")

    assert "SELECTED_RUNTIME=" in out
    assert "RUNTIME_ENV_FILE=absent" in out
    assert "CONTAINER_RUNTIME=" not in out


def test_skip_keeps_an_existing_runtime_env() -> None:
    """--runtime skip on a root with a persisted preference keeps it."""
    out = _run_harness("skip-keeps-existing")

    assert "RUNTIME_ENV_FILE=present" in out
    assert "CONTAINER_RUNTIME=podman" in out
    assert "Kept the existing runtime preference: podman." in out


def test_cli_mode_keeps_an_existing_runtime_env() -> None:
    """--mode cli leaves an existing runtime.env in place and names it."""
    out = _run_harness("cli-keeps-existing")

    assert "RUNTIME_ENV_FILE=present" in out
    assert "CONTAINER_RUNTIME=podman" in out
    assert "Kept the existing runtime preference: podman." in out


def test_failed_podman_keeps_an_existing_runtime_env() -> None:
    """A failed --runtime podman check preserves the existing preference."""
    out = _run_harness("podman-unreachable-keeps-existing")

    assert "RUNTIME_ENV_FILE=present" in out
    assert "CONTAINER_RUNTIME=docker" in out
    assert "Kept the existing runtime preference: docker." in out


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


def test_first_launch_reuses_runtime_for_dashboard_check() -> None:
    """The first-launch dashboard availability check must reuse the selected
    runtime. Otherwise a Podman stack that serve just started is probed as
    Docker and the installer aborts after a successful start (#3441)."""
    out = _run_harness("first-launch-podman")

    assert "SELECTED_RUNTIME=podman" in out

    invocations = [
        line.strip()
        for line in out.split("[FIRST_LAUNCH_ARGS]")[1].splitlines()
        if line.strip() and not line.strip().endswith("--help")
    ]
    serve = next(line for line in invocations if line.startswith("serve"))
    dashboard = next(line for line in invocations if line.startswith("dashboard"))

    assert "--runtime podman" in serve, invocations
    assert "--runtime podman" in dashboard, invocations


def test_first_launch_returns_while_serve_waits_for_setup() -> None:
    """With no config, `vllm-sr serve` waits for the Dashboard to activate one.
    The installer must not wait with it: it returns, prints the Dashboard, and
    leaves serve running to start the Router. A CLI that has
    --container-runtime gets it instead of the deprecated --runtime."""
    out = _run_harness("first-launch-setup-wait")

    assert "First-time serve flow is waiting for setup" in out
    assert "AUTO_LAUNCH_RAN=1" in out
    assert "SERVE_STILL_RUNNING=1" in out
    assert "keeps waiting in the background" in out

    invocations = [
        line.strip()
        for line in out.split("[FIRST_LAUNCH_ARGS]")[1].splitlines()
        if line.strip()
    ]
    serve = next(line for line in invocations if line.startswith("serve"))
    dashboard = next(line for line in invocations if line.startswith("dashboard"))
    assert serve == "serve --container-runtime docker", invocations
    assert "--container-runtime docker" in dashboard, invocations


def test_dashboard_access_uses_the_stack_port_offset() -> None:
    """The first-run link, browser target, and SSH tunnel must agree. The
    Dashboard binds 127.0.0.1 by default, so no network URL is offered."""
    out = _run_harness("print-dashboard-offset")
    access = out.split("[DASHBOARD_ACCESS]\n", 1)[1].split("[NEXT_STEPS]\n", 1)[0]
    next_steps = out.split("[NEXT_STEPS]\n", 1)[1].split("OPENED_URL=", 1)[0]

    assert "http://localhost:9700" in access
    assert "192.0.2.10" not in access
    assert "ssh -L 9700:localhost:9700 fixture-user@fixture.example" in access
    assert "http://localhost:9700" in next_steps
    assert "192.0.2.10" not in next_steps
    assert "ssh -L 9700:localhost:9700 fixture-user@fixture.example" in next_steps
    assert "OPENED_URL=http://localhost:9700" in out
    assert ":8700" not in access + next_steps


def test_network_dashboard_url_needs_a_published_dashboard() -> None:
    """VLLM_SR_DASHBOARD_HOST_BIND=0.0.0.0 publishes the Dashboard on every
    interface; only then does the network URL answer."""
    out = _run_harness("print-dashboard-published")
    access = out.split("[DASHBOARD_ACCESS]\n", 1)[1].split("[NEXT_STEPS]\n", 1)[0]
    next_steps = out.split("[NEXT_STEPS]\n", 1)[1].split("OPENED_URL=", 1)[0]

    assert "http://192.0.2.10:9700" in access
    assert "http://192.0.2.10:9700" in next_steps


def test_dashboard_offset_respects_runtime_host_port_range() -> None:
    """Router gRPC uses 50051 + offset, so 15484 is the largest valid value."""
    script = REPO_ROOT / "install.sh"
    for offset, expected_success in (
        ("15484", True),
        ("15485", False),
        ("-1", False),
        ("invalid", False),
    ):
        result = subprocess.run(
            ["bash", str(script), "--help"],
            env={**os.environ, "VLLM_SR_PORT_OFFSET": offset},
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        assert (result.returncode == 0) is expected_success, (offset, result.stderr)
        if not expected_success:
            assert "VLLM_SR_PORT_OFFSET" in result.stderr
