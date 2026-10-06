import importlib
import os
import subprocess
import sys
from pathlib import Path

from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

core = importlib.import_module("cli.core")
main = importlib.import_module("cli.main").main
runtime_stack = importlib.import_module("cli.runtime_stack")

FLEET_SIM_RUNTIME_MARKERS = (
    "--sim-image",
    "docker-build-vllm-sr-sim",
    "src/fleet-sim",
    "vllm-sr-sim",
)
FLEET_SIM_CLI_MARKERS = (*FLEET_SIM_RUNTIME_MARKERS, "fleet", "simulator")


def test_default_cli_lifecycle_exposes_only_router_envoy_and_dashboard() -> None:
    runner = CliRunner()

    for command in ("serve", "status", "logs", "stop"):
        result = runner.invoke(main, [command, "--help"])
        assert result.exit_code == 0, result.output
        lowered_output = result.output.lower()
        for marker in FLEET_SIM_CLI_MARKERS:
            assert marker not in lowered_output, (
                f"{command} unexpectedly exposes Fleet Simulator runtime marker: "
                f"{marker}"
            )

    layout = runtime_stack.resolve_runtime_stack(stack_name="vllm-sr", port_offset=0)
    assert layout.runtime_container_names == (
        "vllm-sr-router-container",
        "vllm-sr-envoy-container",
        "vllm-sr-dashboard-container",
    )
    assert core.RUNTIME_LOG_SERVICES == ("router", "dashboard", "envoy")


def test_default_make_build_and_serve_closure_excludes_fleet_sim() -> None:
    environment = os.environ.copy()
    environment["CONTAINER_RUNTIME"] = "true"
    result = subprocess.run(
        [
            "make",
            "--dry-run",
            "--no-print-directory",
            "vllm-sr-dev",
            "vllm-sr-build",
            "vllm-sr-start",
            "vllm-sr-test-integration",
            "memory-test-integration",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    lowered_output = output.lower()
    for marker in FLEET_SIM_RUNTIME_MARKERS:
        assert marker not in lowered_output, (
            "default Make build/serve closure unexpectedly includes Fleet Simulator "
            f"runtime marker: {marker}\n{output}"
        )
