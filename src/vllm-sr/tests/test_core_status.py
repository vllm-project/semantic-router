import importlib
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

core = importlib.import_module("cli.core")
runtime_stack = importlib.import_module("cli.runtime_stack")


def test_check_envoy_status_uses_ready_probe_when_available(monkeypatch):
    stack_layout = runtime_stack.resolve_runtime_stack()
    captured = []

    def fake_exec(container_name, command):
        captured.append((container_name, command))
        return (0, "200", "")

    monkeypatch.setattr(core, "container_exec", fake_exec)
    monkeypatch.setattr(core, "container_status", lambda _name: "running")

    assert core._check_envoy_status(stack_layout.envoy_container_name, stack_layout)
    assert captured == [
        (
            stack_layout.envoy_container_name,
            [
                "curl",
                "-f",
                "-s",
                "-o",
                "/dev/null",
                "-w",
                "%{http_code}",
                "http://localhost:9901/ready",
            ],
        )
    ]


def test_check_envoy_status_falls_back_to_envoy_validate(monkeypatch):
    stack_layout = runtime_stack.resolve_runtime_stack()
    captured = []
    responses = iter(((127, "", "curl missing"), (0, "", "")))

    def fake_exec(container_name, command):
        captured.append((container_name, command))
        return next(responses)

    monkeypatch.setattr(core, "container_exec", fake_exec)
    monkeypatch.setattr(core, "container_status", lambda _name: "running")

    assert core._check_envoy_status(stack_layout.envoy_container_name, stack_layout)
    assert captured == [
        (
            stack_layout.envoy_container_name,
            [
                "curl",
                "-f",
                "-s",
                "-o",
                "/dev/null",
                "-w",
                "%{http_code}",
                "http://localhost:9901/ready",
            ],
        ),
        (
            stack_layout.envoy_container_name,
            [
                "/usr/local/bin/envoy",
                "--mode",
                "validate",
                "-c",
                "/etc/envoy/envoy.yaml",
            ],
        ),
    ]


def test_check_envoy_status_does_not_fallback_for_non_envoy_container(monkeypatch):
    stack_layout = runtime_stack.resolve_runtime_stack()
    captured = []

    def fake_exec(container_name, command):
        captured.append((container_name, command))
        return (127, "", "curl missing")

    monkeypatch.setattr(core, "container_exec", fake_exec)
    monkeypatch.setattr(core, "container_status", lambda _name: "running")

    assert (
        core._check_envoy_status(stack_layout.router_container_name, stack_layout)
        is False
    )
    assert captured == [
        (
            stack_layout.router_container_name,
            [
                "curl",
                "-f",
                "-s",
                "-o",
                "/dev/null",
                "-w",
                "%{http_code}",
                "http://localhost:9901/ready",
            ],
        )
    ]


def test_show_status_reports_not_running_when_docker_daemon_unreachable(
    monkeypatch, capsys
):
    stack_layout = runtime_stack.resolve_runtime_stack()

    monkeypatch.setattr(core, "resolve_runtime_stack", lambda: stack_layout)
    monkeypatch.setattr(
        core,
        "_runtime_stack_status",
        lambda _stack_layout: (_ for _ in ()).throw(SystemExit(1)),
    )
    core.show_status()

    captured = capsys.readouterr()
    assert "State  Not running" in captured.out
    assert "Start with: vllm-sr serve" in captured.out
    assert "Docker daemon is not reachable" in captured.err


def test_stop_reports_noop_result_on_stdout(monkeypatch, capsys):
    stack_layout = runtime_stack.resolve_runtime_stack()
    removed_networks = []
    managed_names = (
        *stack_layout.runtime_container_names,
        stack_layout.fleet_sim_container_name,
        stack_layout.grafana_container_name,
        stack_layout.prometheus_container_name,
        stack_layout.jaeger_container_name,
        *stack_layout.storage_container_names,
    )

    monkeypatch.setattr(core, "resolve_runtime_stack", lambda: stack_layout)
    monkeypatch.setattr(
        core,
        "_managed_container_statuses",
        lambda _stack_layout: dict.fromkeys(managed_names, "not found"),
    )
    monkeypatch.setattr(core, "resolve_openclaw_data_dir", lambda _cwd: "/unused")
    monkeypatch.setattr(core, "load_openclaw_registry", lambda _path: [])
    monkeypatch.setattr(
        core,
        "container_remove_network",
        lambda name: (removed_networks.append(name) or 0, "", ""),
    )

    core.stop_vllm_sr()

    captured = capsys.readouterr()
    assert captured.out == "Nothing to stop.\n"
    assert "Stopping vLLM Semantic Router" in captured.err
    assert removed_networks == [stack_layout.network_name]


def test_stop_propagates_orphan_network_removal_failure(monkeypatch, capsys):
    stack_layout = runtime_stack.resolve_runtime_stack()
    managed_names = (
        *stack_layout.runtime_container_names,
        stack_layout.fleet_sim_container_name,
        stack_layout.grafana_container_name,
        stack_layout.prometheus_container_name,
        stack_layout.jaeger_container_name,
        *stack_layout.storage_container_names,
    )

    monkeypatch.setattr(core, "resolve_runtime_stack", lambda: stack_layout)
    monkeypatch.setattr(
        core,
        "_managed_container_statuses",
        lambda _stack_layout: dict.fromkeys(managed_names, "not found"),
    )
    monkeypatch.setattr(core, "resolve_openclaw_data_dir", lambda _cwd: "/unused")
    monkeypatch.setattr(core, "load_openclaw_registry", lambda _path: [])
    monkeypatch.setattr(
        core,
        "container_remove_network",
        lambda _name: (1, "", "network has active endpoints"),
    )

    with pytest.raises(RuntimeError, match=stack_layout.network_name):
        core.stop_vllm_sr()

    captured = capsys.readouterr()
    assert "Nothing to stop" not in captured.out
    assert "network has active endpoints" in captured.err


def test_stop_reports_success_when_only_simulator_exists(monkeypatch, capsys):
    stack_layout = runtime_stack.resolve_runtime_stack()
    managed_names = (
        *stack_layout.runtime_container_names,
        stack_layout.fleet_sim_container_name,
        stack_layout.grafana_container_name,
        stack_layout.prometheus_container_name,
        stack_layout.jaeger_container_name,
        *stack_layout.storage_container_names,
    )
    statuses = dict.fromkeys(managed_names, "not found")
    statuses[stack_layout.fleet_sim_container_name] = "running"

    monkeypatch.setattr(core, "resolve_runtime_stack", lambda: stack_layout)
    monkeypatch.setattr(
        core, "_managed_container_statuses", lambda _stack_layout: statuses
    )
    monkeypatch.setattr(core, "resolve_openclaw_data_dir", lambda _cwd: "/unused")
    monkeypatch.setattr(core, "load_openclaw_registry", lambda _path: [])
    monkeypatch.setattr(core, "container_stop_container", lambda _name: True)
    monkeypatch.setattr(core, "container_remove_container", lambda _name: True)
    monkeypatch.setattr(core, "container_remove_network", lambda _name: (0, "", ""))

    core.stop_vllm_sr()

    captured = capsys.readouterr()
    assert captured.out == "✓ vLLM Semantic Router stopped\n"
    assert "Stopping vLLM Semantic Router" in captured.err


def test_stop_does_not_report_success_when_container_removal_fails(monkeypatch, capsys):
    stack_layout = runtime_stack.resolve_runtime_stack()
    managed_names = (
        *stack_layout.runtime_container_names,
        stack_layout.fleet_sim_container_name,
        stack_layout.grafana_container_name,
        stack_layout.prometheus_container_name,
        stack_layout.jaeger_container_name,
        *stack_layout.storage_container_names,
    )
    statuses = dict.fromkeys(managed_names, "not found")
    statuses[stack_layout.fleet_sim_container_name] = "running"

    monkeypatch.setattr(core, "resolve_runtime_stack", lambda: stack_layout)
    monkeypatch.setattr(
        core, "_managed_container_statuses", lambda _stack_layout: statuses
    )
    monkeypatch.setattr(core, "resolve_openclaw_data_dir", lambda _cwd: "/unused")
    monkeypatch.setattr(core, "load_openclaw_registry", lambda _path: [])
    monkeypatch.setattr(core, "container_stop_container", lambda _name: True)
    monkeypatch.setattr(core, "container_remove_container", lambda _name: False)
    monkeypatch.setattr(core, "container_remove_network", lambda _name: (0, "", ""))

    try:
        core.stop_vllm_sr()
    except RuntimeError as error:
        assert stack_layout.fleet_sim_container_name in str(error)
    else:
        raise AssertionError("stop must propagate container removal failures")

    captured = capsys.readouterr()
    assert "✓ vLLM Semantic Router stopped" not in captured.out


def test_stop_does_not_report_success_when_network_removal_fails(monkeypatch, capsys):
    stack_layout = runtime_stack.resolve_runtime_stack()
    managed_names = (
        *stack_layout.runtime_container_names,
        stack_layout.fleet_sim_container_name,
        stack_layout.grafana_container_name,
        stack_layout.prometheus_container_name,
        stack_layout.jaeger_container_name,
        *stack_layout.storage_container_names,
    )
    statuses = dict.fromkeys(managed_names, "not found")
    statuses[stack_layout.fleet_sim_container_name] = "running"

    monkeypatch.setattr(core, "resolve_runtime_stack", lambda: stack_layout)
    monkeypatch.setattr(
        core, "_managed_container_statuses", lambda _stack_layout: statuses
    )
    monkeypatch.setattr(core, "resolve_openclaw_data_dir", lambda _cwd: "/unused")
    monkeypatch.setattr(core, "load_openclaw_registry", lambda _path: [])
    monkeypatch.setattr(core, "container_stop_container", lambda _name: True)
    monkeypatch.setattr(core, "container_remove_container", lambda _name: True)
    monkeypatch.setattr(
        core,
        "container_remove_network",
        lambda _name: (1, "", "network has active endpoints"),
    )

    with pytest.raises(RuntimeError, match=stack_layout.network_name):
        core.stop_vllm_sr()

    captured = capsys.readouterr()
    assert "✓ vLLM Semantic Router stopped" not in captured.out
    assert "network has active endpoints" in captured.err


def test_show_logs_reports_empty_result_on_stdout(monkeypatch, capsys):
    stack_layout = runtime_stack.resolve_runtime_stack()

    monkeypatch.setattr(core, "resolve_runtime_stack", lambda: stack_layout)
    monkeypatch.setattr(core, "_ensure_runtime_container_available", lambda _name: None)
    monkeypatch.setattr(
        core, "container_logs_output", lambda *_args, **_kwargs: (0, "")
    )

    core.show_logs("router")

    captured = capsys.readouterr()
    assert captured.out == "No recent router logs found\n"
    assert captured.err == ""


def test_simulator_logs_merge_container_stderr_into_raw_stdout(monkeypatch):
    stack_layout = runtime_stack.resolve_runtime_stack()
    calls = []

    monkeypatch.setattr(core, "resolve_runtime_stack", lambda: stack_layout)
    monkeypatch.setattr(core, "_ensure_runtime_container_available", lambda _name: None)
    monkeypatch.setattr(
        core,
        "container_logs",
        lambda *args, **kwargs: calls.append((args, kwargs)) or True,
    )

    core.show_logs("simulator", follow=True)

    assert calls == [
        (
            (stack_layout.fleet_sim_container_name,),
            {"follow": True, "tail": 200, "merge_output": True},
        )
    ]


def test_log_command_failure_exits_nonzero_without_empty_success(monkeypatch, capsys):
    stack_layout = runtime_stack.resolve_runtime_stack()

    monkeypatch.setattr(core, "resolve_runtime_stack", lambda: stack_layout)
    monkeypatch.setattr(core, "_ensure_runtime_container_available", lambda _name: None)
    monkeypatch.setattr(
        core,
        "container_logs_output",
        lambda *_args, **_kwargs: (125, "runtime unavailable\n"),
    )

    try:
        core.show_logs("router")
    except SystemExit as error:
        assert error.code == 125
    else:
        raise AssertionError("logs must propagate container runtime failures")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Error: Failed to get router logs: runtime unavailable" in captured.err


def test_container_logs_can_merge_raw_streams(monkeypatch):
    container_services = importlib.import_module("cli.container_services")
    captured = {}

    monkeypatch.setattr(container_services, "get_container_runtime", lambda: "docker")

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs

    monkeypatch.setattr(container_services.subprocess, "run", fake_run)

    container_services.container_logs(
        "simulator-container", follow=True, tail=200, merge_output=True
    )

    assert captured == {
        "command": ["docker", "logs", "-f", "--tail", "200", "simulator-container"],
        "kwargs": {"check": True, "stderr": subprocess.STDOUT},
    }


def test_never_pull_preflight_skips_dashboard_when_disabled(monkeypatch):
    captured = {}
    sim_checks = []

    def fake_get_runtime_images(**kwargs):
        captured.update(kwargs)
        return {"router": "router:test", "envoy": "envoy:test"}

    monkeypatch.setattr(core, "get_runtime_images", fake_get_runtime_images)
    monkeypatch.setattr(
        core,
        "get_fleet_sim_container_image",
        lambda image=None, pull_policy=None: sim_checks.append((image, pull_policy)),
    )

    core.ensure_runtime_images_for_pull_policy(
        image="router:test",
        router_image=None,
        envoy_image=None,
        dashboard_image="dashboard:missing",
        sim_image="sim:test",
        pull_policy="never",
        env_vars={"VLLM_SR_PLATFORM": "amd"},
        dashboard_disabled=True,
    )

    assert captured["dashboard_image"] is None
    assert captured["include_dashboard"] is False
    assert sim_checks == [("sim:test", "never")]


def test_never_pull_preflight_skips_sim_when_disabled(monkeypatch):
    sim_checks = []
    monkeypatch.setattr(
        core,
        "get_runtime_images",
        lambda **kwargs: {"router": "router:test", "envoy": "envoy:test"},
    )
    monkeypatch.setattr(
        core,
        "get_fleet_sim_container_image",
        lambda image=None, pull_policy=None: sim_checks.append((image, pull_policy)),
    )

    core.ensure_runtime_images_for_pull_policy(
        image="router:test",
        router_image=None,
        envoy_image=None,
        dashboard_image=None,
        sim_image="sim:test",
        pull_policy="never",
        env_vars={"VLLM_SR_SIM_ENABLED": "false"},
    )

    assert sim_checks == []
