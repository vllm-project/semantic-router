import importlib
import sys
from pathlib import Path

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

    monkeypatch.setattr(core, "docker_exec", fake_exec)
    monkeypatch.setattr(core, "docker_container_status", lambda _name: "running")

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

    monkeypatch.setattr(core, "docker_exec", fake_exec)
    monkeypatch.setattr(core, "docker_container_status", lambda _name: "running")

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

    monkeypatch.setattr(core, "docker_exec", fake_exec)
    monkeypatch.setattr(core, "docker_container_status", lambda _name: "running")

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


def test_show_status_reports_not_running_when_docker_daemon_unreachable(monkeypatch):
    messages = []
    stack_layout = runtime_stack.resolve_runtime_stack()

    monkeypatch.setattr(core, "resolve_runtime_stack", lambda: stack_layout)
    monkeypatch.setattr(
        core,
        "_runtime_stack_status",
        lambda _stack_layout: (_ for _ in ()).throw(SystemExit(1)),
    )
    monkeypatch.setattr(core.log, "info", messages.append)

    core.show_status()

    assert "Status: Not running" in messages
    assert any("Docker daemon is not reachable" in message for message in messages)
    assert "Start with: vllm-sr serve" in messages


def test_never_pull_preflight_skips_dashboard_when_disabled(monkeypatch):
    captured = {}
    sim_checks = []

    def fake_get_runtime_images(**kwargs):
        captured.update(kwargs)
        return {"router": "router:test", "envoy": "envoy:test"}

    monkeypatch.setattr(core, "get_runtime_images", fake_get_runtime_images)
    monkeypatch.setattr(
        core,
        "get_fleet_sim_docker_image",
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
        "get_fleet_sim_docker_image",
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
