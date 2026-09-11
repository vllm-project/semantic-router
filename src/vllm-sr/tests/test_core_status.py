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
runtime_service_status = importlib.import_module("cli.runtime_service_status")
storage_backends = importlib.import_module("cli.storage_backends")


def test_check_envoy_status_uses_ready_probe_when_available(monkeypatch):
    stack_layout = runtime_stack.resolve_runtime_stack()
    captured = []

    def fake_exec(container_name, command):
        captured.append((container_name, command))
        return (0, "200", "")

    monkeypatch.setattr(runtime_service_status, "container_exec", fake_exec)
    monkeypatch.setattr(
        runtime_service_status, "container_status", lambda _name: "running"
    )

    assert runtime_service_status._check_envoy_status(
        stack_layout.envoy_container_name, stack_layout
    )
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

    monkeypatch.setattr(runtime_service_status, "container_exec", fake_exec)
    monkeypatch.setattr(
        runtime_service_status, "container_status", lambda _name: "running"
    )

    assert runtime_service_status._check_envoy_status(
        stack_layout.envoy_container_name, stack_layout
    )
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

    monkeypatch.setattr(runtime_service_status, "container_exec", fake_exec)
    monkeypatch.setattr(
        runtime_service_status, "container_status", lambda _name: "running"
    )

    assert (
        runtime_service_status._check_envoy_status(
            stack_layout.router_container_name, stack_layout
        )
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
    assert removed_networks == [
        stack_layout.network_name,
        stack_layout.data_network_name,
    ]


def test_stop_propagates_orphan_network_removal_failure(monkeypatch, capsys):
    stack_layout = runtime_stack.resolve_runtime_stack()
    managed_names = (
        *stack_layout.runtime_container_names,
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


def test_stop_reports_success_when_only_dashboard_exists(monkeypatch, capsys):
    stack_layout = runtime_stack.resolve_runtime_stack()
    managed_names = (
        *stack_layout.runtime_container_names,
        stack_layout.grafana_container_name,
        stack_layout.prometheus_container_name,
        stack_layout.jaeger_container_name,
        *stack_layout.storage_container_names,
    )
    statuses = dict.fromkeys(managed_names, "not found")
    statuses[stack_layout.dashboard_container_name] = "running"

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
        stack_layout.grafana_container_name,
        stack_layout.prometheus_container_name,
        stack_layout.jaeger_container_name,
        *stack_layout.storage_container_names,
    )
    statuses = dict.fromkeys(managed_names, "not found")
    statuses[stack_layout.dashboard_container_name] = "running"

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
        assert stack_layout.dashboard_container_name in str(error)
    else:
        raise AssertionError("stop must propagate container removal failures")

    captured = capsys.readouterr()
    assert "✓ vLLM Semantic Router stopped" not in captured.out


def test_stop_does_not_report_success_when_network_removal_fails(monkeypatch, capsys):
    stack_layout = runtime_stack.resolve_runtime_stack()
    managed_names = (
        *stack_layout.runtime_container_names,
        stack_layout.grafana_container_name,
        stack_layout.prometheus_container_name,
        stack_layout.jaeger_container_name,
        *stack_layout.storage_container_names,
    )
    statuses = dict.fromkeys(managed_names, "not found")
    statuses[stack_layout.dashboard_container_name] = "running"

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


def test_followed_router_logs_merge_container_stderr_into_raw_stdout(monkeypatch):
    stack_layout = runtime_stack.resolve_runtime_stack()
    calls = []

    monkeypatch.setattr(core, "resolve_runtime_stack", lambda: stack_layout)
    monkeypatch.setattr(core, "_ensure_runtime_container_available", lambda _name: None)
    monkeypatch.setattr(
        core,
        "container_logs",
        lambda *args, **kwargs: calls.append((args, kwargs)) or True,
    )

    core.show_logs("router", follow=True)

    assert calls == [
        (
            (stack_layout.router_container_name,),
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
        "service-container", follow=True, tail=200, merge_output=True
    )

    assert captured == {
        "command": ["docker", "logs", "-f", "--tail", "200", "service-container"],
        "kwargs": {"check": True, "stderr": subprocess.STDOUT},
    }


def test_never_pull_preflight_skips_dashboard_when_disabled(monkeypatch):
    captured = {}

    def fake_get_runtime_images(**kwargs):
        captured.update(kwargs)
        return {"router": "router:test", "envoy": "envoy:test"}

    monkeypatch.setattr(core, "get_runtime_images", fake_get_runtime_images)
    core.ensure_runtime_images_for_pull_policy(
        image="router:test",
        router_image=None,
        envoy_image=None,
        dashboard_image="dashboard:missing",
        pull_policy="never",
        env_vars={"VLLM_SR_PLATFORM": "amd"},
        dashboard_disabled=True,
    )

    assert captured["dashboard_image"] is None
    assert captured["include_dashboard"] is False


def _stop_environment(monkeypatch, stack_layout, statuses, stopped, removed):
    monkeypatch.setattr(core, "resolve_runtime_stack", lambda: stack_layout)
    monkeypatch.setattr(
        core, "_managed_container_statuses", lambda _stack_layout: statuses
    )
    monkeypatch.setattr(core, "resolve_openclaw_data_dir", lambda _cwd: "/unused")
    monkeypatch.setattr(core, "load_openclaw_registry", lambda _path: [])
    monkeypatch.setattr(
        core,
        "container_stop_container",
        lambda name: stopped.append(name) or True,
    )
    monkeypatch.setattr(
        core,
        "container_remove_container",
        lambda name: removed.append(name) or True,
    )
    monkeypatch.setattr(core, "container_remove_network", lambda _name: (0, "", ""))
    monkeypatch.setattr(
        core,
        "container_network_disconnect_if_attached",
        lambda _network, _name: (0, "", ""),
    )


def _all_managed_names(stack_layout):
    return (
        *stack_layout.runtime_container_names,
        stack_layout.grafana_container_name,
        stack_layout.prometheus_container_name,
        stack_layout.jaeger_container_name,
        *stack_layout.storage_container_names,
    )


def test_stop_keeps_a_storage_container_whose_data_volume_nobody_recorded(
    monkeypatch, capsys
):
    stack_layout = runtime_stack.resolve_runtime_stack()
    statuses = dict.fromkeys(_all_managed_names(stack_layout), "not found")
    statuses[stack_layout.redis_container_name] = "running"
    statuses[stack_layout.postgres_container_name] = "running"
    stopped, removed = [], []
    _stop_environment(monkeypatch, stack_layout, statuses, stopped, removed)
    # An old container carries neither credential mount, so its anonymous
    # volume has no recorded name and removing it would orphan the data.
    monkeypatch.setattr(
        storage_backends, "container_mount_destinations", lambda _name: {"/etc/hosts"}
    )

    disconnected = []
    monkeypatch.setattr(
        storage_backends,
        "container_network_disconnect_if_attached",
        lambda network, name: disconnected.append((network, name)) or (0, "", ""),
    )

    core.stop_vllm_sr()

    assert stopped == [
        stack_layout.redis_container_name,
        stack_layout.postgres_container_name,
    ]
    assert removed == []
    # A container kept on a stack network blocks removing that network under
    # Podman, so a preserved container is detached from both. A container old
    # enough to be preserved sits on the application network, but `stop` cannot
    # tell that from the container alone and both networks have to go.
    assert disconnected == [
        (stack_layout.network_name, stack_layout.redis_container_name),
        (stack_layout.data_network_name, stack_layout.redis_container_name),
        (stack_layout.network_name, stack_layout.postgres_container_name),
        (stack_layout.data_network_name, stack_layout.postgres_container_name),
    ]
    captured = capsys.readouterr()
    assert "stopped but kept" in captured.err
    assert "next `vllm-sr serve` adopts the volume" in captured.err


def test_stop_removes_a_storage_container_this_cli_provisioned(monkeypatch):
    stack_layout = runtime_stack.resolve_runtime_stack()
    statuses = dict.fromkeys(_all_managed_names(stack_layout), "not found")
    statuses[stack_layout.redis_container_name] = "running"
    statuses[stack_layout.postgres_container_name] = "exited"
    stopped, removed = [], []
    _stop_environment(monkeypatch, stack_layout, statuses, stopped, removed)
    mounts = {
        stack_layout.redis_container_name: {
            storage_backends.CONTAINER_REDIS_CONF_PATH,
            "/data",
        },
        stack_layout.postgres_container_name: {
            storage_backends.CONTAINER_POSTGRES_PASSWORD_PATH,
            "/var/lib/postgresql/data",
        },
    }
    monkeypatch.setattr(storage_backends, "container_mount_destinations", mounts.get)

    core.stop_vllm_sr()

    assert stopped == [stack_layout.redis_container_name]
    assert removed == [
        stack_layout.redis_container_name,
        stack_layout.postgres_container_name,
    ]


def test_stop_keeps_a_storage_container_when_the_runtime_cannot_report_mounts(
    monkeypatch,
):
    stack_layout = runtime_stack.resolve_runtime_stack()
    statuses = dict.fromkeys(_all_managed_names(stack_layout), "not found")
    statuses[stack_layout.redis_container_name] = "running"
    stopped, removed = [], []
    _stop_environment(monkeypatch, stack_layout, statuses, stopped, removed)
    monkeypatch.setattr(
        storage_backends, "container_mount_destinations", lambda _name: None
    )
    monkeypatch.setattr(
        core,
        "detach_preserved_storage_container",
        lambda _networks, _name: None,
    )

    core.stop_vllm_sr()

    assert stopped == [stack_layout.redis_container_name]
    assert removed == []


def test_stop_still_removes_milvus_which_keeps_its_data_on_a_bind_mount(monkeypatch):
    stack_layout = runtime_stack.resolve_runtime_stack()
    statuses = dict.fromkeys(_all_managed_names(stack_layout), "not found")
    statuses[stack_layout.milvus_container_name] = "running"
    stopped, removed = [], []
    _stop_environment(monkeypatch, stack_layout, statuses, stopped, removed)

    def unreachable(name):
        raise AssertionError(f"Milvus must not be mount-inspected: {name}")

    monkeypatch.setattr(storage_backends, "container_mount_destinations", unreachable)

    core.stop_vllm_sr()

    assert removed == [stack_layout.milvus_container_name]
