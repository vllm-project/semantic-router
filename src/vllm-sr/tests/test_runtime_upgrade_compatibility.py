import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

core = importlib.import_module("cli.core")
runtime_stack = importlib.import_module("cli.runtime_stack")


def test_start_vllm_sr_defaults_to_split_and_cleans_legacy_runtime_residue(
    monkeypatch,
):
    stack_layout = runtime_stack.resolve_runtime_stack()
    cleaned = []
    captured = {}

    def _record_cleaned_container(container_name):
        cleaned.append(container_name)

    monkeypatch.delenv("VLLM_SR_TOPOLOGY", raising=False)
    monkeypatch.setattr(core, "print_vllm_logo", lambda: None)
    monkeypatch.setattr(
        core,
        "load_config",
        lambda _path: {
            "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}]
        },
    )
    monkeypatch.setattr(core, "log_startup_banner", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        core,
        "ensure_clean_runtime_container",
        _record_cleaned_container,
    )
    monkeypatch.setattr(core, "ensure_shared_network", lambda _name: None)
    monkeypatch.setattr(core, "start_observability_stack", lambda *args, **kwargs: None)
    monkeypatch.setattr(core, "start_fleet_sim_sidecar", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        core,
        "docker_start_vllm_sr",
        lambda **kwargs: captured.update(kwargs) or (0, "", ""),
    )
    monkeypatch.setattr(core, "connect_runtime_container", lambda *args, **kwargs: None)
    monkeypatch.setattr(core, "maybe_finish_setup_mode", lambda *args, **kwargs: False)
    monkeypatch.setattr(core, "wait_for_router_health", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        core, "ensure_runtime_container_not_exited", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        core, "recover_openclaw_containers", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(core, "log_runtime_summary", lambda *args, **kwargs: None)

    core.start_vllm_sr("/tmp/config.yaml", env_vars={}, enable_observability=False)

    assert captured["topology"] == "split"
    assert cleaned == list(stack_layout.runtime_container_names)


def test_show_status_reports_running_when_only_legacy_runtime_container_exists(
    monkeypatch,
):
    stack_layout = runtime_stack.resolve_runtime_stack()
    info_messages = []
    reported_services = []

    def _record_status_log(message):
        info_messages.append(message)

    monkeypatch.setattr(core, "resolve_runtime_stack", lambda: stack_layout)
    monkeypatch.setattr(
        core,
        "docker_container_status",
        lambda name: "running" if name == stack_layout.container_name else "not found",
    )
    monkeypatch.setattr(
        core,
        "_report_service_status",
        lambda service, _stack_layout: reported_services.append(service),
    )
    monkeypatch.setattr(core.log, "info", _record_status_log)

    core.show_status("all")

    assert "Container Status: Running" in info_messages
    assert reported_services == ["router", "envoy", "dashboard", "simulator"]


def test_stop_vllm_sr_stops_legacy_monolith_after_default_split_rollout(monkeypatch):
    calls = []
    stack_layout = runtime_stack.resolve_runtime_stack()
    status_map = {
        stack_layout.container_name: "running",
        stack_layout.router_container_name: "not found",
        stack_layout.envoy_container_name: "not found",
        stack_layout.dashboard_container_name: "not found",
        stack_layout.fleet_sim_container_name: "not found",
        stack_layout.grafana_container_name: "not found",
        stack_layout.prometheus_container_name: "not found",
        stack_layout.jaeger_container_name: "not found",
    }

    def record(name, ret=(0, "", "")):
        def _fn(*args, **kwargs):
            calls.append((name, args, kwargs))
            return ret

        return _fn

    monkeypatch.setattr(core, "resolve_runtime_stack", lambda: stack_layout)
    monkeypatch.setattr(
        core,
        "docker_container_status",
        lambda name: status_map.get(name, "not found"),
    )
    monkeypatch.setattr(core, "resolve_openclaw_data_dir", lambda _cwd: "/tmp/openclaw")
    monkeypatch.setattr(core, "load_openclaw_registry", lambda _path: [])
    monkeypatch.setattr(core, "docker_stop_container", record("docker_stop_container"))
    monkeypatch.setattr(
        core,
        "docker_remove_container",
        record("docker_remove_container"),
    )
    monkeypatch.setattr(
        core,
        "docker_network_disconnect",
        record("docker_network_disconnect"),
    )
    monkeypatch.setattr(
        core,
        "docker_remove_network",
        record("docker_remove_network"),
    )

    core.stop_vllm_sr()

    stop_calls = [args[0] for name, args, _ in calls if name == "docker_stop_container"]
    remove_calls = [
        args[0] for name, args, _ in calls if name == "docker_remove_container"
    ]
    remove_network_calls = [
        args for name, args, _ in calls if name == "docker_remove_network"
    ]

    assert stop_calls == [stack_layout.container_name]
    assert remove_calls == [stack_layout.container_name]
    assert remove_network_calls == [(stack_layout.network_name,)]
