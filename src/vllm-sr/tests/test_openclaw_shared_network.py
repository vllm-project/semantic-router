from types import SimpleNamespace

from cli import core, docker_cli, docker_services, docker_start, runtime_lifecycle
from cli.consts import (
    DEFAULT_API_PORT,
    DEFAULT_DASHBOARD_PORT,
    DEFAULT_FLEET_SIM_PORT,
    DEFAULT_METRICS_PORT,
    DEFAULT_ROUTER_PORT,
)
from cli.runtime_stack import resolve_runtime_stack


def test_docker_start_vllm_sr_sets_openclaw_shared_network_env(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.1\nlisteners:\n  - name: http-8899\n    address: 0.0.0.0\n    port: 8899\n"
    )

    socket_path = tmp_path / "docker.sock"
    socket_path.write_text("")
    docker_bin = tmp_path / "docker"
    docker_bin.write_text("")

    captured = {}

    def fake_run(cmd, capture_output, text, check):
        captured["cmd"] = cmd
        return SimpleNamespace(stdout="container-id\n", stderr="")

    monkeypatch.setattr(docker_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(docker_start, "get_docker_image", lambda **kwargs: "test-image")
    monkeypatch.setattr(docker_start.subprocess, "run", fake_run)
    monkeypatch.setattr(
        docker_start.shutil,
        "which",
        lambda name: str(docker_bin) if name == "docker" else None,
    )
    monkeypatch.setenv("VLLM_SR_DOCKER_SOCKET", str(socket_path))

    rc, _, _ = docker_cli.docker_start_vllm_sr(
        str(config_path),
        {},
        [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        network_name=None,
        openclaw_network_name="vllm-sr-network",
        minimal=True,
    )

    assert rc == 0
    assert "cmd" in captured
    assert "-e" in captured["cmd"]
    assert "OPENCLAW_DEFAULT_NETWORK_MODE=vllm-sr-network" in captured["cmd"]


def test_docker_start_vllm_sr_mounts_dashboard_data_dir(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.1\nlisteners:\n  - name: http-8899\n    address: 0.0.0.0\n    port: 8899\n"
    )

    captured = {}

    def fake_run(cmd, capture_output, text, check):
        captured["cmd"] = cmd
        return SimpleNamespace(stdout="container-id\n", stderr="")

    monkeypatch.setattr(docker_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(docker_start, "get_docker_image", lambda **kwargs: "test-image")
    monkeypatch.setattr(docker_start.subprocess, "run", fake_run)
    monkeypatch.setattr(docker_start.shutil, "which", lambda name: None)

    rc, _, _ = docker_cli.docker_start_vllm_sr(
        str(config_path),
        {},
        [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        network_name=None,
        openclaw_network_name="vllm-sr-network",
        minimal=True,
    )

    dashboard_data_dir = tmp_path / ".vllm-sr" / "dashboard-data"

    assert rc == 0
    assert dashboard_data_dir.is_dir()
    assert f"{dashboard_data_dir}:/app/data:z" in captured["cmd"]


def test_docker_start_vllm_sr_uses_state_root_for_mounts(tmp_path, monkeypatch):
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    config_path = tmp_path / "runtime-overrides" / "config-with-platform-overrides.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        "version: v0.3\nlisteners:\n  - name: http-8899\n    address: 0.0.0.0\n    port: 8899\n"
    )

    captured = {}

    def fake_run(cmd, capture_output, text, check):
        captured["cmd"] = cmd
        return SimpleNamespace(stdout="container-id\n", stderr="")

    monkeypatch.setattr(docker_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(docker_start, "get_docker_image", lambda **kwargs: "test-image")
    monkeypatch.setattr(docker_start.subprocess, "run", fake_run)
    monkeypatch.setattr(docker_start.shutil, "which", lambda name: None)

    rc, _, _ = docker_cli.docker_start_vllm_sr(
        str(config_path),
        {},
        [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        network_name=None,
        openclaw_network_name="vllm-sr-network",
        minimal=True,
        state_root_dir=str(workspace_dir),
    )

    dashboard_data_dir = workspace_dir / ".vllm-sr" / "dashboard-data"
    models_dir = workspace_dir / "models"

    assert rc == 0
    assert dashboard_data_dir.is_dir()
    assert models_dir.is_dir()
    assert f"{dashboard_data_dir}:/app/data:z" in captured["cmd"]
    assert f"{models_dir}:/app/models:z" in captured["cmd"]


def test_docker_start_vllm_sr_keeps_source_config_mount_with_runtime_override(
    tmp_path, monkeypatch
):
    workspace_dir = tmp_path / "workspace"
    runtime_dir = workspace_dir / ".vllm-sr"
    workspace_dir.mkdir()
    runtime_dir.mkdir()
    source_config_path = workspace_dir / "config.yaml"
    source_config_path.write_text(
        "version: v0.3\nlisteners:\n  - name: http-8899\n    address: 0.0.0.0\n    port: 8899\n"
    )
    runtime_config_path = runtime_dir / "runtime-config.yaml"
    runtime_config_path.write_text(
        "version: v0.3\nlisteners:\n  - name: http-8899\n    address: 0.0.0.0\n    port: 8899\n"
    )

    captured = {}

    def fake_run(cmd, capture_output, text, check):
        captured["cmd"] = cmd
        return SimpleNamespace(stdout="container-id\n", stderr="")

    monkeypatch.setattr(docker_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(docker_start, "get_docker_image", lambda **kwargs: "test-image")
    monkeypatch.setattr(docker_start.subprocess, "run", fake_run)
    monkeypatch.setattr(docker_start.shutil, "which", lambda name: None)

    rc, _, _ = docker_cli.docker_start_vllm_sr(
        str(source_config_path),
        {},
        [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        network_name=None,
        openclaw_network_name="vllm-sr-network",
        minimal=True,
        state_root_dir=str(workspace_dir),
        runtime_config_file=str(runtime_config_path),
    )

    assert rc == 0
    assert f"{source_config_path}:/app/config.yaml:z" in captured["cmd"]
    assert (
        "VLLM_SR_RUNTIME_CONFIG_PATH=/app/.vllm-sr/runtime-config.yaml"
        in captured["cmd"]
    )


def test_start_vllm_sr_creates_and_connects_shared_network_without_observability(
    monkeypatch,
):
    calls = []

    def record(name, ret=(0, "", "")):
        def _fn(*args, **kwargs):
            calls.append((name, args, kwargs))
            return ret

        return _fn

    monkeypatch.setattr(core, "print_vllm_logo", lambda: None)
    monkeypatch.setattr(
        core,
        "load_config",
        lambda path: {
            "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}]
        },
    )
    monkeypatch.setattr(
        runtime_lifecycle,
        "docker_container_status",
        lambda name: "not found" if name == "vllm-sr-container" else "running",
    )
    monkeypatch.setattr(
        runtime_lifecycle, "docker_create_network", record("docker_create_network")
    )
    monkeypatch.setattr(
        core, "start_fleet_sim_sidecar", record("start_fleet_sim_sidecar", True)
    )
    monkeypatch.setattr(core, "docker_start_vllm_sr", record("docker_start_vllm_sr"))
    monkeypatch.setattr(
        runtime_lifecycle, "docker_network_connect", record("docker_network_connect")
    )
    monkeypatch.setattr(
        runtime_lifecycle, "docker_logs_since", lambda *args, **kwargs: (0, "", "")
    )
    monkeypatch.setattr(
        runtime_lifecycle, "docker_exec", lambda *args, **kwargs: (0, "ok", "")
    )
    monkeypatch.setattr(
        runtime_lifecycle, "load_openclaw_registry", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(runtime_lifecycle, "docker_logs", lambda *args, **kwargs: None)

    core.start_vllm_sr("/tmp/config.yaml", env_vars={}, enable_observability=False)

    create_calls = [c for c in calls if c[0] == "docker_create_network"]
    fleet_sim_calls = [c for c in calls if c[0] == "start_fleet_sim_sidecar"]
    start_calls = [c for c in calls if c[0] == "docker_start_vllm_sr"]
    connect_calls = [c for c in calls if c[0] == "docker_network_connect"]

    assert create_calls[0][1] == ("vllm-sr-network",)
    assert fleet_sim_calls[0][1][0] == "/tmp"
    assert fleet_sim_calls[0][1][2].fleet_sim_container_name == "vllm-sr-sim-container"
    assert start_calls[0][2]["network_name"] is None
    assert start_calls[0][2]["openclaw_network_name"] == "vllm-sr-network"
    assert start_calls[0][2]["runtime_config_file"] == "/tmp/config.yaml"
    assert (
        start_calls[0][2]["env_vars"]["TARGET_FLEET_SIM_URL"]
        == "http://vllm-sr-sim-container:8000"
    )
    assert connect_calls[0][1] == ("vllm-sr-network", "vllm-sr-container")


def test_resolve_runtime_stack_supports_custom_stack_name_and_port_offset():
    stack_layout = resolve_runtime_stack(stack_name="audit-a", port_offset=200)

    assert stack_layout.container_name == "audit-a-vllm-sr-container"
    assert stack_layout.fleet_sim_container_name == "audit-a-vllm-sr-sim"
    assert stack_layout.network_name == "audit-a-vllm-sr-network"
    assert stack_layout.jaeger_container_name == "audit-a-vllm-sr-jaeger"
    assert stack_layout.prometheus_container_name == "audit-a-vllm-sr-prometheus"
    assert stack_layout.grafana_container_name == "audit-a-vllm-sr-grafana"
    assert stack_layout.router_port == DEFAULT_ROUTER_PORT + 200
    assert stack_layout.metrics_port == DEFAULT_METRICS_PORT + 200
    assert stack_layout.dashboard_port == DEFAULT_DASHBOARD_PORT + 200
    assert stack_layout.api_port == DEFAULT_API_PORT + 200
    assert stack_layout.fleet_sim_port == DEFAULT_FLEET_SIM_PORT + 200


def test_docker_start_vllm_sr_applies_custom_stack_name_and_port_offset(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.1\nlisteners:\n  - name: http-8899\n    address: 0.0.0.0\n    port: 8899\n"
    )

    captured = {}

    def fake_run(cmd, capture_output, text, check):
        captured["cmd"] = cmd
        return SimpleNamespace(stdout="container-id\n", stderr="")

    monkeypatch.setattr(docker_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(docker_start, "get_docker_image", lambda **kwargs: "test-image")
    monkeypatch.setattr(docker_start.subprocess, "run", fake_run)
    monkeypatch.setattr(docker_start.shutil, "which", lambda name: None)
    monkeypatch.setenv("VLLM_SR_STACK_NAME", "audit-a")
    monkeypatch.setenv("VLLM_SR_PORT_OFFSET", "200")

    rc, _, _ = docker_cli.docker_start_vllm_sr(
        str(config_path),
        {},
        [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        network_name=None,
        openclaw_network_name=None,
        minimal=False,
    )

    assert rc == 0
    assert "--name" in captured["cmd"]
    assert "audit-a-vllm-sr-container" in captured["cmd"]
    assert "OPENCLAW_DEFAULT_NETWORK_MODE=audit-a-vllm-sr-network" in captured["cmd"]
    assert "9099:8899" in captured["cmd"]
    assert "50251:50051" in captured["cmd"]
    assert "9390:9190" in captured["cmd"]
    assert "8900:8700" in captured["cmd"]
    assert "8280:8080" in captured["cmd"]


def test_render_observability_template_uses_stack_specific_container_hosts():
    stack_layout = resolve_runtime_stack(stack_name="audit-a")
    template = (
        "router: vllm-sr-container\n"
        "prometheus: vllm-sr-prometheus\n"
        "jaeger: vllm-sr-jaeger\n"
    )

    rendered = docker_services.render_observability_template(template, stack_layout)

    assert "audit-a-vllm-sr-container" in rendered
    assert "audit-a-vllm-sr-prometheus" in rendered
    assert "audit-a-vllm-sr-jaeger" in rendered


def test_start_vllm_sr_uses_isolated_network_and_container_names(monkeypatch):
    calls = []

    def record(name, ret=(0, "", "")):
        def _fn(*args, **kwargs):
            calls.append((name, args, kwargs))
            return ret

        return _fn

    monkeypatch.setenv("VLLM_SR_STACK_NAME", "audit-a")
    monkeypatch.setenv("VLLM_SR_PORT_OFFSET", "200")
    monkeypatch.setattr(core, "print_vllm_logo", lambda: None)
    monkeypatch.setattr(
        core,
        "load_config",
        lambda path: {
            "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}]
        },
    )
    monkeypatch.setattr(
        runtime_lifecycle,
        "docker_container_status",
        lambda name: "not found" if name == "audit-a-vllm-sr-container" else "running",
    )
    monkeypatch.setattr(
        runtime_lifecycle, "docker_create_network", record("docker_create_network")
    )
    monkeypatch.setattr(
        core, "start_fleet_sim_sidecar", record("start_fleet_sim_sidecar", True)
    )
    monkeypatch.setattr(core, "docker_start_vllm_sr", record("docker_start_vllm_sr"))
    monkeypatch.setattr(
        runtime_lifecycle, "docker_network_connect", record("docker_network_connect")
    )
    monkeypatch.setattr(
        runtime_lifecycle, "docker_logs_since", lambda *args, **kwargs: (0, "", "")
    )
    monkeypatch.setattr(
        runtime_lifecycle, "docker_exec", lambda *args, **kwargs: (0, "ok", "")
    )
    monkeypatch.setattr(
        runtime_lifecycle, "load_openclaw_registry", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(runtime_lifecycle, "docker_logs", lambda *args, **kwargs: None)

    core.start_vllm_sr("/tmp/config.yaml", env_vars={}, enable_observability=False)

    create_calls = [c for c in calls if c[0] == "docker_create_network"]
    fleet_sim_calls = [c for c in calls if c[0] == "start_fleet_sim_sidecar"]
    start_calls = [c for c in calls if c[0] == "docker_start_vllm_sr"]
    connect_calls = [c for c in calls if c[0] == "docker_network_connect"]

    assert create_calls[0][1] == ("audit-a-vllm-sr-network",)
    assert fleet_sim_calls[0][1][2].fleet_sim_container_name == "audit-a-vllm-sr-sim"
    assert start_calls[0][2]["openclaw_network_name"] == "audit-a-vllm-sr-network"
    assert (
        start_calls[0][2]["stack_layout"].container_name == "audit-a-vllm-sr-container"
    )
    assert (
        start_calls[0][2]["env_vars"]["TARGET_FLEET_SIM_URL"]
        == "http://audit-a-vllm-sr-sim:8000"
    )
    assert connect_calls[0][1] == (
        "audit-a-vllm-sr-network",
        "audit-a-vllm-sr-container",
    )


def test_stop_vllm_sr_cleans_residual_observability_and_openclaw(monkeypatch):
    calls = []
    stack_layout = resolve_runtime_stack(stack_name="audit-a")
    status_map = {
        stack_layout.container_name: "not found",
        stack_layout.fleet_sim_container_name: "not found",
        stack_layout.grafana_container_name: "running",
        stack_layout.prometheus_container_name: "exited",
        stack_layout.jaeger_container_name: "not found",
        "openclaw-a": "running",
        "openclaw-b": "exited",
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
    monkeypatch.setattr(core, "resolve_openclaw_data_dir", lambda cwd: "/tmp/openclaw")
    monkeypatch.setattr(
        core,
        "load_openclaw_registry",
        lambda _path: [{"name": "openclaw-a"}, {"containerName": "openclaw-b"}, {}],
    )
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
    disconnect_calls = [
        args for name, args, _ in calls if name == "docker_network_disconnect"
    ]
    remove_network_calls = [
        args for name, args, _ in calls if name == "docker_remove_network"
    ]

    assert stop_calls == ["openclaw-a", stack_layout.grafana_container_name]
    assert remove_calls == [
        stack_layout.grafana_container_name,
        stack_layout.prometheus_container_name,
    ]
    assert disconnect_calls == [
        (stack_layout.network_name, "openclaw-a"),
        (stack_layout.network_name, "openclaw-b"),
    ]
    assert remove_network_calls == [(stack_layout.network_name,)]
