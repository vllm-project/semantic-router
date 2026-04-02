from types import SimpleNamespace

import pytest
from cli import core, docker_cli, docker_services, docker_start, runtime_lifecycle
from cli.runtime_stack import resolve_runtime_stack


@pytest.fixture(autouse=True)
def _split_runtime_topology(monkeypatch):
    monkeypatch.setenv("VLLM_SR_TOPOLOGY", "split")


def _capture_run_commands(monkeypatch):
    captured = []

    def fake_run(cmd, capture_output, text, check):
        captured.append(cmd)
        return SimpleNamespace(stdout="container-id\n", stderr="")

    monkeypatch.setattr(docker_start.subprocess, "run", fake_run)
    monkeypatch.setattr(
        docker_start, "_render_split_envoy_config", lambda *args, **kwargs: None
    )
    return captured


def _find_container_run_cmd(commands, container_name):
    for cmd in commands:
        if "--name" not in cmd:
            continue
        if cmd[cmd.index("--name") + 1] == container_name:
            return cmd
    raise AssertionError(
        f"container command for {container_name} not found: {commands!r}"
    )


def _stub_valid_docker_cli(monkeypatch, tmp_path):
    docker_bin = tmp_path / "docker"
    docker_bin.write_text("")
    monkeypatch.setattr(
        docker_start,
        "resolve_docker_cli_path",
        lambda preferred_path=None: str(docker_bin),
    )
    return docker_bin


def test_docker_start_vllm_sr_honors_legacy_override(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.1\nlisteners:\n  - name: http-8899\n    address: 0.0.0.0\n    port: 8899\n"
    )

    monkeypatch.setenv("VLLM_SR_TOPOLOGY", "legacy")
    monkeypatch.setattr(docker_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        docker_start, "get_docker_image", lambda **kwargs: "legacy-image:latest"
    )
    monkeypatch.setattr(
        docker_start,
        "get_runtime_images",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("split image resolution should not run for legacy topology")
        ),
    )
    captured = _capture_run_commands(monkeypatch)
    _stub_valid_docker_cli(monkeypatch, tmp_path)

    rc, _, _ = docker_cli.docker_start_vllm_sr(
        str(config_path),
        {},
        [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        network_name="vllm-sr-network",
        openclaw_network_name="vllm-sr-network",
        minimal=False,
    )

    assert rc == 0
    assert len(captured) == 1
    container_cmd = _find_container_run_cmd(captured, "vllm-sr-container")
    assert "legacy-image:latest" in container_cmd
    assert "VLLM_SR_ROUTER_CONTAINER_NAME=vllm-sr-router-container" not in container_cmd
    assert "VLLM_SR_ENVOY_CONTAINER_NAME=vllm-sr-envoy-container" not in container_cmd
    assert (
        "VLLM_SR_DASHBOARD_CONTAINER_NAME=vllm-sr-dashboard-container"
        not in container_cmd
    )
    assert "-p" in container_cmd
    assert "8700:8700" in " ".join(container_cmd)


def test_docker_start_vllm_sr_sets_openclaw_shared_network_env(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.1\nlisteners:\n  - name: http-8899\n    address: 0.0.0.0\n    port: 8899\n"
    )

    socket_path = tmp_path / "docker.sock"
    socket_path.write_text("")
    _stub_valid_docker_cli(monkeypatch, tmp_path)

    monkeypatch.setattr(docker_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        docker_start,
        "get_runtime_images",
        lambda **kwargs: {
            "router": "test-image",
            "envoy": "test-image",
            "dashboard": "test-image",
        },
    )
    captured = _capture_run_commands(monkeypatch)
    monkeypatch.setenv("VLLM_SR_DOCKER_SOCKET", str(socket_path))

    rc, _, _ = docker_cli.docker_start_vllm_sr(
        str(config_path),
        {},
        [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        network_name=None,
        openclaw_network_name="vllm-sr-network",
        minimal=False,
    )

    assert rc == 0
    dashboard_cmd = _find_container_run_cmd(captured, "vllm-sr-dashboard-container")
    assert "--network" in dashboard_cmd
    assert "vllm-sr-network" in dashboard_cmd
    assert "-e" in dashboard_cmd
    assert "OPENCLAW_DEFAULT_NETWORK_MODE=vllm-sr-network" in dashboard_cmd
    assert "TARGET_ENVOY_ADMIN_URL=http://vllm-sr-envoy-container:9901" in dashboard_cmd
    assert "VLLM_SR_ENVOY_CONFIG_PATH=/app/.vllm-sr/envoy.yaml" in dashboard_cmd


def test_docker_start_vllm_sr_places_dashboard_openclaw_runtime_flags_before_image(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.1\nlisteners:\n  - name: http-8899\n    address: 0.0.0.0\n    port: 8899\n"
    )

    socket_path = tmp_path / "docker.sock"
    socket_path.write_text("")
    monkeypatch.setenv("VLLM_SR_DOCKER_SOCKET", str(socket_path))
    monkeypatch.setattr(docker_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        docker_start,
        "get_runtime_images",
        lambda **kwargs: {
            "router": "router-image:latest",
            "envoy": "envoy-image:latest",
            "dashboard": "dashboard-image:latest",
        },
    )
    captured = _capture_run_commands(monkeypatch)
    docker_bin = _stub_valid_docker_cli(monkeypatch, tmp_path)

    rc, _, _ = docker_cli.docker_start_vllm_sr(
        str(config_path),
        {},
        [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        network_name="vllm-sr-network",
        openclaw_network_name="vllm-sr-network",
        minimal=False,
    )

    assert rc == 0
    dashboard_cmd = _find_container_run_cmd(captured, "vllm-sr-dashboard-container")
    image_index = dashboard_cmd.index("dashboard-image:latest")
    socket_mount_index = dashboard_cmd.index(f"{socket_path}:/var/run/docker.sock")
    docker_mount_index = dashboard_cmd.index(f"{docker_bin}:/usr/local/bin/docker:ro")
    runtime_env_index = dashboard_cmd.index(
        "OPENCLAW_CONTAINER_RUNTIME=/usr/local/bin/docker"
    )

    assert socket_mount_index < image_index
    assert docker_mount_index < image_index
    assert runtime_env_index < image_index


def test_docker_start_vllm_sr_mounts_host_docker_cli_by_default(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.1\nlisteners:\n  - name: http-8899\n    address: 0.0.0.0\n    port: 8899\n"
    )

    docker_bin = _stub_valid_docker_cli(monkeypatch, tmp_path)
    monkeypatch.setattr(docker_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        docker_start,
        "get_runtime_images",
        lambda **kwargs: {
            "router": "test-image",
            "envoy": "test-image",
            "dashboard": "test-image",
        },
    )
    captured = _capture_run_commands(monkeypatch)

    rc, _, _ = docker_cli.docker_start_vllm_sr(
        str(config_path),
        {},
        [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        network_name="vllm-sr-network",
        openclaw_network_name="vllm-sr-network",
        minimal=False,
    )

    assert rc == 0
    dashboard_cmd = _find_container_run_cmd(captured, "vllm-sr-dashboard-container")
    assert "OPENCLAW_CONTAINER_RUNTIME=/usr/local/bin/docker" in dashboard_cmd
    assert f"{docker_bin}:/usr/local/bin/docker:ro" in dashboard_cmd


def test_docker_start_vllm_sr_uses_in_image_docker_cli_when_opted_out(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.1\nlisteners:\n  - name: http-8899\n    address: 0.0.0.0\n    port: 8899\n"
    )

    docker_bin = _stub_valid_docker_cli(monkeypatch, tmp_path)
    monkeypatch.setenv("VLLM_SR_MOUNT_DOCKER_CLI", "0")
    monkeypatch.setattr(docker_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        docker_start,
        "get_runtime_images",
        lambda **kwargs: {
            "router": "test-image",
            "envoy": "test-image",
            "dashboard": "test-image",
        },
    )
    captured = _capture_run_commands(monkeypatch)

    rc, _, _ = docker_cli.docker_start_vllm_sr(
        str(config_path),
        {},
        [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        network_name="vllm-sr-network",
        openclaw_network_name="vllm-sr-network",
        minimal=False,
    )

    assert rc == 0
    dashboard_cmd = _find_container_run_cmd(captured, "vllm-sr-dashboard-container")
    assert "OPENCLAW_CONTAINER_RUNTIME=docker" in dashboard_cmd
    assert f"{docker_bin}:/usr/local/bin/docker:ro" not in dashboard_cmd


def test_docker_start_vllm_sr_mounts_host_docker_cli_when_requested(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.1\nlisteners:\n  - name: http-8899\n    address: 0.0.0.0\n    port: 8899\n"
    )

    docker_bin = _stub_valid_docker_cli(monkeypatch, tmp_path)
    monkeypatch.setattr(docker_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        docker_start,
        "get_runtime_images",
        lambda **kwargs: {
            "router": "test-image",
            "envoy": "test-image",
            "dashboard": "test-image",
        },
    )
    captured = _capture_run_commands(monkeypatch)
    monkeypatch.setenv("VLLM_SR_MOUNT_DOCKER_CLI", "1")

    rc, _, _ = docker_cli.docker_start_vllm_sr(
        str(config_path),
        {},
        [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        network_name="vllm-sr-network",
        openclaw_network_name="vllm-sr-network",
        minimal=False,
    )

    assert rc == 0
    dashboard_cmd = _find_container_run_cmd(captured, "vllm-sr-dashboard-container")
    assert "OPENCLAW_CONTAINER_RUNTIME=/usr/local/bin/docker" in dashboard_cmd
    assert f"{docker_bin}:/usr/local/bin/docker:ro" in dashboard_cmd


def test_docker_start_vllm_sr_mounts_dashboard_data_dir(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.1\nlisteners:\n  - name: http-8899\n    address: 0.0.0.0\n    port: 8899\n"
    )

    monkeypatch.setattr(docker_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        docker_start,
        "get_runtime_images",
        lambda **kwargs: {
            "router": "test-image",
            "envoy": "test-image",
            "dashboard": "test-image",
        },
    )
    captured = _capture_run_commands(monkeypatch)
    _stub_valid_docker_cli(monkeypatch, tmp_path)

    rc, _, _ = docker_cli.docker_start_vllm_sr(
        str(config_path),
        {},
        [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        network_name=None,
        openclaw_network_name="vllm-sr-network",
        minimal=False,
    )

    dashboard_data_dir = tmp_path / ".vllm-sr" / "dashboard-data"

    assert rc == 0
    assert dashboard_data_dir.is_dir()
    dashboard_cmd = _find_container_run_cmd(captured, "vllm-sr-dashboard-container")
    assert f"{dashboard_data_dir}:/app/data:z" in dashboard_cmd


def test_docker_start_vllm_sr_uses_state_root_for_mounts(tmp_path, monkeypatch):
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    config_path = tmp_path / "runtime-overrides" / "config-with-platform-overrides.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        "version: v0.3\nlisteners:\n  - name: http-8899\n    address: 0.0.0.0\n    port: 8899\n"
    )

    monkeypatch.setattr(docker_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        docker_start,
        "get_runtime_images",
        lambda **kwargs: {
            "router": "test-image",
            "envoy": "test-image",
            "dashboard": "test-image",
        },
    )
    captured = _capture_run_commands(monkeypatch)
    _stub_valid_docker_cli(monkeypatch, tmp_path)

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
    router_cmd = _find_container_run_cmd(captured, "vllm-sr-router-container")
    assert f"{models_dir}:/app/models:z" in router_cmd


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

    monkeypatch.setattr(docker_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        docker_start,
        "get_runtime_images",
        lambda **kwargs: {
            "router": "test-image",
            "envoy": "test-image",
            "dashboard": "test-image",
        },
    )
    captured = _capture_run_commands(monkeypatch)
    _stub_valid_docker_cli(monkeypatch, tmp_path)

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
    router_cmd = _find_container_run_cmd(captured, "vllm-sr-router-container")
    envoy_cmd = _find_container_run_cmd(captured, "vllm-sr-envoy-container")
    assert f"{source_config_path}:/app/config.yaml:z" in router_cmd
    assert (
        f"{workspace_dir / '.vllm-sr' / 'envoy.yaml'}:/etc/envoy/envoy.yaml:z"
        in envoy_cmd
    )
    assert "VLLM_SR_RUNTIME_CONFIG_PATH=/app/.vllm-sr/runtime-config.yaml" in router_cmd
    assert (
        "VLLM_SR_RUNTIME_CONFIG_PATH=/app/.vllm-sr/runtime-config.yaml" not in envoy_cmd
    )


def test_render_observability_template_uses_stack_specific_container_hosts():
    stack_layout = resolve_runtime_stack(stack_name="audit-a")
    template = (
        "router: vllm-sr-container\n"
        "metrics: vllm-sr-container:9190\n"
        "prometheus: vllm-sr-prometheus\n"
        "jaeger: vllm-sr-jaeger\n"
    )

    rendered = docker_services.render_observability_template(template, stack_layout)

    assert "router: audit-a-vllm-sr-container" in rendered
    assert "metrics: audit-a-vllm-sr-router-container:9190" in rendered
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
    monkeypatch.setattr(core, "ensure_clean_runtime_container", lambda _name: None)
    monkeypatch.setattr(
        core,
        "load_config",
        lambda path: {
            "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}]
        },
    )
    monkeypatch.setattr(
        core, "provision_storage_backends", lambda *args, **kwargs: set()
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
    assert start_calls[0][2]["network_name"] == "audit-a-vllm-sr-network"
    assert start_calls[0][2]["openclaw_network_name"] == "audit-a-vllm-sr-network"
    assert (
        start_calls[0][2]["stack_layout"].container_name == "audit-a-vllm-sr-container"
    )
    assert (
        start_calls[0][2]["env_vars"]["TARGET_FLEET_SIM_URL"]
        == "http://audit-a-vllm-sr-sim:8000"
    )
    assert [call[1] for call in connect_calls] == [
        ("audit-a-vllm-sr-network", "audit-a-vllm-sr-router-container"),
        ("audit-a-vllm-sr-network", "audit-a-vllm-sr-envoy-container"),
        ("audit-a-vllm-sr-network", "audit-a-vllm-sr-dashboard-container"),
    ]


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


def test_runtime_service_container_name_prefers_split_runtime_container(monkeypatch):
    stack_layout = resolve_runtime_stack(stack_name="audit-a")
    monkeypatch.setattr(
        core,
        "docker_container_status",
        lambda name: (
            "running" if name == stack_layout.router_container_name else "not found"
        ),
    )

    assert (
        core._runtime_service_container_name("router", stack_layout)
        == stack_layout.router_container_name
    )


def test_runtime_service_container_name_falls_back_to_legacy_container(monkeypatch):
    stack_layout = resolve_runtime_stack(stack_name="audit-a")
    monkeypatch.setattr(
        core,
        "docker_container_status",
        lambda name: "running" if name == stack_layout.container_name else "not found",
    )

    assert (
        core._runtime_service_container_name("router", stack_layout)
        == stack_layout.container_name
    )


def test_runtime_stack_status_detects_split_runtime_when_legacy_container_missing(
    monkeypatch,
):
    stack_layout = resolve_runtime_stack(stack_name="audit-a")
    monkeypatch.setattr(
        core,
        "docker_container_status",
        lambda name: (
            "running" if name == stack_layout.dashboard_container_name else "not found"
        ),
    )

    assert core._runtime_stack_status(stack_layout) == "running"


def test_stop_vllm_sr_stops_split_runtime_containers(monkeypatch):
    calls = []
    stack_layout = resolve_runtime_stack(stack_name="audit-a")
    status_map = {
        stack_layout.container_name: "not found",
        stack_layout.router_container_name: "running",
        stack_layout.envoy_container_name: "exited",
        stack_layout.dashboard_container_name: "running",
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
    monkeypatch.setattr(core, "resolve_openclaw_data_dir", lambda cwd: "/tmp/openclaw")
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

    assert stop_calls == [
        stack_layout.router_container_name,
        stack_layout.dashboard_container_name,
    ]
    assert remove_calls == [
        stack_layout.router_container_name,
        stack_layout.envoy_container_name,
        stack_layout.dashboard_container_name,
    ]


def test_ensure_clean_runtime_container_removes_created_container_without_stop(
    monkeypatch,
):
    calls = []

    monkeypatch.setattr(
        runtime_lifecycle,
        "docker_container_status",
        lambda _name: "created",
    )
    monkeypatch.setattr(
        runtime_lifecycle,
        "docker_stop_container",
        lambda name: calls.append(("stop", name)),
    )
    monkeypatch.setattr(
        runtime_lifecycle,
        "docker_remove_container",
        lambda name: calls.append(("remove", name)),
    )

    runtime_lifecycle.ensure_clean_runtime_container("vllm-sr-router-container")

    assert calls == [("remove", "vllm-sr-router-container")]
