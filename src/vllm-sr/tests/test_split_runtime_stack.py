import pytest
from cli import (
    container_cli,
    container_issuer_egress,
    container_start,
    core,
    runtime_lifecycle,
)
from cli.consts import (
    DEFAULT_API_PORT,
    DEFAULT_DASHBOARD_PORT,
    DEFAULT_FLEET_SIM_PORT,
    DEFAULT_METRICS_PORT,
    DEFAULT_ROUTER_PORT,
)
from cli.runtime_stack import resolve_runtime_stack
from split_runtime_test_support import (
    CONFIG_BODY as _CONFIG_BODY,
)
from split_runtime_test_support import (
    capture_run_commands as _capture_run_commands,
)
from split_runtime_test_support import (
    find_container_run_cmd as _find_container_run_cmd,
)
from split_runtime_test_support import (
    option_values as _option_values,
)
from split_runtime_test_support import (
    stub_valid_container_cli as _stub_valid_container_cli,
)


@pytest.fixture(autouse=True)
def _split_runtime_topology(monkeypatch):
    monkeypatch.setenv("VLLM_SR_TOPOLOGY", "split")


def test_core_resolves_management_port_for_startup_health_check():
    assert (
        core._configured_management_port(
            {"global": {"services": {"management_api": {"port": 9090}}}}
        )
        == 9090
    )
    assert core._configured_management_port({}) == 8080


def test_container_start_vllm_sr_sets_split_service_urls_for_dashboard(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_CONFIG_BODY)

    monkeypatch.setattr(container_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_start,
        "get_runtime_images",
        lambda **kwargs: {
            "router": "test-image",
            "envoy": "test-image",
            "dashboard": "test-image",
        },
    )
    issuer_policy_path = tmp_path / "management-issuer-egress.yaml"
    issuer_policy_path.write_text("version: v1\n")
    monkeypatch.setattr(
        container_start,
        "_local_dashboard_runtime_environment",
        lambda *_args: {"DASHBOARD_ISSUER": "https://issuer.local"},
    )
    monkeypatch.setattr(
        container_start,
        "materialize_management_issuer_egress_policy",
        lambda **_kwargs: container_issuer_egress.ManagementIssuerEgressPolicy(
            host_path=issuer_policy_path
        ),
    )
    captured = _capture_run_commands(monkeypatch)
    _stub_valid_container_cli(monkeypatch, tmp_path)

    rc, _, _ = container_cli.container_start_vllm_sr(
        str(config_path),
        {"TARGET_ROUTER_API_URL": "http://stale-router:8080"},
        [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        network_name="vllm-sr-network",
        openclaw_network_name="vllm-sr-network",
        minimal=False,
    )

    assert rc == 0
    dashboard_cmd = _find_container_run_cmd(captured, "vllm-sr-dashboard-container")
    assert "TARGET_ROUTER_API_URL=http://vllm-sr-router-container:8080" in dashboard_cmd
    assert (
        "TARGET_ROUTER_METRICS_URL=http://vllm-sr-router-container:9190/metrics"
        in dashboard_cmd
    )
    assert "TARGET_ENVOY_URL=http://vllm-sr-envoy-container:8899" in dashboard_cmd
    assert "DASHBOARD_ROUTER_PUBLIC_URL=http://localhost:8899" in dashboard_cmd
    assert not any("VLLM_SR_ENVOY_CONFIG_PATH=" in token for token in dashboard_cmd)
    assert "ENVOY_EXTPROC_ADDRESS=vllm-sr-router-container" in dashboard_cmd
    assert "ENVOY_ROUTER_API_ADDRESS=vllm-sr-router-container" in dashboard_cmd
    assert "ROUTER_CONFIG_PATH=/app/.vllm-sr/compiled-bootstrap.yaml" in dashboard_cmd
    assert not any("VLLM_SR_RUNTIME_CONFIG_PATH=" in token for token in dashboard_cmd)
    assert not any("VLLM_SR_SOURCE_CONFIG_PATH=" in token for token in dashboard_cmd)
    router_cmd = _find_container_run_cmd(captured, "vllm-sr-router-container")
    assert (
        "VLLM_SR_INTERNAL_MANAGEMENT_ISSUER_EGRESS_POLICY_FILE="
        "/app/.vllm-sr/management-issuer-egress-policy.yaml"
    ) in router_cmd
    assert (
        f"{issuer_policy_path}:/app/.vllm-sr/management-issuer-egress-policy.yaml:ro,z"
    ) in router_cmd
    assert "127.0.0.1:8080:8080" in router_cmd
    assert "127.0.0.1:50051:50051" in router_cmd
    assert "127.0.0.1:9190:9190" in router_cmd
    envoy_cmd = _find_container_run_cmd(captured, "vllm-sr-envoy-container")
    assert "0.0.0.0:8899:8899" in envoy_cmd

    for command in (router_cmd, envoy_cmd, dashboard_cmd):
        assert command[command.index("--restart") + 1] == "unless-stopped"
    assert (
        "http://127.0.0.1:8080/ready"
        in router_cmd[router_cmd.index("--health-cmd") + 1]
    )
    assert "/ready" in envoy_cmd[envoy_cmd.index("--health-cmd") + 1]
    assert dashboard_cmd[dashboard_cmd.index("--health-cmd") + 1].endswith(
        "http://127.0.0.1:8700/healthz"
    )

    for component, command in (
        ("router", router_cmd),
        ("envoy", envoy_cmd),
        ("dashboard", dashboard_cmd),
    ):
        assert command[command.index("--entrypoint") + 1] == "/bin/sh"
        producer_mounts = [
            mount
            for mount in _option_values(command, "-v")
            if mount.endswith(":/var/log/vllm-sr-producer/current.log:z")
        ]
        assert len(producer_mounts) == 1
        assert producer_mounts[0].split(":", 1)[0].endswith(f"/{component}.log")

    for command in (router_cmd, dashboard_cmd):
        compiled_mounts = [
            mount
            for mount in _option_values(command, "-v")
            if ":/app/.vllm-sr/compiled-bootstrap.yaml:" in mount
        ]
        assert len(compiled_mounts) == 1
        assert compiled_mounts[0].endswith(
            ":/app/.vllm-sr/compiled-bootstrap.yaml:ro,z"
        )

    router_mounts = _option_values(router_cmd, "-v")
    assert not any(
        mount.endswith(":/app/source-config.yaml:ro,z") for mount in router_mounts
    )
    assert not any(mount.endswith(":/app/.vllm-sr:z") for mount in router_mounts)
    assert not any("/app/.vllm-sr/logs" in mount for mount in router_mounts)
    assert not any("/app/.vllm-sr/knowledge_bases" in mount for mount in router_mounts)
    assert router_cmd[-2:] == [
        "/app/start-router.sh",
        "/app/.vllm-sr/compiled-bootstrap.yaml",
    ]
    assert any(
        mount.endswith(":/var/log/vllm-sr:ro,z")
        for mount in _option_values(dashboard_cmd, "-v")
    )
    assert not any(
        mount.endswith(":/app/.vllm-sr:z")
        for mount in _option_values(dashboard_cmd, "-v")
    )
    assert "VLLM_SR_LOG_SPOOL_DIR=/var/log/vllm-sr" in dashboard_cmd
    assert any(
        value.startswith("VLLM_SR_LOG_SPOOL_GID=")
        for value in _option_values(dashboard_cmd, "-e")
    )
    assert "--group-add" in envoy_cmd


def test_split_runtime_honors_management_listener_port(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.3\n"
        "listeners:\n  - name: public\n    address: 0.0.0.0\n    port: 8899\n"
        "global:\n  services:\n"
        "    backend_egress:\n"
        "      policy_file: /app/config/backend-egress-policy.yaml\n"
        "    backend_dispatch:\n"
        "      bind_address: 0.0.0.0\n"
        "      port: 8187\n"
        "      audience: vllm-sr.backend-dispatch\n"
        "      capability_ttl: 30s\n"
        "      max_request_body_bytes: 67108864\n"
        "    management_api:\n"
        "      bind_address: 0.0.0.0\n      port: 9090\n"
    )
    monkeypatch.setattr(container_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_start,
        "get_runtime_images",
        lambda **_kwargs: {
            "router": "test-image",
            "envoy": "test-image",
            "dashboard": "test-image",
        },
    )
    captured = _capture_run_commands(monkeypatch)
    _stub_valid_container_cli(monkeypatch, tmp_path)

    rc, _, _ = container_cli.container_start_vllm_sr(
        str(config_path),
        {"TARGET_ROUTER_API_URL": "http://stale-router:8080"},
        [{"name": "public", "address": "0.0.0.0", "port": 8899}],
        network_name="vllm-sr-network",
        minimal=False,
    )

    assert rc == 0
    router_cmd = _find_container_run_cmd(captured, "vllm-sr-router-container")
    dashboard_cmd = _find_container_run_cmd(captured, "vllm-sr-dashboard-container")
    assert "127.0.0.1:9090:9090" in router_cmd
    assert (
        "http://127.0.0.1:9090/ready"
        in router_cmd[router_cmd.index("--health-cmd") + 1]
    )
    assert "127.0.0.1:8080:8080" not in router_cmd
    assert "TARGET_ROUTER_API_URL=http://vllm-sr-router-container:9090" in dashboard_cmd
    assert "TARGET_ROUTER_API_URL=http://stale-router:8080" not in dashboard_cmd


@pytest.mark.parametrize("bind_address", ["127.0.0.1", "localhost", "::1"])
def test_split_runtime_rejects_management_listener_unreachable_from_dashboard(
    tmp_path, monkeypatch, bind_address
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.3\n"
        "listeners:\n  - name: public\n    address: 0.0.0.0\n    port: 8899\n"
        "global:\n  services:\n"
        "    backend_egress:\n"
        "      policy_file: /app/config/backend-egress-policy.yaml\n"
        "    backend_dispatch:\n"
        "      bind_address: 0.0.0.0\n"
        "      port: 8187\n"
        "      audience: vllm-sr.backend-dispatch\n"
        "      capability_ttl: 30s\n"
        "      max_request_body_bytes: 67108864\n"
        "    management_api:\n"
        f"      bind_address: {bind_address}\n      port: 8080\n"
    )
    monkeypatch.setattr(container_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_start,
        "get_runtime_images",
        lambda **_kwargs: {
            "router": "test-image",
            "envoy": "test-image",
            "dashboard": "test-image",
        },
    )
    _stub_valid_container_cli(monkeypatch, tmp_path)

    with pytest.raises(
        ValueError, match=r"requires management_api\.bind_address 0\.0\.0\.0"
    ):
        container_cli.container_start_vllm_sr(
            str(config_path),
            {},
            [{"name": "public", "address": "0.0.0.0", "port": 8899}],
            network_name="vllm-sr-network",
            minimal=False,
        )


def test_split_runtime_rejects_invalid_management_auth_exposure(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.3\n"
        "listeners:\n  - name: public\n    address: 0.0.0.0\n    port: 8899\n"
        "global:\n  services:\n"
        "    backend_egress:\n"
        "      policy_file: /app/config/backend-egress-policy.yaml\n"
        "    backend_dispatch:\n"
        "      bind_address: 0.0.0.0\n"
        "      port: 8187\n"
        "      audience: vllm-sr.backend-dispatch\n"
        "      capability_ttl: 30s\n"
        "      max_request_body_bytes: 67108864\n"
        "    management_api:\n"
        "      bind_address: 0.0.0.0\n      port: 8080\n"
        "      remote_exposure: true\n      auth:\n        mode: disabled\n"
    )
    monkeypatch.setattr(container_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_start,
        "get_runtime_images",
        lambda **_kwargs: {
            "router": "test-image",
            "envoy": "test-image",
            "dashboard": "test-image",
        },
    )
    _stub_valid_container_cli(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="requires bearer auth tokens"):
        container_cli.container_start_vllm_sr(
            str(config_path),
            {},
            [{"name": "public", "address": "0.0.0.0", "port": 8899}],
            network_name="vllm-sr-network",
            minimal=False,
        )


def test_envoy_host_publish_preserves_loopback_listener_address(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.3\n"
        "listeners:\n  - name: local-http\n    address: 127.0.0.1\n    port: 8899\n"
        "global:\n  services:\n    backend_egress:\n"
        "      policy_file: /app/config/backend-egress-policy.yaml\n"
    )
    monkeypatch.setattr(container_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_start,
        "get_runtime_images",
        lambda **_kwargs: {
            "router": "test-image",
            "envoy": "test-image",
            "dashboard": "test-image",
        },
    )
    captured = _capture_run_commands(monkeypatch)
    _stub_valid_container_cli(monkeypatch, tmp_path)

    rc, _, _ = container_cli.container_start_vllm_sr(
        str(config_path),
        {},
        [{"name": "local-http", "address": "127.0.0.1", "port": 8899}],
        network_name="vllm-sr-network",
        openclaw_network_name="vllm-sr-network",
        minimal=False,
    )

    assert rc == 0
    envoy_cmd = _find_container_run_cmd(captured, "vllm-sr-envoy-container")
    assert "127.0.0.1:8899:8899" in envoy_cmd


def test_envoy_host_publish_brackets_ipv6_listener_address(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.3\n"
        "listeners:\n  - name: local-v6\n    address: ::1\n    port: 8899\n"
        "global:\n  services:\n    backend_egress:\n"
        "      policy_file: /app/config/backend-egress-policy.yaml\n"
    )
    monkeypatch.setattr(container_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_start,
        "get_runtime_images",
        lambda **_kwargs: {
            "router": "test-image",
            "envoy": "test-image",
            "dashboard": "test-image",
        },
    )
    captured = _capture_run_commands(monkeypatch)
    _stub_valid_container_cli(monkeypatch, tmp_path)

    rc, _, _ = container_cli.container_start_vllm_sr(
        str(config_path),
        {},
        [{"name": "local-v6", "address": "::1", "port": 8899}],
        network_name="vllm-sr-network",
        openclaw_network_name="vllm-sr-network",
        minimal=False,
    )

    assert rc == 0
    envoy_cmd = _find_container_run_cmd(captured, "vllm-sr-envoy-container")
    assert "[::1]:8899:8899" in envoy_cmd


def test_container_start_vllm_sr_uses_role_specific_runtime_images(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_CONFIG_BODY)

    monkeypatch.setattr(container_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_start,
        "get_runtime_images",
        lambda **kwargs: {
            "router": "router-image:latest",
            "envoy": "envoy-image:latest",
            "dashboard": "dashboard-image:latest",
        },
    )
    captured = _capture_run_commands(monkeypatch)
    _stub_valid_container_cli(monkeypatch, tmp_path)

    rc, _, _ = container_cli.container_start_vllm_sr(
        str(config_path),
        {},
        [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        network_name="vllm-sr-network",
        openclaw_network_name="vllm-sr-network",
        minimal=False,
    )

    assert rc == 0
    router_cmd = _find_container_run_cmd(captured, "vllm-sr-router-container")
    envoy_cmd = _find_container_run_cmd(captured, "vllm-sr-envoy-container")
    dashboard_cmd = _find_container_run_cmd(captured, "vllm-sr-dashboard-container")
    assert "router-image:latest" in router_cmd
    assert "envoy-image:latest" in envoy_cmd
    assert "dashboard-image:latest" in dashboard_cmd
    assert "/usr/local/bin/envoy" in envoy_cmd
    assert "/etc/envoy/envoy.yaml" in " ".join(envoy_cmd)


def test_container_start_vllm_sr_skips_dashboard_image_resolution_in_minimal_mode(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_CONFIG_BODY)
    captured_kwargs = {}

    def fake_get_runtime_images(**kwargs):
        captured_kwargs.update(kwargs)
        return {
            "router": "router-image:latest",
            "envoy": "envoy-image:latest",
        }

    monkeypatch.setattr(container_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(container_start, "get_runtime_images", fake_get_runtime_images)
    captured = _capture_run_commands(monkeypatch)
    _stub_valid_container_cli(monkeypatch, tmp_path)

    rc, _, _ = container_cli.container_start_vllm_sr(
        str(config_path),
        {},
        [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        network_name="vllm-sr-network",
        openclaw_network_name="vllm-sr-network",
        minimal=True,
    )

    assert rc == 0
    assert captured_kwargs["include_dashboard"] is False
    _find_container_run_cmd(captured, "vllm-sr-router-container")
    _find_container_run_cmd(captured, "vllm-sr-envoy-container")
    with pytest.raises(AssertionError):
        _find_container_run_cmd(captured, "vllm-sr-dashboard-container")


def test_container_start_vllm_sr_connects_router_before_starting_it(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_CONFIG_BODY)

    monkeypatch.setattr(container_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_start,
        "get_runtime_images",
        lambda **kwargs: {
            "router": "router-image:latest",
            "envoy": "envoy-image:latest",
            "dashboard": "dashboard-image:latest",
        },
    )
    captured = _capture_run_commands(monkeypatch)
    _stub_valid_container_cli(monkeypatch, tmp_path)

    rc, _, _ = container_cli.container_start_vllm_sr(
        str(config_path),
        {},
        [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        network_name="vllm-sr-network",
        openclaw_network_name="vllm-sr-network",
        minimal=False,
    )

    assert rc == 0
    router_cmd = _find_container_run_cmd(captured, "vllm-sr-router-container")
    envoy_cmd = _find_container_run_cmd(captured, "vllm-sr-envoy-container")
    dashboard_cmd = _find_container_run_cmd(captured, "vllm-sr-dashboard-container")
    assert router_cmd[1] == "create"
    assert envoy_cmd[1:3] == ["run", "-d"]
    assert dashboard_cmd[1:3] == ["run", "-d"]
    assert ["docker", "start", "vllm-sr-router-container"] in captured


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
        "container_status",
        lambda _name: "running",
    )
    monkeypatch.setattr(
        runtime_lifecycle,
        "container_create_network",
        record("container_create_network"),
    )
    monkeypatch.setattr(
        core, "start_fleet_sim_sidecar", record("start_fleet_sim_sidecar", True)
    )
    monkeypatch.setattr(
        core, "container_start_vllm_sr", record("container_start_vllm_sr")
    )
    monkeypatch.setattr(
        runtime_lifecycle,
        "container_network_connect",
        record("container_network_connect"),
    )
    monkeypatch.setattr(
        runtime_lifecycle, "container_logs_since", lambda *args, **kwargs: (0, "", "")
    )
    monkeypatch.setattr(
        runtime_lifecycle, "container_exec", lambda *args, **kwargs: (0, "ok", "")
    )
    monkeypatch.setattr(
        runtime_lifecycle, "load_openclaw_registry", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(
        runtime_lifecycle, "container_logs", lambda *args, **kwargs: None
    )

    core.start_vllm_sr("/tmp/config.yaml", env_vars={}, enable_observability=False)

    create_calls = [c for c in calls if c[0] == "container_create_network"]
    fleet_sim_calls = [c for c in calls if c[0] == "start_fleet_sim_sidecar"]
    start_calls = [c for c in calls if c[0] == "container_start_vllm_sr"]
    connect_calls = [c for c in calls if c[0] == "container_network_connect"]

    assert create_calls[0][1] == ("vllm-sr-network",)
    assert fleet_sim_calls[0][1][0] == "/tmp"
    assert fleet_sim_calls[0][1][2].fleet_sim_container_name == "vllm-sr-sim-container"
    assert start_calls[0][2]["network_name"] == "vllm-sr-network"
    assert start_calls[0][2]["openclaw_network_name"] == "vllm-sr-network"
    assert start_calls[0][2]["compiled_bootstrap_file"] == "/tmp/config.yaml"
    assert (
        start_calls[0][2]["env_vars"]["TARGET_FLEET_SIM_URL"]
        == "http://vllm-sr-sim-container:8000"
    )
    assert [call[1] for call in connect_calls] == [
        ("vllm-sr-network", "vllm-sr-router-container"),
        ("vllm-sr-network", "vllm-sr-envoy-container"),
        ("vllm-sr-network", "vllm-sr-dashboard-container"),
    ]


def test_resolve_runtime_stack_supports_custom_stack_name_and_port_offset():
    stack_layout = resolve_runtime_stack(stack_name="audit-a", port_offset=200)

    assert stack_layout.router_container_name == "audit-a-vllm-sr-router-container"
    assert stack_layout.envoy_container_name == "audit-a-vllm-sr-envoy-container"
    assert (
        stack_layout.dashboard_container_name == "audit-a-vllm-sr-dashboard-container"
    )
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
    assert (
        stack_layout.dashboard_service_url
        == "http://audit-a-vllm-sr-dashboard-container:8700"
    )
    assert (
        stack_layout.router_api_service_url
        == "http://audit-a-vllm-sr-router-container:8080"
    )
    assert (
        stack_layout.router_metrics_service_url
        == "http://audit-a-vllm-sr-router-container:9190/metrics"
    )
    assert (
        stack_layout.envoy_admin_service_url
        == "http://audit-a-vllm-sr-envoy-container:9901"
    )
    assert (
        stack_layout.envoy_listener_service_url(8899)
        == "http://audit-a-vllm-sr-envoy-container:8899"
    )


def test_start_vllm_sr_uses_state_root_override(monkeypatch, tmp_path):
    calls = []
    state_root = tmp_path / "workspace-root"
    state_root.mkdir()

    def record(name, ret=(0, "", "")):
        def _fn(*args, **kwargs):
            calls.append((name, args, kwargs))
            return ret

        return _fn

    monkeypatch.setenv("VLLM_SR_STATE_ROOT_DIR", str(state_root))
    monkeypatch.setattr(core, "ensure_clean_runtime_container", lambda _name: None)
    monkeypatch.setattr(
        core,
        "load_config",
        lambda _path: {
            "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}]
        },
    )
    monkeypatch.setattr(
        core, "provision_storage_backends", lambda *args, **kwargs: set()
    )
    monkeypatch.setattr(
        runtime_lifecycle,
        "container_status",
        lambda _name: "running",
    )
    monkeypatch.setattr(
        runtime_lifecycle,
        "container_create_network",
        record("container_create_network"),
    )
    monkeypatch.setattr(
        core, "start_fleet_sim_sidecar", record("start_fleet_sim_sidecar", True)
    )
    monkeypatch.setattr(
        core, "container_start_vllm_sr", record("container_start_vllm_sr")
    )
    monkeypatch.setattr(
        runtime_lifecycle,
        "container_network_connect",
        record("container_network_connect"),
    )
    monkeypatch.setattr(
        runtime_lifecycle, "container_logs_since", lambda *args, **kwargs: (0, "", "")
    )
    monkeypatch.setattr(
        runtime_lifecycle, "container_exec", lambda *args, **kwargs: (0, "ok", "")
    )
    monkeypatch.setattr(
        runtime_lifecycle, "load_openclaw_registry", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(
        runtime_lifecycle, "container_logs", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        core, "recover_openclaw_containers", record("recover_openclaw_containers")
    )

    core.start_vllm_sr("/tmp/config.yaml", env_vars={}, enable_observability=False)

    fleet_sim_calls = [c for c in calls if c[0] == "start_fleet_sim_sidecar"]
    start_calls = [c for c in calls if c[0] == "container_start_vllm_sr"]
    recover_calls = [c for c in calls if c[0] == "recover_openclaw_containers"]

    assert fleet_sim_calls[0][1][0] == str(state_root)
    assert start_calls[0][2]["state_root_dir"] == str(state_root)
    assert recover_calls[0][1][0] == str(state_root)
