import pytest
from cli import container_services
from cli.runtime_stack import resolve_runtime_stack


def _capture_service_command(monkeypatch):
    commands = []
    monkeypatch.setattr(container_services, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_services, "container_status", lambda _name: "not found"
    )
    monkeypatch.setattr(
        container_services,
        "_running_container_for_network_alias",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        container_services, "_replace_existing_container", lambda _: None
    )
    monkeypatch.setattr(container_services, "_is_port_in_use", lambda _port: False)
    monkeypatch.setattr(container_services, "_render_template_copy", lambda *args: None)
    monkeypatch.setattr(
        container_services,
        "get_fleet_sim_container_image",
        lambda **kwargs: "fleet-sim:test",
    )
    monkeypatch.setattr(
        container_services,
        "_run_service_start",
        lambda cmd, _label: commands.append(cmd) or (0, "", ""),
    )
    return commands


def _start_service(service_name, layout, tmp_path):
    common = {"network_name": "test-network", "stack_layout": layout}
    if service_name == "jaeger":
        return container_services.container_start_jaeger(**common)
    if service_name == "prometheus":
        return container_services.container_start_prometheus(
            config_dir=str(tmp_path), **common
        )
    if service_name == "grafana":
        return container_services.container_start_grafana(
            config_dir=str(tmp_path), **common
        )
    if service_name == "redis":
        return container_services.container_start_redis(**common)
    if service_name == "postgres":
        return container_services.container_start_postgres(**common)
    if service_name == "milvus":
        return container_services.container_start_milvus(
            state_root_dir=str(tmp_path), **common
        )
    if service_name == "fleet-sim":
        return container_services.container_start_fleet_sim(
            config_dir=str(tmp_path), **common
        )
    raise AssertionError(f"unknown service: {service_name}")


@pytest.mark.parametrize(
    ("service_name", "port_attributes"),
    [
        ("jaeger", [("jaeger_otlp_port", 4317), ("jaeger_ui_port", 16686)]),
        ("prometheus", [("prometheus_port", 9090)]),
        ("grafana", [("grafana_port", 3000)]),
        ("redis", [("redis_port", 6379)]),
        ("postgres", [("postgres_port", 5432)]),
        ("milvus", [("milvus_port", 19530)]),
        ("fleet-sim", [("fleet_sim_port", 8000)]),
    ],
)
def test_supporting_services_publish_to_loopback_by_default(
    monkeypatch, tmp_path, service_name, port_attributes
):
    monkeypatch.delenv("VLLM_SR_INTERNAL_BIND_ADDRESS", raising=False)
    commands = _capture_service_command(monkeypatch)
    layout = resolve_runtime_stack(port_offset=0)

    assert _start_service(service_name, layout, tmp_path) == (0, "", "")

    assert len(commands) == 1
    expected = {
        f"127.0.0.1:{getattr(layout, attribute)}:{container_port}"
        for attribute, container_port in port_attributes
    }
    published = {
        commands[0][index + 1]
        for index, token in enumerate(commands[0])
        if token == "-p"
    }
    assert published == expected


def test_supporting_service_allows_explicit_internal_bind_override(
    monkeypatch, tmp_path, caplog
):
    monkeypatch.setenv("VLLM_SR_INTERNAL_BIND_ADDRESS", "192.0.2.10")
    commands = _capture_service_command(monkeypatch)
    layout = resolve_runtime_stack(port_offset=200)

    with caplog.at_level("WARNING"):
        assert _start_service("redis", layout, tmp_path) == (0, "", "")

    assert f"192.0.2.10:{layout.redis_port}:6379" in commands[0]
    assert "publishes internal, data, and observability ports" in caplog.text


def test_supporting_service_uses_preflight_bind_without_rereading_environment(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("VLLM_SR_INTERNAL_BIND_ADDRESS", "invalid-after-preflight")
    commands = _capture_service_command(monkeypatch)
    layout = resolve_runtime_stack(port_offset=0)

    result = container_services.container_start_redis(
        network_name="test-network",
        stack_layout=layout,
        bind_address="127.0.0.1",
    )

    assert result == (0, "", "")
    assert f"127.0.0.1:{layout.redis_port}:6379" in commands[0]


def test_supporting_service_rejects_invalid_bind_before_container_mutation(
    monkeypatch,
):
    monkeypatch.setenv("VLLM_SR_INTERNAL_BIND_ADDRESS", "http://0.0.0.0")
    monkeypatch.setattr(container_services, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_services,
        "_replace_existing_container",
        lambda _name: pytest.fail("container mutation must not happen"),
    )

    with pytest.raises(ValueError, match="VLLM_SR_INTERNAL_BIND_ADDRESS"):
        container_services.container_start_jaeger(
            network_name="test-network", stack_layout=resolve_runtime_stack()
        )
