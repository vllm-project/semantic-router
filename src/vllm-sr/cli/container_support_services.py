"""Starters for the auxiliary containers a local stack runs alongside the router.

Jaeger, Prometheus, and Grafana are optional companions
to a `vllm-sr serve`: nothing in the request path depends on them, and
`runtime_lifecycle` starts them as one group. They live here rather
than in `container_services` so that module keeps its one dominant
responsibility -- the container primitives and the storage backends the runtime
genuinely needs -- as this file's rendered-template wiring grows.

Container replacement still goes through `container_services`, which owns the
lifecycle primitives; the dependency runs one way only.
"""

import os

from cli.container_observability import (
    _ensure_hidden_config_dir,
    _render_template_copy,
    _run_service_start,
)
from cli.container_runtime import get_container_runtime
from cli.container_services import _replace_existing_container
from cli.grafana_credentials import (
    CONTAINER_GRAFANA_PASSWORD_PATH,
    GRAFANA_ADMIN_PASSWORD_FILE_ENV,
    GRAFANA_ADMIN_USER,
    ensure_grafana_admin_password_file,
    grafana_admin_username,
)
from cli.runtime_stack import RuntimeStackLayout, resolve_runtime_stack
from cli.utils import get_logger

log = get_logger(__name__)


def container_start_jaeger(
    network_name=None, stack_layout: RuntimeStackLayout | None = None
):
    """Start Jaeger container for distributed tracing."""
    runtime = get_container_runtime()
    stack_layout = stack_layout or resolve_runtime_stack()
    container_name = stack_layout.jaeger_container_name
    network_name = network_name or stack_layout.network_name
    _replace_existing_container(container_name)

    cmd = [
        runtime,
        "run",
        "-d",
        "--name",
        container_name,
        "--network",
        network_name,
        "-e",
        "COLLECTOR_OTLP_ENABLED=true",
        "-p",
        f"{stack_layout.jaeger_otlp_port}:4317",
        "-p",
        f"{stack_layout.jaeger_ui_port}:16686",
        "docker.io/jaegertracing/all-in-one:1.76.0",
    ]
    return _run_service_start(cmd, "Jaeger")


def container_start_prometheus(
    network_name=None,
    config_dir=None,
    stack_layout: RuntimeStackLayout | None = None,
):
    """Start Prometheus container for metrics collection."""
    runtime = get_container_runtime()
    stack_layout = stack_layout or resolve_runtime_stack()
    container_name = stack_layout.prometheus_container_name
    network_name = network_name or stack_layout.network_name
    _replace_existing_container(container_name)

    config_dir = _ensure_hidden_config_dir(config_dir)
    prometheus_data_dir = os.path.join(config_dir, "prometheus-data")
    os.makedirs(prometheus_data_dir, exist_ok=True)

    prometheus_tsdb_dir = os.path.join(prometheus_data_dir, "data")
    os.makedirs(prometheus_tsdb_dir, exist_ok=True)
    try:
        os.chmod(prometheus_data_dir, 0o777)
        os.chmod(prometheus_tsdb_dir, 0o777)
    except Exception as exc:
        log.warning(f"Failed to set permissions on Prometheus data directory: {exc}")
        log.warning(
            "Prometheus may fail to start if it cannot write to the data directory"
        )

    prometheus_config_dir = os.path.join(config_dir, "prometheus-config")
    os.makedirs(prometheus_config_dir, exist_ok=True)
    prometheus_config = os.path.join(prometheus_config_dir, "prometheus.yaml")
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    _render_template_copy(
        os.path.join(template_dir, "prometheus.serve.yaml"),
        prometheus_config,
        stack_layout,
    )

    cmd = [
        runtime,
        "run",
        "-d",
        "--name",
        container_name,
        "--network",
        network_name,
        "-v",
        f"{os.path.abspath(prometheus_config)}:/etc/prometheus/prometheus.yaml:ro",
        "-v",
        f"{os.path.abspath(prometheus_data_dir)}:/prometheus",
        "-p",
        f"{stack_layout.prometheus_port}:9090",
        "docker.io/prom/prometheus:v2.53.0",
        "--config.file=/etc/prometheus/prometheus.yaml",
        "--storage.tsdb.path=/prometheus/data",
        "--storage.tsdb.retention.time=15d",
    ]
    return _run_service_start(cmd, "Prometheus")


def container_start_grafana(
    network_name=None,
    config_dir=None,
    stack_layout: RuntimeStackLayout | None = None,
):
    """Start Grafana container for visualization."""
    runtime = get_container_runtime()
    stack_layout = stack_layout or resolve_runtime_stack()
    container_name = stack_layout.grafana_container_name
    network_name = network_name or stack_layout.network_name
    _replace_existing_container(container_name)

    grafana_dir = os.path.join(_ensure_hidden_config_dir(config_dir), "grafana")
    os.makedirs(grafana_dir, exist_ok=True)

    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    for filename in [
        "grafana.serve.ini",
        "grafana-datasource.serve.yaml",
        "grafana-datasource-jaeger.serve.yaml",
        "grafana-dashboard.serve.yaml",
        "llm-router-dashboard.serve.json",
    ]:
        _render_template_copy(
            os.path.join(template_dir, filename),
            os.path.join(grafana_dir, filename),
            stack_layout,
        )

    # Resolves and materializes the admin password into a file the value is
    # bind-mounted from, so the mount source always exists (never in argv/env).
    password_file = ensure_grafana_admin_password_file(
        config_dir, stack_layout=stack_layout
    )
    admin_user = grafana_admin_username()

    cmd = [
        runtime,
        "run",
        "-d",
        "--name",
        container_name,
        "--network",
        network_name,
        "-e",
        f"{GRAFANA_ADMIN_PASSWORD_FILE_ENV}={CONTAINER_GRAFANA_PASSWORD_PATH}",
        "-v",
        (
            f"{os.path.abspath(password_file)}:"
            f"{CONTAINER_GRAFANA_PASSWORD_PATH}:ro,z"
        ),
    ]
    if admin_user and admin_user != GRAFANA_ADMIN_USER:
        cmd += ["-e", f"GF_SECURITY_ADMIN_USER={admin_user}"]
    cmd += [
        "-e",
        f"PROMETHEUS_URL={stack_layout.prometheus_container_name}:9090",
        "-v",
        f"{os.path.abspath(os.path.join(grafana_dir, 'grafana.serve.ini'))}:/etc/grafana/grafana.ini:ro",
        "-v",
        f"{os.path.abspath(os.path.join(grafana_dir, 'grafana-datasource.serve.yaml'))}:/etc/grafana/provisioning/datasources/datasource.yaml:ro",
        "-v",
        f"{os.path.abspath(os.path.join(grafana_dir, 'grafana-datasource-jaeger.serve.yaml'))}:/etc/grafana/provisioning/datasources/datasource_jaeger.yaml:ro",
        "-v",
        f"{os.path.abspath(os.path.join(grafana_dir, 'grafana-dashboard.serve.yaml'))}:/etc/grafana/provisioning/dashboards/dashboard.yaml:ro",
        "-v",
        f"{os.path.abspath(os.path.join(grafana_dir, 'llm-router-dashboard.serve.json'))}:/etc/grafana/provisioning/dashboards/llm-router-dashboard.json:ro",
        "-p",
        f"{stack_layout.grafana_port}:3000",
        "docker.io/grafana/grafana:11.5.1",
    ]
    return _run_service_start(cmd, "Grafana")
