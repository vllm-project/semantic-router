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
import subprocess
import time

from cli.container_observability import (
    _ensure_hidden_config_dir,
    _render_template_copy,
    _run_service_start,
)
from cli.container_run_command import restart_policy_args
from cli.container_runtime import get_container_runtime
from cli.container_services import _replace_existing_container, container_stop_container
from cli.grafana_credentials import (
    CONTAINER_GRAFANA_PASSWORD_PATH,
    GRAFANA_ADMIN_PASSWORD_FILE_ENV,
    ensure_grafana_admin_password_file,
)
from cli.runtime_stack import RuntimeStackLayout, resolve_runtime_stack
from cli.utils import get_logger

log = get_logger(__name__)

GRAFANA_REKEY_TIMEOUT_SECONDS = 15
GRAFANA_REKEY_POLL_INTERVAL_SECONDS = 0.5
GRAFANA_READY_TIMEOUT_SECONDS = 60
GRAFANA_EXIT_LOG_LINES = 20

JAEGER_IMAGE = "docker.io/jaegertracing/all-in-one:1.76.0"
PROMETHEUS_IMAGE = "docker.io/prom/prometheus:v2.53.0"
GRAFANA_IMAGE = "docker.io/grafana/grafana:11.5.1"

GRAFANA_LIVE_ALLOWED_ORIGINS_ENV = "GF_LIVE_ALLOWED_ORIGINS"
_GRAFANA_LIVE_ALLOWED_ORIGINS_PLACEHOLDER = "__GF_LIVE_ALLOWED_ORIGINS__"


def _wait_for_grafana_ready(
    container_runtime: str, container_name: str, timeout: int
) -> tuple[int, str, str]:
    """Wait until Grafana answers its health check, or say why it never will.

    On a fresh volume Grafana migrates its database before it serves. A
    ``grafana cli`` started during the migrations opens the same SQLite file,
    and the server's migrator then fails with ``database is locked`` and exits.
    """

    running = [container_runtime, "inspect", "-f", "{{.State.Running}}"]
    health = [container_runtime, "exec", container_name, "wget", "-q", "-T", "5"]
    health += ["-O", "/dev/null", "http://127.0.0.1:3000/api/health"]
    logs = [container_runtime, "logs", "--tail", str(GRAFANA_EXIT_LOG_LINES)]
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = subprocess.run(
            [*running, container_name], capture_output=True, text=True, check=False
        )
        if state.returncode != 0 or state.stdout.strip() != "true":
            exited = subprocess.run(
                [*logs, container_name],
                capture_output=True,
                text=True,
                check=False,
            )
            tail = f"{exited.stdout}{exited.stderr}".strip()
            return (1, "", f"Grafana exited before it was ready:\n{tail}")
        if subprocess.run(health, capture_output=True, check=False).returncode == 0:
            return (0, "", "")
        time.sleep(GRAFANA_REKEY_POLL_INTERVAL_SECONDS)
    return (1, "", f"Grafana did not answer its health check within {timeout}s")


def rekey_grafana_admin(
    container_name: str,
    runtime: str | None = None,
    *,
    timeout: int = GRAFANA_REKEY_TIMEOUT_SECONDS,
) -> tuple[int, str, str]:
    """Synchronize the persisted Grafana admin password with the mounted file.

    Grafana applies ``GF_SECURITY_ADMIN_PASSWORD__FILE`` only while it creates
    the database. A reused ``/var/lib/grafana`` volume keeps the old password.
    The reset reads the mounted file inside the container, so the password is
    not placed in process arguments or logs. It waits for the server first.
    """

    container_runtime = runtime or get_container_runtime()
    ready = _wait_for_grafana_ready(
        container_runtime, container_name, GRAFANA_READY_TIMEOUT_SECONDS
    )
    if ready[0] != 0:
        return ready
    command = [
        container_runtime,
        "exec",
        container_name,
        "sh",
        "-c",
        (
            "grafana cli --homepath /usr/share/grafana admin reset-admin-password "
            f"--password-from-stdin < {CONTAINER_GRAFANA_PASSWORD_PATH}"
        ),
    ]
    deadline = time.time() + timeout
    last_result = (1, "", "timeout waiting to rekey Grafana admin")
    while time.time() < deadline:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
            if result.returncode == 0:
                log.info(f"Synchronized Grafana admin credential in {container_name}")
                return (0, result.stdout, result.stderr)
            last_result = (result.returncode, result.stdout, result.stderr)
        except subprocess.TimeoutExpired:
            break
        except Exception as exc:
            last_result = (1, "", str(exc))
        time.sleep(GRAFANA_REKEY_POLL_INTERVAL_SECONDS)
    return last_result


def _grafana_live_allowed_origins() -> str:
    raw_value = os.getenv(GRAFANA_LIVE_ALLOWED_ORIGINS_ENV, "")
    if "\r" in raw_value or "\n" in raw_value:
        raise ValueError(
            f"{GRAFANA_LIVE_ALLOWED_ORIGINS_ENV} must be a comma-separated "
            "list on one line"
        )
    return ", ".join(
        origin.strip() for origin in raw_value.split(",") if origin.strip()
    )


def _observability_host_bind_address() -> str:
    """Keep auxiliary dashboards and telemetry private unless explicitly exposed."""
    address = os.getenv("VLLM_SR_OBSERVABILITY_HOST_BIND", "127.0.0.1").strip()
    if address not in {"0.0.0.0", "127.0.0.1", "::", "::1"}:
        raise ValueError(
            "VLLM_SR_OBSERVABILITY_HOST_BIND must be an explicit wildcard or loopback IP"
        )
    return address


def _published_observability_port(host_port: int, container_port: int) -> str:
    address = _observability_host_bind_address()
    if ":" in address:
        address = f"[{address}]"
    return f"{address}:{host_port}:{container_port}"


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
        *restart_policy_args(),
        "--network",
        network_name,
        "-e",
        "COLLECTOR_OTLP_ENABLED=true",
        "-e",
        "SPAN_STORAGE_TYPE=badger",
        "-e",
        "BADGER_EPHEMERAL=false",
        "-e",
        "BADGER_DIRECTORY_KEY=/tmp/badger/keys",
        "-e",
        "BADGER_DIRECTORY_VALUE=/tmp/badger/values",
        "-e",
        "BADGER_SPAN_STORE_TTL=168h",
        # The pinned image owns writable /tmp (UID10001). A named volume
        # preserves it across replacements without a privileged init container.
        "-v",
        f"{container_name}-data:/tmp",
        "-p",
        _published_observability_port(stack_layout.jaeger_otlp_port, 4317),
        "-p",
        _published_observability_port(stack_layout.jaeger_ui_port, 16686),
        JAEGER_IMAGE,
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
        *restart_policy_args(),
        "--network",
        network_name,
        "-v",
        f"{os.path.abspath(prometheus_config)}:/etc/prometheus/prometheus.yaml:ro",
        "-v",
        f"{os.path.abspath(prometheus_data_dir)}:/prometheus",
        "-p",
        _published_observability_port(stack_layout.prometheus_port, 9090),
        PROMETHEUS_IMAGE,
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
    live_allowed_origins = _grafana_live_allowed_origins()
    runtime = get_container_runtime()
    stack_layout = stack_layout or resolve_runtime_stack()
    container_name = stack_layout.grafana_container_name
    network_name = network_name or stack_layout.network_name
    _replace_existing_container(container_name)

    state_root_dir = config_dir if config_dir is not None else os.getcwd()
    grafana_dir = os.path.join(_ensure_hidden_config_dir(state_root_dir), "grafana")
    os.makedirs(grafana_dir, exist_ok=True)

    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    grafana_config = os.path.join(grafana_dir, "grafana.serve.ini")
    _render_template_copy(
        os.path.join(template_dir, "grafana.serve.ini"),
        grafana_config,
        stack_layout,
        replacements=(
            (_GRAFANA_LIVE_ALLOWED_ORIGINS_PLACEHOLDER, live_allowed_origins),
        ),
    )
    for filename in [
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

    password_file = ensure_grafana_admin_password_file(
        state_root_dir, stack_layout=stack_layout
    )

    cmd = [
        runtime,
        "run",
        "-d",
        "--name",
        container_name,
        *restart_policy_args(),
        "--network",
        network_name,
        "-e",
        f"{GRAFANA_ADMIN_PASSWORD_FILE_ENV}={CONTAINER_GRAFANA_PASSWORD_PATH}",
        "-v",
        f"{os.path.abspath(password_file)}:{CONTAINER_GRAFANA_PASSWORD_PATH}:ro",
        "-e",
        f"PROMETHEUS_URL={stack_layout.prometheus_container_name}:9090",
        "-v",
        f"{os.path.abspath(grafana_config)}:/etc/grafana/grafana.ini:ro",
        "-v",
        f"{os.path.abspath(os.path.join(grafana_dir, 'grafana-datasource.serve.yaml'))}:/etc/grafana/provisioning/datasources/datasource.yaml:ro",
        "-v",
        f"{os.path.abspath(os.path.join(grafana_dir, 'grafana-datasource-jaeger.serve.yaml'))}:/etc/grafana/provisioning/datasources/datasource_jaeger.yaml:ro",
        "-v",
        f"{os.path.abspath(os.path.join(grafana_dir, 'grafana-dashboard.serve.yaml'))}:/etc/grafana/provisioning/dashboards/dashboard.yaml:ro",
        "-v",
        f"{os.path.abspath(os.path.join(grafana_dir, 'llm-router-dashboard.serve.json'))}:/etc/grafana/provisioning/dashboards/llm-router-dashboard.json:ro",
        "-v",
        f"{container_name}-data:/var/lib/grafana",
        "-p",
        _published_observability_port(stack_layout.grafana_port, 3000),
        GRAFANA_IMAGE,
    ]
    status = _run_service_start(cmd, "Grafana")
    if isinstance(status, tuple) and status and status[0] == 0:
        rekey_status = rekey_grafana_admin(container_name, runtime=runtime)
        if rekey_status[0] != 0:
            error_message = (
                f"Failed to synchronize Grafana admin password for {container_name}: "
                f"{rekey_status[2].strip() or f'exit code {rekey_status[0]}'}"
            )
            log.error(error_message)
            container_stop_container(container_name)
            return (rekey_status[0], rekey_status[1], error_message)
    return status
