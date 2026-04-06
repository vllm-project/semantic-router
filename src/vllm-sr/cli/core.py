"""Core management functions for vLLM Semantic Router."""

import os

from cli import runtime_status as runtime_status_support
from cli.docker_cli import (
    docker_container_status,
    docker_exec,
    docker_logs,
    docker_network_disconnect,
    docker_remove_container,
    docker_remove_network,
    docker_start_vllm_sr,
    docker_stop_container,
    load_openclaw_registry,
)
from cli.docker_start import preflight_runtime_images
from cli.logo import print_vllm_logo
from cli.runtime_lifecycle import (
    connect_runtime_container,
    ensure_clean_runtime_container,
    ensure_runtime_container_not_exited,
    ensure_shared_network,
    log_runtime_summary,
    log_startup_banner,
    maybe_finish_setup_mode,
    recover_openclaw_containers,
    resolve_openclaw_data_dir,
    start_fleet_sim_sidecar,
    start_observability_stack,
    wait_for_router_health,
)
from cli.runtime_stack import RuntimeStackLayout, resolve_runtime_stack
from cli.runtime_topology import resolve_runtime_topology
from cli.storage_backends import provision_storage_backends
from cli.utils import get_logger, load_config

log = get_logger(__name__)
STATE_ROOT_DIR_ENV = "VLLM_SR_STATE_ROOT_DIR"


def _resolve_state_root_dir(
    source_config_file: str, env_vars: dict[str, str] | None = None
) -> str:
    env_vars = env_vars or {}
    override = env_vars.get(STATE_ROOT_DIR_ENV) or os.getenv(STATE_ROOT_DIR_ENV)
    if override:
        return os.path.abspath(override)
    return os.path.dirname(os.path.abspath(source_config_file))


def _load_runtime_startup_inputs(
    config_file,
    source_config_file,
    runtime_config_file,
    stack_layout: RuntimeStackLayout,
):
    source_config_file = source_config_file or config_file
    runtime_config_file = runtime_config_file or config_file
    user_config = load_config(runtime_config_file) or {}
    listeners = user_config.get("listeners", [])
    if not listeners:
        log.error("No listeners configured in config.yaml")
        raise SystemExit(1)
    log_startup_banner(source_config_file, listeners, stack_layout)
    return source_config_file, runtime_config_file, user_config, listeners


def _prepare_runtime_startup(
    *,
    env_vars,
    image,
    router_image,
    envoy_image,
    dashboard_image,
    pull_policy,
    enable_observability,
    source_config_file,
    user_config,
    stack_layout: RuntimeStackLayout,
):
    dashboard_disabled = env_vars.get("DISABLE_DASHBOARD") == "true"
    preflight_runtime_images(
        env_vars=env_vars,
        image=image,
        router_image=router_image,
        envoy_image=envoy_image,
        dashboard_image=dashboard_image,
        pull_policy=pull_policy,
        minimal=dashboard_disabled,
    )
    for container_name in stack_layout.runtime_container_names:
        ensure_clean_runtime_container(container_name)

    shared_network_name = stack_layout.network_name
    state_root_dir = _resolve_state_root_dir(source_config_file, env_vars)
    ensure_shared_network(shared_network_name)

    started_backends = provision_storage_backends(
        user_config, shared_network_name, stack_layout
    )
    observability_network_name = start_observability_stack(
        enable_observability,
        shared_network_name,
        state_root_dir,
        env_vars,
        stack_layout,
    )
    runtime_network_name = observability_network_name or shared_network_name
    fleet_sim_enabled = start_fleet_sim_sidecar(
        state_root_dir,
        env_vars,
        stack_layout,
        pull_policy=pull_policy,
    )
    if fleet_sim_enabled:
        env_vars.setdefault("TARGET_FLEET_SIM_URL", stack_layout.fleet_sim_service_url)
    return (
        dashboard_disabled,
        shared_network_name,
        state_root_dir,
        started_backends,
        runtime_network_name,
        fleet_sim_enabled,
    )


def _finalize_runtime_startup(
    *,
    listeners,
    env_vars,
    stack_layout: RuntimeStackLayout,
    dashboard_disabled: bool,
    enable_observability: bool,
    fleet_sim_enabled: bool,
    shared_network_name: str,
    state_root_dir: str,
    started_backends: set[str],
) -> None:
    setup_mode = str(env_vars.get("VLLM_SR_SETUP_MODE", "")).lower() == "true"
    log.info("vLLM Semantic Router container started successfully")
    connect_runtime_container(shared_network_name, stack_layout)
    if maybe_finish_setup_mode(setup_mode, dashboard_disabled, stack_layout):
        return

    _wait_and_verify_runtime(stack_layout, dashboard_disabled)
    recover_openclaw_containers(state_root_dir, env_vars, shared_network_name)
    log_runtime_summary(
        listeners,
        stack_layout,
        dashboard_disabled,
        enable_observability,
        fleet_sim_enabled,
        started_backends=started_backends,
    )


def start_vllm_sr(
    config_file,
    env_vars=None,
    image=None,
    router_image=None,
    envoy_image=None,
    dashboard_image=None,
    topology=None,
    pull_policy=None,
    enable_observability=True,
    source_config_file=None,
    runtime_config_file=None,
):
    """Start vLLM Semantic Router."""
    if env_vars is None:
        env_vars = {}
    stack_layout = resolve_runtime_stack()
    runtime_topology = resolve_runtime_topology(topology)

    print_vllm_logo()
    source_config_file, runtime_config_file, user_config, listeners = (
        _load_runtime_startup_inputs(
            config_file,
            source_config_file,
            runtime_config_file,
            stack_layout,
        )
    )
    log.info(f"Runtime topology: {runtime_topology}")
    (
        dashboard_disabled,
        shared_network_name,
        state_root_dir,
        started_backends,
        runtime_network_name,
        fleet_sim_enabled,
    ) = _prepare_runtime_startup(
        env_vars=env_vars,
        image=image,
        router_image=router_image,
        envoy_image=envoy_image,
        dashboard_image=dashboard_image,
        pull_policy=pull_policy,
        enable_observability=enable_observability,
        source_config_file=source_config_file,
        user_config=user_config,
        stack_layout=stack_layout,
    )

    return_code, _stdout, stderr = docker_start_vllm_sr(
        config_file=source_config_file,
        env_vars=env_vars,
        listeners=listeners,
        image=image,
        router_image=router_image,
        envoy_image=envoy_image,
        dashboard_image=dashboard_image,
        topology=runtime_topology,
        pull_policy=pull_policy,
        network_name=runtime_network_name,
        openclaw_network_name=shared_network_name,
        minimal=dashboard_disabled,
        stack_layout=stack_layout,
        state_root_dir=state_root_dir,
        runtime_config_file=runtime_config_file,
    )
    if return_code != 0:
        log.error(f"Failed to start container: {stderr}")
        raise SystemExit(1)

    _finalize_runtime_startup(
        listeners=listeners,
        env_vars=env_vars,
        stack_layout=stack_layout,
        dashboard_disabled=dashboard_disabled,
        enable_observability=enable_observability,
        fleet_sim_enabled=fleet_sim_enabled,
        shared_network_name=shared_network_name,
        state_root_dir=state_root_dir,
        started_backends=started_backends,
    )


def _wait_and_verify_runtime(stack_layout, dashboard_disabled):
    """Wait for health check and verify core runtime containers are still running."""
    wait_for_router_health(stack_layout)
    for service in ("router", "envoy"):
        ensure_runtime_container_not_exited(
            stack_layout.service_container_name(service)
        )
    if not dashboard_disabled:
        ensure_runtime_container_not_exited(stack_layout.dashboard_container_name)


def stop_vllm_sr():
    """Stop vLLM Semantic Router and observability containers."""
    log.info("Stopping vLLM Semantic Router...")
    stack_layout = resolve_runtime_stack()
    container_statuses = _managed_container_statuses(stack_layout)
    if _all_containers_absent(container_statuses):
        log.info("Container not found. Nothing to stop.")
        return

    openclaw_data_dir = resolve_openclaw_data_dir(os.getcwd())
    network_name = stack_layout.network_name
    _disconnect_openclaw_registry_containers(
        network_name,
        load_openclaw_registry(openclaw_data_dir),
    )
    runtime_stopped = False
    for container_name in _runtime_container_names(stack_layout):
        _stop_managed_container(
            container_name,
            container_statuses[container_name],
            stop_message=f"Stopping {container_name}...",
            stopped_message=f"{container_name} stopped",
        )
        if container_statuses[container_name] != "not found":
            runtime_stopped = True
    _stop_managed_container(
        stack_layout.fleet_sim_container_name,
        container_statuses[stack_layout.fleet_sim_container_name],
        stop_message=f"Stopping {stack_layout.fleet_sim_container_name}...",
        stopped_message=f"{stack_layout.fleet_sim_container_name} stopped",
    )
    for container_name in _observability_container_names(stack_layout):
        _stop_managed_container(
            container_name,
            container_statuses[container_name],
            stop_message=f"Stopping {container_name}...",
            stopped_message=f"{container_name} stopped",
        )
    for container_name in _storage_container_names(stack_layout):
        _stop_managed_container(
            container_name,
            container_statuses[container_name],
            stop_message=f"Stopping {container_name}...",
            stopped_message=f"{container_name} stopped",
        )
    if runtime_stopped:
        log.info("vLLM Semantic Router stopped")
    _remove_runtime_network(network_name)


def _managed_container_statuses(stack_layout: RuntimeStackLayout) -> dict[str, str]:
    container_names = [
        *_runtime_container_names(stack_layout),
        stack_layout.fleet_sim_container_name,
        *_observability_container_names(stack_layout),
        *_storage_container_names(stack_layout),
    ]
    return {
        container_name: docker_container_status(container_name)
        for container_name in container_names
    }


def _all_containers_absent(container_statuses: dict[str, str]) -> bool:
    return all(status == "not found" for status in container_statuses.values())


def _runtime_container_names(stack_layout: RuntimeStackLayout) -> tuple[str, ...]:
    return stack_layout.runtime_container_names


def _runtime_service_container_name(
    service: str, stack_layout: RuntimeStackLayout
) -> str:
    return runtime_status_support.runtime_service_container_name(service, stack_layout)


def _runtime_stack_status(stack_layout: RuntimeStackLayout) -> str:
    return runtime_status_support.runtime_stack_status(
        stack_layout,
        docker_container_status=docker_container_status,
    )


def _disconnect_openclaw_registry_containers(
    network_name: str, openclaw_entries: list[dict[str, str]]
) -> None:
    for entry in openclaw_entries:
        name = entry.get("name") or entry.get("containerName")
        if not name:
            continue
        _disconnect_openclaw_container(network_name, name)


def _disconnect_openclaw_container(network_name: str, container_name: str) -> None:
    container_status = docker_container_status(container_name)
    if container_status == "not found":
        return
    if container_status == "running":
        log.info(f"Stopping OpenClaw container: {container_name}")
        docker_stop_container(container_name)
    log.info(f"Disconnecting {container_name} from {network_name}")
    docker_network_disconnect(network_name, container_name)


def _stop_managed_container(
    container_name: str,
    container_status: str,
    *,
    stop_message: str | None = None,
    stopped_message: str | None = None,
) -> None:
    if container_status == "not found":
        return
    if stop_message and container_status == "running":
        log.info(stop_message)
    if container_status == "running":
        docker_stop_container(container_name)
    docker_remove_container(container_name)
    if stopped_message:
        log.info(stopped_message)


def _observability_container_names(stack_layout: RuntimeStackLayout) -> tuple[str, ...]:
    return (
        stack_layout.grafana_container_name,
        stack_layout.prometheus_container_name,
        stack_layout.jaeger_container_name,
    )


def _storage_container_names(stack_layout: RuntimeStackLayout) -> tuple[str, ...]:
    return stack_layout.storage_container_names


def _remove_runtime_network(network_name: str) -> None:
    return_code, _stdout, _stderr = docker_remove_network(network_name)
    if return_code == 0:
        log.info(f"Network {network_name} removed")


def show_logs(service: str, follow: bool = False):
    return runtime_status_support.show_logs(
        service,
        follow=follow,
        validate_runtime_service=_validate_runtime_service,
        resolve_runtime_stack=resolve_runtime_stack,
        ensure_runtime_container_available=_ensure_runtime_container_available,
        docker_logs=docker_logs,
        runtime_service_container_name=_runtime_service_container_name,
        log_command=_log_command,
        log=log,
    )


def show_status(service: str = "all"):
    return runtime_status_support.show_status(
        service,
        resolve_runtime_stack=resolve_runtime_stack,
        resolve_runtime_status_snapshot=_resolve_runtime_status_snapshot,
        requested_services=_requested_services,
        report_service_status=_report_service_status,
        log=log,
    )


def _resolve_runtime_status_snapshot(
    stack_layout: RuntimeStackLayout,
) -> tuple[str, str]:
    return runtime_status_support.resolve_runtime_status_snapshot(
        stack_layout,
        runtime_stack_status=_runtime_stack_status,
        docker_container_status=docker_container_status,
        log=log,
    )


def _validate_runtime_service(service: str) -> None:
    runtime_status_support.validate_runtime_service(service, log=log)


def _requested_services(service: str) -> list[str]:
    return runtime_status_support.requested_services(
        service,
        validate_runtime_service=_validate_runtime_service,
    )


def _ensure_runtime_container_available(container_name: str) -> None:
    runtime_status_support.ensure_runtime_container_available(
        container_name,
        docker_container_status=docker_container_status,
        log=log,
    )


def _log_command(container_name: str, grep_pattern: str, follow: bool) -> str:
    return runtime_status_support.log_command(container_name, grep_pattern, follow)


def _report_service_status(service: str, stack_layout) -> None:
    runtime_status_support.report_service_status(
        service,
        stack_layout,
        runtime_service_container_name=_runtime_service_container_name,
        check_router_status=_check_router_status,
        check_envoy_status=_check_envoy_status,
        check_dashboard_status=_check_dashboard_status,
        check_fleet_sim_status=_check_fleet_sim_status,
        log_service_status=_log_service_status,
        log=log,
    )


def _check_router_status(container_name: str) -> bool:
    return runtime_status_support.check_router_status(
        container_name,
        docker_exec=docker_exec,
    )


def _check_envoy_status(container_name: str, stack_layout: RuntimeStackLayout) -> bool:
    return runtime_status_support.check_envoy_status(
        container_name,
        stack_layout,
        docker_exec=docker_exec,
        docker_container_status=docker_container_status,
    )


def _check_dashboard_status(container_name: str) -> bool:
    return runtime_status_support.check_dashboard_status(
        container_name,
        docker_exec=docker_exec,
    )


def _check_fleet_sim_status(container_name: str) -> bool:
    return runtime_status_support.check_fleet_sim_status(
        container_name,
        docker_exec=docker_exec,
    )


def _log_service_status(
    label: str, is_running: bool, detail: str | None = None
) -> None:
    runtime_status_support.log_service_status(
        label,
        is_running,
        detail,
        log=log,
    )
