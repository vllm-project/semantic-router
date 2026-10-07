"""Core management functions for vLLM Semantic Router."""

from contextlib import nullcontext
from pathlib import Path

from cli.bootstrap import DASHBOARD_SETUP_MODE_ENV, SETUP_MODE_ENV
from cli.commands.runtime_management_credentials import (
    management_credential_env_names,
)
from cli.commands.runtime_paths import _runtime_config_filename, resolve_state_root_dir
from cli.consts import HEALTH_CHECK_TIMEOUT, IMAGE_PULL_POLICY_NEVER
from cli.container_cli import (
    container_logs,
    container_logs_output,
    container_remove_container,
    container_remove_network,
    container_start_vllm_sr,
    container_status,
    container_status_strict,
    container_stop_container,
)
from cli.container_images import get_runtime_images
from cli.container_mounts import (
    ContainerMountsUnavailableError,
    inspect_container_mounts,
)
from cli.gateway_mode import GATEWAY_EXTPROC, runs_envoy
from cli.logo import print_vllm_logo
from cli.management_credential import stack_management_credential
from cli.pending_activation import (
    REASON_RESTART,
    SERVE_STARTING,
    SERVE_WAITING,
    clear_pending_activation,
    read_pending_activation,
    serve_heartbeat,
    wait_for_pending_activation,
)
from cli.recipe_activation_recovery import (
    active_recipe_package_for_stack,
    recover_pending_recipe_activation_for_stack,
)
from cli.recipe_directory import resolve_active_recipe_directory
from cli.recipe_topology_contract import MANAGEMENT_CREDENTIAL_ENV
from cli.runtime_config_coordination import runtime_config_lock_scope
from cli.runtime_config_lock import RuntimeConfigLock
from cli.runtime_lifecycle import (
    connect_runtime_container,
    ensure_clean_runtime_container,
    ensure_data_network,
    ensure_shared_network,
    log_runtime_summary,
    log_startup_banner,
    maybe_finish_setup_mode,
    start_observability_stack,
    validate_startup_timeout,
)
from cli.runtime_lifecycle import (
    wait_and_verify_runtime as _wait_and_verify_runtime,
)
from cli.runtime_management_config import (
    _configured_management_port,
    _configured_management_readiness_token_env,
)
from cli.runtime_service_status import (
    report_service_status,
    runtime_service_container_name,
)
from cli.runtime_stack import RuntimeStackLayout, resolve_runtime_stack
from cli.runtime_topology import resolve_runtime_topology
from cli.storage_backends import (
    detach_preserved_storage_container,
    provision_storage_backends,
    storage_data_volume_is_recorded,
)
from cli.terminal import echo, fields, heading, hint, success, warning
from cli.utils import get_logger, load_config

log = get_logger(__name__)
MANAGED_STORAGE_BACKENDS_ENV = "VLLM_SR_MANAGED_STORAGE_BACKENDS"

RUNTIME_LOG_SERVICES = ("router", "dashboard", "envoy")
# Where every stack container mounts the stack's .vllm-sr state directory.
CONTAINER_STATE_DIR = "/app/.vllm-sr"


def _load_runtime_config(runtime_config_file):
    user_config = load_config(runtime_config_file) or {}
    listeners = user_config.get("listeners", [])
    if not listeners:
        log.error("No listeners configured in config.yaml")
        raise SystemExit(1)
    return user_config, listeners


def _prepare_runtime_network(
    source_config_file,
    env_vars,
    stack_layout,
    image,
    router_image,
    envoy_image,
    dashboard_image,
    pull_policy,
    dashboard_disabled,
    include_envoy=True,
):
    shared_network_name = stack_layout.network_name
    state_root_dir = resolve_state_root_dir(source_config_file, env_vars)
    ensure_shared_network(shared_network_name)
    # The data network has to exist before any storage container is created on
    # it, which is the very next step in the serve sequence.
    ensure_data_network(stack_layout.data_network_name)
    ensure_runtime_images_for_pull_policy(
        image,
        router_image,
        envoy_image,
        dashboard_image,
        pull_policy,
        env_vars,
        dashboard_disabled=dashboard_disabled,
        include_envoy=include_envoy,
    )
    return shared_network_name, state_root_dir


def ensure_runtime_images_for_pull_policy(
    image,
    router_image,
    envoy_image,
    dashboard_image,
    pull_policy=None,
    env_vars=None,
    dashboard_disabled=False,
    include_envoy=True,
):
    env_vars = env_vars or {}
    if pull_policy != IMAGE_PULL_POLICY_NEVER:
        return
    get_runtime_images(
        image=image,
        router_image=router_image,
        envoy_image=envoy_image,
        dashboard_image=None if dashboard_disabled else dashboard_image,
        pull_policy=pull_policy,
        platform=env_vars.get("VLLM_SR_PLATFORM"),
        include_dashboard=not dashboard_disabled,
        include_envoy=include_envoy,
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
    runtime_config_lock: RuntimeConfigLock | None = None,
    startup_timeout: int = HEALTH_CHECK_TIMEOUT,
    gateway: str = GATEWAY_EXTPROC,
):
    """Start vLLM Semantic Router."""
    validate_startup_timeout(startup_timeout)
    env_vars = env_vars if env_vars is not None else {}
    stack_layout = resolve_runtime_stack()
    runtime_topology = resolve_runtime_topology(topology)
    source_config_file = source_config_file or config_file
    runtime_config_file = runtime_config_file or config_file
    state_root_dir = resolve_state_root_dir(source_config_file, env_vars)
    with runtime_config_lock_scope(
        runtime_config_lock,
        runtime_config_file,
        state_root_dir,
        stack_layout,
    ) as runtime_lock:
        return _start_vllm_sr_locked(
            source_config_file=source_config_file,
            runtime_config_file=runtime_config_file,
            env_vars=env_vars,
            stack_layout=stack_layout,
            runtime_topology=runtime_topology,
            state_root_dir=state_root_dir,
            image=image,
            router_image=router_image,
            envoy_image=envoy_image,
            dashboard_image=dashboard_image,
            pull_policy=pull_policy,
            enable_observability=enable_observability,
            startup_timeout=startup_timeout,
            gateway=gateway,
            runtime_lock=runtime_lock,
        )


def _preflight_runtime_config(
    source_config_file,
    runtime_config_file,
    state_root_dir,
    stack_layout,
):
    print_vllm_logo()
    recover_pending_recipe_activation_for_stack(
        runtime_config_path=runtime_config_file,
        state_root_dir=state_root_dir,
        stack_name=stack_layout.stack_name,
        managed_container_names=stack_layout.runtime_container_names,
        status_provider=container_status_strict,
    )
    active_recipe_package_for_stack(
        state_root_dir=state_root_dir,
        stack_name=stack_layout.stack_name,
    )
    # Reject an incomplete managed Recipe before stopping an existing stack or
    # provisioning any support services.
    resolve_active_recipe_directory(source_config_file)
    return _load_runtime_config(runtime_config_file)


def _start_vllm_sr_locked(
    *,
    source_config_file,
    runtime_config_file,
    env_vars,
    stack_layout,
    runtime_topology,
    state_root_dir,
    image,
    router_image,
    envoy_image,
    dashboard_image,
    pull_policy,
    enable_observability,
    startup_timeout=HEALTH_CHECK_TIMEOUT,
    gateway=GATEWAY_EXTPROC,
    runtime_lock: RuntimeConfigLock | None = None,
):
    user_config, listeners = _preflight_runtime_config(
        source_config_file,
        runtime_config_file,
        state_root_dir,
        stack_layout,
    )
    pending = read_pending_activation(runtime_config_file)
    if pending is not None and pending.reason == REASON_RESTART:
        log.info(
            f"Applying the change saved {pending.saved_by()} that needed a restart"
            + (f": {pending.detail}" if pending.detail else "")
        )
    # This start serves whatever the Dashboard saved, so a pending activation
    # is spent.
    clear_pending_activation(runtime_config_file)
    management_port = _configured_management_port(user_config)
    stack_layout.host_port(management_port, name="management API host port")
    for listener in listeners:
        stack_layout.host_port(
            listener["port"],
            name=f"listener {listener.get('name', 'unknown')} host port",
        )
    readiness_token_env = _readiness_token_env(
        user_config, env_vars, runtime_config_file, state_root_dir, stack_layout
    )

    log_startup_banner(source_config_file, listeners, stack_layout)
    log.info(f"Runtime topology: {runtime_topology}")
    for container_name in stack_layout.runtime_container_names:
        ensure_clean_runtime_container(container_name)

    dashboard_disabled = env_vars.get("DISABLE_DASHBOARD") == "true"
    shared_network_name, state_root_dir = _prepare_runtime_network(
        source_config_file,
        env_vars,
        stack_layout,
        image,
        router_image,
        envoy_image,
        dashboard_image,
        pull_policy,
        dashboard_disabled,
        include_envoy=runs_envoy(gateway),
    )

    started_backends, runtime_network_name = _start_support_services(
        user_config,
        shared_network_name,
        state_root_dir,
        env_vars,
        stack_layout,
        enable_observability,
    )

    setup_mode = str(env_vars.get("VLLM_SR_SETUP_MODE", "")).lower() == "true"
    # The Dashboard reads the heartbeat to tell a Router this command is
    # starting from a stopped one; it has no container runtime to ask.
    with serve_heartbeat(runtime_config_file) as heartbeat:
        return_code, _stdout, stderr = _start_runtime_containers(
            source_config_file,
            runtime_config_file,
            listeners,
            runtime_topology,
            runtime_network_name,
            stack_layout,
            state_root_dir,
            dashboard_disabled,
            env_vars,
            image,
            router_image,
            envoy_image,
            dashboard_image,
            pull_policy,
            gateway,
        )
        if return_code != 0:
            log.error(f"Failed to start container: {stderr}")
            raise SystemExit(1)

        log.info("vLLM Semantic Router container started successfully")
        connect_runtime_container(shared_network_name, stack_layout)
        if setup_mode:
            # Until setup activates a config this command changes nothing, and
            # the Dashboard's activation needs the runtime config lock.
            heartbeat(SERVE_WAITING)
            with runtime_lock.released() if runtime_lock else nullcontext():
                maybe_finish_setup_mode(
                    setup_mode,
                    dashboard_disabled,
                    stack_layout,
                    startup_timeout=startup_timeout,
                    envoy=runs_envoy(gateway),
                )
                activated = _wait_for_setup_activation(
                    runtime_config_file, stack_layout
                )
            if not activated:
                return
            heartbeat(SERVE_STARTING)
            user_config, listeners = _start_activated_runtime(
                source_config_file,
                runtime_config_file,
                runtime_topology,
                runtime_network_name,
                shared_network_name,
                stack_layout,
                state_root_dir,
                env_vars,
                image,
                router_image,
                envoy_image,
                dashboard_image,
                pull_policy,
                gateway,
            )
            management_port = _configured_management_port(user_config)
            readiness_token_env = _readiness_token_env(
                user_config,
                env_vars,
                runtime_config_file,
                state_root_dir,
                stack_layout,
            )

        _wait_and_verify_runtime(
            stack_layout,
            dashboard_disabled,
            management_port,
            readiness_token_env,
            startup_timeout=startup_timeout,
            envoy=runs_envoy(gateway),
        )
    log_runtime_summary(
        listeners,
        stack_layout,
        dashboard_disabled,
        enable_observability,
        started_backends=started_backends,
        config=user_config,
        gateway=gateway,
    )


def _readiness_token_env(
    user_config, env_vars, runtime_config_file, state_root_dir, stack_layout
):
    """The env name the Router's readiness check authenticates with.

    The check runs in the Router container and reads the value there. The
    stack's management credential counts as available whenever the config
    binds it, because this command gives it to the Router then.
    """

    available = dict(env_vars)
    if MANAGEMENT_CREDENTIAL_ENV in management_credential_env_names(
        runtime_config_file
    ):
        available[MANAGEMENT_CREDENTIAL_ENV] = stack_management_credential(
            state_root_dir, stack_layout=stack_layout
        )
    return _configured_management_readiness_token_env(user_config, available)


def _start_support_services(
    user_config,
    shared_network_name,
    state_root_dir,
    env_vars,
    stack_layout,
    enable_observability,
):
    if not enable_observability:
        _stop_observability_containers(stack_layout)
    started_backends = provision_storage_backends(
        user_config, stack_layout, state_root_dir=state_root_dir
    )
    env_vars[MANAGED_STORAGE_BACKENDS_ENV] = ",".join(sorted(started_backends))
    observability_network_name = start_observability_stack(
        enable_observability,
        shared_network_name,
        state_root_dir,
        env_vars,
        stack_layout,
    )
    runtime_network_name = observability_network_name or shared_network_name
    return started_backends, runtime_network_name


def _wait_for_setup_activation(
    runtime_config_file, stack_layout: RuntimeStackLayout
) -> bool:
    """Wait until the Dashboard activates a config."""

    dashboard = stack_layout.dashboard_container_name
    log.info(
        "Waiting for setup: the Router starts here once you activate a config "
        "in the Dashboard (Ctrl-C stops waiting)"
    )
    activated = (
        wait_for_pending_activation(
            runtime_config_file,
            dashboard_running=lambda: container_status(dashboard) == "running",
        )
        is not None
    )
    if not activated:
        log.info(
            "Stopped waiting for setup. The Dashboard keeps running; after you "
            "activate a config there, run `vllm-sr serve` to start the Router."
        )
    return activated


def _start_activated_runtime(
    source_config_file,
    runtime_config_file,
    runtime_topology,
    runtime_network_name,
    shared_network_name,
    stack_layout,
    state_root_dir,
    env_vars,
    image,
    router_image,
    envoy_image,
    dashboard_image,
    pull_policy,
    gateway,
):
    """Recreate the Router (and Envoy) that setup mode left stopped, from the activated config.

    Setup may change the listeners, so the containers are created anew rather
    than started: they publish the activated config's ports.
    """

    log.info("Setup activated: starting the Router")
    user_config, listeners = _load_runtime_config(runtime_config_file)
    services = ("router", "envoy") if runs_envoy(gateway) else ("router",)
    for service in services:
        ensure_clean_runtime_container(
            runtime_service_container_name(service, stack_layout)
        )
    env_vars.pop(SETUP_MODE_ENV, None)
    env_vars.pop(DASHBOARD_SETUP_MODE_ENV, None)
    return_code, _stdout, stderr = _start_runtime_containers(
        source_config_file,
        runtime_config_file,
        listeners,
        runtime_topology,
        runtime_network_name,
        stack_layout,
        state_root_dir,
        False,
        env_vars,
        image,
        router_image,
        envoy_image,
        dashboard_image,
        pull_policy,
        gateway,
        services=services,
    )
    if return_code != 0:
        log.error(f"Failed to start the Router after setup: {stderr}")
        raise SystemExit(1)
    connect_runtime_container(shared_network_name, stack_layout)
    clear_pending_activation(runtime_config_file)
    return user_config, listeners


def _start_runtime_containers(
    source_config_file,
    runtime_config_file,
    listeners,
    runtime_topology,
    runtime_network_name,
    stack_layout,
    state_root_dir,
    dashboard_disabled,
    env_vars,
    image,
    router_image,
    envoy_image,
    dashboard_image,
    pull_policy,
    gateway=GATEWAY_EXTPROC,
    services=None,
):
    return container_start_vllm_sr(
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
        minimal=dashboard_disabled,
        stack_layout=stack_layout,
        state_root_dir=state_root_dir,
        runtime_config_file=runtime_config_file,
        gateway=gateway,
        services=services,
    )


def stop_vllm_sr():
    """Stop vLLM Semantic Router and observability containers."""
    log.info("Stopping vLLM Semantic Router...")
    stack_layout = resolve_runtime_stack()
    container_statuses = _managed_container_statuses(stack_layout)
    containers_absent = _all_containers_absent(container_statuses)

    network_name = stack_layout.network_name
    stack_network_names = (network_name, stack_layout.data_network_name)
    failures = []
    for container_name in (
        *_runtime_container_names(stack_layout),
        stack_layout.sr_bench_container_name,
    ):
        if not _stop_managed_container(
            container_name,
            container_statuses[container_name],
            stop_message=f"Stopping {container_name}...",
            stopped_message=f"{container_name} stopped",
        ):
            failures.append(container_name)
    for container_name in _observability_container_names(stack_layout):
        if not _stop_managed_container(
            container_name,
            container_statuses[container_name],
            stop_message=f"Stopping {container_name}...",
            stopped_message=f"{container_name} stopped",
        ):
            failures.append(container_name)
    # Only Redis and Postgres keep their data in a volume this CLI has to name;
    # Milvus writes to a host bind mount, so removing it orphans nothing.
    credentialed_storage = {
        stack_layout.redis_container_name,
        stack_layout.postgres_container_name,
    }
    for container_name in _storage_container_names(stack_layout):
        if not _stop_managed_container(
            container_name,
            container_statuses[container_name],
            stop_message=f"Stopping {container_name}...",
            stopped_message=f"{container_name} stopped",
            preserve_unadopted_data=container_name in credentialed_storage,
            network_names=stack_network_names,
        ):
            failures.append(container_name)
    for stack_network_name in stack_network_names:
        if not _remove_runtime_network(stack_network_name):
            failures.append(stack_network_name)
    if failures:
        raise RuntimeError("Failed to stop managed containers: " + ", ".join(failures))
    if containers_absent:
        echo("Nothing to stop.")
        return
    success("vLLM Semantic Router stopped")


def _managed_container_statuses(stack_layout: RuntimeStackLayout) -> dict[str, str]:
    container_names = [
        *_runtime_container_names(stack_layout),
        stack_layout.sr_bench_container_name,
        *_observability_container_names(stack_layout),
        *_storage_container_names(stack_layout),
    ]
    return {
        container_name: container_status(container_name)
        for container_name in container_names
    }


def _all_containers_absent(container_statuses: dict[str, str]) -> bool:
    return all(status == "not found" for status in container_statuses.values())


def _runtime_container_names(stack_layout: RuntimeStackLayout) -> tuple[str, ...]:
    return stack_layout.runtime_container_names


def _runtime_stack_status(stack_layout: RuntimeStackLayout) -> str:
    fallback_status = "not found"
    for container_name in _runtime_container_names(stack_layout):
        status = container_status(container_name)
        if status == "running":
            return status
        if status != "not found" and fallback_status == "not found":
            fallback_status = status
    return fallback_status


def _stop_managed_container(
    container_name: str,
    container_status: str,
    *,
    stop_message: str | None = None,
    stopped_message: str | None = None,
    preserve_unadopted_data: bool = False,
    network_names: tuple[str, ...] = (),
) -> bool:
    if container_status == "not found":
        return True
    if stop_message and container_status == "running":
        log.info(stop_message)
    if container_status == "running" and not container_stop_container(container_name):
        return False
    if preserve_unadopted_data and not storage_data_volume_is_recorded(container_name):
        log.info(
            f"{container_name} stopped but kept: it predates this CLI's "
            "per-stack storage credentials, so nothing has recorded which "
            "volume holds its data and removing it would orphan that volume. "
            "The next `vllm-sr serve` adopts the volume and takes the "
            "container over."
        )
        # A kept container stays attached to a stack network, which Podman
        # counts as a user of that network and refuses to remove. Detaching it
        # costs nothing: the next `serve` recreates it on the new network
        # anyway.
        if network_names:
            detach_preserved_storage_container(network_names, container_name)
        return True
    if not container_remove_container(container_name):
        return False
    if stopped_message:
        log.info(stopped_message)
    return True


def _observability_container_names(stack_layout: RuntimeStackLayout) -> tuple[str, ...]:
    return (
        stack_layout.grafana_container_name,
        stack_layout.prometheus_container_name,
        stack_layout.jaeger_container_name,
    )


def _stop_observability_containers(stack_layout: RuntimeStackLayout) -> None:
    """Stop this stack's old collectors when switching to minimal mode.

    Keep containers and their data for the next full-mode start. Strict checks
    prevent an unavailable container runtime from looking like an empty stack.
    The caller holds the stack lifecycle lock throughout deployment.
    """
    names = _observability_container_names(stack_layout)
    stopped_states = {"not found", "exited", "created", "dead"}
    states = {name: container_status_strict(name) for name in names}
    for name, state in states.items():
        if state in {"running", "paused", "restarting"}:
            if not container_stop_container(name):
                raise RuntimeError(f"Failed to stop observability container: {name}")
        elif state not in stopped_states:
            raise RuntimeError(f"Observability container is not stopped: {name}")
    for name in names:
        if container_status_strict(name) not in stopped_states:
            raise RuntimeError(f"Observability container is not stopped: {name}")


def _storage_container_names(stack_layout: RuntimeStackLayout) -> tuple[str, ...]:
    return stack_layout.storage_container_names


def _remove_runtime_network(network_name: str) -> bool:
    return_code, _stdout, stderr = container_remove_network(network_name)
    if return_code == 0:
        log.info(f"Network {network_name} removed")
        return True
    detail = (stderr or "").strip()
    if "not found" in detail.lower() or "no such network" in detail.lower():
        return True
    if "active endpoints" in detail.lower():
        log.warning(
            f"Keeping network {network_name}: external containers are still attached"
        )
        return True
    log.error(
        f"Failed to remove network {network_name}: "
        f"{detail or f'exit code {return_code}'}"
    )
    return False


def show_logs(service: str, follow: bool = False):
    """Show logs from a runtime service."""
    _validate_runtime_service(service)
    stack_layout = resolve_runtime_stack()
    container_name = runtime_service_container_name(service, stack_layout)
    if (
        service == "envoy"
        and container_status(container_name) == "not found"
        and container_status(stack_layout.router_container_name) != "not found"
    ):
        log.error("This stack runs in standalone mode, with no Envoy container")
        hint("Envoy runs with: vllm-sr serve --gateway extproc")
        raise SystemExit(1)
    _ensure_runtime_container_available(container_name)

    if follow:
        log.info(f"Following {service} logs (Ctrl+C to stop)...")
        log.info("")
        if not container_logs(container_name, follow=True, tail=200, merge_output=True):
            raise SystemExit(1)
        return

    return_code, output = container_logs_output(container_name, tail=200)
    if return_code != 0:
        message = output.strip() or f"container runtime exited with code {return_code}"
        log.error(f"Failed to get {service} logs: {message}")
        raise SystemExit(return_code)
    if output:
        echo(output, nl=False)
    else:
        echo(f"No recent {service} logs found")


def show_status(service: str = "all"):
    """Show runtime service status."""
    stack_layout = resolve_runtime_stack()
    status = _resolve_runtime_status_snapshot(stack_layout)
    if status == "not found":
        heading("Runtime status")
        fields((("State", "Not running"),))
        echo("Start with: vllm-sr serve")
        return
    if status == "exited":
        heading("Runtime status")
        fields((("State", "Container exited (error)"),))
        echo("View logs with: vllm-sr logs router")
        return
    if status != "running":
        heading("Runtime status")
        fields((("State", status),))
        return

    heading("Runtime status")
    fields((("State", "Running"),))
    echo()
    activation_state = _activation_state(stack_layout)
    if activation_state:
        echo(activation_state)
        echo()

    services = _requested_services(service)
    if (
        service == "all"
        and container_status(stack_layout.envoy_container_name) == "not found"
    ):
        # A standalone stack has no Envoy container.
        services.remove("envoy")
    for requested_service in services:
        report_service_status(requested_service, stack_layout)

    echo()
    echo(f"Detailed logs: vllm-sr logs <{'|'.join(services)}>")


def _activation_state(stack_layout: RuntimeStackLayout) -> str | None:
    """What a saved config waits for before the stack serves it, if anything."""

    setup_waiting = container_status(stack_layout.router_container_name) == "created"
    try:
        mounts = inspect_container_mounts(stack_layout.dashboard_container_name)
    except ContainerMountsUnavailableError:
        return None
    state_dir = next(
        (
            mount.get("Source")
            for mount in mounts
            if mount.get("Destination") == CONTAINER_STATE_DIR
        ),
        None,
    )
    if not state_dir:
        return None
    runtime_config = Path(state_dir) / _runtime_config_filename(stack_layout.stack_name)
    pending = read_pending_activation(runtime_config)
    if pending is not None and pending.reason == REASON_RESTART:
        return (
            f"Restart required: a change saved {pending.saved_by()} needs "
            "`vllm-sr serve` to apply it"
            + (f" ({pending.detail})." if pending.detail else ".")
        )
    if not setup_waiting:
        return None
    if pending is not None:
        return "Setup is complete; run `vllm-sr serve` to start the Router."
    return f"Setup mode: activate a config in the Dashboard ({stack_layout.dashboard_url})."


def _resolve_runtime_status_snapshot(
    stack_layout: RuntimeStackLayout,
) -> str:
    try:
        return _runtime_stack_status(stack_layout)
    except SystemExit:
        warning(
            "Docker daemon is not reachable, so local container status cannot be inspected"
        )
        return "not found"


def _validate_runtime_service(service: str) -> None:
    if service in RUNTIME_LOG_SERVICES:
        return
    log.error(f"Invalid service: {service}")
    log.error("Must be 'envoy', 'router', or 'dashboard'")
    raise SystemExit(1)


def _requested_services(service: str) -> list[str]:
    if service == "all":
        return ["router", "envoy", "dashboard"]
    _validate_runtime_service(service)
    return [service]


def _ensure_runtime_container_available(container_name: str) -> None:
    if container_status(container_name) != "not found":
        return
    log.error("Container not found. Is vLLM Semantic Router running?")
    hint("Start it with: vllm-sr serve")
    raise SystemExit(1)
