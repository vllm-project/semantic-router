"""Container startup orchestration for vLLM Semantic Router."""

import os
import subprocess

from cli.commands.runtime_support import (
    RECIPE_ENV_ALLOWLIST_ENV,
    sensitive_env_names,
)
from cli.config_generator import generate_envoy_config_from_user_config
from cli.consts import (
    DEFAULT_NOFILE_LIMIT,
    MIN_NOFILE_LIMIT,
    PLATFORM_AMD,
    PLATFORM_NVIDIA,
)
from cli.container_gpu_isolation import router_runtime_env
from cli.container_images import (
    _normalize_platform,
    get_runtime_images,
)
from cli.container_log_spool import (
    LOG_SPOOL_GID_ENV,
    LOG_SPOOL_READER_DIR,
    LOG_SPOOL_ROOT_ENV,
    bounded_log_spool_entrypoint,
)
from cli.container_management_listener import _managed_management_listener
from cli.container_openclaw_support import configure_openclaw_support
from cli.container_run_command import (
    append_custom_dns,
    append_env_vars,
    append_host_gateway,
    append_mount_specs,
    append_port_mappings,
    append_supplemental_gids,
    build_base_run_command,
    maybe_append_amd_gpu_passthrough,
    maybe_append_nvidia_gpu_passthrough,
)
from cli.container_runtime import get_container_runtime, resolve_container_cli_path
from cli.container_services import (
    container_remove_container,
    container_status,
    container_stop_container,
)
from cli.container_start_paths import (
    _active_recipe_mount_specs,
    _prepare_runtime_paths,
    _runtime_mount_specs,
)
from cli.parser import parse_user_config
from cli.runtime_stack import PORT_OFFSET_ENV, RuntimeStackLayout, resolve_runtime_stack
from cli.runtime_topology import resolve_runtime_topology
from cli.storage_secrets import (
    STORAGE_SECRET_ENV_NAMES,
    load_storage_secrets,
    storage_secret_env,
    storage_state_path,
)
from cli.utils import get_logger

log = get_logger(__name__)


def container_start_vllm_sr(
    config_file,
    env_vars,
    listeners,
    image=None,
    router_image=None,
    envoy_image=None,
    dashboard_image=None,
    topology=None,
    pull_policy=None,
    network_name=None,
    openclaw_network_name=None,
    minimal=False,
    stack_layout: RuntimeStackLayout | None = None,
    state_root_dir: str | None = None,
    runtime_config_file: str | None = None,
):
    """Start the runtime containers and return code, stdout, and stderr."""
    runtime = get_container_runtime()
    env_vars = dict(env_vars or {})
    stack_layout = stack_layout or resolve_runtime_stack()
    resolve_runtime_topology(topology)

    normalized_platform = _resolve_platform(env_vars)
    nofile_limit = _resolve_nofile_limit()

    runtime_network_name = (
        network_name or openclaw_network_name or stack_layout.network_name
    )
    config_dir, runtime_paths, runtime_container_config = _prepare_runtime_paths(
        config_file,
        runtime_config_file=runtime_config_file,
        state_root_dir=state_root_dir,
        stack_layout=stack_layout,
    )
    common_env = _build_common_runtime_env(
        env_vars,
        stack_layout,
        runtime_container_config=runtime_container_config,
        recipe_store_dir=runtime_paths["container_recipe_store_dir"],
    )
    storage_secret_values = _resolve_storage_secret_env(config_dir, stack_layout)
    _render_split_envoy_config(
        runtime_paths["effective_config_path"],
        runtime_paths["envoy_config_path"],
        stack_layout,
    )
    container_specs = _resolve_container_specs(
        runtime=runtime,
        image=image,
        router_image=router_image,
        envoy_image=envoy_image,
        dashboard_image=dashboard_image,
        pull_policy=pull_policy,
        normalized_platform=normalized_platform,
        nofile_limit=nofile_limit,
        runtime_network_name=runtime_network_name,
        common_env=common_env,
        listeners=listeners,
        minimal=minimal,
        config_dir=config_dir,
        runtime_paths=runtime_paths,
        openclaw_network_name=openclaw_network_name,
        stack_layout=stack_layout,
        storage_secret_names=tuple(storage_secret_values),
    )

    log.info(f"Starting vLLM Semantic Router runtime with {runtime}...")
    started_containers = []
    stdout_chunks = []
    stderr_chunks = []

    for service_name, container_name, cmd in container_specs:
        log.info(f"Starting {service_name} container: {container_name}")
        log.debug(f"Container command: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                env=_service_child_env(service_name, storage_secret_values),
            )
        except subprocess.CalledProcessError as exc:
            if exc.stdout:
                stdout_chunks.append(exc.stdout)
            stderr_chunks.append(exc.stderr)
            _cleanup_started_containers(started_containers)
            return (exc.returncode, "\n".join(stdout_chunks), "\n".join(stderr_chunks))

        started_containers.append(container_name)
        if result.stdout:
            stdout_chunks.append(result.stdout)
        if result.stderr:
            stderr_chunks.append(result.stderr)

    return (0, "\n".join(stdout_chunks), "\n".join(stderr_chunks))


def _resolve_storage_secret_env(
    state_root_dir: str, stack_layout: RuntimeStackLayout
) -> dict[str, str]:
    """Return this stack's storage credentials, for one child process only.

    No credential state means this stack provisions no managed storage, so
    there is nothing to hand Router. An unreadable state is a different thing
    and is left to raise: continuing would start Router with no credential and
    turn a damaged state file into an unexplained authentication failure.
    """

    path = storage_state_path(state_root_dir, stack_layout=stack_layout)
    if not os.path.lexists(path):
        log.debug(f"No storage credential state for this stack: {path}")
        return {}
    return storage_secret_env(
        load_storage_secrets(state_root_dir, stack_layout=stack_layout)
    )


def _service_child_env(
    service_name: str, storage_secret_values: dict[str, str]
) -> dict[str, str] | None:
    """Hand the storage credentials to the Router's ``run`` invocation alone.

    The values travel in this one child's environment, paired with the
    inheriting ``-e NAME`` flag, so they never appear in a command line. They
    are never assigned into ``os.environ``: that would expose them to every
    other process the CLI spawns, including ``docker exec`` and OpenClaw
    workloads. ``None`` means "inherit", which is what every other service
    gets.
    """

    if service_name != "router" or not storage_secret_values:
        return None
    return {**os.environ, **storage_secret_values}


def _build_common_runtime_env(
    env_vars: dict[str, str],
    stack_layout: RuntimeStackLayout,
    *,
    runtime_container_config: str | None,
    recipe_store_dir: str | None = None,
):
    common_env = dict(env_vars or {})
    common_env["VLLM_SR_RUNTIME_CONFIG_PATH"] = runtime_container_config
    common_env["VLLM_SR_SOURCE_CONFIG_PATH"] = runtime_container_config
    common_env["VLLM_SR_STATE_ROOT_DIR"] = "/app"
    common_env["VLLM_SR_CONFIG_BASE_DIR"] = "/app"
    common_env[PORT_OFFSET_ENV] = str(stack_layout.port_offset)
    if recipe_store_dir:
        common_env["VLLM_SR_RECIPE_STORE_DIR"] = recipe_store_dir
    stack_name_value = os.getenv("VLLM_SR_STACK_NAME", "").strip()
    if stack_name_value:
        common_env.setdefault("VLLM_SR_STACK_NAME", stack_name_value)
    common_env.setdefault(
        "VLLM_SR_ROUTER_CONTAINER_NAME", stack_layout.router_container_name
    )
    common_env.setdefault(
        "VLLM_SR_ENVOY_CONTAINER_NAME", stack_layout.envoy_container_name
    )
    common_env.setdefault(
        "VLLM_SR_DASHBOARD_CONTAINER_NAME", stack_layout.dashboard_container_name
    )
    return common_env


def _resolve_container_specs(
    *,
    runtime: str,
    image: str | None,
    router_image: str | None,
    envoy_image: str | None,
    dashboard_image: str | None,
    pull_policy: str | None,
    normalized_platform: str,
    nofile_limit: int,
    runtime_network_name: str,
    common_env: dict[str, str],
    listeners,
    minimal: bool,
    config_dir: str,
    runtime_paths: dict[str, str],
    openclaw_network_name: str | None,
    stack_layout: RuntimeStackLayout,
    storage_secret_names: tuple[str, ...] = (),
):
    runtime_images = get_runtime_images(
        image=image,
        router_image=router_image,
        envoy_image=envoy_image,
        dashboard_image=dashboard_image,
        pull_policy=pull_policy,
        platform=normalized_platform,
        include_dashboard=not minimal,
    )
    return _runtime_container_specs(
        runtime=runtime,
        image_by_service=runtime_images,
        nofile_limit=nofile_limit,
        runtime_network_name=runtime_network_name,
        normalized_platform=normalized_platform,
        common_env=common_env,
        listeners=listeners,
        minimal=minimal,
        config_dir=config_dir,
        runtime_paths=runtime_paths,
        openclaw_network_name=openclaw_network_name,
        stack_layout=stack_layout,
        storage_secret_names=storage_secret_names,
    )


def _runtime_container_specs(
    *,
    runtime: str,
    image_by_service: dict[str, str],
    nofile_limit: int,
    runtime_network_name: str,
    normalized_platform: str,
    common_env: dict[str, str],
    listeners,
    minimal: bool,
    config_dir: str,
    runtime_paths: dict[str, str],
    openclaw_network_name: str | None,
    stack_layout: RuntimeStackLayout,
    storage_secret_names: tuple[str, ...] = (),
):
    listener_port = _primary_listener_port(listeners)
    management_listener = _managed_management_listener(
        runtime_paths["effective_config_path"], stack_layout
    )
    listener_host_ports = {
        listener["port"] + stack_layout.port_offset
        for listener in listeners
        if listener.get("port")
    }
    if management_listener["host_port"] in listener_host_ports:
        raise ValueError(
            "management API host port conflicts with an Envoy listener host port"
        )
    setup_mode = str(common_env.get("VLLM_SR_SETUP_MODE", "")).lower() == "true"
    inherited_sensitive_env = _sensitive_runtime_env_names(common_env, runtime_paths)
    router_cmd = _build_router_runtime_command(
        runtime=runtime,
        router_image=image_by_service["router"],
        nofile_limit=nofile_limit,
        runtime_network_name=runtime_network_name,
        normalized_platform=normalized_platform,
        common_env=common_env,
        runtime_paths=runtime_paths,
        setup_mode=setup_mode,
        stack_layout=stack_layout,
        inherited_sensitive_env=inherited_sensitive_env,
        management_listener=management_listener,
        storage_secret_names=storage_secret_names,
    )
    envoy_cmd = _build_envoy_runtime_command(
        runtime=runtime,
        envoy_image=image_by_service["envoy"],
        nofile_limit=nofile_limit,
        runtime_network_name=runtime_network_name,
        common_env=common_env,
        listeners=listeners,
        runtime_paths=runtime_paths,
        setup_mode=setup_mode,
        stack_layout=stack_layout,
    )

    specs = [
        ("router", stack_layout.router_container_name, router_cmd),
        ("envoy", stack_layout.envoy_container_name, envoy_cmd),
    ]

    if minimal:
        return specs

    dashboard_cmd = _build_dashboard_runtime_command(
        runtime=runtime,
        dashboard_image=image_by_service["dashboard"],
        nofile_limit=nofile_limit,
        runtime_network_name=runtime_network_name,
        common_env=common_env,
        config_dir=config_dir,
        listener_port=listener_port,
        openclaw_network_name=openclaw_network_name,
        runtime_paths=runtime_paths,
        stack_layout=stack_layout,
        inherited_sensitive_env=inherited_sensitive_env,
        management_listener=management_listener,
    )
    specs.append(("dashboard", stack_layout.dashboard_container_name, dashboard_cmd))

    return specs


def _build_router_runtime_command(
    *,
    runtime: str,
    router_image: str,
    nofile_limit: int,
    runtime_network_name: str,
    normalized_platform: str,
    common_env: dict[str, str],
    runtime_paths: dict[str, str],
    setup_mode: bool,
    stack_layout: RuntimeStackLayout,
    inherited_sensitive_env: set[str],
    management_listener: dict[str, int | str],
    storage_secret_names: tuple[str, ...] = (),
):
    router_env = router_runtime_env(common_env, normalized_platform)
    # Names only. Each one is rendered as an inheriting `-e NAME` flag, and the
    # value reaches Docker through the Router child process environment. They
    # stay out of `common_env` on purpose: `_build_dashboard_runtime_env()`
    # copies that mapping, which would hand two unused credentials to the one
    # container holding the Docker socket.
    for name in storage_secret_names:
        router_env.setdefault(name, "")
    service_entrypoint, service_args = bounded_log_spool_entrypoint(
        "/app/start-router.sh",
        [
            runtime_paths["runtime_container_config"],
            "/app/.vllm-sr",
        ],
    )
    return _build_service_run_command(
        runtime=runtime,
        image=router_image,
        container_name=stack_layout.router_container_name,
        nofile_limit=nofile_limit,
        network_name=runtime_network_name,
        env_vars=router_env,
        mount_specs=[
            *_runtime_mount_specs(runtime_paths, include_models=True),
            runtime_paths["log_spool_router_mount"],
        ],
        port_mappings=[
            ("127.0.0.1", stack_layout.router_port, 50051),
            ("127.0.0.1", stack_layout.metrics_port, 9190),
            (
                "127.0.0.1",
                int(management_listener["host_port"]),
                int(management_listener["port"]),
            ),
        ],
        entrypoint=service_entrypoint,
        command_args=service_args,
        enable_amd_gpu=normalized_platform == PLATFORM_AMD,
        enable_nvidia_gpu=normalized_platform == PLATFORM_NVIDIA,
        start_immediately=not setup_mode,
        inherited_env_keys=inherited_sensitive_env,
    )


def _build_envoy_runtime_command(
    *,
    runtime: str,
    envoy_image: str,
    nofile_limit: int,
    runtime_network_name: str,
    common_env: dict[str, str],
    listeners,
    runtime_paths: dict[str, str],
    setup_mode: bool,
    stack_layout: RuntimeStackLayout,
):
    service_entrypoint, service_args = bounded_log_spool_entrypoint(
        "/usr/local/bin/envoy",
        [
            "-c",
            "/etc/envoy/envoy.yaml",
            "--log-level",
            "debug",
        ],
    )
    return _build_service_run_command(
        runtime=runtime,
        image=envoy_image,
        container_name=stack_layout.envoy_container_name,
        nofile_limit=nofile_limit,
        network_name=runtime_network_name,
        env_vars={},
        mount_specs=[
            f"{runtime_paths['envoy_config_path']}:/etc/envoy/envoy.yaml:z",
            runtime_paths["log_spool_envoy_mount"],
        ],
        port_mappings=[
            (
                _listener_host_address(listener),
                listener["port"] + stack_layout.port_offset,
                listener["port"],
            )
            for listener in listeners
            if listener.get("port")
        ],
        entrypoint=service_entrypoint,
        command_args=service_args,
        start_immediately=not setup_mode,
        supplemental_gids=[int(runtime_paths["log_spool_gid"])],
    )


def _listener_host_address(listener: dict) -> str:
    address = str(listener.get("address") or "").strip()
    if address in {"0.0.0.0", "::", "127.0.0.1", "::1"}:
        return address
    raise ValueError(
        "listener address must be an explicit wildcard or loopback IP for host publication"
    )


def _build_dashboard_runtime_command(
    *,
    runtime: str,
    dashboard_image: str,
    nofile_limit: int,
    runtime_network_name: str,
    common_env: dict[str, str],
    config_dir: str,
    listener_port: int,
    openclaw_network_name: str | None,
    runtime_paths: dict[str, str],
    stack_layout: RuntimeStackLayout,
    inherited_sensitive_env: set[str],
    management_listener: dict[str, int | str],
):
    dashboard_env = _build_dashboard_runtime_env(
        common_env=common_env,
        listener_port=listener_port,
        stack_layout=stack_layout,
        management_port=int(management_listener["port"]),
    )
    dashboard_mount_specs = _runtime_mount_specs(
        runtime_paths, include_dashboard_data=True
    )
    dashboard_mount_specs.extend(
        [
            runtime_paths["log_spool_dashboard_mount"],
            f"{runtime_paths['log_spool_root']}:{LOG_SPOOL_READER_DIR}:ro,z",
        ]
    )
    dashboard_mount_specs.extend(_active_recipe_mount_specs(runtime_paths))
    if runtime_paths.get("active_recipe_root"):
        dashboard_env["VLLM_SR_ACTIVE_RECIPE_DIR"] = "/app/recipe"
    else:
        dashboard_env.pop("VLLM_SR_ACTIVE_RECIPE_DIR", None)
    dashboard_env["DASHBOARD_CONFIG_DIR"] = "/app"
    dashboard_env["ROUTER_CONFIG_PATH"] = runtime_paths["runtime_container_config"]
    dashboard_env["VLLM_SR_RECIPE_STORE_DIR"] = runtime_paths[
        "container_recipe_store_dir"
    ]
    dashboard_env[LOG_SPOOL_ROOT_ENV] = LOG_SPOOL_READER_DIR
    dashboard_env[LOG_SPOOL_GID_ENV] = runtime_paths["log_spool_gid"]
    configure_openclaw_support(
        dashboard_mount_specs,
        dashboard_env,
        config_dir,
        openclaw_network_name,
        runtime,
        stack_layout,
        resolve_container_cli=resolve_container_cli_path,
    )
    service_entrypoint, service_args = bounded_log_spool_entrypoint(
        "/app/entrypoint.sh",
        [
            "/app/start-dashboard.sh",
            runtime_paths["runtime_container_config"],
        ],
    )
    return _build_service_run_command(
        runtime=runtime,
        image=dashboard_image,
        container_name=stack_layout.dashboard_container_name,
        nofile_limit=nofile_limit,
        network_name=runtime_network_name,
        env_vars=dashboard_env,
        mount_specs=dashboard_mount_specs,
        port_mappings=[(stack_layout.dashboard_port, 8700)],
        entrypoint=service_entrypoint,
        command_args=service_args,
        inherited_env_keys={"DASHBOARD_ADMIN_PASSWORD"} | inherited_sensitive_env,
    )


def _recipe_env_binding_names(common_env: dict[str, str]) -> set[str]:
    raw = common_env.get(RECIPE_ENV_ALLOWLIST_ENV, "")
    return {name for item in raw.split(",") if (name := item.strip())}


def _sensitive_runtime_env_names(
    common_env: dict[str, str], runtime_paths: dict[str, str]
) -> set[str]:
    """Return value-bearing env names that must stay out of Docker argv/logs.

    The storage credential names are added after the intersection rather than
    through it: they are deliberately kept out of ``common_env`` so only the
    Router command carries them, which is exactly what the intersection would
    otherwise filter away.
    """

    names = set(sensitive_env_names(runtime_paths.get("source_config_path")))
    names.update(_recipe_env_binding_names(common_env))
    return (names & set(common_env)) | set(STORAGE_SECRET_ENV_NAMES)


def _build_dashboard_runtime_env(
    *,
    common_env: dict[str, str],
    listener_port: int,
    stack_layout: RuntimeStackLayout,
    management_port: int = 8080,
):
    dashboard_env = dict(common_env)
    for name in (
        "DASHBOARD_ADMIN_EMAIL",
        "DASHBOARD_ADMIN_PASSWORD",
        "DASHBOARD_ADMIN_NAME",
    ):
        value = os.getenv(name)
        if value:
            dashboard_env[name] = value

    bootstrap_policy_env = "DASHBOARD_ALLOW_OPEN_BOOTSTRAP"
    if bootstrap_policy_env in os.environ:
        dashboard_env[bootstrap_policy_env] = os.environ[bootstrap_policy_env]
    elif bootstrap_policy_env not in dashboard_env:
        bootstrap_email = dashboard_env.get("DASHBOARD_ADMIN_EMAIL", "").strip()
        bootstrap_password = dashboard_env.get("DASHBOARD_ADMIN_PASSWORD", "").strip()
        if not (bootstrap_email and bootstrap_password):
            dashboard_env[bootstrap_policy_env] = "true"

    dashboard_env["TARGET_ROUTER_API_URL"] = (
        f"http://{stack_layout.router_container_name}:{management_port}"
    )
    dashboard_env.setdefault(
        "TARGET_ROUTER_METRICS_URL", stack_layout.router_metrics_service_url
    )
    dashboard_env.setdefault(
        "TARGET_ENVOY_URL", stack_layout.envoy_listener_service_url(listener_port)
    )
    dashboard_env.setdefault(
        "TARGET_ENVOY_ADMIN_URL", stack_layout.envoy_admin_service_url
    )
    dashboard_env.setdefault(
        "ENVOY_EXTPROC_ADDRESS", stack_layout.router_container_name
    )
    dashboard_env.setdefault(
        "ENVOY_ROUTER_API_ADDRESS", stack_layout.router_container_name
    )
    dashboard_env.setdefault("VLLM_SR_ENVOY_CONFIG_PATH", "/app/.vllm-sr/envoy.yaml")
    dashboard_env.setdefault(
        "OPENCLAW_DASHBOARD_CONTAINER_NAME", stack_layout.dashboard_container_name
    )
    dashboard_env.setdefault(
        "OPENCLAW_MODEL_GATEWAY_CONTAINER_NAME", stack_layout.envoy_container_name
    )
    return dashboard_env


def _resolve_platform(env_vars):
    platform = (
        env_vars.get("DASHBOARD_PLATFORM")
        or env_vars.get("VLLM_SR_PLATFORM")
        or os.getenv("VLLM_SR_PLATFORM")
    )
    return _normalize_platform(platform)


def _resolve_nofile_limit():
    nofile_limit = int(os.getenv("VLLM_SR_NOFILE_LIMIT", DEFAULT_NOFILE_LIMIT))
    if nofile_limit < MIN_NOFILE_LIMIT:
        log.warning(
            f"File descriptor limit {nofile_limit} is below minimum {MIN_NOFILE_LIMIT}. "
            "Using minimum value."
        )
        return MIN_NOFILE_LIMIT
    if nofile_limit != DEFAULT_NOFILE_LIMIT:
        log.info(f"Using custom file descriptor limit: {nofile_limit}")
    return nofile_limit


def _primary_listener_port(listeners):
    for listener in listeners:
        port = listener.get("port")
        if port:
            return port
    return 8888


def _render_split_envoy_config(
    config_path: str,
    output_path: str,
    stack_layout: RuntimeStackLayout,
) -> None:
    original_extproc = os.environ.get("ENVOY_EXTPROC_ADDRESS")
    original_router_api = os.environ.get("ENVOY_ROUTER_API_ADDRESS")
    os.environ["ENVOY_EXTPROC_ADDRESS"] = stack_layout.router_container_name
    os.environ["ENVOY_ROUTER_API_ADDRESS"] = stack_layout.router_container_name
    try:
        generate_envoy_config_from_user_config(
            parse_user_config(config_path),
            output_path,
        )
        log.info(f"Rendered split Envoy config: {output_path}")
    finally:
        _restore_env_var("ENVOY_EXTPROC_ADDRESS", original_extproc)
        _restore_env_var("ENVOY_ROUTER_API_ADDRESS", original_router_api)


def _restore_env_var(name: str, original_value: str | None) -> None:
    if original_value is None:
        os.environ.pop(name, None)
        return
    os.environ[name] = original_value


def _build_service_run_command(
    *,
    runtime: str,
    image: str,
    container_name: str,
    nofile_limit: int,
    network_name: str,
    env_vars: dict[str, str],
    mount_specs: list[str],
    port_mappings: list[tuple[int, int] | tuple[str, int, int]],
    entrypoint: str,
    command_args: list[str],
    enable_amd_gpu: bool = False,
    enable_nvidia_gpu: bool = False,
    start_immediately: bool = True,
    inherited_env_keys: set[str] | None = None,
    supplemental_gids: list[int] | None = None,
):
    cmd = build_base_run_command(
        runtime,
        nofile_limit,
        network_name,
        container_name,
        start_immediately=start_immediately,
    )
    maybe_append_amd_gpu_passthrough(cmd, enable_amd_gpu)
    maybe_append_nvidia_gpu_passthrough(cmd, enable_nvidia_gpu, runtime)
    append_supplemental_gids(cmd, supplemental_gids or [], runtime)
    append_host_gateway(cmd, runtime)
    append_custom_dns(cmd)
    append_mount_specs(cmd, mount_specs)
    append_port_mappings(cmd, port_mappings)
    cmd.extend(["--entrypoint", entrypoint])
    append_env_vars(cmd, env_vars, inherited_env_keys)
    cmd.append(image)
    cmd.extend(command_args)
    return cmd


def _cleanup_started_containers(container_names: list[str]) -> None:
    for container_name in reversed(container_names):
        if container_status(container_name) == "running":
            container_stop_container(container_name)
        container_remove_container(container_name)
