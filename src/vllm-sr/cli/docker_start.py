"""Container startup orchestration for vLLM Semantic Router."""

import os
import subprocess

from cli.config_generator import generate_envoy_config_from_user_config
from cli.consts import (
    DEFAULT_NOFILE_LIMIT,
    MIN_NOFILE_LIMIT,
    PLATFORM_AMD,
)
from cli.docker_images import (
    _normalize_platform,
    get_docker_image,
    get_runtime_images,
)
from cli.docker_openclaw_support import configure_openclaw_support
from cli.docker_run_command import (
    append_env_vars,
    append_host_gateway,
    append_mount_specs,
    append_port_mappings,
    build_base_run_command,
    maybe_append_amd_gpu_passthrough,
)
from cli.docker_runtime import get_container_runtime, resolve_docker_cli_path
from cli.docker_services import (
    docker_container_status,
    docker_remove_container,
    docker_stop_container,
)
from cli.parser import parse_user_config
from cli.runtime_stack import RuntimeStackLayout, resolve_runtime_stack
from cli.runtime_topology import resolve_runtime_topology, split_runtime_enabled
from cli.utils import get_logger

log = get_logger(__name__)


def docker_start_vllm_sr(
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
    """
    Start vLLM Semantic Router container.

    Returns:
        (return_code, stdout, stderr)
    """
    runtime = get_container_runtime()
    env_vars = dict(env_vars or {})
    stack_layout = stack_layout or resolve_runtime_stack()
    runtime_topology = resolve_runtime_topology(topology)
    use_split_runtime = split_runtime_enabled(runtime_topology)

    normalized_platform = _resolve_platform(env_vars)
    nofile_limit = _resolve_nofile_limit()

    runtime_network_name = (
        network_name or openclaw_network_name or stack_layout.network_name
    )
    config_dir, runtime_paths, runtime_container_config = _prepare_runtime_paths(
        config_file,
        runtime_config_file=runtime_config_file,
        state_root_dir=state_root_dir,
    )
    common_env = _build_common_runtime_env(
        env_vars,
        stack_layout,
        runtime_container_config=runtime_container_config,
        split_runtime=use_split_runtime,
    )
    if use_split_runtime:
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
        use_split_runtime=use_split_runtime,
    )

    log.info(f"Starting vLLM Semantic Router runtime with {runtime}...")
    started_containers = []
    stdout_chunks = []
    stderr_chunks = []

    for service_name, container_name, cmd in container_specs:
        log.info(f"Starting {service_name} container: {container_name}")
        log.debug(f"Container command: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
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


def _build_common_runtime_env(
    env_vars: dict[str, str],
    stack_layout: RuntimeStackLayout,
    *,
    runtime_container_config: str | None,
    split_runtime: bool,
):
    common_env = dict(env_vars or {})
    if runtime_container_config:
        common_env.setdefault("VLLM_SR_RUNTIME_CONFIG_PATH", runtime_container_config)
    if split_runtime:
        common_env.setdefault(
            "VLLM_SR_ROUTER_CONTAINER_NAME", stack_layout.router_container_name
        )
        common_env.setdefault(
            "VLLM_SR_ENVOY_CONTAINER_NAME", stack_layout.envoy_container_name
        )
        common_env.setdefault(
            "VLLM_SR_DASHBOARD_CONTAINER_NAME", stack_layout.dashboard_container_name
        )
    else:
        _strip_split_runtime_envs(common_env)
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
    use_split_runtime: bool,
):
    if use_split_runtime:
        runtime_images = get_runtime_images(
            image=image,
            router_image=router_image,
            envoy_image=envoy_image,
            dashboard_image=dashboard_image,
            pull_policy=pull_policy,
            platform=normalized_platform,
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
        )

    if any(value for value in (router_image, envoy_image, dashboard_image)):
        log.warning(
            "Ignoring router/envoy/dashboard image overrides for legacy topology"
        )

    selected_image = get_docker_image(
        image=image,
        pull_policy=pull_policy,
        platform=normalized_platform,
    )
    return [
        (
            "runtime",
            stack_layout.container_name,
            _build_legacy_runtime_command(
                runtime=runtime,
                image=selected_image,
                container_name=stack_layout.container_name,
                nofile_limit=nofile_limit,
                network_name=runtime_network_name,
                env_vars=common_env,
                mount_specs=_legacy_runtime_mount_specs(
                    runtime_paths,
                    include_dashboard_data=not minimal,
                ),
                listeners=listeners,
                minimal=minimal,
                enable_amd_gpu=normalized_platform == PLATFORM_AMD,
                config_dir=config_dir,
                openclaw_network_name=openclaw_network_name,
                stack_layout=stack_layout,
            ),
        )
    ]


def _strip_split_runtime_envs(env_vars: dict[str, str]) -> None:
    for key in (
        "VLLM_SR_ROUTER_CONTAINER_NAME",
        "VLLM_SR_ENVOY_CONTAINER_NAME",
        "VLLM_SR_DASHBOARD_CONTAINER_NAME",
        "TARGET_ROUTER_API_URL",
        "TARGET_ROUTER_METRICS_URL",
        "TARGET_ENVOY_URL",
        "TARGET_ENVOY_ADMIN_URL",
        "OPENCLAW_DASHBOARD_CONTAINER_NAME",
        "OPENCLAW_MODEL_GATEWAY_CONTAINER_NAME",
    ):
        env_vars.pop(key, None)


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
):
    listener_port = _primary_listener_port(listeners)
    setup_mode = str(common_env.get("VLLM_SR_SETUP_MODE", "")).lower() == "true"
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
    )
    specs.append(("dashboard", stack_layout.dashboard_container_name, dashboard_cmd))

    return specs


def _legacy_runtime_mount_specs(
    runtime_paths: dict[str, str],
    *,
    include_dashboard_data: bool,
):
    return _runtime_mount_specs(
        runtime_paths,
        include_models=True,
        include_dashboard_data=include_dashboard_data,
    )


def _build_legacy_runtime_command(
    *,
    runtime: str,
    image: str,
    container_name: str,
    nofile_limit: int,
    network_name: str,
    env_vars: dict[str, str],
    mount_specs: list[str],
    listeners,
    minimal: bool,
    enable_amd_gpu: bool,
    config_dir: str,
    openclaw_network_name: str | None,
    stack_layout: RuntimeStackLayout,
):
    legacy_env = _build_legacy_runtime_env(
        env_vars=env_vars,
        minimal=minimal,
        mount_specs=mount_specs,
        config_dir=config_dir,
        openclaw_network_name=openclaw_network_name,
        runtime=runtime,
        stack_layout=stack_layout,
    )

    cmd = build_base_run_command(
        runtime,
        nofile_limit,
        network_name,
        container_name,
        start_immediately=True,
    )
    maybe_append_amd_gpu_passthrough(cmd, enable_amd_gpu)
    append_host_gateway(cmd, runtime)
    append_mount_specs(cmd, mount_specs)
    if not minimal:
        cmd.extend(["-p", f"{stack_layout.dashboard_port}:8700"])
    append_port_mappings(
        cmd,
        _legacy_runtime_port_mappings(listeners, stack_layout),
    )
    append_env_vars(cmd, legacy_env)
    cmd.append(image)
    return cmd


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
):
    return _build_service_run_command(
        runtime=runtime,
        image=router_image,
        container_name=stack_layout.router_container_name,
        nofile_limit=nofile_limit,
        network_name=runtime_network_name,
        env_vars=common_env,
        mount_specs=_runtime_mount_specs(runtime_paths, include_models=True),
        port_mappings=[
            (stack_layout.router_port, 50051),
            (stack_layout.metrics_port, 9190),
            (stack_layout.api_port, 8080),
        ],
        entrypoint="/app/start-router.sh",
        command_args=[
            common_env.get("VLLM_SR_RUNTIME_CONFIG_PATH", "/app/config.yaml"),
            "/app/.vllm-sr",
        ],
        enable_amd_gpu=normalized_platform == PLATFORM_AMD,
        start_immediately=not setup_mode,
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
    return _build_service_run_command(
        runtime=runtime,
        image=envoy_image,
        container_name=stack_layout.envoy_container_name,
        nofile_limit=nofile_limit,
        network_name=runtime_network_name,
        env_vars={},
        mount_specs=[
            f"{runtime_paths['envoy_config_path']}:/etc/envoy/envoy.yaml:z",
        ],
        port_mappings=[
            (listener["port"] + stack_layout.port_offset, listener["port"])
            for listener in listeners
            if listener.get("port")
        ],
        entrypoint="/usr/local/bin/envoy",
        command_args=[
            "-c",
            "/etc/envoy/envoy.yaml",
            "--log-level",
            "debug",
        ],
        start_immediately=not setup_mode,
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
):
    dashboard_env = _build_dashboard_runtime_env(
        common_env=common_env,
        listener_port=listener_port,
        stack_layout=stack_layout,
    )
    dashboard_mount_specs = _runtime_mount_specs(
        runtime_paths, include_dashboard_data=True
    )
    configure_openclaw_support(
        dashboard_mount_specs,
        dashboard_env,
        config_dir,
        openclaw_network_name,
        runtime,
        stack_layout,
        resolve_docker_cli=resolve_docker_cli_path,
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
        entrypoint="/app/start-dashboard.sh",
        command_args=["/app/config.yaml"],
    )


def _build_dashboard_runtime_env(
    *,
    common_env: dict[str, str],
    listener_port: int,
    stack_layout: RuntimeStackLayout,
):
    dashboard_env = dict(common_env)
    dashboard_env.setdefault(
        "TARGET_ROUTER_API_URL", stack_layout.router_api_service_url
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


def _build_legacy_runtime_env(
    *,
    env_vars: dict[str, str],
    minimal: bool,
    mount_specs: list[str],
    config_dir: str,
    openclaw_network_name: str | None,
    runtime: str,
    stack_layout: RuntimeStackLayout,
):
    legacy_env = dict(env_vars)
    if minimal:
        return legacy_env

    legacy_env.setdefault(
        "OPENCLAW_DASHBOARD_CONTAINER_NAME", stack_layout.container_name
    )
    legacy_env.setdefault(
        "OPENCLAW_MODEL_GATEWAY_CONTAINER_NAME", stack_layout.container_name
    )
    configure_openclaw_support(
        mount_specs,
        legacy_env,
        config_dir,
        openclaw_network_name,
        runtime,
        stack_layout,
        resolve_docker_cli=resolve_docker_cli_path,
    )
    return legacy_env


def _legacy_runtime_port_mappings(listeners, stack_layout: RuntimeStackLayout):
    port_mappings = [
        (stack_layout.router_port, 50051),
        (stack_layout.metrics_port, 9190),
        (stack_layout.api_port, 8080),
    ]
    port_mappings.extend(
        (port + stack_layout.port_offset, port)
        for listener in listeners
        if (port := listener.get("port"))
    )
    return port_mappings


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


def _prepare_runtime_paths(
    config_file,
    runtime_config_file=None,
    state_root_dir=None,
):
    source_config_path = os.path.abspath(config_file)
    runtime_config_path = os.path.abspath(runtime_config_file or config_file)
    config_dir = (
        os.path.abspath(state_root_dir)
        if state_root_dir
        else os.path.dirname(source_config_path)
    )

    vllm_sr_dir = os.path.join(config_dir, ".vllm-sr")
    os.makedirs(vllm_sr_dir, exist_ok=True)
    log.info(f"Mounting .vllm-sr directory: {vllm_sr_dir}")

    models_dir = os.path.join(config_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    dashboard_data_dir = os.path.join(config_dir, ".vllm-sr", "dashboard-data")
    os.makedirs(dashboard_data_dir, exist_ok=True)
    log.info(f"Mounting dashboard data directory: {dashboard_data_dir}")

    effective_config_path = runtime_config_path
    envoy_config_path = os.path.join(vllm_sr_dir, "envoy.yaml")

    runtime_container_config = None
    if runtime_config_path != source_config_path:
        runtime_container_config = (
            f"/app/.vllm-sr/{os.path.basename(runtime_config_path)}"
        )
        log.info(
            "Using source config %s with runtime override %s",
            source_config_path,
            runtime_container_config,
        )

    return (
        config_dir,
        {
            "source_config_path": source_config_path,
            "effective_config_path": effective_config_path,
            "vllm_sr_dir": vllm_sr_dir,
            "models_dir": models_dir,
            "dashboard_data_dir": dashboard_data_dir,
            "envoy_config_path": envoy_config_path,
        },
        runtime_container_config,
    )


def _runtime_mount_specs(
    runtime_paths: dict[str, str],
    *,
    include_models: bool = False,
    include_dashboard_data: bool = False,
):
    mounts = [
        f"{runtime_paths['source_config_path']}:/app/config.yaml:z",
        f"{runtime_paths['vllm_sr_dir']}:/app/.vllm-sr:z",
    ]
    if include_models:
        mounts.append(f"{runtime_paths['models_dir']}:/app/models:z")
    if include_dashboard_data:
        mounts.append(f"{runtime_paths['dashboard_data_dir']}:/app/data:z")
    return mounts


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
    port_mappings: list[tuple[int, int]],
    entrypoint: str,
    command_args: list[str],
    enable_amd_gpu: bool = False,
    start_immediately: bool = True,
):
    cmd = build_base_run_command(
        runtime,
        nofile_limit,
        network_name,
        container_name,
        start_immediately=start_immediately,
    )
    maybe_append_amd_gpu_passthrough(cmd, enable_amd_gpu)
    append_host_gateway(cmd, runtime)
    append_mount_specs(cmd, mount_specs)
    append_port_mappings(cmd, port_mappings)
    cmd.extend(["--entrypoint", entrypoint])
    append_env_vars(cmd, env_vars)
    cmd.append(image)
    cmd.extend(command_args)
    return cmd


def _cleanup_started_containers(container_names: list[str]) -> None:
    for container_name in reversed(container_names):
        if docker_container_status(container_name) == "running":
            docker_stop_container(container_name)
        docker_remove_container(container_name)
