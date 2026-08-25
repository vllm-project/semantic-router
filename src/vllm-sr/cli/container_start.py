"""Container startup orchestration for vLLM Semantic Router."""

import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

from cli.bootstrap import local_dashboard_environment
from cli.commands.runtime_support import sensitive_env_names
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
from cli.container_management_listener import _management_listener
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
from cli.container_runtime_policy import (
    DockerHealthcheck,
    append_docker_runtime_policy,
    dashboard_healthcheck,
    envoy_healthcheck,
    router_healthcheck,
)
from cli.container_runtime_preparation import (
    prepare_runtime_paths as _prepare_runtime_paths,
)
from cli.container_runtime_preparation import (
    primary_listener_port as _primary_listener_port,
)
from cli.container_runtime_preparation import (
    render_split_envoy_config as _render_split_envoy_config,
)
from cli.container_runtime_preparation import (
    router_runtime_mount_specs as _router_runtime_mount_specs,
)
from cli.container_start_environment import (
    cleanup_started_containers,
    service_child_environment,
)
from cli.control_plane_deployment import (
    runtime_capabilities,
)
from cli.runtime_stack import PORT_OFFSET_ENV, RuntimeStackLayout, resolve_runtime_stack
from cli.runtime_topology import resolve_runtime_topology
from cli.storage_secrets import (
    STORAGE_SECRET_ENV_NAMES,
)
from cli.utils import get_logger, load_config

log = get_logger(__name__)

DASHBOARD_SECRET_ENV_NAMES = frozenset(
    {"DASHBOARD_ADMIN_PASSWORD", "DASHBOARD_JWT_SECRET"}
)


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
    compiled_bootstrap_file: str | None = None,
    router_child_env: Mapping[str, str] | None = None,
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
    config_dir, runtime_paths, _runtime_container_config = _prepare_runtime_paths(
        config_file,
        compiled_bootstrap_file=compiled_bootstrap_file,
        state_root_dir=state_root_dir,
        stack_layout=stack_layout,
    )
    common_env = _build_common_runtime_env(
        env_vars,
        stack_layout,
    )
    dashboard_runtime_env = _local_dashboard_runtime_environment(
        runtime_paths["effective_config_path"], config_dir, stack_layout
    )
    trust_bundle = dashboard_runtime_env.get("SSL_CERT_FILE", "").strip()
    if trust_bundle:
        common_env.setdefault("SSL_CERT_FILE", trust_bundle)
    router_secret_values = _router_child_environment(
        common_env,
        runtime_paths,
        router_child_env or {},
    )
    dashboard_secret_values = {
        name: value
        for name in DASHBOARD_SECRET_ENV_NAMES
        if (
            value := (
                common_env.get(name, "").strip()
                or os.environ.get(name, "").strip()
                or dashboard_runtime_env.get(name, "").strip()
            )
        )
    }
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
        router_child_env_names=tuple(router_secret_values),
        dashboard_runtime_env=dashboard_runtime_env,
    )

    return _launch_container_specs(
        runtime,
        container_specs,
        router_secret_values,
        dashboard_secret_values,
    )


def _launch_container_specs(
    runtime: str,
    container_specs: list[tuple[str, str, list[str]]],
    router_secret_values: Mapping[str, str],
    dashboard_secret_values: Mapping[str, str],
) -> tuple[int, str, str]:
    """Start one prepared stack and roll back only containers started here."""

    log.info("Starting vLLM Semantic Router runtime with %s...", runtime)
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
                env=service_child_environment(
                    service_name,
                    router_secret_values,
                    dashboard_secret_values,
                ),
            )
        except subprocess.CalledProcessError as exc:
            if exc.stdout:
                stdout_chunks.append(exc.stdout)
            stderr_chunks.append(exc.stderr)
            cleanup_started_containers(started_containers)
            return (exc.returncode, "\n".join(stdout_chunks), "\n".join(stderr_chunks))

        started_containers.append(container_name)
        if result.stdout:
            stdout_chunks.append(result.stdout)
        if result.stderr:
            stderr_chunks.append(result.stderr)

    return 0, "\n".join(stdout_chunks), "\n".join(stderr_chunks)


def _router_child_environment(
    common_env: Mapping[str, str],
    runtime_paths: Mapping[str, str],
    explicit_bindings: Mapping[str, str],
) -> dict[str, str]:
    """Collect only Router-owned secret values for its Docker child."""

    sensitive_names = _sensitive_runtime_env_names(
        dict(common_env), dict(runtime_paths)
    )
    bindings = {
        name: value
        for name, value in common_env.items()
        if name in sensitive_names and value
    }
    for name, value in explicit_bindings.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(value, str)
            or not value
        ):
            raise ValueError("Router child environment bindings must be non-empty text")
        existing = bindings.get(name)
        if existing is not None and existing != value:
            raise ValueError(f"conflicting Router child environment binding: {name}")
        bindings[name] = value
    return bindings


def _local_dashboard_runtime_environment(
    effective_config_path: str,
    state_root_dir: str,
    stack_layout: RuntimeStackLayout,
) -> dict[str, str]:
    """Read stack-owned Dashboard material for durable local Management."""

    config = load_config(effective_config_path) or {}
    if not runtime_capabilities(config).durable_management:
        return {}
    return local_dashboard_environment(state_root_dir, stack_layout)


def _build_common_runtime_env(
    env_vars: dict[str, str],
    stack_layout: RuntimeStackLayout,
):
    common_env = dict(env_vars or {})
    common_env["VLLM_SR_STATE_ROOT_DIR"] = "/app"
    common_env["VLLM_SR_CONFIG_BASE_DIR"] = "/app"
    common_env[PORT_OFFSET_ENV] = str(stack_layout.port_offset)
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
    router_child_env_names: tuple[str, ...] = (),
    dashboard_runtime_env: Mapping[str, str] | None = None,
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
        router_child_env_names=router_child_env_names,
        dashboard_runtime_env=dashboard_runtime_env or {},
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
    router_child_env_names: tuple[str, ...] = (),
    dashboard_runtime_env: Mapping[str, str] | None = None,
):
    listener_port = _primary_listener_port(listeners)
    management_listener = _management_listener(
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
    inherited_sensitive_env = _sensitive_runtime_env_names(common_env, runtime_paths)
    router_cmd = _build_router_runtime_command(
        runtime=runtime,
        router_image=image_by_service["router"],
        nofile_limit=nofile_limit,
        runtime_network_name=runtime_network_name,
        normalized_platform=normalized_platform,
        common_env=common_env,
        runtime_paths=runtime_paths,
        stack_layout=stack_layout,
        inherited_sensitive_env=inherited_sensitive_env,
        management_listener=management_listener,
        router_child_env_names=router_child_env_names,
    )
    envoy_cmd = _build_envoy_runtime_command(
        runtime=runtime,
        envoy_image=image_by_service["envoy"],
        nofile_limit=nofile_limit,
        runtime_network_name=runtime_network_name,
        common_env=common_env,
        listeners=listeners,
        runtime_paths=runtime_paths,
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
        router_secret_env_names=(inherited_sensitive_env | set(router_child_env_names)),
        dashboard_runtime_env=dashboard_runtime_env or {},
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
    stack_layout: RuntimeStackLayout,
    inherited_sensitive_env: set[str],
    management_listener: dict[str, int | str | bool],
    router_child_env_names: tuple[str, ...] = (),
):
    router_env = router_runtime_env(
        {
            name: value
            for name, value in common_env.items()
            if name not in DASHBOARD_SECRET_ENV_NAMES
        },
        normalized_platform,
    )
    # Names only. Each explicit binding is rendered as an inheriting `-e NAME`
    # flag, while its value exists only in the Router Docker child environment.
    for name in router_child_env_names:
        router_env.setdefault(name, "")
    service_entrypoint, service_args = bounded_log_spool_entrypoint(
        "/app/start-router.sh",
        [runtime_paths["runtime_container_config"]],
    )
    return _build_service_run_command(
        runtime=runtime,
        image=router_image,
        container_name=stack_layout.router_container_name,
        nofile_limit=nofile_limit,
        network_name=runtime_network_name,
        env_vars=router_env,
        mount_specs=[
            *_router_runtime_mount_specs(runtime_paths),
            *(
                [f"{router_env['SSL_CERT_FILE']}:{router_env['SSL_CERT_FILE']}:ro"]
                if router_env.get("SSL_CERT_FILE")
                else []
            ),
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
        start_immediately=True,
        inherited_env_keys=inherited_sensitive_env | set(router_child_env_names),
        docker_healthcheck=router_healthcheck(
            int(management_listener["port"]),
            tls_enabled=bool(management_listener["tls_enabled"]),
        ),
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
        start_immediately=True,
        supplemental_gids=[int(runtime_paths["log_spool_gid"])],
        docker_healthcheck=envoy_healthcheck(),
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
    router_secret_env_names: set[str],
    dashboard_runtime_env: Mapping[str, str],
    management_listener: dict[str, int | str | bool],
):
    dashboard_env = _build_dashboard_runtime_env(
        common_env=common_env,
        excluded_env_names=router_secret_env_names,
        dashboard_runtime_env=dashboard_runtime_env,
        listener_port=listener_port,
        stack_layout=stack_layout,
        management_port=int(management_listener["port"]),
        management_tls=bool(management_listener["tls_enabled"]),
        management_ca_file=str(management_listener["certificate_file"]),
    )
    # The Dashboard is an API client, not a second runtime authority. Mount
    # only its compiled bootstrap view and private state; never expose the
    # Router state directory that also contains unrelated control-plane keys.
    dashboard_mount_specs = [
        runtime_paths["compiled_bootstrap_mount"],
        f"{runtime_paths['dashboard_data_dir']}:/app/data:z",
        runtime_paths["log_spool_dashboard_mount"],
        f"{runtime_paths['log_spool_root']}:{LOG_SPOOL_READER_DIR}:ro,z",
    ]
    for name in (
        "DASHBOARD_SIGNING_KEY_FILE",
        "DASHBOARD_ISSUER_TLS_CERT_FILE",
        "DASHBOARD_ISSUER_TLS_KEY_FILE",
        "SSL_CERT_FILE",
    ):
        secret_path = dashboard_env.get(name, "").strip()
        if secret_path:
            dashboard_mount_specs.append(f"{secret_path}:{secret_path}:ro")
    bootstrap_token_path = dashboard_env.get(
        "DASHBOARD_ROUTER_BOOTSTRAP_TOKEN_FILE", ""
    ).strip()
    if bootstrap_token_path:
        bootstrap_token_directory = str(Path(bootstrap_token_path).parent)
        dashboard_mount_specs.append(
            f"{bootstrap_token_directory}:{bootstrap_token_directory}:rw"
        )
    dashboard_env["DASHBOARD_CONFIG_DIR"] = "/app"
    dashboard_env["ROUTER_CONFIG_PATH"] = runtime_paths["runtime_container_config"]
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
        inherited_env_keys=set(DASHBOARD_SECRET_ENV_NAMES),
        docker_healthcheck=dashboard_healthcheck(),
    )


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
    return (names & set(common_env)) | set(STORAGE_SECRET_ENV_NAMES)


def _build_dashboard_runtime_env(
    *,
    common_env: dict[str, str],
    excluded_env_names: set[str] | frozenset[str] = frozenset(),
    dashboard_runtime_env: Mapping[str, str] | None = None,
    listener_port: int,
    stack_layout: RuntimeStackLayout,
    management_port: int = 8080,
    management_tls: bool = False,
    management_ca_file: str = "",
):
    dashboard_env = {
        name: value
        for name, value in common_env.items()
        if name not in excluded_env_names
    }
    for name in (
        "DASHBOARD_ADMIN_EMAIL",
        "DASHBOARD_ADMIN_PASSWORD",
        "DASHBOARD_ADMIN_NAME",
        "DASHBOARD_ISSUER",
        "DASHBOARD_ISSUER_ID",
        "DASHBOARD_SIGNING_KEY_FILE",
        "DASHBOARD_KEY_ID",
        "DASHBOARD_ISSUER_TLS_LISTEN_ADDR",
        "DASHBOARD_ISSUER_TLS_CERT_FILE",
        "DASHBOARD_ISSUER_TLS_KEY_FILE",
        "DASHBOARD_ROUTER_BOOTSTRAP_TOKEN_FILE",
        "DASHBOARD_JWT_SECRET",
    ):
        value = os.getenv(name)
        if value:
            dashboard_env[name] = value
    for name, value in (dashboard_runtime_env or {}).items():
        if value and name not in dashboard_env:
            dashboard_env[name] = value

    bootstrap_policy_env = "DASHBOARD_ALLOW_OPEN_BOOTSTRAP"
    if bootstrap_policy_env in os.environ:
        dashboard_env[bootstrap_policy_env] = os.environ[bootstrap_policy_env]
    elif bootstrap_policy_env not in dashboard_env:
        bootstrap_email = dashboard_env.get("DASHBOARD_ADMIN_EMAIL", "").strip()
        bootstrap_password = dashboard_env.get("DASHBOARD_ADMIN_PASSWORD", "").strip()
        router_bootstrap = dashboard_env.get(
            "DASHBOARD_ROUTER_BOOTSTRAP_TOKEN_FILE", ""
        ).strip()
        if router_bootstrap or not (bootstrap_email and bootstrap_password):
            dashboard_env[bootstrap_policy_env] = "true"

    scheme = "https" if management_tls else "http"
    dashboard_env["TARGET_ROUTER_API_URL"] = (
        f"{scheme}://{stack_layout.router_container_name}:{management_port}"
    )
    if management_tls and management_ca_file:
        dashboard_env.setdefault("SSL_CERT_FILE", management_ca_file)
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
    docker_healthcheck: DockerHealthcheck,
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
    append_docker_runtime_policy(cmd, runtime, docker_healthcheck)
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
