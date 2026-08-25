"""Prepare immutable local-stack paths and the split Envoy configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from cli.commands.runtime_paths import (
    _compiled_bootstrap_output_path,
    _container_compiled_bootstrap_path,
    materialize_compiled_bootstrap,
)
from cli.config_generator import generate_envoy_config_from_user_config
from cli.container_log_spool import prepare_runtime_log_spool
from cli.control_plane_deployment import local_control_plane_secret_mounts
from cli.envoy_dispatch_contract import (
    BACKEND_DISPATCH_ADDRESS_ENV,
    validate_networked_backend_dispatch,
)
from cli.parser import parse_user_config
from cli.runtime_stack import RuntimeStackLayout, resolve_runtime_stack
from cli.utils import get_logger, load_config

log = get_logger(__name__)


def prepare_runtime_paths(
    config_file: str,
    compiled_bootstrap_file: str | None = None,
    state_root_dir: str | None = None,
    stack_layout: RuntimeStackLayout | None = None,
) -> tuple[str, dict[str, Any], str]:
    """Materialize private runtime paths without mutating the source config."""

    stack_layout = stack_layout or resolve_runtime_stack()
    source_config_path = os.path.abspath(config_file)
    config_dir = (
        os.path.abspath(state_root_dir)
        if state_root_dir
        else os.path.dirname(source_config_path)
    )
    state_dir = os.path.join(config_dir, ".vllm-sr")
    os.makedirs(state_dir, exist_ok=True)
    log.info("Using private runtime state directory: %s", state_dir)

    compiled_bootstrap_path = _prepare_compiled_bootstrap(
        source_config_path,
        config_dir,
        stack_layout,
        compiled_bootstrap_file,
    )
    models_dir = os.path.join(config_dir, "models")
    dashboard_data_dir = os.path.join(state_dir, "dashboard-data")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(dashboard_data_dir, exist_ok=True)
    log.info("Mounting dashboard data directory: %s", dashboard_data_dir)

    log_spool = prepare_runtime_log_spool(state_dir, stack_layout.stack_name)
    control_plane_secret_mounts = local_control_plane_secret_mounts(
        load_config(str(compiled_bootstrap_path)) or {}
    )
    knowledge_bases_mount = _knowledge_bases_mount(Path(state_dir))
    runtime_container_config = _container_compiled_bootstrap_path(
        Path(source_config_path), stack_name=stack_layout.stack_name
    )
    log.info(
        "Using immutable source config %s with read-only compiled bootstrap %s",
        source_config_path,
        runtime_container_config,
    )

    runtime_paths: dict[str, Any] = {
        "source_config_path": source_config_path,
        "effective_config_path": str(compiled_bootstrap_path),
        "compiled_bootstrap_mount": (
            f"{compiled_bootstrap_path}:{runtime_container_config}:ro,z"
        ),
        "knowledge_bases_mount": knowledge_bases_mount,
        "models_dir": models_dir,
        "dashboard_data_dir": dashboard_data_dir,
        "log_spool_root": str(log_spool.root),
        "log_spool_gid": str(log_spool.gid),
        **{
            f"log_spool_{component}_mount": log_spool.producer_mount(component)
            for component in ("router", "envoy", "dashboard")
        },
        "envoy_config_path": os.path.join(state_dir, "envoy.yaml"),
        "runtime_container_config": runtime_container_config,
        "control_plane_secret_mounts": control_plane_secret_mounts,
    }
    return config_dir, runtime_paths, runtime_container_config


def _prepare_compiled_bootstrap(
    source_config_path: str,
    config_dir: str,
    stack_layout: RuntimeStackLayout,
    compiled_bootstrap_file: str | None,
) -> Path:
    expected = _compiled_bootstrap_output_path(
        Path(source_config_path),
        state_root_dir=config_dir,
        stack_name=stack_layout.stack_name,
    )
    if compiled_bootstrap_file is None:
        return materialize_compiled_bootstrap(
            Path(source_config_path),
            Path(source_config_path).read_bytes(),
            state_root_dir=config_dir,
            stack_name=stack_layout.stack_name,
        )
    supplied = Path(compiled_bootstrap_file).expanduser().absolute()
    if supplied.resolve() != expected.resolve():
        raise ValueError(
            "compiled_bootstrap_file must match this stack's CLI-compiled bootstrap"
        )
    if not supplied.is_file() or supplied.is_symlink():
        raise ValueError("compiled bootstrap must be a private regular file")
    return supplied


def _knowledge_bases_mount(state_dir: Path) -> str:
    knowledge_bases_dir = state_dir / "knowledge_bases"
    if not (knowledge_bases_dir.exists() or knowledge_bases_dir.is_symlink()):
        return ""
    if knowledge_bases_dir.is_symlink() or not knowledge_bases_dir.is_dir():
        raise ValueError("Runtime knowledge-base path must be an owned directory")
    return f"{knowledge_bases_dir}:/app/.vllm-sr/knowledge_bases:ro,z"


def router_runtime_mount_specs(runtime_paths: dict[str, Any]) -> list[str]:
    """Return only the files and data roots consumed by the Router."""

    mounts = [runtime_paths["compiled_bootstrap_mount"]]
    if knowledge_bases_mount := runtime_paths.get("knowledge_bases_mount", ""):
        mounts.append(knowledge_bases_mount)
    mounts.append(f"{runtime_paths['models_dir']}:/app/models:z")
    mounts.extend(runtime_paths.get("control_plane_secret_mounts", ()))
    return mounts


def primary_listener_port(listeners: list[dict[str, Any]]) -> int:
    """Return the first configured listener port or the public default."""

    for listener in listeners:
        if port := listener.get("port"):
            return int(port)
    return 8888


def render_split_envoy_config(
    config_path: str,
    output_path: str,
    stack_layout: RuntimeStackLayout,
) -> None:
    """Compile the Envoy side of a split Router/Envoy local stack."""

    user_config = parse_user_config(config_path)
    validate_networked_backend_dispatch(user_config, "Docker")
    original_values = {
        "ENVOY_EXTPROC_ADDRESS": os.environ.get("ENVOY_EXTPROC_ADDRESS"),
        "ENVOY_ROUTER_API_ADDRESS": os.environ.get("ENVOY_ROUTER_API_ADDRESS"),
        BACKEND_DISPATCH_ADDRESS_ENV: os.environ.get(BACKEND_DISPATCH_ADDRESS_ENV),
    }
    os.environ.update(
        {
            "ENVOY_EXTPROC_ADDRESS": stack_layout.router_container_name,
            "ENVOY_ROUTER_API_ADDRESS": stack_layout.router_container_name,
            BACKEND_DISPATCH_ADDRESS_ENV: stack_layout.router_container_name,
        }
    )
    try:
        generate_envoy_config_from_user_config(user_config, output_path)
        log.info("Rendered split Envoy config: %s", output_path)
    finally:
        for name, value in original_values.items():
            _restore_env_var(name, value)


def _restore_env_var(name: str, original_value: str | None) -> None:
    if original_value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = original_value
