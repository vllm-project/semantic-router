"""Where one runtime start puts its host state, and how it mounts it.

A `vllm-sr serve` materializes a set of directories beside the config it was
handed -- the active runtime config, the model cache, the Dashboard database,
the Recipe store, the log spool -- and every service command then mounts some
subset of them. Resolving those paths and rendering the mount specs is one
concern, separate from deciding what each container runs, so it lives beside
`container_start` rather than inside it.

Nothing here reaches back into `container_start`: the paths are computed once
and handed to the command builders as a plain mapping.
"""

import os
from pathlib import Path

from cli.commands.runtime_paths import (
    _container_readonly_source_config_path,
    _container_runtime_config_path,
    _runtime_config_output_path,
    materialize_runtime_config,
)
from cli.container_log_spool import prepare_runtime_log_spool
from cli.recipe_directory import resolve_active_recipe_directory
from cli.runtime_stack import RuntimeStackLayout, resolve_runtime_stack
from cli.utils import get_logger

log = get_logger(__name__)


def _prepare_runtime_directories(
    config_dir: str,
    vllm_sr_dir: str,
    stack_layout: RuntimeStackLayout,
    *,
    managed_recipe: bool,
) -> tuple[str, str, str, str]:
    """Create mutable runtime roots and return their resolved host paths."""

    evaluation_staging_root = os.path.join(config_dir, ".vllm-sr-evaluation-staging")
    models_dir = (
        os.path.join(vllm_sr_dir, "models")
        if managed_recipe
        else os.path.join(config_dir, "models")
    )
    dashboard_data_dir = os.path.join(vllm_sr_dir, "dashboard-data")
    recipe_store_dir = os.path.join(
        vllm_sr_dir, "recipe-store", stack_layout.stack_name
    )
    for directory in (models_dir, dashboard_data_dir, recipe_store_dir):
        os.makedirs(directory, exist_ok=True)
    log.info("Mounting dashboard data directory: %s", dashboard_data_dir)
    return evaluation_staging_root, models_dir, dashboard_data_dir, recipe_store_dir


def _prepare_runtime_paths(
    config_file,
    runtime_config_file=None,
    state_root_dir=None,
    stack_layout: RuntimeStackLayout | None = None,
):
    stack_layout = stack_layout or resolve_runtime_stack()
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
    # Keep immutable Evaluation deployment snapshots outside .vllm-sr. The
    # latter is mounted read-write into Dashboard, which would otherwise give
    # the container an alias around the snapshot's dedicated read-only mount.
    active_config_path = _runtime_config_output_path(
        Path(source_config_path),
        state_root_dir=config_dir,
        stack_name=stack_layout.stack_name,
    )
    if Path(runtime_config_path).resolve() != active_config_path.resolve():
        active_config_path = materialize_runtime_config(
            Path(source_config_path),
            Path(runtime_config_path).read_bytes(),
            state_root_dir=config_dir,
            stack_name=stack_layout.stack_name,
        )
    runtime_config_path = str(active_config_path)

    active_recipe = resolve_active_recipe_directory(source_config_path)
    # A managed Recipe source is a distributable five-file directory. Keep
    # mutable model/runtime state under its explicitly ignored .vllm-sr area
    # so serving it never changes the package contract.
    (
        evaluation_deployment_staging_root,
        models_dir,
        dashboard_data_dir,
        recipe_store_dir,
    ) = _prepare_runtime_directories(
        config_dir,
        vllm_sr_dir,
        stack_layout,
        managed_recipe=active_recipe is not None,
    )

    log_spool = prepare_runtime_log_spool(vllm_sr_dir, stack_layout.stack_name)

    effective_config_path = runtime_config_path
    envoy_config_path = os.path.join(vllm_sr_dir, "envoy.yaml")

    runtime_container_config = _container_runtime_config_path(
        Path(source_config_path), stack_name=stack_layout.stack_name
    )
    log.info(
        "Using read-only source config %s with active runtime config %s",
        source_config_path,
        runtime_container_config,
    )

    active_recipe_paths = {
        f"active_recipe_{name.replace('.', '_').replace('-', '_')}_path": str(path)
        for name, path in (active_recipe.assets if active_recipe else ())
    }

    return (
        config_dir,
        {
            "source_config_path": source_config_path,
            "effective_config_path": effective_config_path,
            "vllm_sr_dir": vllm_sr_dir,
            "evaluation_deployment_staging_root": (evaluation_deployment_staging_root),
            "models_dir": models_dir,
            "dashboard_data_dir": dashboard_data_dir,
            "recipe_store_dir": recipe_store_dir,
            "log_spool_logs_root": str(log_spool.root.parent),
            "log_spool_root": str(log_spool.root),
            "log_spool_gid": str(log_spool.gid),
            **{
                f"log_spool_{component}_mount": log_spool.producer_mount(component)
                for component in ("router", "envoy", "dashboard")
            },
            "container_recipe_store_dir": (
                f"/app/.vllm-sr/recipe-store/{stack_layout.stack_name}"
            ),
            "envoy_config_path": envoy_config_path,
            "runtime_container_config": runtime_container_config,
            "active_recipe_root": str(active_recipe.root) if active_recipe else "",
            **active_recipe_paths,
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
        f"{runtime_paths['source_config_path']}:{_container_readonly_source_config_path()}:ro,z",
        f"{runtime_paths['vllm_sr_dir']}:/app/.vllm-sr:z",
        f"{runtime_paths['log_spool_logs_root']}:/app/.vllm-sr/logs:ro,z",
    ]
    if include_models:
        mounts.append(f"{runtime_paths['models_dir']}:/app/models:z")
    if include_dashboard_data:
        mounts.append(f"{runtime_paths['dashboard_data_dir']}:/app/data:z")
    return mounts


def _active_recipe_mount_specs(runtime_paths: dict[str, str]) -> list[str]:
    if not runtime_paths.get("active_recipe_root"):
        return []

    mounts = []
    for filename in (
        "config.yaml",
        "metadata.yaml",
        "probes.yaml",
        "recipe.dsl",
        "README.md",
    ):
        key = f"active_recipe_{filename.replace('.', '_').replace('-', '_')}_path"
        mounts.append(f"{runtime_paths[key]}:/app/recipe/{filename}:ro,z")
    return mounts
