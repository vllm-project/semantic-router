"""Turning one `serve` request into the config the target will actually run.

Both halves of that job live here. The local Docker half takes the runtime
config lock, finishes any Recipe activation the last run left pending, and
materializes this stack's active config under its private state root; the
Kubernetes half deliberately keeps its translation in memory so a target-neutral
transform never lands in the local active config path. Before replacing an
existing Docker config, preparation stops its old consumers so the restart
cannot also trigger a reload in the departing process.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from cli.bootstrap import is_setup_mode_config
from cli.commands.runtime_observability import (
    reconcile_runtime_tracing,
    recover_runtime_tracing_projection,
    validate_package_tracing_mode,
)
from cli.commands.runtime_paths import (
    _runtime_config_output_path,
    materialize_runtime_config,
)
from cli.commands.runtime_support import (
    build_effective_config_bytes,
    build_effective_config_document,
    validate_config_recipe_env_bindings,
    validate_setup_mode_flags,
)
from cli.container_services import container_status_strict
from cli.parser import parse_user_config
from cli.recipe_activation_recovery import (
    active_recipe_package_config_path,
    active_recipe_package_for_stack,
    recover_pending_recipe_activation_for_stack,
)
from cli.runtime_config_lock import acquire_runtime_config_lock
from cli.runtime_lifecycle import stop_runtime_before_config_replacement
from cli.runtime_stack import RuntimeStackLayout, resolve_runtime_stack
from cli.utils import get_logger
from cli.validator import validate_user_config

log = get_logger(__name__)


def _prepare_runtime_config_replacement(
    effective_data: bytes,
    stack_layout: RuntimeStackLayout,
    *,
    minimal: bool,
    readonly: bool,
) -> None:
    # Validate the candidate without publishing it to the running file watcher.
    # A restart owns the next generation; the departing process must not build it.
    with tempfile.TemporaryDirectory(prefix="vllm-sr-config-") as directory:
        candidate = Path(directory) / "config.yaml"
        candidate.write_bytes(effective_data)
        prepared = parse_user_config(str(candidate), log_summary=False)
        errors = validate_user_config(prepared, log_summary=False)
        if errors:
            raise ValueError("Invalid runtime config: " + "; ".join(map(str, errors)))
        validate_setup_mode_flags(is_setup_mode_config(candidate), minimal, readonly)
    stop_runtime_before_config_replacement(stack_layout)


def _prepare_docker_runtime_config(
    config_path: Path,
    algorithm: str | None,
    source_setup_mode: bool,
    platform: str | None,
    recipe_env_bindings: tuple[str, ...],
    replace_active_config: bool,
    *,
    minimal: bool = False,
    readonly: bool = False,
):
    stack_layout = resolve_runtime_stack()
    state_root_dir = (
        Path(os.environ["VLLM_SR_STATE_ROOT_DIR"]).expanduser().absolute()
        if os.getenv("VLLM_SR_STATE_ROOT_DIR", "").strip()
        else config_path.expanduser().absolute().parent
    )
    effective_config_path = _runtime_config_output_path(
        config_path,
        state_root_dir=state_root_dir,
        stack_name=stack_layout.stack_name,
    )
    runtime_lock = acquire_runtime_config_lock(
        runtime_config_path=effective_config_path,
        state_root_dir=state_root_dir,
        stack_name=stack_layout.stack_name,
        timeout_seconds=0,
    )
    try:
        recover_pending_recipe_activation_for_stack(
            runtime_config_path=effective_config_path,
            state_root_dir=state_root_dir,
            stack_name=stack_layout.stack_name,
            managed_container_names=stack_layout.runtime_container_names,
            status_provider=container_status_strict,
        )
        package_active = active_recipe_package_for_stack(
            state_root_dir=state_root_dir, stack_name=stack_layout.stack_name
        )
        source_candidate_selected = False
        if package_active:
            if replace_active_config:
                raise ValueError(
                    "--replace-active-config cannot replace an active Recipe "
                    "package; change or deactivate that package through the "
                    "Recipe workflow first"
                )
            # Authorize against what the package declares, not against what
            # the CLI materialized. The runtime config also carries the CLI's
            # own references -- this stack's generated storage credentials --
            # which use reserved names an operator can neither bind nor need
            # to. The package's immutable config.yaml is the authorization
            # surface; validating the realized document would reject the
            # CLI's own output. Falling back keeps the stricter superset when
            # the authored file cannot be resolved.
            validate_config_recipe_env_bindings(
                active_recipe_package_config_path(
                    state_root_dir=state_root_dir,
                    stack_name=stack_layout.stack_name,
                )
                or effective_config_path,
                recipe_env_bindings,
            )
            validate_package_tracing_mode(
                effective_config_path, stack_layout, minimal=minimal
            )
        else:
            recover_runtime_tracing_projection(effective_config_path)
            effective_config_bytes = build_effective_config_bytes(
                config_path, algorithm, source_setup_mode, platform
            )
            effective_config_path = materialize_runtime_config(
                config_path,
                effective_config_bytes,
                state_root_dir=state_root_dir,
                stack_name=stack_layout.stack_name,
                replace_active=replace_active_config,
                before_replace=lambda: _prepare_runtime_config_replacement(
                    effective_config_bytes,
                    stack_layout,
                    minimal=minimal,
                    readonly=readonly,
                ),
            )
            source_candidate_selected = (
                effective_config_path.read_bytes() == effective_config_bytes
            )
        setup_mode = is_setup_mode_config(effective_config_path)
        if not setup_mode and not package_active:
            reconcile_runtime_tracing(
                effective_config_path,
                stack_layout,
                enable_observability=not minimal,
                reset_projection=source_candidate_selected,
                before_replace=lambda data: _prepare_runtime_config_replacement(
                    data, stack_layout, minimal=minimal, readonly=readonly
                ),
            )
        return effective_config_path, setup_mode, runtime_lock
    except Exception:
        runtime_lock.close()
        raise


def _prepare_effective_serve_config(
    config_path: Path,
    *,
    resolved_target: str,
    algorithm: str | None,
    source_setup_mode: bool,
    platform: str | None,
    recipe_env_bindings: tuple[str, ...],
    replace_active_config: bool,
    minimal: bool = False,
    readonly: bool = False,
):
    """Prepare the target-specific active config and its optional runtime lock."""

    if resolved_target != "docker" and recipe_env_bindings:
        raise ValueError(
            "--recipe-env is supported only for local Docker Recipe packages"
        )
    if resolved_target != "docker" and replace_active_config:
        raise ValueError(
            "--replace-active-config is supported only for local Docker deployments"
        )
    if resolved_target == "docker":
        effective_path, setup_mode, runtime_lock = _prepare_docker_runtime_config(
            config_path,
            algorithm,
            source_setup_mode,
            platform,
            recipe_env_bindings,
            replace_active_config,
            minimal=minimal,
            readonly=readonly,
        )
        return effective_path, setup_mode, runtime_lock, None

    # Kubernetes is an in-memory translation flow. Never publish its target-
    # neutral transforms into the local Docker/Dashboard active config path.
    effective_config_document = build_effective_config_document(
        config_path,
        algorithm,
        source_setup_mode,
        platform,
        materialize_local_runtime=False,
    )
    return config_path, source_setup_mode, None, effective_config_document
