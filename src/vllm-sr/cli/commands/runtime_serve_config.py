"""Turning one `serve` request into the config the target will actually run.

Both halves of that job live here. The local Docker half takes the runtime
config lock, finishes any Recipe activation the last run left pending, and
materializes this stack's active config under its private state root; the
Kubernetes half deliberately keeps its translation in memory so a target-neutral
transform never lands in the local active config path. Neither is Click plumbing
and neither is deployment, so they stay out of `commands.runtime`, which owns
the command surface and the user-facing serve flow.
"""

from __future__ import annotations

import os
from pathlib import Path

from cli.bootstrap import is_setup_mode_config
from cli.commands.runtime_model_source import ServeModelSource
from cli.commands.runtime_paths import (
    _runtime_config_output_path,
    materialize_runtime_config,
)
from cli.commands.runtime_support import (
    build_effective_config_bytes,
    build_effective_config_document,
    validate_config_recipe_env_bindings,
)
from cli.container_services import container_status_strict
from cli.recipe_activation_recovery import (
    active_recipe_package_config_path,
    active_recipe_package_for_stack,
    recover_pending_recipe_activation_for_stack,
)
from cli.runtime_config_lock import acquire_runtime_config_lock
from cli.runtime_stack import resolve_runtime_stack
from cli.utils import get_logger

log = get_logger(__name__)


def _prepare_docker_runtime_config(
    config_path: Path,
    algorithm: str | None,
    source_setup_mode: bool,
    platform: str | None,
    recipe_env_bindings: tuple[str, ...],
    *,
    state_root_dir: Path | None = None,
    require_exact_source: bool = False,
):
    stack_layout = resolve_runtime_stack()
    state_root_dir = state_root_dir or (
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
        if package_active:
            if require_exact_source:
                raise ValueError(
                    "This stack has an active managed Recipe, so the requested "
                    "catalog model selection was not applied. Deactivate the Recipe "
                    "in the Dashboard or use a different VLLM_SR_STACK_NAME."
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
        else:
            effective_config_bytes = build_effective_config_bytes(
                config_path, algorithm, source_setup_mode, platform
            )
            effective_config_path = materialize_runtime_config(
                config_path,
                effective_config_bytes,
                state_root_dir=state_root_dir,
                stack_name=stack_layout.stack_name,
            )
            if (
                require_exact_source
                and effective_config_path.read_bytes() != effective_config_bytes
            ):
                raise ValueError(
                    "The requested catalog model selection conflicts with Dashboard "
                    "changes preserved in this stack's runtime config. Use a different "
                    "VLLM_SR_STACK_NAME or serve the edited config explicitly with "
                    "--config."
                )
        setup_mode = is_setup_mode_config(effective_config_path)
        return effective_config_path, setup_mode, runtime_lock
    except Exception:
        runtime_lock.close()
        raise


def _prepare_effective_serve_config(
    config_path: Path,
    *,
    model_source: ServeModelSource | None,
    resolved_target: str,
    algorithm: str | None,
    source_setup_mode: bool,
    platform: str | None,
    recipe_env_bindings: tuple[str, ...],
):
    """Prepare the target-specific active config and its optional runtime lock."""

    if model_source is not None and resolved_target != "docker":
        raise ValueError(
            "serve MODEL currently supports the local Docker target. For Kubernetes, "
            "materialize an editable config with 'vllm-sr model fork' and deploy it "
            "through the chart or operator workflow."
        )
    if resolved_target != "docker" and recipe_env_bindings:
        raise ValueError(
            "--recipe-env is supported only for local Docker Recipe packages"
        )
    if resolved_target == "docker":
        effective_path, setup_mode, runtime_lock = _prepare_docker_runtime_config(
            config_path,
            algorithm,
            source_setup_mode,
            platform,
            recipe_env_bindings,
            state_root_dir=(
                model_source.state_root if model_source is not None else None
            ),
            require_exact_source=model_source is not None,
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
