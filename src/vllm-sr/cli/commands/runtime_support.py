"""Support helpers for runtime-oriented CLI commands."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from cli.bootstrap import (
    DASHBOARD_SETUP_MODE_ENV,
    SETUP_MODE_ENV,
    BootstrapResult,
    is_setup_mode_config,
)
from cli.commands.runtime_config_mutation import (
    _apply_algorithm_override,
    _normalized_algorithm_override,
    _platform_requires_gpu_defaults,
    apply_platform_gpu_defaults,
)
from cli.commands.runtime_kb import _sync_runtime_kb_store
from cli.commands.runtime_paths import (
    _container_runtime_config_path,
    _container_source_config_path,
    _write_runtime_config,
)
from cli.runtime_stack import resolve_runtime_stack
from cli.service_defaults import (
    inject_local_service_runtime_defaults,
    inject_local_store_runtime_defaults,
)
from cli.utils import get_logger

log = get_logger(__name__)

RUNTIME_CONFIG_PATH_ENV = "VLLM_SR_RUNTIME_CONFIG_PATH"
SOURCE_CONFIG_PATH_ENV = "VLLM_SR_SOURCE_CONFIG_PATH"
RUNTIME_ALGORITHM_OVERRIDE_ENV = "VLLM_SR_ALGORITHM_OVERRIDE"

PASSTHROUGH_ENV_RULES = (
    ("HF_ENDPOINT", False),
    ("HF_TOKEN", True),
    ("HF_HOME", False),
    ("HF_HUB_CACHE", False),
    ("VLLM_SR_DETERMINISTIC_EMBEDDINGS", False),
    ("ANTHROPIC_API_KEY", True),
    ("OPENAI_API_KEY", True),
    ("OPENROUTER_API_KEY", True),
    ("OPENCLAW_BASE_IMAGE", False),
    ("SR_LOG_LEVEL", False),
    ("SR_LOG_ENCODING", False),
    ("SR_LOG_DEVELOPMENT", False),
    ("SR_LOG_ADD_CALLER", False),
)


def _finalize_runtime_config_write(
    config_path: Path, config: dict[str, object], changed: bool
) -> Path:
    if not changed:
        return config_path

    effective_config = _write_runtime_config(config_path, config)
    log.info(f"Created effective runtime config: {effective_config}")
    return effective_config


def log_bootstrap_result(requested_config: str, bootstrap: BootstrapResult) -> None:
    """Report any workspace files created during bootstrap."""
    if bootstrap.created_config:
        log.warning(f"Config file not found: {requested_config}")
        log.info(f"Created bootstrap setup config: {bootstrap.config_path}")
    if bootstrap.created_output_dir:
        log.info(f"Created bootstrap output directory: {bootstrap.output_dir}")
    if bootstrap.created_defaults:
        log.info(
            f"Wrote global defaults reference: {bootstrap.output_dir / 'global-defaults.yaml'}"
        )


def validate_setup_mode_flags(setup_mode: bool, minimal: bool, readonly: bool) -> None:
    """Reject option combinations that conflict with dashboard-first bootstrap."""
    if setup_mode and minimal:
        raise ValueError(
            "Setup mode requires the dashboard. Remove --minimal or create a full config first."
        )
    if setup_mode and readonly:
        raise ValueError(
            "Setup mode requires dashboard editing. Remove --readonly or activate a config first."
        )


def append_passthrough_env_vars(env_vars: dict[str, str]) -> None:
    """Pass selected host environment variables into the container runtime."""
    for name, masked in PASSTHROUGH_ENV_RULES:
        value = os.environ.get(name)
        if value is None:
            continue
        env_vars[name] = value
        logged_value = "***" if masked else value
        log.info(f"Passing environment variable: {name}={logged_value}")


def apply_runtime_mode_env_vars(
    env_vars: dict[str, str],
    minimal: bool,
    readonly: bool,
    setup_mode: bool,
    platform: str | None,
    algorithm: str | None = None,
    log_level: str | None = None,
) -> None:
    """Apply runtime-mode environment variables derived from CLI flags."""
    if minimal:
        env_vars["DISABLE_DASHBOARD"] = "true"
        log.info("Minimal mode: ENABLED (no dashboard, no observability)")
        if readonly:
            log.warning("--readonly is ignored in minimal mode (dashboard is disabled)")

    if readonly and not minimal:
        env_vars["DASHBOARD_READONLY"] = "true"
        log.info("Dashboard read-only mode: ENABLED")

    if setup_mode:
        env_vars[SETUP_MODE_ENV] = "true"
        env_vars[DASHBOARD_SETUP_MODE_ENV] = "true"
        log.info(
            "Setup mode: starting dashboard-first bootstrap flow with router/envoy on standby"
        )

    if platform:
        env_vars["DASHBOARD_PLATFORM"] = platform
        env_vars["VLLM_SR_PLATFORM"] = platform
        log.info(f"Platform branding: {platform}")

    if algorithm:
        env_vars[RUNTIME_ALGORITHM_OVERRIDE_ENV] = algorithm.lower()

    if log_level:
        normalized_log_level = log_level.lower()
        env_vars["SR_LOG_LEVEL"] = normalized_log_level
        log.info(f"Router log level: {normalized_log_level}")


def resolve_effective_config_path(
    config_path: Path, algorithm: str | None, setup_mode: bool, platform: str | None
) -> Path:
    """Apply CLI algorithm and platform override translation when appropriate."""
    with config_path.open() as handle:
        config = yaml.safe_load(handle) or {}

    kb_runtime_required, changed = _sync_runtime_kb_store(config, config_path)
    if not setup_mode:
        stack = resolve_runtime_stack()
        changed = inject_local_service_runtime_defaults(config, stack) or changed
        changed = inject_local_store_runtime_defaults(config, stack) or changed
    normalized_algorithm = _normalized_algorithm_override(algorithm, setup_mode)
    apply_gpu_defaults = _platform_requires_gpu_defaults(platform)
    if (
        not kb_runtime_required
        and not normalized_algorithm
        and not apply_gpu_defaults
        and not changed
    ):
        return config_path

    if normalized_algorithm:
        changed = _apply_algorithm_override(config, normalized_algorithm) or changed

    changed = apply_platform_gpu_defaults(config, platform) or changed
    return _finalize_runtime_config_write(
        config_path,
        config,
        changed or kb_runtime_required,
    )


def sync_runtime_config(
    config_path: Path, algorithm: str | None = None, platform: str | None = None
) -> Path:
    """Regenerate the internal runtime config from the user-facing source config."""
    return resolve_effective_config_path(
        config_path=config_path,
        algorithm=algorithm,
        setup_mode=is_setup_mode_config(config_path),
        platform=platform,
    )


def configure_runtime_override_env_vars(
    env_vars: dict[str, str],
    source_config_path: Path,
    effective_config_path: Path,
) -> None:
    """Expose the runtime-only config path to the container when overrides exist."""
    env_vars[SOURCE_CONFIG_PATH_ENV] = _container_source_config_path()

    if source_config_path.resolve() != effective_config_path.resolve() or env_vars.get(
        RUNTIME_ALGORITHM_OVERRIDE_ENV
    ):
        env_vars[RUNTIME_CONFIG_PATH_ENV] = _container_runtime_config_path(
            source_config_path
        )
        return

    env_vars.pop(RUNTIME_CONFIG_PATH_ENV, None)
