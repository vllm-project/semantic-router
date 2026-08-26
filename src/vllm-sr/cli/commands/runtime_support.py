"""Support helpers for runtime-oriented CLI commands."""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

from cli.bootstrap import BootstrapResult
from cli.commands.runtime_kb import (
    _sync_runtime_kb_store,
)
from cli.commands.runtime_looper import apply_local_looper_endpoint
from cli.commands.runtime_management_credentials import (
    management_credential_env_names,
)
from cli.commands.runtime_paths import _write_compiled_bootstrap
from cli.compiled_bootstrap_overrides import (
    _platform_requires_gpu_defaults,
    apply_platform_gpu_defaults,
)
from cli.consts import (
    CONTAINER_RUNTIME_ENV,
    SUPPORTED_CONTAINER_RUNTIMES,
)
from cli.runtime_env_names import (
    RESERVED_RUNTIME_ENV_NAMES,
)
from cli.runtime_stack import resolve_runtime_stack
from cli.service_defaults import (
    inject_local_service_runtime_defaults,
    inject_local_store_runtime_defaults,
)
from cli.storage_secrets import POSTGRES_PASSWORD_ENV, REDIS_PASSWORD_ENV
from cli.utils import get_logger
from cli.yaml_contract import load_yaml

log = get_logger(__name__)

# Preserve trusted source-config passthrough semantics, including references
# with fallbacks.
_ENV_REFERENCE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::?-[^}]*)?\}")

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

_STATIC_SENSITIVE = frozenset(
    {
        *(name for name, masked in PASSTHROUGH_ENV_RULES if masked),
        POSTGRES_PASSWORD_ENV,
        REDIS_PASSWORD_ENV,
    }
)


# Never auto-forward these from a ${VAR} match: POSIX process/identity vars are present
# in every host shell, so an incidental match (a templated path, an example URL) would
# silently leak them into the container. CLI override vars are covered too, so a config
# that happens to reference one can't inject a host value ahead of the CLI's own default.
def _finalize_runtime_config_write(
    config_path: Path, config: dict[str, object], changed: bool
) -> Path:
    if not changed:
        return config_path

    effective_config = _write_compiled_bootstrap(config_path, config)
    log.info(f"Compiled immutable bootstrap: {effective_config}")
    return effective_config


def log_bootstrap_result(requested_config: str, bootstrap: BootstrapResult) -> None:
    """Report any workspace files created during bootstrap."""
    if bootstrap.created_config:
        log.warning(f"Config file not found: {requested_config}")
        log.info(f"Created Management workspace config: {bootstrap.config_path}")
    if bootstrap.created_output_dir:
        log.info(f"Created bootstrap output directory: {bootstrap.output_dir}")
    if bootstrap.created_secrets:
        log.info(f"Created private local trust material: {bootstrap.secret_dir}")


def config_env_references(config_path: Path | str | None) -> set[str]:
    """Env names used by trusted source config passthrough.

    Provider credentials use named ``secret_env`` references. Other subsystem-
    specific ``*_env`` fields and exact braced references are collected
    by the same non-value-bearing traversal.
    """
    if config_path is None:
        return set()
    try:
        document = load_yaml(Path(config_path).read_text())
    except (OSError, yaml.YAMLError):
        return set()

    names: set[str] = set()
    pending: list[object] = [document]
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            for field, value in node.items():
                if (
                    isinstance(field, str)
                    and field.endswith("_env")
                    and isinstance(value, str)
                    and value.strip()
                ):
                    names.add(value.strip())
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)
        elif isinstance(node, str):
            names.update(_ENV_REFERENCE.findall(node))
    return names - RESERVED_RUNTIME_ENV_NAMES


def sensitive_env_names(config_path: Path | str | None = None) -> set[str]:
    """Names that must reach the container as a secret, never as plain-text manifest."""
    return (
        _STATIC_SENSITIVE
        | config_env_references(config_path)
        | management_credential_env_names(config_path)
    )


def append_passthrough_env_vars(
    env_vars: dict[str, str], config_path: Path | str | None = None
) -> None:
    """Pass established host variables for an operator-trusted source config."""
    discovered_names = config_env_references(
        config_path
    ) | management_credential_env_names(config_path)
    mask_rules = dict(PASSTHROUGH_ENV_RULES)
    for name in sorted(discovered_names):
        mask_rules[name] = True
    for name, masked in mask_rules.items():
        if name in env_vars:
            continue
        value = os.environ.get(name)
        if value is None:
            continue
        env_vars[name] = value
        logged_value = "***" if masked else value
        log.info(f"Passing environment variable: {name}={logged_value}")


def apply_container_runtime_override(runtime: str | None) -> None:
    """Apply ``--runtime`` to the process environment so detection picks it up.

    The ``CONTAINER_RUNTIME`` environment variable is the canonical input that
    ``cli.container_runtime._detect_container_runtime`` reads. This helper bridges
    the CLI flag to that env var, normalizes the value, validates it, and
    invalidates the runtime detection cache so the next call re-resolves.
    """
    if runtime is None:
        return
    normalized = runtime.strip().lower()
    if not normalized:
        return
    if normalized not in SUPPORTED_CONTAINER_RUNTIMES:
        raise ValueError(
            f"Unsupported --runtime value: {runtime!r}. "
            f"Choose one of: {', '.join(SUPPORTED_CONTAINER_RUNTIMES)}."
        )
    os.environ[CONTAINER_RUNTIME_ENV] = normalized
    # Reset the cached detection so subsequent get_container_runtime() calls
    # observe the new override instead of a stale answer from earlier in the
    # process.
    from cli.container_runtime import reset_container_runtime_cache  # noqa: PLC0415

    reset_container_runtime_cache()
    log.info(f"Container runtime override: {normalized}")


def apply_runtime_mode_env_vars(
    env_vars: dict[str, str],
    minimal: bool,
    readonly: bool,
    platform: str | None,
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

    if platform:
        env_vars["DASHBOARD_PLATFORM"] = platform
        env_vars["VLLM_SR_PLATFORM"] = platform
        log.info(f"Platform branding: {platform}")

    if log_level:
        normalized_log_level = log_level.lower()
        env_vars["SR_LOG_LEVEL"] = normalized_log_level
        log.info(f"Router log level: {normalized_log_level}")


def _resolve_effective_config_document(
    config_path: Path,
    platform: str | None,
    *,
    materialize_local_runtime: bool = True,
) -> tuple[dict[str, object], bool]:
    with config_path.open() as handle:
        config = load_yaml(handle) or {}

    if materialize_local_runtime:
        kb_runtime_required, changed = _sync_runtime_kb_store(config, config_path)
    else:
        kb_runtime_required, changed = False, False
    if materialize_local_runtime:
        stack = resolve_runtime_stack()
        changed = inject_local_service_runtime_defaults(config, stack) or changed
        changed = inject_local_store_runtime_defaults(config, stack) or changed
        changed = apply_local_looper_endpoint(config, stack) or changed
    apply_gpu_defaults = _platform_requires_gpu_defaults(platform)
    if not kb_runtime_required and not apply_gpu_defaults and not changed:
        return config, False

    changed = apply_platform_gpu_defaults(config, platform) or changed
    return config, changed or kb_runtime_required


def build_effective_config_bytes(
    config_path: Path,
    platform: str | None,
) -> bytes:
    """Build the effective runtime config without touching active runtime state."""

    config, changed = _resolve_effective_config_document(
        config_path,
        platform,
    )
    if not changed:
        return config_path.read_bytes()
    return yaml.dump(config, default_flow_style=False, sort_keys=False).encode("utf-8")


def build_effective_config_document(
    config_path: Path,
    platform: str | None,
    *,
    materialize_local_runtime: bool = True,
) -> dict[str, object]:
    """Build an effective config in memory without publishing runtime state."""

    config, _changed = _resolve_effective_config_document(
        config_path,
        platform,
        materialize_local_runtime=materialize_local_runtime,
    )
    return config


def resolve_effective_config_path(
    config_path: Path,
    platform: str | None,
    *,
    materialize_local_runtime: bool = True,
) -> Path:
    """Compile target-appropriate bootstrap staging from the source config."""

    config, changed = _resolve_effective_config_document(
        config_path,
        platform,
        materialize_local_runtime=materialize_local_runtime,
    )
    return _finalize_runtime_config_write(
        config_path,
        config,
        changed,
    )
