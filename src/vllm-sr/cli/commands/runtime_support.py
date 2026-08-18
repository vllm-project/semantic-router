"""Support helpers for runtime-oriented CLI commands."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
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
from cli.commands.runtime_kb import (
    _sync_runtime_kb_store,
    _validate_package_kb_paths,
)
from cli.commands.runtime_looper import apply_local_looper_endpoint
from cli.commands.runtime_management_credentials import (
    management_credential_env_names,
)
from cli.commands.runtime_paths import (
    _container_runtime_config_path,
    _write_runtime_config,
    write_runtime_config_bytes,
)
from cli.consts import (
    CONTAINER_RUNTIME_ENV,
    SUPPORTED_CONTAINER_RUNTIMES,
)
from cli.runtime_env_names import (
    RESERVED_RUNTIME_ENV_NAMES,
    normalize_runtime_env_names,
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
RECIPE_ENV_ALLOWLIST_ENV = "VLLM_SR_RECIPE_ENV_ALLOWLIST"

# Preserve trusted source-config passthrough semantics, including references
# with fallbacks. A narrower matcher below drives package activation policy.
_ENV_REFERENCE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::?-[^}]*)?\}")

API_KEY_ENV_FIELD = "api_key_env"

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

_STATIC_SENSITIVE = frozenset(name for name, masked in PASSTHROUGH_ENV_RULES if masked)


# Never auto-forward these from a ${VAR} match: POSIX process/identity vars are present
# in every host shell, so an incidental match (a templated path, an example URL) would
# silently leak them into the container. CLI override vars are covered too, so a config
# that happens to reference one can't inject a host value ahead of the CLI's own default.
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


def config_env_references(config_path: Path | str | None) -> set[str]:
    """Env names used by trusted source config passthrough.

    This preserves the established ``api_key_env`` and braced interpolation
    behavior for operator-selected local source configs.
    """
    if config_path is None:
        return set()
    try:
        document = yaml.safe_load(Path(config_path).read_text())
    except (OSError, yaml.YAMLError):
        return set()

    names: set[str] = set()
    pending: list[object] = [document]
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            named = node.get(API_KEY_ENV_FIELD)
            if isinstance(named, str) and named.strip():
                names.add(named.strip())
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)
        elif isinstance(node, str):
            names.update(_ENV_REFERENCE.findall(node))
    return names - RESERVED_RUNTIME_ENV_NAMES


def required_config_env_references(config_path: Path | str | None) -> set[str]:
    """Return every environment name Router interpolation will consult.

    This parser deliberately mirrors Router's permissive tokenization, then
    ``validate_config_recipe_env_bindings`` applies the stricter remote-package
    name and reserved-name policy. That two-step design makes lowercase,
    malformed, fallback, and reserved references fail closed instead of
    disappearing from the preflight inventory.
    """

    if config_path is None:
        return set()
    try:
        document = yaml.safe_load(Path(config_path).read_text())
    except (OSError, yaml.YAMLError):
        return set()

    names: set[str] = set()
    pending: list[object] = [document]
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            named = node.get(API_KEY_ENV_FIELD)
            if isinstance(named, str) and named.strip():
                names.add(named.strip())
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)
        elif isinstance(node, str):
            names.update(_router_consulted_env_names(node))
    return names


def _router_consulted_env_names(value: str) -> set[str]:
    names: set[str] = set()
    index = 0
    while index < len(value):
        if value[index] != "$":
            index += 1
            continue
        if index + 1 < len(value) and value[index + 1] == "$":
            index += 2
            continue
        if index + 1 < len(value) and value[index + 1] == "{":
            close = value.find("}", index + 2)
            if close < 0:
                index += 1
                continue
            inner = value[index + 2 : close]
            separator = inner.find(":-")
            if separator > 0:
                name = inner[:separator]
            else:
                separator = inner.find("-")
                name = inner[:separator] if separator > 0 else inner
            if name:
                names.add(name)
            index = close + 1
            continue
        end = index + 1
        while end < len(value) and (
            value[end].isascii() and (value[end].isalnum() or value[end] == "_")
        ):
            end += 1
        if end > index + 1:
            names.add(value[index + 1 : end])
            index = end
            continue
        index += 1
    return names


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


def normalize_recipe_env_names(names: Iterable[str]) -> tuple[str, ...]:
    """Validate, deduplicate, and stabilize explicit Recipe env bindings."""

    try:
        return normalize_runtime_env_names(names)
    except ValueError as error:
        raise ValueError(
            "Invalid Recipe environment binding name. Use an uppercase, "
            "non-reserved environment variable name, without NAME=value."
        ) from error


def configure_recipe_env_bindings(
    env_vars: dict[str, str], requested_names: Iterable[str]
) -> tuple[str, ...]:
    """Bind only operator-authorized host variables for Recipes.

    The names-only allowlist is safe to expose to the Dashboard. Values are
    never accepted on the command line and never included in log messages.
    """

    names = list(requested_names)
    configured_allowlist = os.getenv(RECIPE_ENV_ALLOWLIST_ENV, "").strip()
    if configured_allowlist:
        raw_names = configured_allowlist.split(",")
        if any(not raw.strip() for raw in raw_names):
            raise ValueError(
                f"{RECIPE_ENV_ALLOWLIST_ENV} must be a comma-separated list of names"
            )
        names.extend(raw_names)

    normalized = normalize_recipe_env_names(names)
    missing = [name for name in normalized if not os.getenv(name)]
    if missing:
        raise ValueError(
            "Recipe environment binding has no non-empty host value: "
            + ", ".join(missing)
        )

    env_vars[RECIPE_ENV_ALLOWLIST_ENV] = ",".join(normalized)
    for name in normalized:
        env_vars[name] = os.environ[name]
        log.info("Binding Recipe environment variable: %s=***", name)
    return normalized


def validate_config_recipe_env_bindings(
    config_path: Path | str, allowed_names: Iterable[str]
) -> None:
    """Require every active config dependency to be explicitly authorized."""

    required = required_config_env_references(config_path)
    try:
        normalized_required = set(normalize_recipe_env_names(required))
    except ValueError as exc:
        raise ValueError(
            "Active config contains an invalid Recipe environment binding"
        ) from exc
    missing = sorted(normalized_required - set(allowed_names))
    if missing:
        flags = " ".join(f"--recipe-env {name}" for name in missing)
        raise ValueError(
            "Active config requires explicit Recipe environment bindings: "
            f"{', '.join(missing)}. Restart serve with {flags}."
        )


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


def _resolve_effective_config_document(
    config_path: Path,
    algorithm: str | None,
    setup_mode: bool,
    platform: str | None,
    *,
    package_activation: bool = False,
    materialize_local_runtime: bool = True,
) -> tuple[dict[str, object], bool]:
    with config_path.open() as handle:
        config = yaml.safe_load(handle) or {}

    if package_activation:
        _validate_package_kb_paths(config)
        kb_runtime_required, changed = False, False
    elif materialize_local_runtime:
        kb_runtime_required, changed = _sync_runtime_kb_store(config, config_path)
    else:
        kb_runtime_required, changed = False, False
    if not setup_mode and materialize_local_runtime:
        stack = resolve_runtime_stack()
        changed = inject_local_service_runtime_defaults(config, stack) or changed
        changed = inject_local_store_runtime_defaults(config, stack) or changed
        changed = apply_local_looper_endpoint(config, stack) or changed
    normalized_algorithm = _normalized_algorithm_override(algorithm, setup_mode)
    apply_gpu_defaults = _platform_requires_gpu_defaults(platform)
    if (
        not kb_runtime_required
        and not normalized_algorithm
        and not apply_gpu_defaults
        and not changed
    ):
        return config, False

    if normalized_algorithm:
        changed = _apply_algorithm_override(config, normalized_algorithm) or changed

    changed = apply_platform_gpu_defaults(config, platform) or changed
    return config, changed or kb_runtime_required


def build_effective_config_bytes(
    config_path: Path,
    algorithm: str | None,
    setup_mode: bool,
    platform: str | None,
    *,
    package_activation: bool = False,
) -> bytes:
    """Build the effective runtime config without touching active runtime state."""

    config, changed = _resolve_effective_config_document(
        config_path,
        algorithm,
        setup_mode,
        platform,
        package_activation=package_activation,
    )
    if not changed:
        return config_path.read_bytes()
    return yaml.dump(config, default_flow_style=False, sort_keys=False).encode("utf-8")


def build_effective_config_document(
    config_path: Path,
    algorithm: str | None,
    setup_mode: bool,
    platform: str | None,
    *,
    materialize_local_runtime: bool = True,
) -> dict[str, object]:
    """Build an effective config in memory without publishing runtime state."""

    config, _changed = _resolve_effective_config_document(
        config_path,
        algorithm,
        setup_mode,
        platform,
        materialize_local_runtime=materialize_local_runtime,
    )
    return config


def realize_runtime_config(
    source_path: Path,
    target_path: Path,
    *,
    algorithm: str | None = None,
    platform: str | None = None,
    package_activation: bool = False,
) -> Path:
    """Realize one raw package config at an explicit runtime-owned path.

    The raw source is never changed. Runtime defaults and current CLI
    algorithm/platform transforms are applied before a strict atomic replace of
    ``target_path``.
    """

    if algorithm is None:
        algorithm = os.getenv(RUNTIME_ALGORITHM_OVERRIDE_ENV, "").strip() or None
    if platform is None:
        platform = (
            os.getenv("VLLM_SR_PLATFORM", "").strip()
            or os.getenv("DASHBOARD_PLATFORM", "").strip()
            or None
        )

    effective = build_effective_config_bytes(
        source_path,
        algorithm=algorithm,
        setup_mode=is_setup_mode_config(source_path),
        platform=platform,
        package_activation=package_activation,
    )
    return write_runtime_config_bytes(target_path, effective)


def resolve_effective_config_path(
    config_path: Path,
    algorithm: str | None,
    setup_mode: bool,
    platform: str | None,
    *,
    materialize_local_runtime: bool = True,
) -> Path:
    """Apply target-appropriate runtime config transformations."""

    config, changed = _resolve_effective_config_document(
        config_path,
        algorithm,
        setup_mode,
        platform,
        materialize_local_runtime=materialize_local_runtime,
    )
    return _finalize_runtime_config_write(
        config_path,
        config,
        changed,
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
    """Expose source and runtime-owned config paths inside local containers."""
    container_runtime_path = _container_runtime_config_path(source_config_path)
    # This env is the Router/Dashboard persistence source, not a provenance
    # hint. Keep mutations on the runtime-owned active file; the original
    # source is separately mounted read-only for inspection.
    env_vars[SOURCE_CONFIG_PATH_ENV] = container_runtime_path
    env_vars[RUNTIME_CONFIG_PATH_ENV] = container_runtime_path
