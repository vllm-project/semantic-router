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
    ("ANTHROPIC_API_KEY", True),
    ("OPENAI_API_KEY", True),
    ("OPENCLAW_BASE_IMAGE", False),
)

ALGORITHM_TYPES = [
    "static",
    "elo",
    "router_dc",
    "automix",
    "hybrid",
    "thompson",
    "gmtrouter",
    "router_r1",
]

ALGORITHM_HINTS = {
    "elo": "  Tip: Submit feedback via POST /api/v1/feedback",
    "router_dc": "  Tip: Ensure models have 'description' fields",
    "automix": "  Tip: Configure model 'pricing' for cost optimization",
    "hybrid": "  Tip: Configure weights in decision.algorithm.hybrid",
    "thompson": "  Tip: Balances exploration vs exploitation automatically",
    "gmtrouter": "  Tip: Learns user preferences via graph neural network",
    "router_r1": "  Tip: Requires Router-R1 server (see training docs)",
}

AMD_OVERRIDE_PREVIEW_LIMIT = 8
AMD_GPU_USE_CPU_PATHS: tuple[tuple[str, ...], ...] = (
    ("global", "model_catalog", "embeddings", "semantic", "use_cpu"),
    ("global", "model_catalog", "modules", "prompt_guard", "use_cpu"),
    ("global", "model_catalog", "modules", "classifier", "domain", "use_cpu"),
    ("global", "model_catalog", "modules", "classifier", "pii", "use_cpu"),
    (
        "global",
        "model_catalog",
        "modules",
        "hallucination_mitigation",
        "fact_check",
        "use_cpu",
    ),
    (
        "global",
        "model_catalog",
        "modules",
        "hallucination_mitigation",
        "detector",
        "use_cpu",
    ),
    (
        "global",
        "model_catalog",
        "modules",
        "hallucination_mitigation",
        "explainer",
        "use_cpu",
    ),
    ("global", "model_catalog", "modules", "feedback_detector", "use_cpu"),
    (
        "global",
        "model_catalog",
        "modules",
        "modality_detector",
        "classifier",
        "use_cpu",
    ),
)


def _normalize_platform(value: str | None) -> str:
    """Normalize platform value for comparisons."""
    if value is None:
        return ""
    return str(value).strip().lower()


def _runtime_config_output_dir(config_path: Path) -> Path:
    runtime_dir = config_path.parent / ".vllm-sr"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir


def _runtime_config_output_path(source_config_path: Path) -> Path:
    stack_name = os.getenv("VLLM_SR_STACK_NAME", "").strip()
    filename = "runtime-config.yaml"
    if stack_name:
        filename = f"runtime-config.{stack_name}.yaml"
    return _runtime_config_output_dir(source_config_path) / filename


def _container_runtime_config_path(source_config_path: Path) -> str:
    return f"/app/.vllm-sr/{_runtime_config_output_path(source_config_path).name}"


def _container_source_config_path() -> str:
    return "/app/config.yaml"


def _write_runtime_config(source_config_path: Path, config: dict[str, object]) -> Path:
    runtime_config_path = _runtime_config_output_path(source_config_path)
    with runtime_config_path.open("w") as handle:
        yaml.dump(config, handle, default_flow_style=False, sort_keys=False)
    return runtime_config_path


def _set_use_cpu_false_for_amd(
    config_node: object, path: str, changed_paths: list[str]
) -> None:
    """Recursively set use_cpu=true to false for AMD GPU defaults."""
    if isinstance(config_node, dict):
        for key, value in config_node.items():
            current_path = f"{path}.{key}" if path else key
            if key == "use_cpu" and value is True:
                config_node[key] = False
                changed_paths.append(current_path)
            else:
                _set_use_cpu_false_for_amd(value, current_path, changed_paths)
        return

    if isinstance(config_node, list):
        for index, item in enumerate(config_node):
            _set_use_cpu_false_for_amd(item, f"{path}[{index}]", changed_paths)


def _ensure_mapping_path(
    root: dict[str, object], path: tuple[str, ...]
) -> dict[str, object] | None:
    current: dict[str, object] = root
    current_path: list[str] = []

    for key in path:
        current_path.append(key)
        next_node = current.get(key)
        if next_node is None:
            next_mapping: dict[str, object] = {}
            current[key] = next_mapping
            current = next_mapping
            continue
        if not isinstance(next_node, dict):
            log.warning(
                "Platform amd detected: skipping GPU default injection for %s because %s is not a mapping",
                ".".join(path),
                ".".join(current_path),
            )
            return None
        current = next_node

    return current


def _inject_missing_amd_gpu_defaults(
    merged_config: dict[str, object], changed_paths: list[str]
) -> None:
    for use_cpu_path in AMD_GPU_USE_CPU_PATHS:
        parent = _ensure_mapping_path(merged_config, use_cpu_path[:-1])
        if parent is None:
            continue

        leaf_key = use_cpu_path[-1]
        existing_value = parent.get(leaf_key)
        if existing_value is False:
            continue

        parent[leaf_key] = False
        changed_paths.append(".".join(use_cpu_path))


def _platform_requires_gpu_defaults(platform: str | None) -> bool:
    normalized_platform = _normalize_platform(
        platform or os.getenv("VLLM_SR_PLATFORM") or os.getenv("DASHBOARD_PLATFORM")
    )
    if normalized_platform != "amd":
        return False

    force_gpu = os.getenv("VLLM_SR_AMD_FORCE_GPU", "1").strip().lower()
    if force_gpu in {"0", "false", "no", "off"}:
        log.info(
            "Platform amd detected but GPU default override disabled by VLLM_SR_AMD_FORCE_GPU"
        )
        return False
    return True


def apply_platform_gpu_defaults(
    merged_config: dict[str, object], platform: str | None
) -> bool:
    """
    Apply platform-specific GPU defaults.

    For AMD platform, default all `use_cpu` flags to false so inference prefers GPU.
    Can be disabled by setting VLLM_SR_AMD_FORCE_GPU=0/false/no/off.
    """
    if not _platform_requires_gpu_defaults(platform):
        return False

    changed_paths: list[str] = []
    _set_use_cpu_false_for_amd(merged_config, "", changed_paths)
    _inject_missing_amd_gpu_defaults(merged_config, changed_paths)
    if not changed_paths:
        log.info("Platform amd detected: no use_cpu flags found to override")
        return False

    preview = ", ".join(changed_paths[:AMD_OVERRIDE_PREVIEW_LIMIT])
    if len(changed_paths) > AMD_OVERRIDE_PREVIEW_LIMIT:
        preview = f"{preview}, ..."
    log.info(
        "Platform amd detected: set %d use_cpu flag(s) to false for GPU default (%s)",
        len(changed_paths),
        preview,
    )
    return True


def inject_algorithm_into_config(config_path: Path, algorithm: str) -> Path:
    """Create a temporary config with algorithm.type injected into all decisions."""
    with config_path.open() as handle:
        config = yaml.safe_load(handle) or {}

    routing = config.get("routing")
    if not isinstance(routing, dict):
        log.warning("No routing section found; skipping --algorithm override")
        decisions = []
    else:
        decisions = routing.get("decisions", [])
        if not isinstance(decisions, list):
            log.warning(
                "routing.decisions is not a list; skipping --algorithm override"
            )
            decisions = []

    for decision in decisions:
        if "algorithm" not in decision:
            decision["algorithm"] = {}
        decision["algorithm"]["type"] = algorithm
        log.info(
            f"  Injected algorithm.type={algorithm} into decision '{decision.get('name', 'unnamed')}'"
        )

    runtime_config_path = _write_runtime_config(config_path, config)
    log.info(f"Created config with algorithm: {runtime_config_path}")
    return runtime_config_path


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


def resolve_effective_config_path(
    config_path: Path, algorithm: str | None, setup_mode: bool, platform: str | None
) -> Path:
    """Apply CLI algorithm and platform override translation when appropriate."""
    apply_algorithm = bool(algorithm)
    apply_gpu_defaults = _platform_requires_gpu_defaults(platform)

    if not apply_algorithm and not apply_gpu_defaults:
        return config_path
    if apply_algorithm and setup_mode:
        log.warning(
            f"--algorithm={algorithm} ignored in setup mode until a runnable config is activated"
        )
        apply_algorithm = False
        if not apply_gpu_defaults:
            return config_path

    with config_path.open() as handle:
        config = yaml.safe_load(handle) or {}

    changed = False
    if apply_algorithm:
        normalized = algorithm.lower()
        log.info(f"Model selection algorithm: {normalized}")
        routing = config.get("routing")
        if not isinstance(routing, dict):
            log.warning("No routing section found; skipping --algorithm override")
            decisions = []
        else:
            decisions = routing.get("decisions", [])
            if not isinstance(decisions, list):
                log.warning(
                    "routing.decisions is not a list; skipping --algorithm override"
                )
                decisions = []

        for decision in decisions:
            if "algorithm" not in decision:
                decision["algorithm"] = {}
            decision["algorithm"]["type"] = normalized
            log.info(
                "  Injected algorithm.type=%s into decision '%s'",
                normalized,
                decision.get("name", "unnamed"),
            )
        if decisions:
            changed = True
        log_algorithm_hint(normalized)

    if apply_platform_gpu_defaults(config, platform):
        changed = True

    if not changed:
        return config_path

    effective_config = _write_runtime_config(config_path, config)
    log.info(f"Created effective runtime config: {effective_config}")
    return effective_config


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


def log_algorithm_hint(algorithm: str) -> None:
    """Emit a targeted hint for the selected routing algorithm."""
    hint = ALGORITHM_HINTS.get(algorithm)
    if hint:
        log.info(hint)
