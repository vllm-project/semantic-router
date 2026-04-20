"""Support helpers for runtime-oriented CLI commands."""

from __future__ import annotations

import os
import posixpath
import shutil
from pathlib import Path

import yaml

from cli.bootstrap import (
    DASHBOARD_SETUP_MODE_ENV,
    SETUP_MODE_ENV,
    BootstrapResult,
    is_setup_mode_config,
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
    ("ANTHROPIC_API_KEY", True),
    ("OPENAI_API_KEY", True),
    ("OPENCLAW_BASE_IMAGE", False),
    ("SR_LOG_LEVEL", False),
    ("SR_LOG_ENCODING", False),
    ("SR_LOG_DEVELOPMENT", False),
    ("SR_LOG_ADD_CALLER", False),
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
KB_RUNTIME_ROOT = "knowledge_bases"
KB_BOOTSTRAP_STATE_FILE = ".bootstrap-state.yaml"
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


def _runtime_kb_state_dir(config_path: Path) -> Path:
    kb_dir = _runtime_config_output_dir(config_path) / KB_RUNTIME_ROOT
    kb_dir.mkdir(parents=True, exist_ok=True)
    return kb_dir


def _clean_kb_source_path(value: str) -> str:
    cleaned = posixpath.normpath(str(value or "").strip().replace("\\", "/"))
    if cleaned == ".":
        return ""
    return cleaned.rstrip("/")


def _runtime_kb_bootstrap_state_path(config_path: Path) -> Path:
    return _runtime_kb_state_dir(config_path) / KB_BOOTSTRAP_STATE_FILE


def _load_runtime_kb_bootstrap_state(config_path: Path) -> set[str]:
    state_path = _runtime_kb_bootstrap_state_path(config_path)
    if not state_path.exists():
        return set()

    data = yaml.safe_load(state_path.read_text(encoding="utf-8")) or {}
    processed = data.get("processed")
    if not isinstance(processed, list):
        return set()

    return {
        cleaned
        for item in processed
        if isinstance(item, str)
        if (cleaned := _clean_kb_source_path(item))
    }


def _write_runtime_kb_bootstrap_state(
    config_path: Path, processed_runtime_paths: set[str]
) -> None:
    state_path = _runtime_kb_bootstrap_state_path(config_path)
    state_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "processed": sorted(processed_runtime_paths),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _candidate_kb_source_roots(config_path: Path, source_path: str) -> list[Path]:
    source_root = Path(source_path)
    if source_root.is_absolute():
        return [source_root]

    candidates: list[Path] = [config_path.parent / source_root]
    for ancestor in Path(__file__).resolve().parents:
        config_root = ancestor / "config"
        if config_root.is_dir():
            candidates.append(config_root / source_root)
    return candidates


def _resolve_kb_source_root(config_path: Path, source_path: str) -> Path | None:
    for candidate in _candidate_kb_source_roots(config_path, source_path):
        if candidate.exists():
            return candidate
    return None


def _configured_knowledge_bases(config: dict[str, object]) -> list[dict[str, object]]:
    global_config = config.get("global")
    if not isinstance(global_config, dict):
        return []

    model_catalog = global_config.get("model_catalog")
    if not isinstance(model_catalog, dict):
        return []

    kb_configs = model_catalog.get("kbs")
    if not isinstance(kb_configs, list):
        return []

    return [kb_config for kb_config in kb_configs if isinstance(kb_config, dict)]


def _kb_source_spec(
    kb_config: dict[str, object],
) -> tuple[str, dict[str, object], str] | None:
    source = kb_config.get("source")
    if not isinstance(source, dict):
        return None

    source_path = str(source.get("path") or "").strip()
    kb_name = str(kb_config.get("name") or "").strip()
    if not source_path or not kb_name:
        return None

    return kb_name, source, source_path


def _runtime_kb_relative_path(source_path: str, kb_name: str) -> str:
    cleaned = _clean_kb_source_path(source_path)
    runtime_root = _clean_kb_source_path(KB_RUNTIME_ROOT)

    if cleaned == runtime_root:
        return f"{runtime_root}/{kb_name}"
    if cleaned.startswith(f"{runtime_root}/"):
        return cleaned

    leaf_name = Path(cleaned).name
    if leaf_name in {"", ".", runtime_root}:
        leaf_name = kb_name
    return f"{runtime_root}/{leaf_name}"


def _runtime_kb_target_root(config_path: Path, runtime_relative_path: str) -> Path:
    return _runtime_config_output_dir(config_path) / Path(runtime_relative_path)


def _sync_runtime_kb_store(
    config: dict[str, object],
    config_path: Path,
) -> tuple[bool, bool]:
    kb_configs = _configured_knowledge_bases(config)
    if not kb_configs:
        return False, False

    changed = False
    state_changed = False
    bootstrapped = _load_runtime_kb_bootstrap_state(config_path)

    for kb_config in kb_configs:
        kb_source_spec = _kb_source_spec(kb_config)
        if kb_source_spec is None:
            continue

        kb_name, source, source_path = kb_source_spec
        runtime_relative_path = _runtime_kb_relative_path(source_path, kb_name)
        runtime_target = _runtime_kb_target_root(config_path, runtime_relative_path)

        normalized_source_path = f"{runtime_relative_path}/"
        if source.get("path") != normalized_source_path:
            source["path"] = normalized_source_path
            changed = True

        if runtime_target.exists():
            if runtime_relative_path not in bootstrapped:
                bootstrapped.add(runtime_relative_path)
                state_changed = True
            continue

        if runtime_relative_path in bootstrapped:
            continue

        resolved_source_root = _resolve_kb_source_root(config_path, source_path)
        if resolved_source_root is None:
            log.warning(
                "Runtime KB bootstrap: could not resolve KB source.path=%s for %s",
                source_path,
                kb_name,
            )
            continue

        runtime_target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copytree(resolved_source_root, runtime_target)
            bootstrapped.add(runtime_relative_path)
            state_changed = True
        except OSError as exc:
            log.warning(
                "Runtime KB bootstrap: failed to import %s from %s to %s: %s",
                kb_name,
                resolved_source_root,
                runtime_target,
                exc,
            )

    if state_changed:
        _write_runtime_kb_bootstrap_state(config_path, bootstrapped)

    return True, changed or state_changed


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


def _normalized_algorithm_override(
    algorithm: str | None, setup_mode: bool
) -> str | None:
    if not algorithm:
        return None

    if setup_mode:
        log.warning(
            f"--algorithm={algorithm} ignored in setup mode until a runnable config is activated"
        )
        return None

    return algorithm.lower()


def _routing_decisions(config: dict[str, object]) -> list[dict[str, object]]:
    routing = config.get("routing")
    if not isinstance(routing, dict):
        log.warning("No routing section found; skipping --algorithm override")
        return []

    decisions = routing.get("decisions", [])
    if not isinstance(decisions, list):
        log.warning("routing.decisions is not a list; skipping --algorithm override")
        return []

    return [decision for decision in decisions if isinstance(decision, dict)]


def _apply_algorithm_override(
    config: dict[str, object], normalized_algorithm: str
) -> bool:
    log.info(f"Model selection algorithm: {normalized_algorithm}")
    log_algorithm_hint(normalized_algorithm)
    decisions = _routing_decisions(config)
    if not decisions:
        return False

    for decision in decisions:
        if "algorithm" not in decision:
            decision["algorithm"] = {}
        decision["algorithm"]["type"] = normalized_algorithm
        log.info(
            "  Injected algorithm.type=%s into decision '%s'",
            normalized_algorithm,
            decision.get("name", "unnamed"),
        )
    return True


def _finalize_runtime_config_write(
    config_path: Path, config: dict[str, object], changed: bool
) -> Path:
    if not changed:
        return config_path

    effective_config = _write_runtime_config(config_path, config)
    log.info(f"Created effective runtime config: {effective_config}")
    return effective_config


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


def log_algorithm_hint(algorithm: str) -> None:
    """Emit a targeted hint for the selected routing algorithm."""
    hint = ALGORITHM_HINTS.get(algorithm)
    if hint:
        log.info(hint)
