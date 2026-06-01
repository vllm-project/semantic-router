"""Runtime config mutation helpers for CLI overrides."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from cli.commands.runtime_paths import _write_runtime_config
from cli.utils import get_logger

log = get_logger(__name__)

ALGORITHM_TYPES = [
    "static",
    "elo",
    "router_dc",
    "automix",
    "hybrid",
    "rl_driven",
    "gmtrouter",
    "knn",
    "kmeans",
    "svm",
    "mlp",
    "multi_factor",
    "session_aware",
]

ALGORITHM_HINTS = {
    "elo": "  Tip: Submit feedback via POST /api/v1/feedback",
    "router_dc": "  Tip: Ensure models have 'description' fields",
    "automix": "  Tip: Configure model 'pricing' for cost optimization",
    "hybrid": "  Tip: Configure weights in decision.algorithm.hybrid",
    "rl_driven": "  Tip: Configure persistence in decision.algorithm.rl_driven.storage_path",
    "knn": "  Tip: Configure global.router.model_selection.ml.knn for trained KNN routing",
    "kmeans": "  Tip: Configure global.router.model_selection.ml.kmeans for cluster routing",
    "svm": "  Tip: Configure global.router.model_selection.ml.svm for trained SVM routing",
    "mlp": "  Tip: Configure global.router.model_selection.ml.mlp for trained MLP routing",
    "multi_factor": "  Tip: Configure decision.algorithm.multi_factor for SLO-aware scoring",
    "session_aware": "  Tip: Preserve agentic continuity with decision.algorithm.session_aware",
    "gmtrouter": "  Tip: Learns user preferences via graph neural network",
}

ALGORITHM_CONFIG_BLOCKS = (
    "confidence",
    "ratings",
    "remom",
    "elo",
    "router_dc",
    "automix",
    "hybrid",
    "rl_driven",
    "gmtrouter",
    "latency_aware",
    "multi_factor",
    "session_aware",
)

EXPECTED_CONFIG_BLOCK_BY_ALGORITHM = {
    "elo": "elo",
    "router_dc": "router_dc",
    "automix": "automix",
    "hybrid": "hybrid",
    "rl_driven": "rl_driven",
    "gmtrouter": "gmtrouter",
    "multi_factor": "multi_factor",
    "session_aware": "session_aware",
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
    if value is None:
        return ""
    return str(value).strip().lower()


def _set_use_cpu_false_for_amd(
    config_node: object, path: str, changed_paths: list[str]
) -> None:
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


def log_algorithm_hint(algorithm: str) -> None:
    hint = ALGORITHM_HINTS.get(algorithm)
    if hint:
        log.info(hint)


def _replace_algorithm_config(
    algorithm_config: dict[str, object],
    normalized_algorithm: str,
) -> None:
    expected_block = EXPECTED_CONFIG_BLOCK_BY_ALGORITHM.get(normalized_algorithm)
    for block in ALGORITHM_CONFIG_BLOCKS:
        if block != expected_block:
            algorithm_config.pop(block, None)
    algorithm_config["type"] = normalized_algorithm


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
        if not isinstance(decision["algorithm"], dict):
            decision["algorithm"] = {}
        _replace_algorithm_config(decision["algorithm"], normalized_algorithm)
        log.info(
            "  Injected algorithm.type=%s into decision '%s'",
            normalized_algorithm,
            decision.get("name", "unnamed"),
        )
    return True


def inject_algorithm_into_config(config_path: Path, algorithm: str) -> Path:
    """Create a temporary config with algorithm.type injected into all decisions."""
    with config_path.open() as handle:
        config = yaml.safe_load(handle) or {}

    normalized_algorithm = algorithm.lower()
    for decision in _routing_decisions(config):
        if "algorithm" not in decision:
            decision["algorithm"] = {}
        if not isinstance(decision["algorithm"], dict):
            decision["algorithm"] = {}
        _replace_algorithm_config(decision["algorithm"], normalized_algorithm)
        log.info(
            f"  Injected algorithm.type={normalized_algorithm} into decision '{decision.get('name', 'unnamed')}'"
        )

    runtime_config_path = _write_runtime_config(config_path, config)
    log.info(f"Created config with algorithm: {runtime_config_path}")
    return runtime_config_path
