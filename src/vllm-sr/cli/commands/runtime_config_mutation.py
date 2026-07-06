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
    "router_dc",
    "automix",
    "hybrid",
    "workflows",
    "latency_aware",
    "knn",
    "kmeans",
    "svm",
    "mlp",
    "multi_factor",
]

ALGORITHM_HINTS = {
    "router_dc": "  Tip: Ensure models have 'description' fields",
    "automix": "  Tip: Configure model 'pricing' for cost optimization",
    "hybrid": "  Tip: Configure weights in decision.algorithm.hybrid",
    "workflows": "  Tip: Configure decision.algorithm.workflows; dynamic mode requires planner.model",
    "latency_aware": "  Tip: Configure decision.algorithm.latency_aware with TPOT or TTFT percentiles",
    "knn": "  Tip: Configure global.router.model_selection.ml.knn for trained KNN routing",
    "kmeans": "  Tip: Configure global.router.model_selection.ml.kmeans for cluster routing",
    "svm": "  Tip: Configure global.router.model_selection.ml.svm for trained SVM routing",
    "mlp": "  Tip: Configure global.router.model_selection.ml.mlp for trained MLP routing",
    "multi_factor": "  Tip: Configure decision.algorithm.multi_factor for SLO-aware scoring",
}

ALGORITHM_CONFIG_BLOCKS = (
    "confidence",
    "ratings",
    "remom",
    "fusion",
    "workflows",
    "router_dc",
    "automix",
    "hybrid",
    "latency_aware",
    "multi_factor",
)

RETIRED_ALGORITHM_CONFIG_BLOCKS = (
    "session_aware",
    "elo",
    "rl_driven",
    "gmtrouter",
    "bandit",
    "personalization",
)

EXPECTED_CONFIG_BLOCK_BY_ALGORITHM = {
    "router_dc": "router_dc",
    "automix": "automix",
    "hybrid": "hybrid",
    "workflows": "workflows",
    "latency_aware": "latency_aware",
    "multi_factor": "multi_factor",
}

DEFAULT_CONFIG_BLOCK_BY_ALGORITHM: dict[str, dict[str, object]] = {
    "latency_aware": {
        "tpot_percentile": 90,
        "ttft_percentile": 95,
    },
    "workflows": {
        "template": "micro_agent",
    },
}

AMD_OVERRIDE_PREVIEW_LIMIT = 8
TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}
FALSEY_ENV_VALUES = {"0", "false", "no", "off"}
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

    force_gpu = os.getenv("VLLM_SR_AMD_FORCE_GPU", "").strip().lower()
    if force_gpu in TRUTHY_ENV_VALUES:
        return True
    if force_gpu in FALSEY_ENV_VALUES:
        log.info(
            "Platform amd detected: keeping router internal model use_cpu settings; "
            "VLLM_SR_AMD_FORCE_GPU is explicitly disabled"
        )
        return False

    preserve_cpu = os.getenv("VLLM_SR_AMD_PRESERVE_CPU", "").strip().lower()
    if preserve_cpu in TRUTHY_ENV_VALUES:
        log.info(
            "Platform amd detected: keeping router internal model use_cpu settings; "
            "VLLM_SR_AMD_PRESERVE_CPU is enabled"
        )
        return False
    return True


def apply_platform_gpu_defaults(
    merged_config: dict[str, object], platform: str | None
) -> bool:
    """
    Apply platform-specific GPU defaults.

    For AMD platform, rewrite router internal model `use_cpu` flags to false by
    default so `--platform amd` uses ROCm for local signal models. Set
    VLLM_SR_AMD_PRESERVE_CPU=1/true/yes/on or VLLM_SR_AMD_FORCE_GPU=0/false/no/off
    to preserve CPU settings when the router does not have dedicated GPU headroom.
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
    decision_config: dict[str, object] | None = None,
) -> None:
    expected_block = EXPECTED_CONFIG_BLOCK_BY_ALGORITHM.get(normalized_algorithm)
    for block in (*ALGORITHM_CONFIG_BLOCKS, *RETIRED_ALGORITHM_CONFIG_BLOCKS):
        if block != expected_block:
            algorithm_config.pop(block, None)
    algorithm_config["type"] = normalized_algorithm
    if (
        expected_block
        and expected_block not in algorithm_config
        and normalized_algorithm in DEFAULT_CONFIG_BLOCK_BY_ALGORITHM
    ):
        default_block = _default_algorithm_config_block(
            normalized_algorithm, decision_config
        )
        if default_block:
            algorithm_config[expected_block] = default_block


def _default_algorithm_config_block(
    normalized_algorithm: str,
    decision_config: dict[str, object] | None,
) -> dict[str, object]:
    block = dict(DEFAULT_CONFIG_BLOCK_BY_ALGORITHM[normalized_algorithm])
    if normalized_algorithm != "workflows":
        return block

    models = []
    if decision_config is not None:
        model_refs = decision_config.get("modelRefs", [])
        if isinstance(model_refs, list):
            for model_ref in model_refs:
                if isinstance(model_ref, dict) and isinstance(
                    model_ref.get("model"), str
                ):
                    models.append(model_ref["model"])

    if models:
        block["mode"] = "static"
        block["roles"] = [{"name": "worker", "models": [models[0]]}]
    else:
        block = {}
    return block


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
        _replace_algorithm_config(decision["algorithm"], normalized_algorithm, decision)
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
        _replace_algorithm_config(decision["algorithm"], normalized_algorithm, decision)
        log.info(
            f"  Injected algorithm.type={normalized_algorithm} into decision '{decision.get('name', 'unnamed')}'"
        )

    runtime_config_path = _write_runtime_config(config_path, config)
    log.info(f"Created config with algorithm: {runtime_config_path}")
    return runtime_config_path
