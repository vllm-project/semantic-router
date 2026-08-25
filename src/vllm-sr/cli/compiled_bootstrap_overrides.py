"""Target-specific overrides for an immutable compiled bootstrap."""

from __future__ import annotations

import os

from cli.consts import PLATFORM_AMD, PLATFORM_NVIDIA
from cli.utils import get_logger

log = get_logger(__name__)

GPU_OVERRIDE_PREVIEW_LIMIT = 8
TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}
FALSEY_ENV_VALUES = {"0", "false", "no", "off"}
# Platforms that flip router internal-model `use_cpu` flags to false by default
# so local signal models run on the platform GPU.
GPU_DEFAULT_PLATFORMS = (PLATFORM_AMD, PLATFORM_NVIDIA)
GPU_USE_CPU_PATHS: tuple[tuple[str, ...], ...] = (
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


def _set_use_cpu_false(
    config_node: object, path: str, changed_paths: list[str]
) -> None:
    if isinstance(config_node, dict):
        for key, value in config_node.items():
            current_path = f"{path}.{key}" if path else key
            if key == "use_cpu" and value is True:
                config_node[key] = False
                changed_paths.append(current_path)
            else:
                _set_use_cpu_false(value, current_path, changed_paths)
        return

    if isinstance(config_node, list):
        for index, item in enumerate(config_node):
            _set_use_cpu_false(item, f"{path}[{index}]", changed_paths)


def _ensure_mapping_path(
    root: dict[str, object], path: tuple[str, ...], platform: str
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
                "Platform %s detected: skipping GPU default injection for %s because %s is not a mapping",
                platform,
                ".".join(path),
                ".".join(current_path),
            )
            return None
        current = next_node

    return current


def _inject_missing_gpu_defaults(
    merged_config: dict[str, object], changed_paths: list[str], platform: str
) -> None:
    for use_cpu_path in GPU_USE_CPU_PATHS:
        parent = _ensure_mapping_path(merged_config, use_cpu_path[:-1], platform)
        if parent is None:
            continue

        leaf_key = use_cpu_path[-1]
        existing_value = parent.get(leaf_key)
        if existing_value is False:
            continue

        parent[leaf_key] = False
        changed_paths.append(".".join(use_cpu_path))


def _resolve_gpu_default_platform(platform: str | None) -> str:
    """Return the normalized GPU platform requiring use_cpu defaults, else ""."""
    normalized_platform = _normalize_platform(
        platform or os.getenv("VLLM_SR_PLATFORM") or os.getenv("DASHBOARD_PLATFORM")
    )
    if normalized_platform not in GPU_DEFAULT_PLATFORMS:
        return ""

    env_prefix = f"VLLM_SR_{normalized_platform.upper()}"
    force_gpu = os.getenv(f"{env_prefix}_FORCE_GPU", "").strip().lower()
    if force_gpu in TRUTHY_ENV_VALUES:
        return normalized_platform
    if force_gpu in FALSEY_ENV_VALUES:
        log.info(
            "Platform %s detected: keeping router internal model use_cpu settings; "
            "%s_FORCE_GPU is explicitly disabled",
            normalized_platform,
            env_prefix,
        )
        return ""

    preserve_cpu = os.getenv(f"{env_prefix}_PRESERVE_CPU", "").strip().lower()
    if preserve_cpu in TRUTHY_ENV_VALUES:
        log.info(
            "Platform %s detected: keeping router internal model use_cpu settings; "
            "%s_PRESERVE_CPU is enabled",
            normalized_platform,
            env_prefix,
        )
        return ""
    return normalized_platform


def _platform_requires_gpu_defaults(platform: str | None) -> bool:
    return bool(_resolve_gpu_default_platform(platform))


def apply_platform_gpu_defaults(
    merged_config: dict[str, object], platform: str | None
) -> bool:
    """
    Apply platform-specific GPU defaults.

    For AMD (ROCm) and NVIDIA (CUDA) platforms, rewrite router internal model
    `use_cpu` flags to false by default so `--platform amd` / `--platform nvidia`
    run local signal models on the platform GPU. Set
    VLLM_SR_<PLATFORM>_PRESERVE_CPU=1/true/yes/on (e.g. VLLM_SR_NVIDIA_PRESERVE_CPU)
    or VLLM_SR_<PLATFORM>_FORCE_GPU=0/false/no/off
    to preserve CPU settings when the router does not have dedicated GPU headroom.
    """
    resolved_platform = _resolve_gpu_default_platform(platform)
    if not resolved_platform:
        return False

    changed_paths: list[str] = []
    _set_use_cpu_false(merged_config, "", changed_paths)
    _inject_missing_gpu_defaults(merged_config, changed_paths, resolved_platform)
    if not changed_paths:
        log.info(
            "Platform %s detected: no use_cpu flags found to override",
            resolved_platform,
        )
        return False

    preview = ", ".join(changed_paths[:GPU_OVERRIDE_PREVIEW_LIMIT])
    if len(changed_paths) > GPU_OVERRIDE_PREVIEW_LIMIT:
        preview = f"{preview}, ..."
    log.info(
        "Platform %s detected: set %d use_cpu flag(s) to false for GPU default (%s)",
        resolved_platform,
        len(changed_paths),
        preview,
    )
    return True
