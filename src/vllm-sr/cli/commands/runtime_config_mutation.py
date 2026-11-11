"""Runtime config mutation helpers for CLI overrides."""

from __future__ import annotations

from pathlib import Path

import yaml

from cli.commands.runtime_paths import _write_runtime_config
from cli.config_recipe_scopes import recipe_scopes
from cli.config_schema import routing_surface_catalog
from cli.utils import get_logger

log = get_logger(__name__)

_ALGORITHM_SURFACES = tuple(routing_surface_catalog()["algorithms"])
_CANONICAL_ALGORITHM_TYPES = frozenset(
    surface["type"] for surface in _ALGORITHM_SURFACES
)

# `serve --algorithm` is a mutation convenience, not the canonical algorithm
# inventory. Keep this explicit opt-in policy because some algorithms require
# user-authored payloads that the CLI must not invent. The full Router surface
# remains discoverable through `vllm-sr config schema`.
ALGORITHM_OVERRIDE_TYPES = (
    "static",
    "random",
    "multi_factor",
    "workflows",
)

if not set(ALGORITHM_OVERRIDE_TYPES).issubset(_CANONICAL_ALGORITHM_TYPES):
    raise RuntimeError("CLI algorithm override policy is stale against Router schema")

ALGORITHM_HINTS = {
    "workflows": "  Tip: Configure decision.algorithm.workflows; dynamic mode requires planner.model",
    "multi_factor": "  Tip: Configure decision.algorithm.multi_factor for SLO-aware scoring",
    "random": "  Tip: No configuration needed",
}

ALGORITHM_CONFIG_BLOCKS = tuple(
    surface["config_field"]
    for surface in _ALGORITHM_SURFACES
    if surface.get("config_field")
)

EXECUTOR_ALGORITHM_TYPES = frozenset(
    surface["type"]
    for surface in _ALGORITHM_SURFACES
    if surface.get("kind") == "executor"
)

# The settings every executor reads; a selector carries neither.
EXECUTOR_SETTINGS = ("budget", "on_error")

# Blocks of retired and removed algorithm types, and the native cascade's
# settings, which an override replaces with the overriding type's own.
RETIRED_ALGORITHM_CONFIG_BLOCKS = (
    "session_aware",
    "elo",
    "rl_driven",
    "gmtrouter",
    "bandit",
    "personalization",
    "confidence",
    "prompt",
    "router_dc",
    "automix",
    "hybrid",
    "latency_aware",
    "quality",
    "stages",
)

EXPECTED_CONFIG_BLOCK_BY_ALGORITHM = {
    surface["type"]: surface["config_field"]
    for surface in _ALGORITHM_SURFACES
    if surface.get("config_field")
}

DEFAULT_CONFIG_BLOCK_BY_ALGORITHM: dict[str, dict[str, object]] = {
    "workflows": {
        "template": "micro_agent",
    },
    "multi_factor": {},
}


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
    scopes = recipe_scopes(config)
    if not scopes:
        log.warning("No recipe found; skipping --algorithm override")
        return []

    found: list[dict[str, object]] = []
    for path, recipe in scopes:
        decisions = recipe.get("decisions", [])
        if not isinstance(decisions, list):
            log.warning(
                f"{path}.decisions is not a list; skipping --algorithm override"
            )
            continue
        found.extend(decision for decision in decisions if isinstance(decision, dict))
    return found


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
    if normalized_algorithm not in EXECUTOR_ALGORITHM_TYPES:
        for setting in EXECUTOR_SETTINGS:
            algorithm_config.pop(setting, None)
    algorithm_config["type"] = normalized_algorithm
    if (
        expected_block
        and expected_block not in algorithm_config
        and normalized_algorithm in DEFAULT_CONFIG_BLOCK_BY_ALGORITHM
    ):
        default_block = _default_algorithm_config_block(
            normalized_algorithm, decision_config
        )
        if default_block is not None:
            algorithm_config[expected_block] = default_block


def _default_algorithm_config_block(
    normalized_algorithm: str,
    decision_config: dict[str, object] | None,
) -> dict[str, object] | None:
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
        return None
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
    with config_path.open(encoding="utf-8") as handle:
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
