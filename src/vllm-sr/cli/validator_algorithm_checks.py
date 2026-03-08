"""Algorithm-related validation helpers."""

from cli.models import UserConfig
from cli.validator_common import ValidationError, iter_condition_nodes

_PERCENTILE_MIN = 1
_PERCENTILE_MAX = 100
_HYBRID_WEIGHT_TOLERANCE = 0.01
_LOOPER_ALGORITHM_TYPES = {"confidence", "concurrent", "sequential", "remom"}
_SELECTION_ALGORITHM_TYPES = {
    "static",
    "elo",
    "router_dc",
    "automix",
    "hybrid",
    "latency_aware",
    "thompson",
    "gmtrouter",
    "router_r1",
}
_ALL_ALGORITHM_TYPES = _LOOPER_ALGORITHM_TYPES | _SELECTION_ALGORITHM_TYPES
_EXPECTED_BLOCK_BY_TYPE = {
    "confidence": "confidence",
    "concurrent": "concurrent",
    "remom": "remom",
    "latency_aware": "latency_aware",
}


def validate_latency_compatibility(config: UserConfig) -> list[ValidationError]:
    """Reject legacy latency conditions in favor of latency-aware algorithms."""
    has_legacy_conditions = any(
        _is_latency_condition(condition.type)
        for decision in config.decisions
        for condition in iter_condition_nodes(decision.rules.conditions)
    )
    if not has_legacy_conditions:
        return []
    return [
        ValidationError(
            "legacy latency config is no longer supported; use "
            "decision.algorithm.type=latency_aware and remove "
            "conditions.type=latency",
            field="decisions.rules.conditions",
        )
    ]


def validate_latency_aware_algorithm_config(
    config: UserConfig,
) -> list[ValidationError]:
    """Validate latency-aware algorithm settings."""
    errors: list[ValidationError] = []
    for decision in config.decisions:
        if not _is_latency_aware_algorithm(decision):
            continue
        errors.extend(_latency_aware_errors(decision))
    return errors


def validate_algorithm_one_of(config: UserConfig) -> list[ValidationError]:
    """Validate that only the config block matching algorithm.type is set."""
    errors: list[ValidationError] = []
    for decision in config.decisions:
        if decision.algorithm is None:
            continue
        errors.extend(_algorithm_block_errors(decision))
    return errors


def validate_algorithm_configurations(config: UserConfig) -> list[ValidationError]:
    """Validate decision algorithm-specific constraints."""
    errors: list[ValidationError] = []
    for decision in config.decisions:
        if not decision.algorithm:
            continue
        errors.extend(_algorithm_type_errors(decision))
        errors.extend(_router_dc_description_errors(config, decision))
        errors.extend(_hybrid_weight_errors(decision))
    return errors


def _is_latency_condition(condition_type: str | None) -> bool:
    if not condition_type:
        return False
    return condition_type.strip().lower() == "latency"


def _is_latency_aware_algorithm(decision) -> bool:
    if not decision.algorithm:
        return False
    return (decision.algorithm.type or "").strip().lower() == "latency_aware"


def _latency_aware_errors(decision) -> list[ValidationError]:
    latency_cfg = decision.algorithm.latency_aware
    if latency_cfg is None:
        return [
            ValidationError(
                f"decision '{decision.name}' requires algorithm.latency_aware when "
                "algorithm.type=latency_aware",
                field=f"decisions.{decision.name}.algorithm.latency_aware",
            )
        ]

    errors: list[ValidationError] = []
    has_tpot = (
        latency_cfg.tpot_percentile is not None and latency_cfg.tpot_percentile > 0
    )
    has_ttft = (
        latency_cfg.ttft_percentile is not None and latency_cfg.ttft_percentile > 0
    )
    if not has_tpot and not has_ttft:
        errors.append(
            ValidationError(
                f"decision '{decision.name}' must set tpot_percentile or "
                "ttft_percentile in algorithm.latency_aware",
                field=f"decisions.{decision.name}.algorithm.latency_aware",
            )
        )
    errors.extend(
        _percentile_range_errors(
            decision.name,
            "tpot_percentile",
            latency_cfg.tpot_percentile,
            enabled=has_tpot,
        )
    )
    errors.extend(
        _percentile_range_errors(
            decision.name,
            "ttft_percentile",
            latency_cfg.ttft_percentile,
            enabled=has_ttft,
        )
    )
    return errors


def _percentile_range_errors(
    decision_name: str,
    field_name: str,
    percentile: int | None,
    *,
    enabled: bool,
) -> list[ValidationError]:
    if not enabled or percentile is None:
        return []
    if _PERCENTILE_MIN <= percentile <= _PERCENTILE_MAX:
        return []
    return [
        ValidationError(
            f"decision '{decision_name}' algorithm.latency_aware.{field_name} must "
            f"be between {_PERCENTILE_MIN} and {_PERCENTILE_MAX}",
            field=f"decisions.{decision_name}.algorithm.latency_aware.{field_name}",
        )
    ]


def _algorithm_block_errors(decision) -> list[ValidationError]:
    algorithm = decision.algorithm
    configured_blocks = _configured_algorithm_blocks(algorithm)
    display_type = (algorithm.type or "").strip() or "<empty>"
    normalized_type = (algorithm.type or "").strip().lower()

    if len(configured_blocks) > 1:
        return [
            ValidationError(
                f"decision '{decision.name}' algorithm.type={display_type} cannot be "
                "combined with multiple algorithm config blocks: "
                + ", ".join(configured_blocks),
                field=f"decisions.{decision.name}.algorithm",
            )
        ]

    expected_block = _EXPECTED_BLOCK_BY_TYPE.get(normalized_type)
    if expected_block is None:
        return _unexpected_algorithm_block_errors(
            decision.name,
            display_type,
            configured_blocks,
        )

    if len(configured_blocks) == 1 and configured_blocks[0] != expected_block:
        found_block = configured_blocks[0]
        return [
            ValidationError(
                f"decision '{decision.name}' algorithm.type={display_type} requires "
                f"algorithm.{expected_block} configuration; found algorithm.{found_block}",
                field=f"decisions.{decision.name}.algorithm.{found_block}",
            )
        ]

    return []


def _configured_algorithm_blocks(algorithm) -> list[str]:
    configured_blocks: list[str] = []
    for block_name in ("confidence", "concurrent", "remom", "latency_aware"):
        if getattr(algorithm, block_name) is not None:
            configured_blocks.append(block_name)
    return configured_blocks


def _unexpected_algorithm_block_errors(
    decision_name: str,
    display_type: str,
    configured_blocks: list[str],
) -> list[ValidationError]:
    if not configured_blocks:
        return []
    configured_block = configured_blocks[0]
    return [
        ValidationError(
            f"decision '{decision_name}' algorithm.type={display_type} cannot be used "
            f"with algorithm.{configured_block} configuration",
            field=f"decisions.{decision_name}.algorithm.{configured_block}",
        )
    ]


def _algorithm_type_errors(decision) -> list[ValidationError]:
    algo_type = decision.algorithm.type
    if algo_type in _ALL_ALGORITHM_TYPES:
        return []
    return [
        ValidationError(
            f"Decision '{decision.name}' has invalid algorithm type '{algo_type}'. "
            f"Valid types: {', '.join(sorted(_ALL_ALGORITHM_TYPES))}",
            field=f"decisions.{decision.name}.algorithm.type",
        )
    ]


def _router_dc_description_errors(
    config: UserConfig, decision
) -> list[ValidationError]:
    algo = decision.algorithm
    if algo.type != "router_dc" or not algo.router_dc:
        return []
    if not algo.router_dc.require_descriptions:
        return []

    errors: list[ValidationError] = []
    for model_ref in decision.modelRefs:
        model = next(
            (
                model
                for model in config.providers.models
                if model.name == model_ref.model
            ),
            None,
        )
        if model is None or model.description:
            continue
        errors.append(
            ValidationError(
                f"Decision '{decision.name}' uses router_dc with "
                "require_descriptions=true, but model "
                f"'{model.name}' has no description",
                field=f"providers.models.{model.name}.description",
            )
        )
    return errors


def _hybrid_weight_errors(decision) -> list[ValidationError]:
    algo = decision.algorithm
    if algo.type != "hybrid" or not algo.hybrid:
        return []

    hybrid_cfg = algo.hybrid
    total = (
        (0.3 if hybrid_cfg.elo_weight is None else hybrid_cfg.elo_weight)
        + (0.3 if hybrid_cfg.router_dc_weight is None else hybrid_cfg.router_dc_weight)
        + (0.2 if hybrid_cfg.automix_weight is None else hybrid_cfg.automix_weight)
        + (0.2 if hybrid_cfg.cost_weight is None else hybrid_cfg.cost_weight)
    )
    if abs(total - 1.0) <= _HYBRID_WEIGHT_TOLERANCE:
        return []

    return [
        ValidationError(
            f"Decision '{decision.name}' hybrid weights sum to {total:.2f}, "
            "should sum to 1.0",
            field=f"decisions.{decision.name}.algorithm.hybrid",
        )
    ]
