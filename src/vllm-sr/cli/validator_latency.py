"""Latency-algorithm compatibility validation."""

from cli.config_contract import iter_routing_profiles
from cli.models import UserConfig
from cli.validation_error import ValidationError

MIN_PERCENTILE = 1
MAX_PERCENTILE = 100


def _all_decisions(config: UserConfig):
    for profile_name, routing in iter_routing_profiles(config):
        field_prefix = (
            "routing.decisions"
            if profile_name == "default"
            else f"recipes.{profile_name}.routing.decisions"
        )
        for decision in routing.decisions:
            yield field_prefix, decision


def _iter_condition_nodes(conditions):
    if not conditions:
        return
    for condition in conditions:
        yield condition
        if getattr(condition, "conditions", None):
            yield from _iter_condition_nodes(condition.conditions)


def validate_latency_compatibility(
    config: UserConfig,
) -> list[ValidationError]:
    errors: list[ValidationError] = []
    for field_prefix, decision in _all_decisions(config):
        has_legacy_conditions = any(
            (condition.type or "").strip().lower() == "latency"
            for condition in _iter_condition_nodes(decision.rules.conditions)
        )
        if has_legacy_conditions:
            errors.append(
                ValidationError(
                    "legacy latency config is no longer supported; use decision.algorithm.type=latency_aware and remove conditions.type=latency",
                    field=f"{field_prefix}.{decision.name}.rules.conditions",
                )
            )
    return errors


def validate_latency_aware_algorithm_config(
    config: UserConfig,
) -> list[ValidationError]:
    errors: list[ValidationError] = []
    for field_prefix, decision in _all_decisions(config):
        if (
            not decision.algorithm
            or (decision.algorithm.type or "").strip().lower() != "latency_aware"
        ):
            continue
        latency_cfg = decision.algorithm.latency_aware
        field = f"{field_prefix}.{decision.name}.algorithm.latency_aware"
        if latency_cfg is None:
            errors.append(
                ValidationError(
                    f"decision '{decision.name}' requires algorithm.latency_aware when algorithm.type=latency_aware",
                    field=field,
                )
            )
            continue
        has_tpot = bool(latency_cfg.tpot_percentile)
        has_ttft = bool(latency_cfg.ttft_percentile)
        if not has_tpot and not has_ttft:
            errors.append(
                ValidationError(
                    f"decision '{decision.name}' must set tpot_percentile or ttft_percentile in algorithm.latency_aware",
                    field=field,
                )
            )
        if (
            has_tpot
            and not MIN_PERCENTILE <= latency_cfg.tpot_percentile <= MAX_PERCENTILE
        ):
            errors.append(
                ValidationError(
                    f"decision '{decision.name}' algorithm.latency_aware.tpot_percentile must be between 1 and 100",
                    field=f"{field}.tpot_percentile",
                )
            )
        if (
            has_ttft
            and not MIN_PERCENTILE <= latency_cfg.ttft_percentile <= MAX_PERCENTILE
        ):
            errors.append(
                ValidationError(
                    f"decision '{decision.name}' algorithm.latency_aware.ttft_percentile must be between 1 and 100",
                    field=f"{field}.ttft_percentile",
                )
            )
    return errors
