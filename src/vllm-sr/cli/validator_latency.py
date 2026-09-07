"""Latency-algorithm compatibility validation."""

from cli.models import UserConfig
from cli.validation_error import ValidationError

MIN_PERCENTILE = 1
MAX_PERCENTILE = 100


def _all_decisions(config: UserConfig):
    for decision in config.routing.decisions:
        yield "decisions", decision
    for recipe in config.recipes:
        field_prefix = f"recipes.{recipe.name}.decisions"
        for decision in recipe.routing.decisions:
            yield field_prefix, decision


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
