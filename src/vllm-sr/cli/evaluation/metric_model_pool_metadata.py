"""Presentation metadata for the fixed and per-arm model-pool metrics."""

from __future__ import annotations

from dataclasses import dataclass

from cli.evaluation.metric_model_pool_contract import parse_model_pool_arm_metric_id


@dataclass(frozen=True)
class ModelPoolMetricMetadata:
    name: str
    unit: str
    direction: str


_METADATA = {
    "model_pool.all_arm_failure_rate": ModelPoolMetricMetadata(
        "Cases where every arm failed", "fraction", "lower_is_better"
    ),
    "model_pool.arm_count": ModelPoolMetricMetadata(
        "Observed model arms", "arms", "target"
    ),
    "model_pool.best_single_quality": ModelPoolMetricMetadata(
        "Best single-arm quality", "score", "higher_is_better"
    ),
    "model_pool.mean_pairwise_failure_jaccard": ModelPoolMetricMetadata(
        "Mean pairwise arm failure overlap", "fraction", "lower_is_better"
    ),
    "model_pool.oracle_gain": ModelPoolMetricMetadata(
        "Oracle gain over best single arm", "score", "higher_is_better"
    ),
    "model_pool.oracle_quality": ModelPoolMetricMetadata(
        "Pool oracle quality", "score", "higher_is_better"
    ),
    "model_pool.pareto_dominated_arm_count": ModelPoolMetricMetadata(
        "Quality-cost Pareto-dominated arms", "arms", "lower_is_better"
    ),
    "model_pool.pareto_evaluable_arm_count": ModelPoolMetricMetadata(
        "Arms with complete comparable quality and cost",
        "arms",
        "higher_is_better",
    ),
    "model_pool.quality_cost_shared_support_cases": ModelPoolMetricMetadata(
        "Cases with complete arm quality and cost support",
        "cases",
        "higher_is_better",
    ),
    "model_pool.quality_cost_shared_support_fraction": ModelPoolMetricMetadata(
        "Complete arm quality and cost support rate",
        "fraction",
        "higher_is_better",
    ),
    "model_pool.quality_dominated_arm_count": ModelPoolMetricMetadata(
        "Quality-dominated arms on complete common cases",
        "arms",
        "lower_is_better",
    ),
    "model_pool.quality_shared_support_cases": ModelPoolMetricMetadata(
        "Cases with complete arm quality support", "cases", "higher_is_better"
    ),
    "model_pool.quality_shared_support_fraction": ModelPoolMetricMetadata(
        "Complete arm quality support rate", "fraction", "higher_is_better"
    ),
    "model_pool.selection_arm_coverage": ModelPoolMetricMetadata(
        "Selected-arm coverage", "fraction", "higher_is_better"
    ),
    "model_pool.selection_entropy_bits": ModelPoolMetricMetadata(
        "Arm selection entropy", "bits", "target"
    ),
    "model_pool.unique_win_rate": ModelPoolMetricMetadata(
        "Unique-win case rate", "fraction", "higher_is_better"
    ),
    "model_pool.unique_wins": ModelPoolMetricMetadata(
        "Cases with a unique winning arm", "cases", "higher_is_better"
    ),
    "model_pool.worst_arm_reliability": ModelPoolMetricMetadata(
        "Reliability of the least reliable frozen arm",
        "fraction",
        "higher_is_better",
    ),
}


def metric_metadata(metric_id: str) -> ModelPoolMetricMetadata:
    metadata = _METADATA.get(metric_id)
    if metadata is not None:
        return metadata
    parsed = parse_model_pool_arm_metric_id(metric_id)
    if parsed is None:
        raise ValueError(f"unknown model-pool metric {metric_id}")
    arm_id, measure = parsed
    if measure == "quality":
        return ModelPoolMetricMetadata(f"{arm_id} quality", "score", "higher_is_better")
    if measure == "success_rate":
        return ModelPoolMetricMetadata(
            f"{arm_id} success rate", "fraction", "higher_is_better"
        )
    return ModelPoolMetricMetadata(
        f"{arm_id} marginal contribution", "score", "higher_is_better"
    )
