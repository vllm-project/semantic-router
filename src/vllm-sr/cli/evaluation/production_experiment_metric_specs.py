"""Metric specifications for production experiment reductions."""

from __future__ import annotations

from cli.evaluation.production_experiment_reducer import (
    ProductionExperimentReduction,
)

ExperimentMetricSpec = tuple[
    str,
    str,
    float | None,
    str,
    str,
    tuple[float, float] | None,
]


def _control_specs(
    reduced: ProductionExperimentReduction,
) -> tuple[ExperimentMetricSpec, ...]:
    return (
        (
            "experiment.assignment_support",
            "Randomized policy-arm support",
            reduced.assignment_support,
            "fraction",
            "higher_is_better",
            None,
        ),
        (
            "experiment.assignment_balance_p_value",
            "Policy-arm assignment-balance p-value",
            reduced.assignment_balance_p_value,
            "p-value",
            "higher_is_better",
            None,
        ),
        (
            "experiment.risk_event_rate",
            "Production experiment risk-event rate",
            reduced.risk_event_rate,
            "fraction",
            "lower_is_better",
            None,
        ),
        (
            "experiment.risk_event_upper_confidence_bound",
            "One-sided 95% production risk-event upper bound",
            reduced.risk_event_upper_confidence_bound,
            "fraction",
            "lower_is_better",
            None,
        ),
        (
            "experiment.risk_budget_max_rate",
            "Frozen production risk-budget maximum",
            reduced.risk_budget_max_rate,
            "fraction",
            "lower_is_better",
            None,
        ),
        (
            "experiment.minimum_assignment_count",
            "Frozen minimum production assignment count",
            (
                float(reduced.minimum_assignment_count)
                if reduced.minimum_assignment_count is not None
                else None
            ),
            "assignments",
            "higher_is_better",
            None,
        ),
        (
            "experiment.controls_operational",
            "Stop and rollback control-plane readiness",
            (
                None
                if reduced.controls_operational is None
                else float(reduced.controls_operational)
            ),
            "boolean",
            "higher_is_better",
            None,
        ),
        (
            "experiment.candidate_safe",
            "Production candidate safety result",
            None if reduced.candidate_safe is None else float(reduced.candidate_safe),
            "boolean",
            "higher_is_better",
            None,
        ),
    )


def _sample_specs(
    reduced: ProductionExperimentReduction,
) -> tuple[ExperimentMetricSpec, ...]:
    return (
        (
            "preference.online_outcome_coverage",
            "Assignment/exposure/outcome coverage",
            reduced.outcome_coverage,
            "fraction",
            "higher_is_better",
            None,
        ),
        (
            "preference.online_effective_sample_size",
            "Target-policy effective sample size",
            reduced.effective_sample_size,
            "effective samples",
            "higher_is_better",
            None,
        ),
        (
            "preference.online_minimum_effective_sample_size",
            "Frozen minimum effective sample size",
            reduced.minimum_effective_sample_size,
            "effective samples",
            "higher_is_better",
            None,
        ),
        (
            "preference.online_effective_sample_ratio",
            "Target-policy effective-sample ratio",
            reduced.effective_sample_ratio,
            "fraction",
            "higher_is_better",
            None,
        ),
        (
            "preference.reference_effective_sample_size",
            "Reference-policy effective sample size",
            reduced.reference_effective_sample_size,
            "effective samples",
            "higher_is_better",
            None,
        ),
        (
            "preference.reference_effective_sample_ratio",
            "Reference-policy effective-sample ratio",
            reduced.reference_effective_sample_ratio,
            "fraction",
            "higher_is_better",
            None,
        ),
        (
            "preference.online_minimum_effective_sample_ratio",
            "Frozen minimum effective-sample ratio",
            reduced.minimum_effective_sample_ratio,
            "fraction",
            "higher_is_better",
            None,
        ),
        (
            "preference.online_ips_reward",
            "Inverse-propensity target-policy reward",
            reduced.ips_reward,
            "fraction",
            "higher_is_better",
            None,
        ),
    )


def _causal_result_specs(
    reduced: ProductionExperimentReduction,
) -> tuple[ExperimentMetricSpec, ...]:
    count = reduced.assignment_count
    return (
        (
            "preference.online_snips_reward",
            "Self-normalized IPS target-policy reward",
            reduced.snips_reward,
            "fraction",
            "higher_is_better",
            reduced.snips_confidence_interval,
        ),
        (
            "preference.reference_snips_reward",
            "Reference-policy self-normalized IPS reward",
            reduced.reference_snips_reward,
            "fraction",
            "higher_is_better",
            None,
        ),
        (
            "preference.online_reward_lift",
            "Target-minus-reference SNIPS reward lift",
            reduced.reward_lift,
            "fraction",
            "higher_is_better",
            reduced.reward_lift_confidence_interval,
        ),
        (
            "preference.minimum_reward_lift",
            "Frozen minimum target-vs-reference reward lift",
            reduced.minimum_reward_lift,
            "fraction",
            "higher_is_better",
            None,
        ),
        (
            "preference.online_segment_count",
            "Observed preference segments",
            float(reduced.segment_count) if count else None,
            "segments",
            "higher_is_better",
            None,
        ),
        (
            "preference.online_segment_coverage",
            "Minimum-sample segment coverage",
            reduced.segment_coverage,
            "fraction",
            "higher_is_better",
            None,
        ),
        (
            "preference.online_causal_eligible",
            "Causal estimator eligibility",
            float(reduced.causal_eligible) if count else None,
            "boolean",
            "higher_is_better",
            None,
        ),
        (
            "preference.online_reward_lift_passed",
            "Target-vs-reference lower-confidence-bound result",
            (
                float(reduced.preference_passed)
                if reduced.preference_passed is not None
                else None
            ),
            "boolean",
            "higher_is_better",
            None,
        ),
    )


def production_experiment_metric_specs(
    reduced: ProductionExperimentReduction,
) -> tuple[ExperimentMetricSpec, ...]:
    return (
        *_control_specs(reduced),
        *_sample_specs(reduced),
        *_causal_result_specs(reduced),
    )
