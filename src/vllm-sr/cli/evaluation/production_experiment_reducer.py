"""Cohort validation and causal statistics for production experiments."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import erfc, sqrt

from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.method_evidence import ProductionExperimentMethodEvidence

_NORMAL_95 = 1.959963984540054
_ONE_SIDED_NORMAL_95 = 1.6448536269514722
_SRM_MINIMUM_P_VALUE = 0.01


@dataclass(frozen=True)
class ProductionExperimentReduction:
    assignment_count: int
    candidate_safe: bool | None
    assignment_support: float | None
    assignment_balance_p_value: float | None
    risk_event_rate: float | None
    risk_event_upper_confidence_bound: float | None
    risk_budget_max_rate: float | None
    minimum_assignment_count: int | None
    controls_operational: bool | None
    outcome_coverage: float | None
    causal_eligible: bool
    ips_reward: float | None
    snips_reward: float | None
    snips_confidence_interval: tuple[float, float] | None
    effective_sample_size: float | None
    minimum_effective_sample_size: float | None
    effective_sample_ratio: float | None
    minimum_effective_sample_ratio: float | None
    reference_snips_reward: float | None
    reference_effective_sample_size: float | None
    reference_effective_sample_ratio: float | None
    reward_lift: float | None
    reward_lift_confidence_interval: tuple[float, float] | None
    minimum_reward_lift: float | None
    preference_passed: bool | None
    segment_count: int
    segment_coverage: float | None
    target_id: str | None
    backend_topology_digest: str | None
    mixture_snapshot_digest: str | None


@dataclass(frozen=True)
class _ExperimentCohort:
    first: ProductionExperimentMethodEvidence
    rows: list[ExecutionRecord]
    online: list[ExecutionRecord]
    arm_counts: Counter[str]
    expected_counts: dict[str, float]


@dataclass(frozen=True)
class _ControlStats:
    support: float
    balance_p_value: float
    risk_event_rate: float
    risk_event_upper_bound: float
    sealed_window_complete: bool
    controls_operational: bool
    candidate_safe: bool
    outcome_coverage: float
    complete: bool


@dataclass(frozen=True)
class _PreferenceSample:
    target_weights: list[float]
    reference_weights: list[float]
    rewards: list[float]
    segment_count: int
    segment_coverage: float | None
    target_weight_total: float
    reference_weight_total: float
    effective_sample_size: float | None
    effective_sample_ratio: float | None
    reference_effective_sample_size: float | None
    reference_effective_sample_ratio: float | None


@dataclass(frozen=True)
class _CausalStats:
    eligible: bool
    ips_reward: float | None
    snips_reward: float | None
    snips_confidence_interval: tuple[float, float] | None
    reference_snips_reward: float | None
    reward_lift: float | None
    reward_lift_confidence_interval: tuple[float, float] | None
    preference_passed: bool | None


def _production_rows(records: list[ExecutionRecord]) -> list[ExecutionRecord]:
    return [
        row
        for row in records
        if row.track_id == "preference" and row.production_experiment is not None
    ]


def _wilson_upper_bound(events: int, count: int) -> float:
    if count <= 0:
        raise ValueError("risk bound requires a positive sample count")
    rate = events / count
    z = _ONE_SIDED_NORMAL_95
    denominator = 1 + z * z / count
    center = rate + z * z / (2 * count)
    spread = z * sqrt(rate * (1 - rate) / count + z * z / (4 * count * count))
    return min(1.0, (center + spread) / denominator)


def _effective_sample_size(weights: list[float]) -> float | None:
    total = sum(weights)
    square_total = sum(weight * weight for weight in weights)
    return total * total / square_total if square_total > 0 else None


def _snips(weights: list[float], rewards: list[float]) -> float | None:
    total = sum(weights)
    if total <= 0:
        return None
    return (
        sum(weight * reward for weight, reward in zip(weights, rewards, strict=True))
        / total
    )


def _empty_reduction() -> ProductionExperimentReduction:
    return ProductionExperimentReduction(
        assignment_count=0,
        candidate_safe=None,
        assignment_support=None,
        assignment_balance_p_value=None,
        risk_event_rate=None,
        risk_event_upper_confidence_bound=None,
        risk_budget_max_rate=None,
        minimum_assignment_count=None,
        controls_operational=None,
        outcome_coverage=None,
        causal_eligible=False,
        ips_reward=None,
        snips_reward=None,
        snips_confidence_interval=None,
        effective_sample_size=None,
        minimum_effective_sample_size=None,
        effective_sample_ratio=None,
        minimum_effective_sample_ratio=None,
        reference_snips_reward=None,
        reference_effective_sample_size=None,
        reference_effective_sample_ratio=None,
        reward_lift=None,
        reward_lift_confidence_interval=None,
        minimum_reward_lift=None,
        preference_passed=None,
        segment_count=0,
        segment_coverage=None,
        target_id=None,
        backend_topology_digest=None,
        mixture_snapshot_digest=None,
    )


def _validate_contract(
    experiment: ProductionExperimentMethodEvidence,
    first: ProductionExperimentMethodEvidence,
) -> None:
    if (
        experiment.experiment_id != first.experiment_id
        or experiment.ledger_id != first.ledger_id
        or experiment.ledger_total_assignment_count
        != first.ledger_total_assignment_count
        or experiment.ledger_total_outcome_count != first.ledger_total_outcome_count
        or experiment.source_id != first.source_id
        or experiment.policy_snapshot_digest != first.policy_snapshot_digest
        or experiment.config_digest != first.config_digest
        or experiment.target_id != first.target_id
        or experiment.backend_topology_digest != first.backend_topology_digest
        or experiment.mixture_snapshot_digest != first.mixture_snapshot_digest
        or experiment.policy_arms != first.policy_arms
        or experiment.minimum_effective_sample_size
        != first.minimum_effective_sample_size
        or experiment.minimum_effective_sample_ratio
        != first.minimum_effective_sample_ratio
        or experiment.minimum_segment_sample_size != first.minimum_segment_sample_size
        or experiment.minimum_assignment_count != first.minimum_assignment_count
        or experiment.minimum_reward_lift != first.minimum_reward_lift
        or experiment.confidence_level != first.confidence_level
        or experiment.stop_rule_id != first.stop_rule_id
        or experiment.stop_rule_evaluated_at != first.stop_rule_evaluated_at
        or experiment.rollback_receipt_id != first.rollback_receipt_id
        or experiment.rollback_validated_at != first.rollback_validated_at
        or experiment.ledger_sealed_at != first.ledger_sealed_at
    ):
        raise ValueError("production records mix experiment contracts")


def _collect_cohort(records: list[ExecutionRecord]) -> _ExperimentCohort | None:
    rows = _production_rows(records)
    if not rows:
        return None
    first = rows[0].production_experiment
    if first is None:
        raise ValueError("production assignment row lost typed evidence")
    assignments: set[str] = set()
    exposures: set[str] = set()
    participants: set[str] = set()
    arm_counts: Counter[str] = Counter()
    expected_counts = {arm.id: 0.0 for arm in first.policy_arms}
    online: list[ExecutionRecord] = []
    for row in rows:
        experiment = row.production_experiment
        if experiment is None:
            raise ValueError("production assignment row lost typed evidence")
        _validate_contract(experiment, first)
        if (
            experiment.assignment_id in assignments
            or experiment.exposure_id in exposures
            or experiment.participant_digest in participants
        ):
            raise ValueError("production experiment identities are not unique")
        assignments.add(experiment.assignment_id)
        exposures.add(experiment.exposure_id)
        participants.add(experiment.participant_digest)
        arm_counts[experiment.assigned_policy_arm_id] += 1
        for arm in experiment.policy_arms:
            expected_counts[arm.id] += arm.assignment_probability
        if row.online_preference is not None:
            online.append(row)
    return _ExperimentCohort(first, rows, online, arm_counts, expected_counts)


def _control_stats(cohort: _ExperimentCohort) -> _ControlStats:
    first, rows = cohort.first, cohort.rows
    required_arms = {arm.id for arm in first.policy_arms}
    observed_arms = {arm_id for arm_id, count in cohort.arm_counts.items() if count > 0}
    support = len(required_arms.intersection(observed_arms)) / len(required_arms)
    chi_square = sum(
        (cohort.arm_counts[arm_id] - cohort.expected_counts[arm_id]) ** 2
        / cohort.expected_counts[arm_id]
        for arm_id in sorted(required_arms)
    )
    balance_p = erfc(sqrt(chi_square / 2))
    risk_events = sum(
        bool(row.production_experiment and row.production_experiment.risk_event)
        for row in rows
    )
    risk_upper_bound = _wilson_upper_bound(risk_events, len(rows))
    sealed_window_complete = len(rows) == first.ledger_total_assignment_count
    controls_operational = first.rollback_ready and (
        not first.stop_triggered
        or (first.rollback_executed_at is not None and first.rollback_succeeded is True)
    )
    candidate_safe = (
        sealed_window_complete
        and len(rows) >= first.minimum_assignment_count
        and balance_p >= _SRM_MINIMUM_P_VALUE
        and risk_upper_bound <= first.risk_budget_max_rate
        and first.rollback_ready
        and not first.stop_triggered
    )
    outcome_coverage = len(cohort.online) / len(rows)
    complete = (
        sealed_window_complete
        and first.ledger_total_outcome_count == first.ledger_total_assignment_count
        and len(cohort.online) == first.ledger_total_outcome_count
        and outcome_coverage == 1.0
    )
    return _ControlStats(
        support=support,
        balance_p_value=balance_p,
        risk_event_rate=risk_events / len(rows),
        risk_event_upper_bound=risk_upper_bound,
        sealed_window_complete=sealed_window_complete,
        controls_operational=controls_operational,
        candidate_safe=candidate_safe,
        outcome_coverage=outcome_coverage,
        complete=complete,
    )


def _preference_sample(cohort: _ExperimentCohort) -> _PreferenceSample:
    target_weights: list[float] = []
    reference_weights: list[float] = []
    rewards: list[float] = []
    segments: Counter[str] = Counter()
    for row in cohort.online:
        experiment = row.production_experiment
        preference = row.online_preference
        if experiment is None or preference is None:
            raise ValueError("online outcome lost its production assignment")
        arm = next(
            arm
            for arm in experiment.policy_arms
            if arm.id == experiment.assigned_policy_arm_id
        )
        target_weights.append(
            arm.target_policy_probability / experiment.behavior_propensity
        )
        reference_weights.append(
            arm.reference_policy_probability / experiment.behavior_propensity
        )
        rewards.append(preference.outcome.reward)
        segments[experiment.segment_id] += 1
    effective = _effective_sample_size(target_weights)
    reference_effective = _effective_sample_size(reference_weights)
    target_segments = {
        row.production_experiment.segment_id
        for row in cohort.rows
        if row.production_experiment is not None
    }
    segment_coverage = (
        sum(
            segments[segment] >= cohort.first.minimum_segment_sample_size
            for segment in target_segments
        )
        / len(target_segments)
        if target_segments
        else None
    )
    return _PreferenceSample(
        target_weights=target_weights,
        reference_weights=reference_weights,
        rewards=rewards,
        segment_count=len(segments),
        segment_coverage=segment_coverage,
        target_weight_total=sum(target_weights),
        reference_weight_total=sum(reference_weights),
        effective_sample_size=effective,
        effective_sample_ratio=(
            effective / len(cohort.rows) if effective is not None else None
        ),
        reference_effective_sample_size=reference_effective,
        reference_effective_sample_ratio=(
            reference_effective / len(cohort.rows)
            if reference_effective is not None
            else None
        ),
    )


def _causal_eligible(
    cohort: _ExperimentCohort,
    controls: _ControlStats,
    sample: _PreferenceSample,
) -> bool:
    first = cohort.first
    return bool(
        controls.complete
        and controls.support == 1.0
        and controls.candidate_safe
        and sample.effective_sample_size is not None
        and sample.effective_sample_size >= first.minimum_effective_sample_size
        and sample.effective_sample_ratio is not None
        and sample.effective_sample_ratio >= first.minimum_effective_sample_ratio
        and sample.reference_effective_sample_size is not None
        and sample.reference_effective_sample_size
        >= first.minimum_effective_sample_size
        and sample.reference_effective_sample_ratio is not None
        and sample.reference_effective_sample_ratio
        >= first.minimum_effective_sample_ratio
        and sample.segment_coverage == 1.0
        and sample.target_weight_total > 0
        and sample.reference_weight_total > 0
    )


def _causal_stats(
    cohort: _ExperimentCohort,
    controls: _ControlStats,
    sample: _PreferenceSample,
) -> _CausalStats:
    eligible = _causal_eligible(cohort, controls, sample)
    ips = (
        sum(
            weight * reward
            for weight, reward in zip(
                sample.target_weights, sample.rewards, strict=True
            )
        )
        / len(cohort.rows)
        if eligible
        else None
    )
    snips = _snips(sample.target_weights, sample.rewards) if eligible else None
    reference_snips = (
        _snips(sample.reference_weights, sample.rewards) if eligible else None
    )
    confidence: tuple[float, float] | None = None
    reward_lift: float | None = None
    lift_confidence: tuple[float, float] | None = None
    if eligible and snips is not None:
        variance = sum(
            weight * weight * (reward - snips) ** 2
            for weight, reward in zip(
                sample.target_weights, sample.rewards, strict=True
            )
        ) / (sample.target_weight_total * sample.target_weight_total)
        half_width = _NORMAL_95 * sqrt(max(0.0, variance))
        confidence = (max(0.0, snips - half_width), min(1.0, snips + half_width))
        if reference_snips is None:
            raise ValueError("causally eligible evidence lost its reference policy")
        reward_lift = snips - reference_snips
        influence = _lift_influence(sample, snips, reference_snips)
        count = len(sample.rewards)
        standard_error = sqrt(
            sum(value * value for value in influence) / (count * (count - 1))
        )
        lift_half_width = _NORMAL_95 * standard_error
        lift_confidence = (
            max(-1.0, reward_lift - lift_half_width),
            min(1.0, reward_lift + lift_half_width),
        )
    preference_passed = (
        None
        if not controls.complete
        else bool(
            eligible
            and lift_confidence is not None
            and lift_confidence[0] >= cohort.first.minimum_reward_lift
        )
    )
    return _CausalStats(
        eligible,
        ips,
        snips,
        confidence,
        reference_snips,
        reward_lift,
        lift_confidence,
        preference_passed,
    )


def _lift_influence(
    sample: _PreferenceSample,
    snips: float,
    reference_snips: float,
) -> list[float]:
    count = len(sample.rewards)
    return [
        count
        * (
            target_weight * (reward - snips) / sample.target_weight_total
            - reference_weight
            * (reward - reference_snips)
            / sample.reference_weight_total
        )
        for target_weight, reference_weight, reward in zip(
            sample.target_weights,
            sample.reference_weights,
            sample.rewards,
            strict=True,
        )
    ]


def reduce_production_experiment(
    records: list[ExecutionRecord],
) -> ProductionExperimentReduction:
    cohort = _collect_cohort(records)
    if cohort is None:
        return _empty_reduction()
    controls = _control_stats(cohort)
    sample = _preference_sample(cohort)
    causal = _causal_stats(cohort, controls, sample)
    first = cohort.first
    return ProductionExperimentReduction(
        assignment_count=len(cohort.rows),
        candidate_safe=controls.candidate_safe and controls.support == 1.0,
        assignment_support=controls.support,
        assignment_balance_p_value=controls.balance_p_value,
        risk_event_rate=controls.risk_event_rate,
        risk_event_upper_confidence_bound=controls.risk_event_upper_bound,
        risk_budget_max_rate=first.risk_budget_max_rate,
        minimum_assignment_count=first.minimum_assignment_count,
        controls_operational=controls.controls_operational,
        outcome_coverage=controls.outcome_coverage,
        causal_eligible=causal.eligible,
        ips_reward=causal.ips_reward,
        snips_reward=causal.snips_reward,
        snips_confidence_interval=causal.snips_confidence_interval,
        effective_sample_size=sample.effective_sample_size,
        minimum_effective_sample_size=first.minimum_effective_sample_size,
        effective_sample_ratio=sample.effective_sample_ratio,
        minimum_effective_sample_ratio=first.minimum_effective_sample_ratio,
        reference_snips_reward=causal.reference_snips_reward,
        reference_effective_sample_size=sample.reference_effective_sample_size,
        reference_effective_sample_ratio=sample.reference_effective_sample_ratio,
        reward_lift=causal.reward_lift,
        reward_lift_confidence_interval=causal.reward_lift_confidence_interval,
        minimum_reward_lift=first.minimum_reward_lift,
        preference_passed=causal.preference_passed,
        segment_count=sample.segment_count,
        segment_coverage=sample.segment_coverage,
        target_id=first.target_id,
        backend_topology_digest=first.backend_topology_digest,
        mixture_snapshot_digest=first.mixture_snapshot_digest,
    )
