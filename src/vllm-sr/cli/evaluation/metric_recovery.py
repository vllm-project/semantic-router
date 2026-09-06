"""Server-portable live exact-step fault-recovery reduction."""

from __future__ import annotations

from dataclasses import dataclass

from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.method_evidence import RecoveryMethodEvidence
from cli.evaluation.metric_core import MetricDraft, build_metric

MINIMUM_RECOVERY_PASS_RATE_LOWER_BOUND = 0.8
_ONE_SIDED_95_Z = 1.6448536269514722


def _one_sided_wilson_bounds(successes: int, total: int) -> tuple[float, float] | None:
    if total <= 0:
        return None
    proportion = successes / total
    z_squared = _ONE_SIDED_95_Z * _ONE_SIDED_95_Z
    denominator = 1 + z_squared / total
    center = proportion + z_squared / (2 * total)
    margin = _ONE_SIDED_95_Z * (
        (proportion * (1 - proportion) / total + z_squared / (4 * total * total)) ** 0.5
    )
    return (
        max(0.0, (center - margin) / denominator),
        min(1.0, (center + margin) / denominator),
    )


def one_sided_wilson_lower_bound(successes: int, total: int) -> float | None:
    bounds = _one_sided_wilson_bounds(successes, total)
    return bounds[0] if bounds is not None else None


@dataclass(frozen=True)
class RecoveryReduction:
    pair_count: int
    cluster_count: int
    cluster_pass_rate: float | None
    cluster_pass_rate_confidence_interval: tuple[float, float] | None
    cluster_pass_rate_lower_confidence_bound: float | None
    treatment_success_rate: float | None
    baseline_success_rate: float | None
    success_delta: float | None
    mean_latency_delta_ms: float | None
    maximum_retry_amplification: float | None
    maximum_recovery_latency_ms: float | None
    maximum_retry_amplification_threshold: float | None
    distinct_seed_count: int
    minimum_cluster_count: int | None
    minimum_distinct_seed_count: int | None
    minimum_pair_count: int | None
    passed: bool | None
    target_id: str | None
    backend_topology_digest: str | None
    mixture_snapshot_digest: str | None


_RecoveryClusterRow = tuple[bool, bool, bool, float]


@dataclass(frozen=True)
class _RecoveryObservations:
    first: RecoveryMethodEvidence | None
    pair_count: int
    distinct_seed_count: int
    ordered_clusters: tuple[tuple[_RecoveryClusterRow, ...], ...]
    retry_amplifications: tuple[float, ...]


def _validate_recovery_contract(
    method: RecoveryMethodEvidence,
    first: RecoveryMethodEvidence | None,
    policy_snapshot_digest: str | None,
    config_digest: str | None,
) -> None:
    if method.method_id != "live-fault-recovery.v1":
        raise ValueError("recovery evidence uses an unsupported method")
    if first is None:
        if (
            policy_snapshot_digest is not None
            and method.policy_snapshot_digest != policy_snapshot_digest
        ) or (config_digest is not None and method.config_digest != config_digest):
            raise ValueError("fault-recovery ledger belongs to another snapshot")
        return
    if any(
        (
            method.ledger_id != first.ledger_id,
            method.source_id != first.source_id,
            method.policy_snapshot_digest != first.policy_snapshot_digest,
            method.config_digest != first.config_digest,
            method.target_id != first.target_id,
            method.backend_topology_digest != first.backend_topology_digest,
            method.mixture_snapshot_digest != first.mixture_snapshot_digest,
            method.ledger_total_pair_count != first.ledger_total_pair_count,
            method.minimum_pair_count != first.minimum_pair_count,
            method.minimum_cluster_count != first.minimum_cluster_count,
            method.minimum_distinct_seed_count != first.minimum_distinct_seed_count,
            method.maximum_recovery_latency_ms != first.maximum_recovery_latency_ms,
            method.maximum_retry_amplification != first.maximum_retry_amplification,
        )
    ):
        raise ValueError("recovery records mix sealed ledger contracts")


def _recovery_outcome(method: RecoveryMethodEvidence) -> tuple[bool, float]:
    retry_amplification = (method.treatment_retry_count + 1) / (
        method.baseline_retry_count + 1
    )
    passed = (
        method.injection_observed
        and method.recovered
        and method.state_preserved
        and method.treatment_terminal_success
        and method.duplicate_side_effect_count == 0
        and method.treatment_recovery_latency_ms <= method.maximum_recovery_latency_ms
        and retry_amplification <= method.maximum_retry_amplification
    )
    return passed, retry_amplification


def _collect_recovery_observations(
    records: list[ExecutionRecord],
    policy_snapshot_digest: str | None,
    config_digest: str | None,
) -> _RecoveryObservations:
    fault_ids: set[str] = set()
    pair_ids: set[tuple[str, str]] = set()
    seeds: set[int] = set()
    cluster_rows: dict[str, list[_RecoveryClusterRow]] = {}
    retry_amplifications: list[float] = []
    first: RecoveryMethodEvidence | None = None
    for row in records:
        method = row.recovery
        if row.track_id != "agentic" or method is None:
            continue
        _validate_recovery_contract(
            method,
            first,
            policy_snapshot_digest,
            config_digest,
        )
        if first is None:
            first = method
        pair_id = (method.cohort_pair_id, method.repetition_id)
        if method.fault_id in fault_ids or pair_id in pair_ids:
            raise ValueError("recovery reduction received a duplicate live pair")
        fault_ids.add(method.fault_id)
        pair_ids.add(pair_id)
        seeds.add(method.seed)
        passed, retry_amplification = _recovery_outcome(method)
        if row.success is not passed:
            raise ValueError("agentic result does not bind its recovery evidence")
        cluster_rows.setdefault(method.cluster_id, []).append(
            (
                passed,
                method.baseline_terminal_success,
                method.treatment_terminal_success,
                method.treatment_recovery_latency_ms
                - method.baseline_recovery_latency_ms,
            )
        )
        retry_amplifications.append(retry_amplification)
    return _RecoveryObservations(
        first=first,
        pair_count=len(pair_ids),
        distinct_seed_count=len(seeds),
        ordered_clusters=tuple(
            tuple(cluster_rows[cluster_id]) for cluster_id in sorted(cluster_rows)
        ),
        retry_amplifications=tuple(retry_amplifications),
    )


def reduce_recovery(
    records: list[ExecutionRecord],
    *,
    policy_snapshot_digest: str | None = None,
    config_digest: str | None = None,
) -> RecoveryReduction:
    """Reduce independent clusters conservatively and deterministically.

    Every pair in a cluster must pass, while cluster latency contributes its
    worst pair delta. Sorted cluster identities make floating-point summation
    reproducible with the server implementation.
    """

    observations = _collect_recovery_observations(
        records, policy_snapshot_digest, config_digest
    )
    first = observations.first
    count = observations.pair_count
    ordered_clusters = observations.ordered_clusters
    cluster_passes = [all(row[0] for row in rows) for rows in ordered_clusters]
    cluster_baseline_successes = [
        all(row[1] for row in rows) for rows in ordered_clusters
    ]
    cluster_treatment_successes = [
        all(row[2] for row in rows) for rows in ordered_clusters
    ]
    cluster_latency_deltas = [max(row[3] for row in rows) for rows in ordered_clusters]
    cluster_count = len(ordered_clusters)
    cluster_pass_rate = sum(cluster_passes) / cluster_count if cluster_count else None
    cluster_pass_rate_interval = _one_sided_wilson_bounds(
        sum(cluster_passes), cluster_count
    )
    cluster_pass_rate_lower_bound = (
        cluster_pass_rate_interval[0]
        if cluster_pass_rate_interval is not None
        else None
    )
    baseline_rate = (
        sum(cluster_baseline_successes) / cluster_count if cluster_count else None
    )
    treatment_rate = (
        sum(cluster_treatment_successes) / cluster_count if cluster_count else None
    )
    complete = (
        first is not None
        and count == first.ledger_total_pair_count
        and count >= first.minimum_pair_count
        and cluster_count >= first.minimum_cluster_count
        and observations.distinct_seed_count >= first.minimum_distinct_seed_count
    )
    return RecoveryReduction(
        pair_count=count,
        cluster_count=cluster_count,
        cluster_pass_rate=cluster_pass_rate,
        cluster_pass_rate_confidence_interval=cluster_pass_rate_interval,
        cluster_pass_rate_lower_confidence_bound=cluster_pass_rate_lower_bound,
        treatment_success_rate=treatment_rate,
        baseline_success_rate=baseline_rate,
        success_delta=(treatment_rate - baseline_rate if cluster_count else None),
        mean_latency_delta_ms=(
            sum(cluster_latency_deltas) / cluster_count if cluster_count else None
        ),
        maximum_retry_amplification=(
            max(observations.retry_amplifications) if count else None
        ),
        maximum_recovery_latency_ms=(
            first.maximum_recovery_latency_ms if first is not None else None
        ),
        maximum_retry_amplification_threshold=(
            first.maximum_retry_amplification if first is not None else None
        ),
        distinct_seed_count=observations.distinct_seed_count,
        minimum_cluster_count=(
            first.minimum_cluster_count if first is not None else None
        ),
        minimum_distinct_seed_count=(
            first.minimum_distinct_seed_count if first is not None else None
        ),
        minimum_pair_count=(first.minimum_pair_count if first is not None else None),
        passed=(
            cluster_pass_rate_lower_bound >= MINIMUM_RECOVERY_PASS_RATE_LOWER_BOUND
            if complete and cluster_pass_rate_lower_bound is not None
            else None
        ),
        target_id=first.target_id if first is not None else None,
        backend_topology_digest=(
            first.backend_topology_digest if first is not None else None
        ),
        mixture_snapshot_digest=(
            first.mixture_snapshot_digest if first is not None else None
        ),
    )


_RecoveryMetricSpec = tuple[
    str,
    str,
    float | None,
    str,
    str,
    int,
    tuple[float, float] | None,
]


def _recovery_outcome_specs(
    reduced: RecoveryReduction,
) -> tuple[_RecoveryMetricSpec, ...]:
    cluster_count = reduced.cluster_count
    pair_count = reduced.pair_count
    return (
        (
            "agentic.recovery_pair_count",
            "Live fault-recovery pair count",
            float(pair_count) if pair_count else None,
            "pairs",
            "higher_is_better",
            pair_count,
            None,
        ),
        (
            "agentic.recovery_cluster_count",
            "Independent fault-recovery cluster count",
            float(cluster_count) if cluster_count else None,
            "clusters",
            "higher_is_better",
            cluster_count,
            None,
        ),
        (
            "agentic.recovery_cluster_pass_rate",
            "All-pairs recovery pass rate across independent clusters",
            reduced.cluster_pass_rate,
            "fraction",
            "higher_is_better",
            cluster_count,
            reduced.cluster_pass_rate_confidence_interval,
        ),
        (
            "agentic.recovery_cluster_pass_rate_lower_95",
            "One-sided 95% independent-cluster recovery lower bound",
            reduced.cluster_pass_rate_lower_confidence_bound,
            "fraction",
            "higher_is_better",
            cluster_count,
            None,
        ),
        (
            "agentic.recovery_minimum_cluster_count",
            "Frozen minimum independent fault-recovery clusters",
            (
                float(reduced.minimum_cluster_count)
                if reduced.minimum_cluster_count is not None
                else None
            ),
            "clusters",
            "higher_is_better",
            cluster_count,
            None,
        ),
        (
            "agentic.recovery_treatment_success_rate",
            "All-pairs treatment success across independent clusters",
            reduced.treatment_success_rate,
            "fraction",
            "higher_is_better",
            cluster_count,
            None,
        ),
        (
            "agentic.recovery_baseline_success_rate",
            "All-pairs baseline success across independent clusters",
            reduced.baseline_success_rate,
            "fraction",
            "higher_is_better",
            cluster_count,
            None,
        ),
    )


def _recovery_guardrail_specs(
    reduced: RecoveryReduction,
) -> tuple[_RecoveryMetricSpec, ...]:
    cluster_count = reduced.cluster_count
    pair_count = reduced.pair_count
    return (
        (
            "agentic.recovery_success_delta",
            "Cluster-weighted treatment minus baseline continuity success",
            reduced.success_delta,
            "fraction",
            "higher_is_better",
            cluster_count,
            None,
        ),
        (
            "agentic.recovery_mean_latency_delta_ms",
            "Mean cluster-worst treatment minus baseline recovery latency",
            reduced.mean_latency_delta_ms,
            "ms",
            "lower_is_better",
            cluster_count,
            None,
        ),
        (
            "agentic.recovery_max_retry_amplification",
            "Maximum paired retry amplification",
            reduced.maximum_retry_amplification,
            "ratio",
            "lower_is_better",
            pair_count,
            None,
        ),
        (
            "agentic.recovery_maximum_latency_ms",
            "Frozen maximum treatment recovery latency",
            reduced.maximum_recovery_latency_ms,
            "ms",
            "lower_is_better",
            pair_count,
            None,
        ),
        (
            "agentic.recovery_retry_amplification_threshold",
            "Frozen maximum retry amplification",
            reduced.maximum_retry_amplification_threshold,
            "ratio",
            "lower_is_better",
            pair_count,
            None,
        ),
        (
            "agentic.recovery_distinct_seed_count",
            "Distinct live fault-recovery seeds",
            float(reduced.distinct_seed_count) if reduced.pair_count else None,
            "seeds",
            "higher_is_better",
            pair_count,
            None,
        ),
        (
            "agentic.recovery_minimum_distinct_seed_count",
            "Frozen minimum distinct fault-recovery seeds",
            (
                float(reduced.minimum_distinct_seed_count)
                if reduced.minimum_distinct_seed_count is not None
                else None
            ),
            "seeds",
            "higher_is_better",
            pair_count,
            None,
        ),
    )


def _recovery_metric_specs(
    reduced: RecoveryReduction,
) -> tuple[_RecoveryMetricSpec, ...]:
    return (*_recovery_outcome_specs(reduced), *_recovery_guardrail_specs(reduced))


def recovery_metrics(records: list[ExecutionRecord]) -> list[MetricDraft]:
    reduced = reduce_recovery(records)
    return [
        build_metric(
            metric_id, name, "agentic", value, unit, direction, sample_count
        ).model_copy(update={"confidence_interval": interval})
        for metric_id, name, value, unit, direction, sample_count, interval in (
            _recovery_metric_specs(reduced)
        )
    ]
