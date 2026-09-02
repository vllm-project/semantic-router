"""Deterministic report comparison without rerunning workloads."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from cli.evaluation.reporting import (
    EvaluationComparison,
    EvaluationMetric,
    EvaluationReport,
)

_LATENCY_REGRESSION_BUDGET = 0.05
_PRIMARY_METRICS = frozenset(
    {
        "routing.accuracy",
        "model_pool.oracle_quality",
        "joint.realized_quality",
        "joint.oracle_regret",
        "joint.normalized_regret",
        "joint.reliability",
    }
)

_ALLOWED_TREATMENT_FACTORS = {
    "schema_adapter": frozenset(),
    "recipe": frozenset({"policy_snapshot_digest", "binding_snapshot_digest"}),
    "selector": frozenset({"policy_snapshot_digest", "binding_snapshot_digest"}),
    "model_pool": frozenset({"pool_snapshot_digest", "binding_snapshot_digest"}),
    "runtime_capacity": frozenset({"environment_snapshot_digest"}),
    "agent_multimodal": frozenset(
        {"policy_snapshot_digest", "binding_snapshot_digest"}
    ),
    "online_adaptation": frozenset(
        {"policy_snapshot_digest", "binding_snapshot_digest"}
    ),
}


@dataclass(frozen=True)
class _ComparisonEvidence:
    paired: int
    improvements: int
    regressions: int
    primary_regression: bool
    latency_over_budget: bool


def _validate_compatibility(
    baseline: EvaluationReport, candidate: EvaluationReport
) -> None:
    if baseline.run.id == candidate.run.id:
        raise ValueError("a run cannot be compared with itself")
    if candidate.run.baseline_run_id != baseline.run.id:
        raise ValueError(
            "candidate baseline_run_id must reference the selected baseline run"
        )
    checks = (
        (
            "change_profile",
            baseline.run.change_profile,
            candidate.run.change_profile,
        ),
        ("mode", baseline.run.mode, candidate.run.mode),
        ("target_id", baseline.run.target_id, candidate.run.target_id),
        ("suite_ids", set(baseline.run.suite_ids), set(candidate.run.suite_ids)),
        ("track_ids", set(baseline.run.track_ids), set(candidate.run.track_ids)),
        ("sample_limit", baseline.run.sample_limit, candidate.run.sample_limit),
        ("concurrency", baseline.run.concurrency, candidate.run.concurrency),
        (
            "workload_snapshot_digest",
            baseline.provenance.workload_snapshot_digest,
            candidate.provenance.workload_snapshot_digest,
        ),
        (
            "benchmark_revisions",
            baseline.provenance.benchmark_revisions,
            candidate.provenance.benchmark_revisions,
        ),
        ("seed", baseline.run.seed, candidate.run.seed),
    )
    incompatibilities = [name for name, old, new in checks if old != new]
    if not baseline.provenance.workload_snapshot_digest:
        incompatibilities.append("workload_snapshot_digest")
    if not baseline.provenance.benchmark_revisions:
        incompatibilities.append("benchmark_revisions")
    factor_snapshots = (
        (
            "policy_snapshot_digest",
            baseline.provenance.policy_snapshot_digest,
            candidate.provenance.policy_snapshot_digest,
        ),
        (
            "binding_snapshot_digest",
            baseline.provenance.binding_snapshot_digest,
            candidate.provenance.binding_snapshot_digest,
        ),
        (
            "pool_snapshot_digest",
            baseline.provenance.pool_snapshot_digest,
            candidate.provenance.pool_snapshot_digest,
        ),
        (
            "environment_snapshot_digest",
            baseline.provenance.environment_snapshot_digest,
            candidate.provenance.environment_snapshot_digest,
        ),
    )
    allowed_changes = _ALLOWED_TREATMENT_FACTORS[baseline.run.change_profile]
    changed_factors: set[str] = set()
    for name, old, new in factor_snapshots:
        if not old or not new or (old != new and name not in allowed_changes):
            incompatibilities.append(name)
        elif old != new:
            changed_factors.add(name)
    if baseline.run.change_profile == "schema_adapter":
        if baseline.provenance.code_revision == candidate.provenance.code_revision:
            incompatibilities.append("code_revision treatment")
    elif not changed_factors:
        incompatibilities.append(
            f"{baseline.run.change_profile} treatment factor did not change"
        )
    incompatibilities = list(dict.fromkeys(incompatibilities))
    if incompatibilities:
        raise ValueError(
            "reports are not paired-comparable; incompatible "
            + ", ".join(incompatibilities)
        )


def _is_improvement(metric: EvaluationMetric, delta: float) -> bool:
    positive = delta > 0
    return not positive if metric.direction == "lower_is_better" else positive


def _over_budget(baseline: float, candidate: float) -> bool:
    if not math.isfinite(baseline) or not math.isfinite(candidate):
        return True
    if baseline <= 0:
        return candidate > baseline
    return candidate > baseline * (1 + _LATENCY_REGRESSION_BUDGET)


def _paired_metrics(
    baseline: EvaluationReport, candidate: EvaluationReport
) -> tuple[list[EvaluationMetric], _ComparisonEvidence]:
    baseline_by_id = {metric.id: metric for metric in baseline.metrics}
    compared: list[EvaluationMetric] = []
    paired = improvements = regressions = 0
    primary_regression = latency_over_budget = False
    for metric in candidate.metrics:
        old = baseline_by_id.get(metric.id)
        direction_matches = bool(
            old
            and (
                not old.direction
                or not metric.direction
                or old.direction == metric.direction
            )
        )
        if (
            not old
            or old.value is None
            or metric.value is None
            or not direction_matches
        ):
            compared.append(metric)
            continue
        delta = metric.value - old.value
        paired += 1
        if delta and metric.direction not in {None, "target"}:
            if _is_improvement(metric, delta):
                improvements += 1
            else:
                regressions += 1
                primary_regression |= metric.id in _PRIMARY_METRICS
        if "latency" in metric.id.casefold():
            latency_over_budget |= _over_budget(old.value, metric.value)
        compared.append(
            metric.model_copy(update={"baseline_value": old.value, "delta": delta})
        )
    return compared, _ComparisonEvidence(
        paired=paired,
        improvements=improvements,
        regressions=regressions,
        primary_regression=primary_regression,
        latency_over_budget=latency_over_budget,
    )


def _comparison_verdict(
    baseline: EvaluationReport,
    candidate: EvaluationReport,
    evidence: _ComparisonEvidence,
) -> tuple[str, str]:
    required = [gate for gate in candidate.gates if gate.disposition == "required"]
    if any(gate.verdict == "fail" for gate in required):
        return "fail", "A required candidate gate failed."
    required_unavailable = any(gate.verdict == "unavailable" for gate in required)
    if candidate.summary.verdict == "fail":
        return "fail", "The candidate report failed."
    quality_regressed = (
        baseline.summary.quality_score is not None
        and candidate.summary.quality_score is not None
        and candidate.summary.quality_score < baseline.summary.quality_score
    )
    if evidence.primary_regression or quality_regressed:
        return "fail", "A primary quality or joint-system metric regressed."
    summary_latency_over_budget = (
        baseline.summary.latency_p95_ms is not None
        and candidate.summary.latency_p95_ms is not None
        and _over_budget(
            baseline.summary.latency_p95_ms,
            candidate.summary.latency_p95_ms,
        )
    )
    if evidence.latency_over_budget or summary_latency_over_budget:
        return "fail", "Tail latency exceeded the 5% paired regression budget."
    if candidate.summary.verdict == "unavailable" or required_unavailable:
        return "unavailable", "Required candidate evidence is unavailable."
    if evidence.paired == 0:
        return "unavailable", "No direction-aware paired metric evidence is available."
    return (
        "unavailable",
        "Aggregate point deltas are diagnostic only; a case-aligned paired delta confidence interval is required for promotion.",
    )


def compare_reports(
    baseline: EvaluationReport, candidate: EvaluationReport
) -> EvaluationComparison:
    _validate_compatibility(baseline, candidate)
    compared, evidence = _paired_metrics(baseline, candidate)
    verdict, reason = _comparison_verdict(baseline, candidate, evidence)
    summary = (
        f"Compared {len(compared)} metrics ({evidence.paired} paired): "
        f"{evidence.improvements} improved, {evidence.regressions} regressed. "
        f"{reason}"
    )
    if verdict == "fail":
        recommendations = (
            "Do not promote until required gates and promotion-critical regressions are resolved.",
        )
    elif verdict == "unavailable":
        recommendations = (
            "Collect complete profile-qualified workload, treatment-factor, and benchmark snapshots before promotion.",
        )
    elif evidence.regressions:
        recommendations = (
            "Promotion-critical budgets passed; review advisory metric regressions before rollout.",
        )
    else:
        recommendations = (
            "Paired promotion-critical evidence passed without a detected regression.",
        )
    return EvaluationComparison(
        baseline_run_id=baseline.run.id,
        candidate_run_id=candidate.run.id,
        verdict=verdict,
        summary=summary,
        metrics=tuple(compared),
        gates=candidate.gates,
        recommendations=recommendations,
        created_at=datetime.now(timezone.utc),
    )
