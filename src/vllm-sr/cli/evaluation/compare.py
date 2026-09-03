"""Deterministic standalone draft comparison without rerunning workloads."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from cli.evaluation.builtin_executors import DEFAULT_EXECUTOR_REGISTRY
from cli.evaluation.contracts import RunManifest
from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.executor_registry import ExecutorRegistry
from cli.evaluation.paired_statistics import (
    PairedStatisticResult,
    paired_statistic_results,
)
from cli.evaluation.published_bundle import (
    ValidatedReportBundle,
    load_published_report_bundle,
)
from cli.evaluation.reporting import EvaluationMetric
from cli.evaluation.standalone_comparison import StandaloneComparison
from cli.evaluation.store import LocalArtifactStore
from cli.evaluation.worker_report import WorkerReportDraft

_LATENCY_REGRESSION_BUDGET = 0.05


@dataclass(frozen=True)
class _TreatmentSpec:
    primary: str
    allowed: frozenset[str]


_TREATMENT_SPECS = {
    "schema_adapter": _TreatmentSpec("code_revision", frozenset({"code_revision"})),
    "recipe": _TreatmentSpec(
        "policy_snapshot_digest", frozenset({"policy_snapshot_digest"})
    ),
    "selector": _TreatmentSpec("selector_digest", frozenset({"selector_digest"})),
    # Pool membership is an explicit composite: it may necessarily rewrite the
    # candidate binding and candidate-serving topology, but the pool itself is
    # always the required primary delta.
    "model_pool": _TreatmentSpec(
        "pool_snapshot_digest",
        frozenset(
            {
                "pool_snapshot_digest",
                "binding_snapshot_digest",
                "environment_snapshot_digest",
            }
        ),
    ),
    "runtime_capacity": _TreatmentSpec(
        "environment_snapshot_digest", frozenset({"environment_snapshot_digest"})
    ),
    "online_adaptation": _TreatmentSpec(
        "adaptation_digest", frozenset({"adaptation_digest"})
    ),
}


@dataclass(frozen=True)
class _ComparisonEvidence:
    paired: int
    improvements: int
    regressions: int
    latency_over_budget: bool
    paired_interval_count: int
    paired_interval_failed: bool
    paired_interval_passed: bool


def _common_compatibility_issues(
    baseline: WorkerReportDraft, candidate: WorkerReportDraft
) -> list[str]:
    checks = (
        (
            "change_profile",
            baseline.run.change_profile,
            candidate.run.change_profile,
        ),
        ("mode", baseline.run.mode, candidate.run.mode),
        ("target_id", baseline.run.target_id, candidate.run.target_id),
        (
            "mixture_id",
            baseline.run.mixture.id if baseline.run.mixture is not None else None,
            candidate.run.mixture.id if candidate.run.mixture is not None else None,
        ),
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
    return incompatibilities


def _factor_snapshots(
    baseline: WorkerReportDraft, candidate: WorkerReportDraft
) -> tuple[tuple[str, str | None, str | None], ...]:
    baseline_mixture = baseline.run.mixture
    candidate_mixture = candidate.run.mixture
    return (
        (
            "code_revision",
            baseline.provenance.code_revision,
            candidate.provenance.code_revision,
        ),
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
        (
            "selector_digest",
            baseline_mixture.selector_digest if baseline_mixture is not None else None,
            (
                candidate_mixture.selector_digest
                if candidate_mixture is not None
                else None
            ),
        ),
        (
            "adaptation_digest",
            (
                baseline_mixture.adaptation_digest
                if baseline_mixture is not None
                else None
            ),
            (
                candidate_mixture.adaptation_digest
                if candidate_mixture is not None
                else None
            ),
        ),
    )


def _validate_treatment_factors(
    factor_snapshots: tuple[tuple[str, str | None, str | None], ...],
    treatment: _TreatmentSpec,
    incompatibilities: list[str],
) -> None:
    changed_factors: set[str] = set()
    for name, old, new in factor_snapshots:
        if old is None and new is None:
            if name == treatment.primary:
                incompatibilities.append(f"{name} snapshot unavailable")
            continue
        if not old or not new or (old != new and name not in treatment.allowed):
            incompatibilities.append(name)
        elif old != new:
            changed_factors.add(name)
    if treatment.primary not in changed_factors:
        incompatibilities.append(f"{treatment.primary} treatment factor did not change")


def _validate_compatibility(
    baseline: WorkerReportDraft, candidate: WorkerReportDraft
) -> None:
    if baseline.run.id == candidate.run.id:
        raise ValueError("a run cannot be compared with itself")
    if candidate.run.baseline_run_id != baseline.run.id:
        raise ValueError(
            "candidate baseline_run_id must reference the selected baseline run"
        )
    incompatibilities = _common_compatibility_issues(baseline, candidate)
    treatment = _TREATMENT_SPECS.get(baseline.run.change_profile)
    if treatment is None:
        raise ValueError(
            "reports are not paired-comparable; change_profile "
            f"{baseline.run.change_profile!r} has no independent server-owned "
            "treatment factor"
        )
    _validate_treatment_factors(
        _factor_snapshots(baseline, candidate),
        treatment,
        incompatibilities,
    )

    incompatibilities = list(dict.fromkeys(incompatibilities))
    if incompatibilities:
        raise ValueError(
            "reports are not paired-comparable; incompatible "
            + ", ".join(incompatibilities)
        )


def _has_complete_qualified_track_vector(report: WorkerReportDraft) -> bool:
    return all(
        track.evidence_level != "E0"
        and track.status == "completed"
        and track.coverage.unavailable in {None, 0}
        for track in report.tracks
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
    baseline: WorkerReportDraft,
    candidate: WorkerReportDraft,
    paired_statistics: tuple[PairedStatisticResult, ...],
) -> tuple[list[EvaluationMetric], _ComparisonEvidence]:
    baseline_by_id = {metric.id: metric for metric in baseline.metrics}
    statistics_by_id = {
        statistic.metric_id: statistic for statistic in paired_statistics
    }
    compared: list[EvaluationMetric] = []
    paired = improvements = regressions = 0
    latency_over_budget = False
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
        if "latency" in metric.id.casefold():
            latency_over_budget |= _over_budget(old.value, metric.value)
        statistic = statistics_by_id.get(metric.id)
        update: dict[str, object] = {"baseline_value": old.value, "delta": delta}
        if statistic is not None:
            update.update(
                confidence_interval=statistic.confidence_interval,
                sample_count=statistic.sample_count,
            )
        compared.append(metric.model_copy(update=update))
    qualified = [
        statistic
        for statistic in paired_statistics
        if statistic.confidence_interval is not None
    ]

    def regressed(statistic: PairedStatisticResult) -> bool:
        lower, upper = statistic.confidence_interval or (0.0, 0.0)
        if statistic.direction == "lower_is_better":
            return lower > 0
        return upper < 0

    def passed(statistic: PairedStatisticResult) -> bool:
        lower, upper = statistic.confidence_interval or (0.0, 0.0)
        if statistic.direction == "lower_is_better":
            return upper <= 0
        return lower >= 0

    return compared, _ComparisonEvidence(
        paired=paired,
        improvements=improvements,
        regressions=regressions,
        latency_over_budget=latency_over_budget,
        paired_interval_count=len(qualified),
        paired_interval_failed=any(regressed(statistic) for statistic in qualified),
        paired_interval_passed=bool(qualified)
        and all(passed(statistic) for statistic in qualified),
    )


def _comparison_failure_reason(
    baseline: WorkerReportDraft,
    candidate: WorkerReportDraft,
    evidence: _ComparisonEvidence,
) -> str | None:
    required = [gate for gate in candidate.gates if gate.disposition == "required"]
    if any(gate.verdict == "fail" for gate in required):
        return "A required candidate gate failed."
    if candidate.summary.verdict == "fail":
        return "The candidate report failed."
    if evidence.paired_interval_failed:
        return "A registered paired statistic regressed with 95% confidence."
    summary_latency_over_budget = (
        baseline.summary.latency_p95_ms is not None
        and candidate.summary.latency_p95_ms is not None
        and _over_budget(
            baseline.summary.latency_p95_ms,
            candidate.summary.latency_p95_ms,
        )
    )
    if evidence.latency_over_budget or summary_latency_over_budget:
        return "Tail latency exceeded the 5% paired regression budget."
    return None


def _comparison_verdict(
    baseline: WorkerReportDraft,
    candidate: WorkerReportDraft,
    evidence: _ComparisonEvidence,
) -> tuple[str, str]:
    failure_reason = _comparison_failure_reason(baseline, candidate, evidence)
    if failure_reason is not None:
        return "fail", failure_reason
    if not _has_complete_qualified_track_vector(
        baseline
    ) or not _has_complete_qualified_track_vector(candidate):
        return (
            "unavailable",
            "Every selected track needs complete qualified evidence for paired promotion.",
        )
    required = [gate for gate in candidate.gates if gate.disposition == "required"]
    required_unavailable = any(gate.verdict == "unavailable" for gate in required)
    if candidate.summary.verdict == "unavailable" or required_unavailable:
        return "unavailable", "Required candidate evidence is unavailable."
    if evidence.paired_interval_passed:
        return "pass", "Registered case-aligned paired confidence intervals passed."
    if evidence.paired_interval_count == 0:
        return "unavailable", "No registered paired statistic has a valid interval."
    return (
        "unavailable",
        "A registered paired confidence interval crosses the promotion boundary.",
    )


def compare_worker_drafts(
    baseline: WorkerReportDraft,
    candidate: WorkerReportDraft,
    baseline_records: list[ExecutionRecord],
    candidate_records: list[ExecutionRecord],
) -> StandaloneComparison:
    _validate_compatibility(baseline, candidate)
    paired_statistics = paired_statistic_results(
        baseline_records, candidate_records, seed=candidate.run.seed
    )
    compared, evidence = _paired_metrics(baseline, candidate, paired_statistics)
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
    return StandaloneComparison(
        baseline_run_id=baseline.run.id,
        candidate_run_id=candidate.run.id,
        verdict=verdict,
        summary=summary,
        metrics=tuple(compared),
        gates=candidate.gates,
        recommendations=recommendations,
        created_at=datetime.now(timezone.utc),
    )


def _load_validated_bundle(
    store: LocalArtifactStore,
    run_id: str,
    executor_registry: ExecutorRegistry,
) -> ValidatedReportBundle:
    manifest = RunManifest.model_validate(
        store.read_run_json(run_id, "run-manifest.json")
    )
    if manifest.run_id != run_id:
        raise ValueError("manifest identity does not match its run bundle")
    executor_id = next(iter(manifest.suite_executors.values()))
    bundle = load_published_report_bundle(
        manifest,
        store,
        executor_registry.contract(executor_id),
    )
    if bundle is None:
        raise ValueError("published report bundle does not exist")
    return bundle


def compare_runs(
    store: LocalArtifactStore,
    baseline_run_id: str,
    candidate_run_id: str,
    *,
    executor_registry: ExecutorRegistry = DEFAULT_EXECUTOR_REGISTRY,
) -> StandaloneComparison:
    """Compare two local draft bundles using their private aligned evidence."""

    baseline_bundle = _load_validated_bundle(
        store,
        baseline_run_id,
        executor_registry,
    )
    candidate_bundle = _load_validated_bundle(
        store,
        candidate_run_id,
        executor_registry,
    )
    baseline = baseline_bundle.report
    candidate = candidate_bundle.report
    if baseline.run.id != baseline_run_id or candidate.run.id != candidate_run_id:
        raise ValueError("report identity does not match its run bundle")
    return compare_worker_drafts(
        baseline,
        candidate,
        list(baseline_bundle.records),
        list(candidate_bundle.records),
    )
