"""Registered deterministic statistics over case-aligned private records."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.metric_model_pool import outcome_quality
from cli.evaluation.statistics import bootstrap_interval

_Value = Callable[[ExecutionRecord], float | None]


@dataclass(frozen=True)
class PairedStatistic:
    metric_id: str
    track_id: str
    direction: Literal["higher_is_better", "lower_is_better"]
    analysis_unit: Literal[
        "record", "case_max", "case_oracle_regret", "case_normalized_regret"
    ]
    value: _Value


@dataclass(frozen=True)
class PairedStatisticResult:
    metric_id: str
    baseline_value: float
    candidate_value: float
    delta: float
    confidence_interval: tuple[float, float] | None
    sample_count: int
    direction: Literal["higher_is_better", "lower_is_better"]


def _quality(record: ExecutionRecord) -> float | None:
    return outcome_quality(record) if record.status != "unavailable" else None


def _successful_quality(record: ExecutionRecord) -> float | None:
    return _quality(record)


def _success(record: ExecutionRecord) -> float | None:
    if record.status == "unavailable" or record.success is None:
        return None
    return float(record.success)


def _preference(record: ExecutionRecord) -> float | None:
    if record.status == "unavailable" or record.preference_match is None:
        return None
    return float(record.preference_match)


def _violation(record: ExecutionRecord) -> float | None:
    if record.status == "unavailable" or record.safety_violations is None:
        return None
    return float(record.safety_violations > 0)


_STATISTICS = (
    PairedStatistic(
        "routing.accuracy", "routing", "higher_is_better", "record", _quality
    ),
    PairedStatistic(
        "model_pool.oracle_quality",
        "model_pool",
        "higher_is_better",
        "case_max",
        _successful_quality,
    ),
    PairedStatistic(
        "joint.realized_quality", "joint", "higher_is_better", "record", _quality
    ),
    PairedStatistic(
        "joint.reliability", "joint", "higher_is_better", "record", _success
    ),
    PairedStatistic(
        "joint.oracle_regret",
        "joint",
        "lower_is_better",
        "case_oracle_regret",
        _quality,
    ),
    PairedStatistic(
        "joint.normalized_regret",
        "joint",
        "lower_is_better",
        "case_normalized_regret",
        _quality,
    ),
    PairedStatistic(
        "agentic.task_score", "agentic", "higher_is_better", "record", _quality
    ),
    PairedStatistic(
        "agentic.success_rate", "agentic", "higher_is_better", "record", _success
    ),
    PairedStatistic(
        "multimodal.quality", "multimodal", "higher_is_better", "record", _quality
    ),
    PairedStatistic(
        "preference.agreement",
        "preference",
        "higher_is_better",
        "record",
        _preference,
    ),
    PairedStatistic(
        "safety.violation_case_rate",
        "safety",
        "lower_is_better",
        "record",
        _violation,
    ),
)
PAIRED_STATISTIC_REGISTRY = MappingProxyType(
    {statistic.metric_id: statistic for statistic in _STATISTICS}
)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _records_by_id(
    records: list[ExecutionRecord], label: str
) -> dict[str, ExecutionRecord]:
    indexed = {record.id: record for record in records}
    if len(indexed) != len(records):
        raise ValueError(f"{label} private records contain duplicate ids")
    return indexed


def aligned_record_pairs(
    baseline: list[ExecutionRecord], candidate: list[ExecutionRecord]
) -> tuple[tuple[ExecutionRecord, ExecutionRecord], ...]:
    """Require exact analysis-unit alignment before computing any delta."""

    baseline_by_id = _records_by_id(baseline, "baseline")
    candidate_by_id = _records_by_id(candidate, "candidate")
    if set(baseline_by_id) != set(candidate_by_id):
        raise ValueError("private records are not case-aligned")
    pairs: list[tuple[ExecutionRecord, ExecutionRecord]] = []
    for record_id in sorted(baseline_by_id):
        old = baseline_by_id[record_id]
        new = candidate_by_id[record_id]
        old_identity = (old.track_id, old.case_id, old.attempt_id, old.arm_id)
        new_identity = (new.track_id, new.case_id, new.attempt_id, new.arm_id)
        if old_identity != new_identity:
            raise ValueError("private record analysis identities do not match")
        pairs.append((old, new))
    return tuple(pairs)


def _analysis_unit_values(
    statistic: PairedStatistic,
    pairs: tuple[tuple[ExecutionRecord, ExecutionRecord], ...],
) -> list[tuple[float, float]]:
    if statistic.analysis_unit in {
        "case_oracle_regret",
        "case_normalized_regret",
    }:
        return _case_regret_values(
            pairs, normalized=statistic.analysis_unit == "case_normalized_regret"
        )
    eligible: list[tuple[ExecutionRecord, ExecutionRecord, float, float]] = []
    for old, new in pairs:
        if old.track_id != statistic.track_id:
            continue
        old_value = statistic.value(old)
        new_value = statistic.value(new)
        if old_value is not None and new_value is not None:
            eligible.append((old, new, old_value, new_value))
    if statistic.analysis_unit == "record":
        return [(old_value, new_value) for _, _, old_value, new_value in eligible]

    by_case: dict[str, list[tuple[float, float]]] = {}
    for old, _, old_value, new_value in eligible:
        by_case.setdefault(old.case_id, []).append((old_value, new_value))
    return [
        (
            max(old_value for old_value, _ in case_values),
            max(new_value for _, new_value in case_values),
        )
        for _, case_values in sorted(by_case.items())
    ]


def _case_regret_values(
    pairs: tuple[tuple[ExecutionRecord, ExecutionRecord], ...],
    *,
    normalized: bool,
) -> list[tuple[float, float]]:
    by_case: dict[str, list[tuple[ExecutionRecord, ExecutionRecord]]] = {}
    for old, new in pairs:
        if old.track_id in {"model_pool", "joint"}:
            by_case.setdefault(old.case_id, []).append((old, new))
    values: list[tuple[float, float]] = []
    for _, case_pairs in sorted(by_case.items()):
        old_oracle = [
            quality
            for old, _ in case_pairs
            if old.track_id == "model_pool" and (quality := _quality(old)) is not None
        ]
        new_oracle = [
            quality
            for _, new in case_pairs
            if new.track_id == "model_pool" and (quality := _quality(new)) is not None
        ]
        old_realized = [
            quality
            for old, _ in case_pairs
            if old.track_id == "joint" and (quality := _quality(old)) is not None
        ]
        new_realized = [
            quality
            for _, new in case_pairs
            if new.track_id == "joint" and (quality := _quality(new)) is not None
        ]
        if not old_oracle or not new_oracle:
            continue
        if len(old_realized) != 1 or len(new_realized) != 1:
            if old_realized or new_realized:
                raise ValueError("joint regret requires one realized record per case")
            continue
        old_best, new_best = max(old_oracle), max(new_oracle)
        if normalized and (old_best <= 0 or new_best <= 0):
            continue
        # A finite sampled pool is a lower bound on the true stochastic model
        # support. Realized quality above that sample is zero shortfall, not a
        # negative regret that can offset genuine failures.
        old_regret = max(0.0, old_best - old_realized[0])
        new_regret = max(0.0, new_best - new_realized[0])
        if normalized:
            old_regret /= old_best
            new_regret /= new_best
        values.append((old_regret, new_regret))
    return values


def paired_statistic_results(
    baseline: list[ExecutionRecord],
    candidate: list[ExecutionRecord],
    *,
    seed: int,
) -> tuple[PairedStatisticResult, ...]:
    pairs = aligned_record_pairs(baseline, candidate)
    results: list[PairedStatisticResult] = []
    for statistic in _STATISTICS:
        values = _analysis_unit_values(statistic, pairs)
        if not values:
            continue
        deltas = [new - old for old, new in values]
        metric_seed = seed + int.from_bytes(
            hashlib.sha256(statistic.metric_id.encode()).digest()[:4], "big"
        )
        results.append(
            PairedStatisticResult(
                metric_id=statistic.metric_id,
                baseline_value=_mean([old for old, _ in values]),
                candidate_value=_mean([new for _, new in values]),
                delta=_mean(deltas),
                confidence_interval=bootstrap_interval(deltas, _mean, seed=metric_seed),
                sample_count=len(values),
                direction=statistic.direction,
            )
        )
    return tuple(results)
