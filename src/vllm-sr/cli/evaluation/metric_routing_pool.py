"""Routing, model-pool, and joint system metric reducers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import log2

from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.metric_core import _mean, _metric, percentile
from cli.evaluation.reporting import EvaluationMetric


def _routing(records: list[ExecutionRecord]) -> list[EvaluationMetric]:
    total = len(records)
    covered = sum(record.selected_arm_id is not None for record in records)
    graded = [record.quality for record in records if record.quality is not None]
    latencies = [
        record.latency_ms for record in records if record.latency_ms is not None
    ]
    return [
        _metric(
            "routing.coverage",
            "Routing coverage",
            "routing",
            covered / total if total else None,
            "fraction",
            "higher_is_better",
            total,
        ),
        _metric(
            "routing.abstention_rate",
            "Routing abstention rate",
            "routing",
            (total - covered) / total if total else None,
            "fraction",
            "lower_is_better",
            total,
        ),
        _metric(
            "routing.accuracy",
            "Routing accuracy",
            "routing",
            sum(graded) / len(graded) if graded else None,
            "fraction",
            "higher_is_better",
            len(graded),
        ),
        _metric(
            "routing.fallback_rate",
            "Fallback rate",
            "routing",
            sum(bool(row.fallback) for row in records) / total if total else None,
            "fraction",
            "lower_is_better",
            total,
        ),
        _metric(
            "routing.latency_p50_ms",
            "Route latency p50",
            "routing",
            percentile(latencies, 0.50),
            "ms",
            "lower_is_better",
            len(latencies),
        ),
        _metric(
            "routing.latency_p95_ms",
            "Route latency p95",
            "routing",
            percentile(latencies, 0.95),
            "ms",
            "lower_is_better",
            len(latencies),
        ),
    ]


@dataclass(frozen=True)
class _PoolStats:
    by_case: dict[str, list[ExecutionRecord]]
    by_arm: dict[str, list[ExecutionRecord]]
    oracle_values: list[float]
    unique_wins: int
    marginal: dict[str, list[float]]
    best_single: float | None
    oracle_quality: float | None
    selected: list[str]
    selection_counts: dict[str, int]
    selection_entropy: float | None


def _pool_stats(
    records: list[ExecutionRecord], joint_records: list[ExecutionRecord]
) -> _PoolStats:
    by_case: dict[str, list[ExecutionRecord]] = defaultdict(list)
    by_arm: dict[str, list[ExecutionRecord]] = defaultdict(list)
    for record in records:
        by_case[record.case_id].append(record)
        if record.arm_id:
            by_arm[record.arm_id].append(record)
    oracle_values: list[float] = []
    unique_wins = 0
    marginal: dict[str, list[float]] = defaultdict(list)
    for rows in by_case.values():
        qualified = [row for row in rows if row.success and row.quality is not None]
        if not qualified:
            continue
        best = max(row.quality or 0 for row in qualified)
        oracle_values.append(best)
        if sum((row.quality or 0) == best for row in qualified) == 1:
            unique_wins += 1
        for arm_id in by_arm:
            alternatives = [
                row.quality
                for row in qualified
                if row.arm_id != arm_id and row.quality is not None
            ]
            if alternatives:
                marginal[arm_id].append(best - max(alternatives))
    arm_quality = {
        arm_id: _mean(row.quality for row in rows if row.quality is not None)[0]
        for arm_id, rows in by_arm.items()
    }
    best_single = max(
        (quality for quality in arm_quality.values() if quality is not None),
        default=None,
    )
    oracle_quality = sum(oracle_values) / len(oracle_values) if oracle_values else None
    selected = [
        row.selected_arm_id
        for row in joint_records
        if row.selected_arm_id and row.selected_arm_id in by_arm
    ]
    selection_counts = {arm_id: selected.count(arm_id) for arm_id in set(selected)}
    selection_entropy = (
        -sum(
            (count / len(selected)) * log2(count / len(selected))
            for count in selection_counts.values()
        )
        if selected
        else None
    )
    return _PoolStats(
        by_case=dict(by_case),
        by_arm=dict(by_arm),
        oracle_values=oracle_values,
        unique_wins=unique_wins,
        marginal=dict(marginal),
        best_single=best_single,
        oracle_quality=oracle_quality,
        selected=selected,
        selection_counts=selection_counts,
        selection_entropy=selection_entropy,
    )


def _pool_summary_metrics(stats: _PoolStats) -> list[EvaluationMetric]:
    return [
        _metric(
            "model_pool.oracle_quality",
            "Pool oracle quality",
            "model_pool",
            stats.oracle_quality,
            "score",
            "higher_is_better",
            len(stats.oracle_values),
        ),
        _metric(
            "model_pool.unique_wins",
            "Cases with a unique winning arm",
            "model_pool",
            float(stats.unique_wins) if stats.by_case else None,
            "cases",
            "higher_is_better",
            len(stats.by_case),
        ),
        _metric(
            "model_pool.unique_win_rate",
            "Unique-win case rate",
            "model_pool",
            (
                stats.unique_wins / len(stats.oracle_values)
                if stats.oracle_values
                else None
            ),
            "fraction",
            "higher_is_better",
            len(stats.oracle_values),
        ),
        _metric(
            "model_pool.best_single_quality",
            "Best single-arm quality",
            "model_pool",
            stats.best_single,
            "score",
            "higher_is_better",
            len(stats.by_case),
        ),
        _metric(
            "model_pool.oracle_gain",
            "Oracle gain over best single arm",
            "model_pool",
            (
                stats.oracle_quality - stats.best_single
                if stats.oracle_quality is not None and stats.best_single is not None
                else None
            ),
            "score",
            "higher_is_better",
            len(stats.oracle_values),
        ),
        _metric(
            "model_pool.selection_entropy_bits",
            "Arm selection entropy",
            "model_pool",
            stats.selection_entropy,
            "bits",
            "target",
            len(stats.selected),
        ),
        _metric(
            "model_pool.selection_arm_coverage",
            "Selected-arm coverage",
            "model_pool",
            (len(stats.selection_counts) / len(stats.by_arm) if stats.by_arm else None),
            "fraction",
            "higher_is_better",
            len(stats.selected),
        ),
    ]


def _pool_arm_metrics(stats: _PoolStats) -> list[EvaluationMetric]:
    metrics: list[EvaluationMetric] = []
    for arm_id in sorted(stats.by_arm):
        rows = stats.by_arm[arm_id]
        quality, quality_count = _mean(
            row.quality for row in rows if row.quality is not None
        )
        success, success_count = _mean(
            float(bool(row.success)) for row in rows if row.success is not None
        )
        metrics.extend(
            [
                _metric(
                    f"model_pool.arm.{arm_id}.quality",
                    f"{arm_id} quality",
                    "model_pool",
                    quality,
                    "score",
                    "higher_is_better",
                    quality_count,
                ),
                _metric(
                    f"model_pool.arm.{arm_id}.success_rate",
                    f"{arm_id} success rate",
                    "model_pool",
                    success,
                    "fraction",
                    "higher_is_better",
                    success_count,
                ),
                _metric(
                    f"model_pool.arm.{arm_id}.marginal_contribution",
                    f"{arm_id} marginal contribution",
                    "model_pool",
                    (
                        sum(stats.marginal.get(arm_id, []))
                        / len(stats.marginal[arm_id])
                        if stats.marginal.get(arm_id)
                        else None
                    ),
                    "score",
                    "higher_is_better",
                    len(stats.marginal.get(arm_id, [])),
                ),
            ]
        )
    return metrics


def _model_pool(
    records: list[ExecutionRecord], joint_records: list[ExecutionRecord]
) -> list[EvaluationMetric]:
    stats = _pool_stats(records, joint_records)
    return _pool_summary_metrics(stats) + _pool_arm_metrics(stats)


def _joint(
    records: list[ExecutionRecord], pool_records: list[ExecutionRecord]
) -> list[EvaluationMetric]:
    realized, count = _mean(row.quality for row in records if row.quality is not None)
    by_case: dict[str, list[float]] = defaultdict(list)
    for row in pool_records:
        if row.success and row.quality is not None:
            by_case[row.case_id].append(row.quality)
    oracle_by_case = {
        case_id: max(values) for case_id, values in by_case.items() if values
    }
    regrets = [
        oracle_by_case[row.case_id] - row.quality
        for row in records
        if row.quality is not None and row.case_id in oracle_by_case
    ]
    normalized_regrets = [
        (oracle_by_case[row.case_id] - row.quality) / oracle_by_case[row.case_id]
        for row in records
        if row.quality is not None
        and row.case_id in oracle_by_case
        and oracle_by_case[row.case_id] > 0
    ]
    reliability, reliability_count = _mean(
        float(bool(row.success)) for row in records if row.success is not None
    )
    latencies = [row.latency_ms for row in records if row.latency_ms is not None]
    return [
        _metric(
            "joint.realized_quality",
            "Realized routing quality",
            "joint",
            realized,
            "score",
            "higher_is_better",
            count,
        ),
        _metric(
            "joint.oracle_regret",
            "Pool-oracle regret",
            "joint",
            sum(regrets) / len(regrets) if regrets else None,
            "score",
            "lower_is_better",
            len(regrets),
        ),
        _metric(
            "joint.normalized_regret",
            "Normalized pool-oracle regret",
            "joint",
            (
                sum(normalized_regrets) / len(normalized_regrets)
                if normalized_regrets
                else None
            ),
            "fraction",
            "lower_is_better",
            len(normalized_regrets),
        ),
        _metric(
            "joint.reliability",
            "End-to-end execution reliability",
            "joint",
            reliability,
            "fraction",
            "higher_is_better",
            reliability_count,
        ),
        _metric(
            "joint.latency_p95_ms",
            "End-to-end latency p95",
            "joint",
            percentile(latencies, 0.95),
            "ms",
            "lower_is_better",
            len(latencies),
        ),
    ]
