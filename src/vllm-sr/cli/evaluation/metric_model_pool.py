"""Server-parity model-pool metric reduction over a frozen dense cohort."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from math import inf, isfinite, log2

from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.metric_core import MetricDraft, build_metric
from cli.evaluation.metric_model_pool_contract import (
    ARM_MEASURES as _ARM_MEASURES,
)
from cli.evaluation.metric_model_pool_contract import (
    MIN_DENSE_POOL_ARMS as _MIN_DENSE_POOL_ARMS,
)
from cli.evaluation.metric_model_pool_contract import (
    ModelPoolReductionContext,
    build_dense_model_pool_matrix,
    model_pool_arm_metric_id,
)
from cli.evaluation.metric_model_pool_metadata import (
    metric_metadata as _metric_metadata,
)

_NON_AUTHORITATIVE = "non_authoritative"
_MISSING_ARM_CELL = "missing_arm_cell"
_UNGRADED_SUCCESS = "ungraded_success"
_UNAVAILABLE_RECORD = "unavailable_record"
_MISSING_RUNTIME_COST = "missing_runtime_cost"
_MISSING_SELECTION = "missing_joint_selection"

_STATIC_METRIC_IDS = (
    "model_pool.all_arm_failure_rate",
    "model_pool.arm_count",
    "model_pool.best_single_quality",
    "model_pool.mean_pairwise_failure_jaccard",
    "model_pool.oracle_gain",
    "model_pool.oracle_quality",
    "model_pool.pareto_dominated_arm_count",
    "model_pool.pareto_evaluable_arm_count",
    "model_pool.quality_cost_shared_support_cases",
    "model_pool.quality_cost_shared_support_fraction",
    "model_pool.quality_dominated_arm_count",
    "model_pool.quality_shared_support_cases",
    "model_pool.quality_shared_support_fraction",
    "model_pool.selection_arm_coverage",
    "model_pool.selection_entropy_bits",
    "model_pool.unique_win_rate",
    "model_pool.unique_wins",
    "model_pool.worst_arm_reliability",
)


def outcome_quality(record: ExecutionRecord) -> float | None:
    """Treat execution failure as observed zero and ungraded success as missing."""

    if record.status == "failed" or record.success is False:
        return 0.0
    if record.status == "unavailable":
        return None
    return record.quality


@dataclass(frozen=True)
class _PoolCell:
    success_known: bool
    success: bool
    quality: float | None
    runtime_cost: float | None


@dataclass(frozen=True)
class _ReducedMetric:
    value: float | None
    sample_count: int
    missing_reasons: Counter[str]


@dataclass(frozen=True)
class _SupportSummary:
    cells: dict[str, dict[str, _PoolCell]]
    quality_reasons: Counter[str]
    success_reasons: Counter[str]
    cost_reasons: Counter[str]
    quality_complete_cases: tuple[str, ...]
    quality_cost_complete_cases: tuple[str, ...]
    success_complete: bool


def _cell_from_record(record: ExecutionRecord) -> _PoolCell:
    runtime_cost = record.runtime_cost
    if runtime_cost is not None and (not isfinite(runtime_cost) or runtime_cost < 0):
        raise ValueError("model-pool runtime cost is invalid")
    if record.status == "failed":
        return _PoolCell(True, False, 0.0, runtime_cost)
    if record.status == "succeeded":
        if record.success is None:
            raise ValueError("successful model-pool record omits success")
        if record.success is False:
            return _PoolCell(True, False, 0.0, runtime_cost)
        quality = record.quality
        if quality is not None and (
            not isfinite(quality) or quality < 0 or quality > 1
        ):
            raise ValueError("model-pool quality is invalid")
        return _PoolCell(True, True, quality, runtime_cost)
    if record.status == "unavailable":
        return _PoolCell(False, False, None, runtime_cost)
    raise ValueError("model-pool record status is invalid")


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    total = 0.0
    compensation = 0.0
    for value in values:
        adjusted = value - compensation
        next_total = total + adjusted
        compensation = (next_total - total) - adjusted
        total = next_total
    return total / len(values)


def _metric_ids(arm_ids: tuple[str, ...]) -> tuple[str, ...]:
    dynamic = (
        model_pool_arm_metric_id(arm_id, measure)
        for arm_id in arm_ids
        for measure in _ARM_MEASURES
    )
    return tuple(sorted((*_STATIC_METRIC_IDS, *dynamic)))


def _quality_reduction(
    case_ids: tuple[str, ...],
    arm_ids: tuple[str, ...],
    cells: dict[str, dict[str, _PoolCell]],
) -> tuple[dict[str, float], float, int, dict[str, float]]:
    values_by_arm: dict[str, list[float]] = {arm_id: [] for arm_id in arm_ids}
    marginal_by_arm: dict[str, list[float]] = {arm_id: [] for arm_id in arm_ids}
    oracle_values: list[float] = []
    unique_wins = 0
    for case_id in case_ids:
        best = -inf
        second_best = -inf
        best_count = 0
        for arm_id in arm_ids:
            quality = cells[case_id][arm_id].quality
            if quality is None:  # pragma: no cover - guarded by dense support
                raise ValueError("model-pool quality reduction is not dense")
            values_by_arm[arm_id].append(quality)
            if quality > best:
                second_best, best, best_count = best, quality, 1
            elif quality == best:
                best_count += 1
            elif quality > second_best:
                second_best = quality
        oracle_values.append(best)
        if best_count == 1:
            unique_wins += 1
        for arm_id in arm_ids:
            without = best
            if cells[case_id][arm_id].quality == best and best_count == 1:
                without = second_best
            marginal_by_arm[arm_id].append(best - without)
    quality_by_arm = {arm_id: _mean(values_by_arm[arm_id]) for arm_id in arm_ids}
    marginal = {arm_id: _mean(marginal_by_arm[arm_id]) for arm_id in arm_ids}
    return quality_by_arm, _mean(oracle_values), unique_wins, marginal


def _quality_dominated_arm_count(
    case_ids: tuple[str, ...],
    arm_ids: tuple[str, ...],
    cells: dict[str, dict[str, _PoolCell]],
) -> int:
    dominated = 0
    for candidate in arm_ids:
        for competitor in arm_ids:
            if competitor == candidate:
                continue
            never_worse = True
            strictly_better = False
            for case_id in case_ids:
                candidate_quality = cells[case_id][candidate].quality
                competitor_quality = cells[case_id][competitor].quality
                if candidate_quality is None or competitor_quality is None:
                    raise ValueError("model-pool dominance reduction is not dense")
                if competitor_quality < candidate_quality:
                    never_worse = False
                    break
                strictly_better = strictly_better or (
                    competitor_quality > candidate_quality
                )
            if never_worse and strictly_better:
                dominated += 1
                break
    return dominated


def _failure_jaccard(
    case_ids: tuple[str, ...],
    arm_ids: tuple[str, ...],
    cells: dict[str, dict[str, _PoolCell]],
) -> float:
    overlaps: list[float] = []
    for left, right in combinations(arm_ids, 2):
        intersection = 0
        union = 0
        for case_id in case_ids:
            left_failed = not cells[case_id][left].success
            right_failed = not cells[case_id][right].success
            union += int(left_failed or right_failed)
            intersection += int(left_failed and right_failed)
        overlaps.append(intersection / union if union else 0.0)
    return _mean(overlaps)


def _pareto_counts(
    case_ids: tuple[str, ...],
    arm_ids: tuple[str, ...],
    cells: dict[str, dict[str, _PoolCell]],
) -> tuple[int, int]:
    quality: dict[str, float] = {}
    cost: dict[str, float] = {}
    for arm_id in arm_ids:
        qualities: list[float] = []
        costs: list[float] = []
        for case_id in case_ids:
            cell = cells[case_id][arm_id]
            if cell.quality is None or cell.runtime_cost is None:
                raise ValueError("model-pool Pareto reduction is not dense")
            qualities.append(cell.quality)
            costs.append(cell.runtime_cost)
        quality[arm_id], cost[arm_id] = _mean(qualities), _mean(costs)
    dominated = 0
    for arm_id in arm_ids:
        for competitor in arm_ids:
            if competitor == arm_id:
                continue
            if (
                quality[competitor] >= quality[arm_id]
                and cost[competitor] <= cost[arm_id]
                and (
                    quality[competitor] > quality[arm_id]
                    or cost[competitor] < cost[arm_id]
                )
            ):
                dominated += 1
                break
    return len(arm_ids), dominated


def _selected_arms(
    joint_records: list[ExecutionRecord],
    case_ids: tuple[str, ...],
    arm_ids: tuple[str, ...],
    *,
    strict: bool,
) -> tuple[dict[str, str], Counter[str]]:
    case_set = frozenset(case_ids)
    arm_set = frozenset(arm_ids)
    selected: dict[str, str] = {}
    seen: set[str] = set()
    reasons: Counter[str] = Counter()
    for record in joint_records:
        if record.track_id != "joint" or record.case_id not in case_set:
            raise ValueError(
                "model-pool reducer received joint evidence outside the planned matrix"
            )
        if record.case_id in seen:
            raise ValueError("model-pool reducer received duplicate joint evidence")
        seen.add(record.case_id)
        if record.status == "unavailable":
            if record.selected_arm_id is not None:
                raise ValueError("unavailable joint evidence selects an arm")
            reasons[_MISSING_SELECTION] += 1
            continue
        if record.selected_arm_id not in arm_set:
            if not strict:
                reasons[_MISSING_SELECTION] += 1
                continue
            raise ValueError(
                "model-pool reducer received joint evidence with an invalid selected arm"
            )
        selected[record.case_id] = record.selected_arm_id
    return selected, reasons


def _reduce_selection(
    joint_records: list[ExecutionRecord],
    case_ids: tuple[str, ...],
    arm_ids: tuple[str, ...],
    *,
    strict: bool,
) -> tuple[_ReducedMetric, _ReducedMetric]:
    selected, reasons = _selected_arms(joint_records, case_ids, arm_ids, strict=strict)
    for case_id in case_ids:
        if case_id not in selected:
            reasons[_MISSING_SELECTION] += 1
    if reasons:
        unavailable = _ReducedMetric(None, len(selected), reasons)
        return unavailable, unavailable
    counts = Counter(selected.values())
    entropy = 0.0
    for arm_id in arm_ids:
        count = counts[arm_id]
        if count:
            probability = count / len(case_ids)
            entropy -= probability * log2(probability)
    return (
        _ReducedMetric(entropy, len(case_ids), Counter()),
        _ReducedMetric(len(counts) / len(arm_ids), len(case_ids), Counter()),
    )


def _drafts(
    reduced: dict[str, _ReducedMetric], context: ModelPoolReductionContext
) -> list[MetricDraft]:
    drafts: list[MetricDraft] = []
    for metric_id in _metric_ids(context.frozen_arm_ids):
        metric = reduced[metric_id]
        metadata = _metric_metadata(metric_id)
        drafts.append(
            build_metric(
                metric_id,
                metadata.name,
                "model_pool",
                metric.value,
                metadata.unit,
                metadata.direction,
                metric.sample_count,
                planned_analysis_units=len(context.planned_case_ids),
            ).model_copy(
                update={
                    "model_pool_observed_exclusions": sum(
                        metric.missing_reasons.values()
                    )
                }
            )
        )
    return drafts


def _infer_diagnostic_context(
    records: list[ExecutionRecord], joint_records: list[ExecutionRecord]
) -> ModelPoolReductionContext | None:
    """Preserve direct reducer callers; production always supplies a plan."""

    arm_ids = tuple(
        sorted(
            {
                record.arm_id
                for record in records
                if record.track_id == "model_pool" and record.arm_id is not None
            }
        )
    )
    case_ids = tuple(
        sorted(
            {record.case_id for record in records}
            | {record.case_id for record in joint_records}
        )
    )
    if len(arm_ids) < _MIN_DENSE_POOL_ARMS or not case_ids:
        return None
    return ModelPoolReductionContext(arm_ids, case_ids, authoritative=True)


def _collect_support(
    record_matrix: dict[str, dict[str, ExecutionRecord | None]],
    context: ModelPoolReductionContext,
) -> _SupportSummary:
    cells = {
        case_id: {
            arm_id: _cell_from_record(record)
            for arm_id, record in record_matrix[case_id].items()
            if record is not None
        }
        for case_id in context.planned_case_ids
    }
    quality_reasons: Counter[str] = Counter()
    success_reasons: Counter[str] = Counter()
    cost_reasons: Counter[str] = Counter()
    quality_complete_cases: list[str] = []
    quality_cost_complete_cases: list[str] = []
    success_complete = True
    for case_id in context.planned_case_ids:
        quality_complete = True
        cost_complete = True
        for arm_id in context.frozen_arm_ids:
            cell = cells[case_id].get(arm_id)
            if cell is None:
                quality_complete = False
                cost_complete = False
                success_complete = False
                quality_reasons[_MISSING_ARM_CELL] += 1
                cost_reasons[_MISSING_ARM_CELL] += 1
                success_reasons[_MISSING_ARM_CELL] += 1
                continue
            if not cell.success_known:
                success_complete = False
                success_reasons[_UNAVAILABLE_RECORD] += 1
            if cell.quality is None:
                quality_complete = False
                cost_complete = False
                reason = (
                    _UNGRADED_SUCCESS
                    if cell.success_known and cell.success
                    else _UNAVAILABLE_RECORD
                )
                quality_reasons[reason] += 1
                cost_reasons[reason] += 1
            if cell.runtime_cost is None:
                cost_complete = False
                cost_reasons[_MISSING_RUNTIME_COST] += 1
        if quality_complete:
            quality_complete_cases.append(case_id)
        if quality_complete and cost_complete:
            quality_cost_complete_cases.append(case_id)
    return _SupportSummary(
        cells=cells,
        quality_reasons=quality_reasons,
        success_reasons=success_reasons,
        cost_reasons=cost_reasons,
        quality_complete_cases=tuple(quality_complete_cases),
        quality_cost_complete_cases=tuple(quality_cost_complete_cases),
        success_complete=success_complete,
    )


def _put_reduced(
    reduced: dict[str, _ReducedMetric],
    metric_id: str,
    value: float | None,
    sample_count: int,
    reasons: Counter[str] | None = None,
) -> None:
    reduced[metric_id] = _ReducedMetric(
        value, sample_count, Counter() if reasons is None else Counter(reasons)
    )


def _put_support_metrics(
    reduced: dict[str, _ReducedMetric],
    support: _SupportSummary,
    context: ModelPoolReductionContext,
) -> None:
    case_count = len(context.planned_case_ids)
    quality_count = len(support.quality_complete_cases)
    quality_cost_count = len(support.quality_cost_complete_cases)
    _put_reduced(
        reduced, "model_pool.arm_count", float(len(context.frozen_arm_ids)), case_count
    )
    for metric_id, value, reasons in (
        (
            "model_pool.quality_shared_support_cases",
            float(quality_count),
            support.quality_reasons,
        ),
        (
            "model_pool.quality_shared_support_fraction",
            quality_count / case_count,
            support.quality_reasons,
        ),
        (
            "model_pool.quality_cost_shared_support_cases",
            float(quality_cost_count),
            support.cost_reasons,
        ),
        (
            "model_pool.quality_cost_shared_support_fraction",
            quality_cost_count / case_count,
            support.cost_reasons,
        ),
    ):
        _put_reduced(reduced, metric_id, value, case_count, reasons)


def _put_dense_quality_metrics(
    reduced: dict[str, _ReducedMetric],
    support: _SupportSummary,
    context: ModelPoolReductionContext,
) -> None:
    case_count = len(context.planned_case_ids)
    quality_by_arm, oracle, unique_wins, marginal = _quality_reduction(
        context.planned_case_ids, context.frozen_arm_ids, support.cells
    )
    best_single = max(quality_by_arm.values())
    for arm_id in context.frozen_arm_ids:
        _put_reduced(
            reduced,
            model_pool_arm_metric_id(arm_id, "quality"),
            quality_by_arm[arm_id],
            case_count,
        )
        _put_reduced(
            reduced,
            model_pool_arm_metric_id(arm_id, "marginal_contribution"),
            marginal[arm_id],
            case_count,
        )
    for metric_id, value in (
        ("model_pool.best_single_quality", best_single),
        ("model_pool.oracle_quality", oracle),
        ("model_pool.oracle_gain", oracle - best_single),
        ("model_pool.unique_wins", float(unique_wins)),
        ("model_pool.unique_win_rate", unique_wins / case_count),
        (
            "model_pool.quality_dominated_arm_count",
            float(
                _quality_dominated_arm_count(
                    context.planned_case_ids, context.frozen_arm_ids, support.cells
                )
            ),
        ),
    ):
        _put_reduced(reduced, metric_id, value, case_count)


def _put_unavailable_quality_metrics(
    reduced: dict[str, _ReducedMetric],
    support: _SupportSummary,
    context: ModelPoolReductionContext,
) -> None:
    quality_samples = len(support.quality_complete_cases)
    for arm_id in context.frozen_arm_ids:
        for measure in ("quality", "marginal_contribution"):
            _put_reduced(
                reduced,
                model_pool_arm_metric_id(arm_id, measure),
                None,
                quality_samples,
                support.quality_reasons,
            )
    for metric_id in (
        "model_pool.best_single_quality",
        "model_pool.oracle_quality",
        "model_pool.oracle_gain",
        "model_pool.unique_wins",
        "model_pool.unique_win_rate",
        "model_pool.quality_dominated_arm_count",
    ):
        _put_reduced(reduced, metric_id, None, quality_samples, support.quality_reasons)


def _put_quality_metrics(
    reduced: dict[str, _ReducedMetric],
    support: _SupportSummary,
    context: ModelPoolReductionContext,
) -> None:
    if len(support.quality_complete_cases) == len(context.planned_case_ids):
        _put_dense_quality_metrics(reduced, support, context)
    else:
        _put_unavailable_quality_metrics(reduced, support, context)


def _put_dense_success_metrics(
    reduced: dict[str, _ReducedMetric],
    support: _SupportSummary,
    context: ModelPoolReductionContext,
) -> None:
    case_count = len(context.planned_case_ids)
    all_arm_failures = sum(
        all(
            not support.cells[case_id][arm_id].success
            for arm_id in context.frozen_arm_ids
        )
        for case_id in context.planned_case_ids
    )
    reliability_by_arm: list[float] = []
    for arm_id in context.frozen_arm_ids:
        successes = sum(
            support.cells[case_id][arm_id].success
            for case_id in context.planned_case_ids
        )
        reliability = successes / case_count
        reliability_by_arm.append(reliability)
        _put_reduced(
            reduced,
            model_pool_arm_metric_id(arm_id, "success_rate"),
            reliability,
            case_count,
        )
    for metric_id, value in (
        ("model_pool.worst_arm_reliability", min(reliability_by_arm)),
        ("model_pool.all_arm_failure_rate", all_arm_failures / case_count),
        (
            "model_pool.mean_pairwise_failure_jaccard",
            _failure_jaccard(
                context.planned_case_ids, context.frozen_arm_ids, support.cells
            ),
        ),
    ):
        _put_reduced(reduced, metric_id, value, case_count)


def _put_unavailable_success_metrics(
    reduced: dict[str, _ReducedMetric],
    support: _SupportSummary,
    context: ModelPoolReductionContext,
) -> None:
    for arm_id in context.frozen_arm_ids:
        _put_reduced(
            reduced,
            model_pool_arm_metric_id(arm_id, "success_rate"),
            None,
            0,
            support.success_reasons,
        )
    for metric_id in (
        "model_pool.worst_arm_reliability",
        "model_pool.all_arm_failure_rate",
        "model_pool.mean_pairwise_failure_jaccard",
    ):
        _put_reduced(reduced, metric_id, None, 0, support.success_reasons)


def _put_success_metrics(
    reduced: dict[str, _ReducedMetric],
    support: _SupportSummary,
    context: ModelPoolReductionContext,
) -> None:
    if support.success_complete:
        _put_dense_success_metrics(reduced, support, context)
    else:
        _put_unavailable_success_metrics(reduced, support, context)


def _put_pareto_metrics(
    reduced: dict[str, _ReducedMetric],
    support: _SupportSummary,
    context: ModelPoolReductionContext,
) -> None:
    case_count = len(context.planned_case_ids)
    dense = (
        len(support.quality_complete_cases) == case_count
        and len(support.quality_cost_complete_cases) == case_count
    )
    if dense:
        evaluable, dominated = _pareto_counts(
            context.planned_case_ids, context.frozen_arm_ids, support.cells
        )
        _put_reduced(
            reduced,
            "model_pool.pareto_evaluable_arm_count",
            float(evaluable),
            case_count,
        )
        _put_reduced(
            reduced,
            "model_pool.pareto_dominated_arm_count",
            float(dominated),
            case_count,
        )
        return
    cost_samples = len(support.quality_cost_complete_cases)
    for metric_id in (
        "model_pool.pareto_evaluable_arm_count",
        "model_pool.pareto_dominated_arm_count",
    ):
        _put_reduced(reduced, metric_id, None, cost_samples, support.cost_reasons)


def model_pool_metrics(
    records: list[ExecutionRecord],
    joint_records: list[ExecutionRecord],
    *,
    context: ModelPoolReductionContext | None = None,
) -> list[MetricDraft]:
    """Reduce the exact metric universe attested by the Go server reducer."""

    inferred_context = context is None
    if context is None:
        context = _infer_diagnostic_context(records, joint_records)
        if context is None:
            return []
    metric_ids = _metric_ids(context.frozen_arm_ids)
    if not context.authoritative:
        reason = Counter({_NON_AUTHORITATIVE: 1})
        return _drafts(
            {metric_id: _ReducedMetric(None, 0, reason) for metric_id in metric_ids},
            context,
        )

    record_matrix = build_dense_model_pool_matrix(records, context)
    support = _collect_support(record_matrix, context)
    reduced: dict[str, _ReducedMetric] = {}
    _put_support_metrics(reduced, support, context)

    _put_quality_metrics(reduced, support, context)

    _put_success_metrics(reduced, support, context)

    _put_pareto_metrics(reduced, support, context)

    entropy, coverage = _reduce_selection(
        joint_records,
        context.planned_case_ids,
        context.frozen_arm_ids,
        strict=not inferred_context,
    )
    reduced["model_pool.selection_entropy_bits"] = entropy
    reduced["model_pool.selection_arm_coverage"] = coverage
    if set(reduced) != set(metric_ids):  # pragma: no cover - implementation invariant
        raise RuntimeError("model-pool reducer did not emit its exact metric universe")
    return _drafts(reduced, context)
