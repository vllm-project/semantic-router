"""Server-portable declared-shift pair reduction."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.metric_core import MetricDraft, build_metric


@dataclass(frozen=True)
class RobustnessReduction:
    pair_count: int
    pass_rate: float | None
    worst_slice_pass_rate: float | None
    passed: bool | None


def reduce_robustness(records: list[ExecutionRecord]) -> RobustnessReduction:
    routing = [row for row in records if row.track_id == "routing"]
    by_case: dict[str, ExecutionRecord] = {}
    for row in routing:
        if row.case_id in by_case:
            raise ValueError("robustness reduction requires one routing row per case")
        by_case[row.case_id] = row
    pair_ids: set[str] = set()
    native_pair_count: int | None = None
    source_target_pairs: set[tuple[str, str]] = set()
    passes: list[bool] = []
    by_slice: dict[str, list[bool]] = defaultdict(list)
    for target in routing:
        method = target.robustness
        if method is None:
            continue
        if method.method_id not in {
            "routerarena.robustness.v1",
            "declared-shift.server-live.v1",
        }:
            raise ValueError("robustness evidence uses an unsupported method")
        if native_pair_count is None:
            native_pair_count = method.native_pair_count
        elif native_pair_count != method.native_pair_count:
            raise ValueError("robustness records disagree on native pair count")
        pair = (method.source_case_id, method.target_case_id)
        if method.target_case_id != target.case_id:
            raise ValueError("robustness evidence does not bind its target decision")
        if method.pair_id in pair_ids or pair in source_target_pairs:
            raise ValueError("robustness reduction received a duplicate pair")
        pair_ids.add(method.pair_id)
        source_target_pairs.add(pair)
        source = by_case.get(method.source_case_id)
        if source is None or source.selected_arm_id != method.source_action_id:
            raise ValueError("robustness evidence does not bind its source decision")
        expected = (
            method.source_action_id
            if method.relation == "invariant"
            else method.expected_action_id
        )
        passed = (
            target.selected_arm_id is not None and target.selected_arm_id == expected
        )
        passes.append(passed)
        for slice_id in method.slice_ids:
            by_slice[slice_id].append(passed)
    pass_rate = sum(passes) / len(passes) if passes else None
    slice_rates = [sum(rows) / len(rows) for rows in by_slice.values() if rows]
    worst_slice = min(slice_rates) if slice_rates else None
    return RobustnessReduction(
        pair_count=len(passes),
        pass_rate=pass_rate,
        worst_slice_pass_rate=worst_slice,
        passed=(
            pass_rate == 1.0 and worst_slice == 1.0
            if passes and len(passes) == native_pair_count
            else None
        ),
    )


def robustness_metrics(records: list[ExecutionRecord]) -> list[MetricDraft]:
    reduced = reduce_robustness(records)
    return [
        build_metric(
            "routing.robustness_pass_rate",
            "Pinned declared-shift relation pass rate",
            "routing",
            reduced.pass_rate,
            "fraction",
            "higher_is_better",
            reduced.pair_count,
        ),
        build_metric(
            "routing.robustness_worst_slice_pass_rate",
            "Worst declared robustness-slice pass rate",
            "routing",
            reduced.worst_slice_pass_rate,
            "fraction",
            "higher_is_better",
            reduced.pair_count,
        ),
    ]
