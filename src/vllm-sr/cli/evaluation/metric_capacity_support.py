"""Shared primitives for capacity metric projections."""

from __future__ import annotations

from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.metric_core import complete_sum

MetricSpec = tuple[str, str, float | None, str, str, int]


def measurement_cost(records: list[ExecutionRecord]) -> float | None:
    capacity_tco = complete_sum(row.capacity_tco for row in records)
    return (
        capacity_tco
        if capacity_tco is not None
        else complete_sum(row.runtime_cost for row in records)
    )
