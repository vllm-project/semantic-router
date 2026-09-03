"""Dispatch normalized evidence to narrow track-specific metric reducers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product

from cli.evaluation.capacity_profile import CapacityProfile
from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.metric_analysis_catalog import (
    decode_metric_subject_id,
    resolve_metric_analysis,
)
from cli.evaluation.metric_capacity import capacity_metrics
from cli.evaluation.metric_compound_model_budget import r2_compound_metrics
from cli.evaluation.metric_core import MetricDraft
from cli.evaluation.metric_joint import joint_metrics
from cli.evaluation.metric_methods import method_metrics
from cli.evaluation.metric_model_pool import model_pool_metrics
from cli.evaluation.metric_model_pool_contract import ModelPoolReductionContext
from cli.evaluation.metric_routing import routing_metrics
from cli.evaluation.metric_tracks import (
    agentic_metrics,
    multimodal_metrics,
    preference_metrics,
    safety_metrics,
)
from cli.evaluation.reporting import EvaluationMetric


def compute_metrics(
    records: list[ExecutionRecord],
    *,
    capacity_profile: CapacityProfile | None,
    model_pool_context: ModelPoolReductionContext | None = None,
) -> list[EvaluationMetric]:
    by_track = {
        track: [
            row
            for row in records
            if row.track_id == track and row.status != "unavailable"
        ]
        for track in (
            "routing",
            "model_pool",
            "joint",
            "agentic",
            "multimodal",
            "preference",
            "safety",
            "capacity",
        )
    }
    metrics: list[MetricDraft] = []
    metrics.extend(routing_metrics(by_track["routing"]))
    all_model_pool = [row for row in records if row.track_id == "model_pool"]
    compound_model_budget = [
        row
        for row in all_model_pool
        if row.method_id is not None and row.status != "unavailable"
    ]
    generic_model_pool = [row for row in all_model_pool if row.method_id is None]
    metrics.extend(
        model_pool_metrics(
            generic_model_pool,
            [row for row in records if row.track_id == "joint"],
            context=model_pool_context,
        )
    )
    metrics.extend(joint_metrics(by_track["joint"], by_track["model_pool"]))
    metrics.extend(agentic_metrics(by_track["agentic"]))
    metrics.extend(multimodal_metrics(by_track["multimodal"]))
    metrics.extend(preference_metrics(by_track["preference"]))
    metrics.extend(safety_metrics(by_track["safety"]))
    metrics.extend(capacity_metrics(by_track["capacity"], capacity_profile))
    metrics.extend(r2_compound_metrics(compound_model_budget))
    metrics.extend(
        method_metrics([row for row in records if row.status != "unavailable"])
    )
    return _bind_metric_analysis_provenance(metrics, records)


def _bind_metric_analysis_provenance(
    metrics: list[MetricDraft], records: list[ExecutionRecord]
) -> list[EvaluationMetric]:
    """Bind each reduction to source-observed exclusions before publication.

    Each draft states its reducer-owned planned population; its local
    exclusions are therefore ``planned - sample_count``.  The dispatcher only
    adds unavailable source units through a metric-specific projection.  It
    never infers exclusions from all records in a track, because a case x arm
    metric, a per-arm metric, and a capacity repetition have different units.
    """

    bound: list[EvaluationMetric] = []
    for metric in metrics:
        unavailable_units = (
            0
            if metric.model_pool_observed_exclusions is not None
            else _unavailable_analysis_units(metric, records)
        )
        bound.append(metric.publish(unavailable_analysis_units=unavailable_units))
    return bound


def _unavailable_analysis_units(
    metric: MetricDraft, records: list[ExecutionRecord]
) -> int:
    """Project unavailable evidence into the metric's registered unit.

    This is intentionally narrow and deterministic.  Unknown worker metrics
    have no server-owned projection and therefore cannot be published from the
    canonical reducer path.
    """

    match = resolve_metric_analysis(metric.id)
    projection = match.specification.planned_unit_projection
    projection_track = projection["track_id"]
    unavailable = [
        row
        for row in records
        if row.status == "unavailable" and row.track_id == projection_track
    ]
    if not unavailable:
        return 0
    filtered = [
        row
        for row in unavailable
        if _projection_filters_match(
            metric.id, row, projection.get("filters", ()), match.captures
        )
    ]
    coordinates = projection["coordinates"]
    units: set[tuple[object, ...]] = set()
    for row in filtered:
        values_by_coordinate = [
            _projection_coordinate_values(metric.id, row, coordinate)
            for coordinate in coordinates
        ]
        for unit in product(*values_by_coordinate):
            units.add(unit)
    return len(units)


def _projection_filters_match(
    metric_id: str,
    record: object,
    filters: Sequence[Mapping[str, str]],
    captures: Mapping[str, str],
) -> bool:
    for item in filters:
        field = item["field"]
        capture = item["capture"]
        values = _projection_coordinate_values(metric_id, record, field)
        if not values:
            raise ValueError(
                f"metric {metric_id} cannot project unavailable evidence without {field}"
            )
        expected = decode_metric_subject_id(captures[capture])
        if not any(str(value) == expected for value in values):
            return False
    return True


def _projection_coordinate_values(
    metric_id: str, record: object, coordinate: str
) -> tuple[object, ...]:
    if coordinate in {"measurement_request", "warmup_request"}:
        expected_phase = (
            "measurement" if coordinate == "measurement_request" else "warmup"
        )
        if getattr(record, "load_phase", None) != expected_phase:
            return ()
        repetition = getattr(record, "load_repetition", None)
        request_index = getattr(record, "load_request_index", None)
        if repetition is None or request_index is None:
            raise ValueError(
                f"metric {metric_id} cannot project unavailable {coordinate} evidence"
            )
        return ((repetition, request_index),)
    if coordinate == "agent_task.tool_call_id":
        evidence = getattr(record, "agent_task", None)
        calls = () if evidence is None else evidence.tool_calls
        values = tuple(call.tool_call_id for call in calls)
    else:
        values = _nested_projection_values(record, coordinate)
    if not values:
        raise ValueError(
            f"metric {metric_id} cannot project unavailable evidence without {coordinate}"
        )
    return values


def _nested_projection_values(root: object, path: str) -> tuple[object, ...]:
    values: tuple[object, ...] = (root,)
    for segment in path.split("."):
        expanded: list[object] = []
        for value in values:
            items = value if isinstance(value, (list, tuple)) else (value,)
            for item in items:
                nested = getattr(item, segment, None)
                if isinstance(nested, (list, tuple)):
                    expanded.extend(nested)
                elif nested is not None:
                    expanded.append(nested)
        values = tuple(expanded)
    return values
