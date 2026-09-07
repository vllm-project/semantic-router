from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.metric_model_pool import model_pool_metrics
from cli.evaluation.metric_model_pool_contract import (
    ModelPoolReductionContext,
    build_dense_model_pool_matrix,
    model_pool_arm_metric_id,
    model_pool_arm_segment,
    parse_model_pool_arm_metric_id,
)
from cli.evaluation.metrics import _unavailable_analysis_units, compute_metrics
from cli.evaluation.statistics import attach_confidence_intervals

_FIXTURE = Path(__file__).parent / "fixtures" / "model_pool_metric_parity.v1.json"


def _pool_record(
    case_id: str,
    arm_id: str,
    *,
    status: str = "succeeded",
    success: bool | None = True,
    quality: float | None = 0.5,
    runtime_cost: float | None = 1.0,
    suffix: str = "",
) -> ExecutionRecord:
    coordinate = f"{case_id}-{arm_id}{suffix}"
    return ExecutionRecord(
        id=f"pool-{coordinate}",
        track_id="model_pool",
        case_id=case_id,
        attempt_id=f"attempt-{coordinate}",
        status=status,
        arm_id=arm_id,
        success=success,
        quality=quality,
        runtime_cost=runtime_cost,
    )


def _joint_record(
    case_id: str,
    selected_arm_id: str | None,
    *,
    status: str = "succeeded",
    suffix: str = "",
) -> ExecutionRecord:
    return ExecutionRecord(
        id=f"joint-{case_id}{suffix}",
        track_id="joint",
        case_id=case_id,
        attempt_id=f"joint-attempt-{case_id}{suffix}",
        status=status,
        selected_arm_id=selected_arm_id,
    )


def _context(*, authoritative: bool = True) -> ModelPoolReductionContext:
    return ModelPoolReductionContext(
        frozen_arm_ids=("bravo", "alpha"),
        planned_case_ids=("case-2", "case-1"),
        authoritative=authoritative,
    )


def _complete_records() -> tuple[list[ExecutionRecord], list[ExecutionRecord]]:
    pool = [
        _pool_record("case-1", "alpha", quality=0.7),
        _pool_record("case-1", "bravo", quality=0.6),
        _pool_record("case-2", "alpha", quality=0.8),
        _pool_record("case-2", "bravo", quality=0.5),
    ]
    joint = [
        _joint_record("case-1", "alpha"),
        _joint_record("case-2", "bravo"),
    ]
    return pool, joint


def _by_id(drafts: list[Any]) -> dict[str, Any]:
    return {draft.id: draft for draft in drafts}


def _published(drafts: list[Any]) -> dict[str, Any]:
    return {draft.id: draft.publish(unavailable_analysis_units=0) for draft in drafts}


def _fixture_record(payload: dict[str, Any], index: int) -> ExecutionRecord:
    return ExecutionRecord(
        id=f"pool-golden-{index}",
        track_id="model_pool",
        case_id=payload["case_id"],
        attempt_id=f"pool-golden-attempt-{index}",
        status=payload["status"],
        arm_id=payload["arm_id"],
        success=payload["success"],
        quality=payload["quality"],
        runtime_cost=payload["runtime_cost"],
    )


def _fixture_joint(payload: dict[str, Any], index: int) -> ExecutionRecord:
    return ExecutionRecord(
        id=f"joint-golden-{index}",
        track_id="joint",
        case_id=payload["case_id"],
        attempt_id=f"joint-golden-attempt-{index}",
        status=payload["status"],
        selected_arm_id=payload["selected_arm_id"],
    )


def test_go_python_golden_fixture_matches_every_model_pool_metric() -> None:
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    assert fixture["schema_version"] == "model-pool-metric-parity.v1"
    records = [
        _fixture_record(payload, index)
        for index, payload in enumerate(fixture["pool_records"])
    ]
    joint = [
        _fixture_joint(payload, index)
        for index, payload in enumerate(fixture["joint_records"])
    ]
    context = ModelPoolReductionContext(
        frozen_arm_ids=tuple(fixture["frozen_arm_ids"]),
        planned_case_ids=tuple(fixture["planned_case_ids"]),
        authoritative=True,
    )

    drafts = model_pool_metrics(records, joint, context=context)
    expected = fixture["expected_metrics"]
    assert [draft.id for draft in drafts] == [item["id"] for item in expected]
    assert [draft.id for draft in drafts] == sorted(draft.id for draft in drafts)
    for draft, item in zip(drafts, expected, strict=True):
        assert draft.value == pytest.approx(item["value"])
        assert draft.sample_count == item["sample_count"]
        assert draft.model_pool_observed_exclusions == item["observed_exclusions"]
        published = draft.publish(unavailable_analysis_units=0)
        assert (
            published.analysis_provenance.observed_exclusions
            == item["observed_exclusions"]
        )

    reordered = model_pool_metrics(
        list(reversed(records)), list(reversed(joint)), context=context
    )
    assert reordered == drafts
    decorated = attach_confidence_intervals(
        list(_published(drafts).values()), records + joint, seed=17
    )
    assert all(metric.confidence_interval is None for metric in decorated)


def test_missing_cells_fail_close_each_dense_axis_with_exact_exclusions() -> None:
    pool, joint = _complete_records()
    matrix = build_dense_model_pool_matrix(pool[:-1], _context())
    assert tuple(matrix) == ("case-1", "case-2")
    assert all(tuple(row) == ("alpha", "bravo") for row in matrix.values())
    assert matrix["case-2"]["bravo"] is None
    metrics = _published(model_pool_metrics([], joint, context=_context()))

    quality = metrics["model_pool.oracle_quality"]
    assert quality.value is None
    assert quality.sample_count == 0
    assert quality.analysis_provenance.observed_exclusions == 4
    support = metrics["model_pool.quality_shared_support_cases"]
    assert support.value == 0
    assert support.sample_count == 2
    assert support.analysis_provenance.observed_exclusions == 4
    reliability = metrics["model_pool.worst_arm_reliability"]
    assert reliability.value is None
    assert reliability.sample_count == 0
    assert reliability.analysis_provenance.observed_exclusions == 4


def test_failed_outcome_is_zero_but_ungraded_success_is_unavailable() -> None:
    pool, joint = _complete_records()
    failed = list(pool)
    failed[-1] = failed[-1].model_copy(
        update={"status": "failed", "success": None, "quality": None}
    )
    failed_metrics = _published(model_pool_metrics(failed, joint, context=_context()))
    assert failed_metrics[
        model_pool_arm_metric_id("bravo", "quality")
    ].value == pytest.approx(0.3)
    assert failed_metrics[
        model_pool_arm_metric_id("bravo", "success_rate")
    ].value == pytest.approx(0.5)
    assert (
        failed_metrics[
            "model_pool.oracle_quality"
        ].analysis_provenance.observed_exclusions
        == 0
    )

    ungraded = list(pool)
    ungraded[-1] = ungraded[-1].model_copy(update={"quality": None})
    ungraded_metrics = _published(
        model_pool_metrics(ungraded, joint, context=_context())
    )
    assert ungraded_metrics["model_pool.oracle_quality"].value is None
    assert ungraded_metrics["model_pool.oracle_quality"].sample_count == 1
    assert (
        ungraded_metrics[
            "model_pool.oracle_quality"
        ].analysis_provenance.observed_exclusions
        == 1
    )
    assert ungraded_metrics["model_pool.worst_arm_reliability"].value == 1
    assert (
        ungraded_metrics[
            "model_pool.worst_arm_reliability"
        ].analysis_provenance.observed_exclusions
        == 0
    )


def test_unavailable_record_and_missing_cost_exclude_only_dependent_axes() -> None:
    pool, joint = _complete_records()
    unavailable = list(pool)
    unavailable[-1] = unavailable[-1].model_copy(
        update={
            "status": "unavailable",
            "success": None,
            "quality": None,
        }
    )
    unavailable_metrics = _published(
        model_pool_metrics(unavailable, joint, context=_context())
    )
    assert unavailable_metrics["model_pool.oracle_quality"].value is None
    assert (
        unavailable_metrics[
            "model_pool.oracle_quality"
        ].analysis_provenance.observed_exclusions
        == 1
    )
    assert unavailable_metrics["model_pool.worst_arm_reliability"].value is None
    assert (
        unavailable_metrics[
            "model_pool.worst_arm_reliability"
        ].analysis_provenance.observed_exclusions
        == 1
    )

    missing_cost = list(pool)
    missing_cost[-1] = missing_cost[-1].model_copy(update={"runtime_cost": None})
    cost_metrics = _published(
        model_pool_metrics(missing_cost, joint, context=_context())
    )
    assert cost_metrics["model_pool.oracle_quality"].value == pytest.approx(0.75)
    pareto = cost_metrics["model_pool.pareto_evaluable_arm_count"]
    assert pareto.value is None
    assert pareto.sample_count == 1
    assert pareto.analysis_provenance.observed_exclusions == 1


def test_selection_is_exactly_one_valid_frozen_arm_per_planned_case() -> None:
    pool, joint = _complete_records()
    missing = _published(model_pool_metrics(pool, joint[:1], context=_context()))
    selection = missing["model_pool.selection_entropy_bits"]
    assert selection.value is None
    assert selection.sample_count == 1
    assert selection.analysis_provenance.observed_exclusions == 1

    unavailable = [joint[0], _joint_record("case-2", None, status="unavailable")]
    unavailable_selection = _published(
        model_pool_metrics(pool, unavailable, context=_context())
    )["model_pool.selection_entropy_bits"]
    assert unavailable_selection.sample_count == 1
    assert unavailable_selection.analysis_provenance.observed_exclusions == 2

    invalid_inputs = (
        [joint[0], _joint_record("case-2", "outside")],
        [joint[0], _joint_record("case-1", "alpha", suffix="-duplicate")],
        [joint[0], _joint_record("outside", "alpha")],
        [joint[0], _joint_record("case-2", "alpha", status="unavailable")],
        [joint[0], _joint_record("case-2", None)],
    )
    for invalid in invalid_inputs:
        with pytest.raises(ValueError):
            model_pool_metrics(pool, invalid, context=_context())


def test_non_authoritative_replay_never_claims_reduced_model_pool_values() -> None:
    pool, joint = _complete_records()
    drafts = model_pool_metrics(pool, joint, context=_context(authoritative=False))
    assert len(drafts) == 24
    assert all(draft.value is None and draft.sample_count == 0 for draft in drafts)
    assert all(draft.model_pool_observed_exclusions == 1 for draft in drafts)


def test_compute_metrics_passes_full_live_matrix_and_keeps_replay_unavailable() -> None:
    pool, joint = _complete_records()
    pool[-1] = pool[-1].model_copy(
        update={"status": "unavailable", "success": None, "quality": None}
    )
    live = {
        metric.id: metric
        for metric in compute_metrics(
            pool + joint,
            capacity_profile=None,
            model_pool_context=_context(),
        )
        if metric.id.startswith("model_pool.")
    }
    assert live["model_pool.oracle_quality"].value is None
    assert (
        live["model_pool.oracle_quality"].analysis_provenance.observed_exclusions == 1
    )
    assert live["model_pool.selection_entropy_bits"].value == 1

    replay = [
        metric
        for metric in compute_metrics(
            pool + joint,
            capacity_profile=None,
            model_pool_context=_context(authoritative=False),
        )
        if metric.id.startswith("model_pool.")
    ]
    assert len(replay) == 24
    assert all(
        metric.value is None
        and metric.sample_count == 0
        and metric.analysis_provenance.observed_exclusions == 1
        for metric in replay
    )


def test_portable_arm_codec_and_unavailable_projection_preserve_dotted_id() -> None:
    segment = model_pool_arm_segment("team.a")
    metric_id = model_pool_arm_metric_id("team.a", "quality")
    assert segment == "u-dGVhbS5h"
    assert metric_id == "model_pool.arm.u-dGVhbS5h.quality"
    assert parse_model_pool_arm_metric_id(metric_id) == ("team.a", "quality")

    pool, joint = _complete_records()
    dotted = [
        row.model_copy(update={"arm_id": "team.a"}) if row.arm_id == "bravo" else row
        for row in pool
    ]
    context = ModelPoolReductionContext(
        frozen_arm_ids=("team.a", "alpha"),
        planned_case_ids=("case-1", "case-2"),
        authoritative=True,
    )
    dotted_joint = [
        (
            row.model_copy(update={"selected_arm_id": "team.a"})
            if row.selected_arm_id == "bravo"
            else row
        )
        for row in joint
    ]
    draft = _by_id(model_pool_metrics(dotted, dotted_joint, context=context))[metric_id]
    unavailable = SimpleNamespace(
        track_id="model_pool",
        status="unavailable",
        case_id="case-1",
        arm_id="team.a",
    )
    assert _unavailable_analysis_units(draft, [unavailable]) == 1
