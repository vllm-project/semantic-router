"""Replay fixture adapter producing the common execution record IR."""

from __future__ import annotations

from cli.evaluation.contracts import (
    CaseGrading,
    CaseVisible,
    GradingCaseSet,
    VisibleCaseSet,
)
from cli.evaluation.evidence import (
    ExecutionRecord,
    FixtureCaseEvidence,
    ReplayFixture,
)


def _routing_record(
    case: CaseVisible,
    labels: CaseGrading,
    evidence: FixtureCaseEvidence,
    attempt: str,
) -> ExecutionRecord:
    route = evidence.route
    quality = (
        float(route.selected_model == labels.expected_route)
        if labels.expected_route is not None and route.selected_model is not None
        else None
    )
    return ExecutionRecord(
        id=f"routing-{case.id}",
        track_id="routing",
        case_id=case.id,
        attempt_id=attempt,
        status="succeeded" if route.success else "failed",
        selected_arm_id=route.selected_model,
        selection_status=route.selection_status,
        success=route.success,
        quality=quality,
        fallback=route.fallback,
        latency_ms=route.latency_ms,
    )


def _pool_records(
    case: CaseVisible, evidence: FixtureCaseEvidence, attempt: str
) -> list[ExecutionRecord]:
    return [
        ExecutionRecord(
            id=f"model-pool-{case.id}-{arm.arm_id}",
            track_id="model_pool",
            case_id=case.id,
            attempt_id=f"{attempt}-{arm.arm_id}",
            status="succeeded" if arm.success else "failed",
            arm_id=arm.arm_id,
            success=arm.success,
            quality=arm.quality,
            latency_ms=arm.latency_ms,
            input_tokens=arm.input_tokens,
            output_tokens=arm.output_tokens,
            runtime_cost=arm.runtime_cost,
        )
        for arm in evidence.arms
    ]


def _joint_record(
    case: CaseVisible, evidence: FixtureCaseEvidence, attempt: str
) -> ExecutionRecord:
    selected = next(
        (arm for arm in evidence.arms if arm.arm_id == evidence.route.selected_model),
        None,
    )
    return ExecutionRecord(
        id=f"joint-{case.id}",
        track_id="joint",
        case_id=case.id,
        attempt_id=attempt,
        status="succeeded" if selected is not None and selected.success else "failed",
        selected_arm_id=evidence.route.selected_model,
        success=selected.success if selected else False,
        quality=selected.quality if selected else None,
        latency_ms=(
            (evidence.route.latency_ms or 0) + (selected.latency_ms or 0)
            if selected
            else evidence.route.latency_ms
        ),
        input_tokens=selected.input_tokens if selected else None,
        output_tokens=selected.output_tokens if selected else None,
        runtime_cost=selected.runtime_cost if selected else None,
    )


def _agentic_record(
    case: CaseVisible, evidence: FixtureCaseEvidence, attempt: str
) -> ExecutionRecord:
    trajectory = evidence.trajectory
    return ExecutionRecord(
        id=f"agentic-{case.id}",
        track_id="agentic",
        case_id=case.id,
        attempt_id=attempt,
        status="succeeded" if trajectory.success else "failed",
        success=trajectory.success,
        quality=trajectory.task_score,
        trajectory_steps=trajectory.steps,
        tool_calls=trajectory.tool_calls,
        invalid_tool_calls=trajectory.invalid_tool_calls,
    )


def _multimodal_record(
    case: CaseVisible, evidence: FixtureCaseEvidence, attempt: str
) -> ExecutionRecord:
    multimodal = evidence.multimodal
    return ExecutionRecord(
        id=f"multimodal-{case.id}",
        track_id="multimodal",
        case_id=case.id,
        attempt_id=attempt,
        status="succeeded" if multimodal.supported else "failed",
        success=multimodal.supported,
        quality=multimodal.quality,
        modality=case.modality,
        privacy_violations=multimodal.privacy_violations,
    )


def _preference_record(
    case: CaseVisible, evidence: FixtureCaseEvidence, attempt: str
) -> ExecutionRecord:
    preference = evidence.preference
    return ExecutionRecord(
        id=f"preference-{case.id}",
        track_id="preference",
        case_id=case.id,
        attempt_id=attempt,
        status="succeeded",
        selected_arm_id=preference.chosen_arm_id,
        success=True,
        quality=preference.reward,
        preference_match=preference.chosen_arm_id == preference.preferred_arm_id,
        behavior_propensity=preference.behavior_propensity,
    )


def _safety_record(
    case: CaseVisible, evidence: FixtureCaseEvidence, attempt: str
) -> ExecutionRecord:
    safety = evidence.safety
    return ExecutionRecord(
        id=f"safety-{case.id}",
        track_id="safety",
        case_id=case.id,
        attempt_id=attempt,
        status="succeeded",
        success=True,
        safety_violations=safety.violations,
        should_block=safety.should_block,
        blocked=safety.blocked,
    )


def _capacity_record(
    case: CaseVisible, evidence: FixtureCaseEvidence, attempt: str
) -> ExecutionRecord:
    capacity = evidence.capacity
    return ExecutionRecord(
        id=f"capacity-{case.id}",
        track_id="capacity",
        case_id=case.id,
        attempt_id=attempt,
        status="succeeded" if capacity.success else "failed",
        success=capacity.success,
        latency_ms=capacity.latency_ms,
        concurrency=capacity.concurrency,
        throughput_rps=capacity.throughput_rps,
        capacity_tco=capacity.capacity_tco,
        gpu_seconds=capacity.gpu_seconds,
        energy_kwh=capacity.energy_kwh,
    )


def _case_records(
    case: CaseVisible,
    labels: CaseGrading,
    evidence: FixtureCaseEvidence,
) -> list[ExecutionRecord]:
    attempt = f"attempt-{case.id}"
    track_ids = case.track_ids
    records: list[ExecutionRecord] = []
    if "routing" in track_ids:
        records.append(_routing_record(case, labels, evidence, attempt))
    if "model_pool" in track_ids:
        records.extend(_pool_records(case, evidence, attempt))
    if "joint" in track_ids:
        records.append(_joint_record(case, evidence, attempt))
    if "agentic" in track_ids:
        records.append(_agentic_record(case, evidence, attempt))
    if "multimodal" in track_ids and case.modality != "text":
        records.append(_multimodal_record(case, evidence, attempt))
    if "preference" in track_ids:
        records.append(_preference_record(case, evidence, attempt))
    if "safety" in track_ids:
        records.append(_safety_record(case, evidence, attempt))
    if "capacity" in track_ids:
        records.append(_capacity_record(case, evidence, attempt))
    return records


def execute_fixture(
    visible: VisibleCaseSet,
    grading: GradingCaseSet,
    fixture: ReplayFixture,
    track_ids: tuple[str, ...],
) -> list[ExecutionRecord]:
    visible_by_id = {case.id: case for case in visible.cases}
    grading_by_id = {case.case_id: case for case in grading.cases}
    records: list[ExecutionRecord] = []
    for evidence in fixture.cases:
        case = visible_by_id[evidence.case_id]
        if not set(case.track_ids).issubset(track_ids):
            raise ValueError("fixture case plan exceeds the selected run tracks")
        records.extend(_case_records(case, grading_by_id[evidence.case_id], evidence))
    return records
