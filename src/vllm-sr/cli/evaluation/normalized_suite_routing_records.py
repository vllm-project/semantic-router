"""Build routing, model-pool, and joint replay records."""

from __future__ import annotations

from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.method_evidence import RobustnessMethodEvidence
from cli.evaluation.normalized_suite_inputs import (
    SelectedCase,
    SuiteEvidence,
    evidence_kind,
    opaque_action_id,
    opaque_arm_id,
    opaque_case_id,
)
from cli.evaluation.normalized_suite_record_helpers import (
    is_qualified_outcome,
    one_for_case,
    outcome_status,
    rows_for_case,
    selected_identity,
    unavailable_record,
)
from cli.evaluation.suite_contract import NormalizedDecision, NormalizedOutcome
from cli.evaluation.suite_store_error import SuiteStoreError

_MIN_MODEL_POOL_ARMS = 2


def routing_records(
    case: SelectedCase, evidence: SuiteEvidence
) -> list[ExecutionRecord]:
    if evidence.decisions is None:
        return [
            unavailable_record(
                case,
                "routing",
                "normalized suite lacks routing decision observations",
            )
        ]
    decision = one_for_case(
        evidence.decisions, case.source_visible.id, "routing decisions"
    )
    if decision is None:
        return [
            unavailable_record(
                case,
                "routing",
                "normalized suite has no qualified routing decision for this case",
            )
        ]
    selected_source = decision.selected_arm_id or decision.selected_action_id
    quality = (
        float(selected_source == case.source_grading.expected_route)
        if selected_source is not None
        and case.source_grading.expected_route is not None
        else None
    )
    robustness = _robustness_evidence(case, evidence)
    return [
        ExecutionRecord(
            id=f"routing-{case.visible.id}",
            track_id="routing",
            case_id=case.visible.id,
            attempt_id=f"replay-{case.visible.id}-routing",
            status="succeeded" if decision.success else "failed",
            selected_arm_id=selected_identity(case, decision),
            selection_status=decision.selection_status,
            selection_method=case.executor_id,
            success=decision.success,
            quality=quality,
            fallback=decision.fallback,
            latency_ms=decision.latency_ms,
            robustness=robustness,
            evidence_kind=evidence_kind(case, "routing"),
        )
    ]


def _robustness_evidence(
    case: SelectedCase, evidence: SuiteEvidence
) -> RobustnessMethodEvidence | None:
    pairs = [
        row
        for row in (evidence.perturbations or ())
        if row.perturbed_case_id == case.source_visible.id
    ]
    if len(pairs) > 1:
        raise SuiteStoreError(
            "normalized robustness target belongs to multiple perturbation pairs"
        )
    if not pairs:
        return None
    pair = pairs[0]
    source_decision = one_for_case(
        evidence.decisions, pair.source_case_id, "robustness source decisions"
    )
    if source_decision is None:
        raise SuiteStoreError("normalized robustness pair lacks its source decision")
    source_action = selected_identity(case, source_decision)
    if source_action is None:
        raise SuiteStoreError("normalized robustness source decision lacks an action")
    expected_action = None
    if pair.expected_action_id is not None:
        expected_action = (
            opaque_arm_id(case.manifest, pair.expected_action_id)
            if pair.expected_action_id in case.manifest.arm_ids
            else opaque_action_id(case.manifest, pair.expected_action_id)
        )
    return RobustnessMethodEvidence(
        method_id="routerarena.robustness.v1",
        pair_id=pair.pair_id,
        source_case_id=opaque_case_id(case.manifest, pair.source_case_id),
        target_case_id=case.visible.id,
        shift_type="paraphrase",
        relation=pair.relation,
        source_action_id=source_action,
        expected_action_id=expected_action,
        slice_ids=pair.slice_ids,
        native_pair_count=pair.native_pair_count,
        source_record_digest=pair.source_record_digest,
    )


def model_pool_records(
    case: SelectedCase, evidence: SuiteEvidence
) -> list[ExecutionRecord]:
    if evidence.outcomes is None:
        return [
            unavailable_record(
                case,
                "model_pool",
                "normalized suite lacks dense arm outcome observations",
            )
        ]
    outcomes = sorted(
        rows_for_case(evidence.outcomes, case.source_visible.id),
        key=lambda row: (
            row.arm_id,
            row.action_id or "",
            row.budget_tokens or 0,
            row.source_record_digest,
        ),
    )
    declared = set(case.manifest.arm_ids)
    if declared and any(row.arm_id not in declared for row in outcomes):
        raise SuiteStoreError("normalized outcome references an undeclared arm")
    observed_arms = {row.arm_id for row in outcomes}
    qualified_pool = declared or observed_arms
    if len(qualified_pool) < _MIN_MODEL_POOL_ARMS:
        return [
            unavailable_record(
                case,
                "model_pool",
                "normalized model-pool evidence requires at least two arms",
            )
        ]
    if declared and observed_arms != declared:
        return [
            unavailable_record(
                case,
                "model_pool",
                "normalized dense matrix is incomplete for this case",
                suffix=str(index),
            ).model_copy(update={"arm_id": opaque_arm_id(case.manifest, arm_id)})
            for index, arm_id in enumerate(sorted(declared))
        ]
    records: list[ExecutionRecord] = []
    for index, outcome in enumerate(outcomes):
        if not is_qualified_outcome(outcome):
            records.append(
                unavailable_record(
                    case,
                    "model_pool",
                    "normalized arm outcome lacks a success or quality observation",
                    suffix=str(index),
                )
            )
            continue
        records.append(
            ExecutionRecord(
                id=f"model-pool-{case.visible.id}-{index}",
                track_id="model_pool",
                case_id=case.visible.id,
                attempt_id=f"replay-{case.visible.id}-model-pool-{index}",
                status=outcome_status(outcome),
                arm_id=opaque_arm_id(case.manifest, outcome.arm_id),
                success=(
                    outcome.success
                    if outcome.success is not None
                    else outcome.quality is not None
                ),
                quality=outcome.quality,
                latency_ms=outcome.latency_ms,
                input_tokens=outcome.input_tokens,
                output_tokens=outcome.output_tokens,
                runtime_cost=outcome.runtime_cost_usd,
                grader=(
                    "normalized-grader"
                    if outcome.grader_id or outcome.grader_revision
                    else None
                ),
                evidence_kind=evidence_kind(case, "model_pool"),
            )
        )
    if not records:
        records.append(
            unavailable_record(
                case,
                "model_pool",
                "normalized suite has no arm outcome for this case",
            )
        )
    return records


def _decision_outcome(
    case: SelectedCase,
    evidence: SuiteEvidence,
    decision: NormalizedDecision,
) -> NormalizedOutcome | None:
    if evidence.outcomes is None:
        return None
    selected = decision.selected_action_id or decision.selected_arm_id
    if selected is None:
        return None
    matches = [
        row
        for row in rows_for_case(evidence.outcomes, case.source_visible.id)
        if (row.action_id or row.arm_id) == selected
    ]
    if len(matches) > 1:
        raise SuiteStoreError("normalized joint outcome is ambiguous for its decision")
    return matches[0] if matches else None


def joint_records(case: SelectedCase, evidence: SuiteEvidence) -> list[ExecutionRecord]:
    if evidence.decisions is None or evidence.outcomes is None:
        return [
            unavailable_record(
                case,
                "joint",
                "normalized joint evidence requires both decisions and outcomes",
            )
        ]
    decision = one_for_case(
        evidence.decisions, case.source_visible.id, "routing decisions"
    )
    if decision is None:
        return [
            unavailable_record(
                case,
                "joint",
                "normalized suite has no qualified joint decision for this case",
            )
        ]
    outcome = _decision_outcome(case, evidence, decision)
    if outcome is None or not is_qualified_outcome(outcome):
        return [
            unavailable_record(
                case,
                "joint",
                "normalized suite has no qualified realized outcome for its decision",
            )
        ]
    latencies = [
        value
        for value in (decision.latency_ms, outcome.latency_ms)
        if value is not None
    ]
    success = decision.success and outcome.success is not False
    return [
        ExecutionRecord(
            id=f"joint-{case.visible.id}",
            track_id="joint",
            case_id=case.visible.id,
            attempt_id=f"replay-{case.visible.id}-joint",
            status="succeeded" if success else "failed",
            selected_arm_id=selected_identity(case, decision),
            selection_status=decision.selection_status,
            selection_method=case.executor_id,
            success=success,
            quality=outcome.quality,
            fallback=decision.fallback,
            latency_ms=sum(latencies) if latencies else None,
            input_tokens=outcome.input_tokens,
            output_tokens=outcome.output_tokens,
            runtime_cost=outcome.runtime_cost_usd,
            evidence_kind=evidence_kind(case, "joint"),
        )
    ]
