"""Replay installed normalized suites into the common execution-record IR.

This executor is intentionally data-only.  It never imports or executes code
from an upstream benchmark checkout.  Missing qualification artifacts become
explicit unavailable records; malformed or ambiguous artifacts reject the run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar, cast

from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.normalized_suite_inputs import (
    EXECUTOR_ID as _EXECUTOR_ID,
)
from cli.evaluation.normalized_suite_inputs import (
    NormalizedSuiteInputs,
)
from cli.evaluation.normalized_suite_inputs import (
    SelectedCase as _SelectedCase,
)
from cli.evaluation.normalized_suite_inputs import (
    SuiteEvidence as _SuiteEvidence,
)
from cli.evaluation.normalized_suite_inputs import (
    action_alias as _action_alias,
)
from cli.evaluation.normalized_suite_inputs import (
    arm_alias as _arm_alias,
)
from cli.evaluation.normalized_suite_inputs import (
    build_inputs as _build_inputs,
)
from cli.evaluation.normalized_suite_inputs import (
    evidence_kind as _evidence_kind,
)
from cli.evaluation.normalized_suite_inputs import (
    load_selected_cases as _load_selected_cases,
)
from cli.evaluation.suite_contract import (
    BenchmarkSuiteManifest,
    NormalizedCapacityObservation,
    NormalizedDecision,
    NormalizedMultimodalObservation,
    NormalizedOutcome,
    NormalizedPreference,
    NormalizedSafetyObservation,
    NormalizedTrajectoryStep,
)
from cli.evaluation.suite_install_contract import SuiteArtifactRole
from cli.evaluation.suite_store import NormalizedSuiteStore, SuiteStoreError

_ModelT = TypeVar("_ModelT")


@dataclass(frozen=True)
class NormalizedSuiteExecution:
    inputs: NormalizedSuiteInputs
    records: list[ExecutionRecord]


def _load_optional(
    store: NormalizedSuiteStore,
    manifest: BenchmarkSuiteManifest,
    role: SuiteArtifactRole,
    expected_type: type[_ModelT],
) -> tuple[_ModelT, ...] | None:
    if getattr(manifest.artifacts, role) is None:
        return None
    rows = tuple(store.load_jsonl(manifest.id, role))
    if not all(isinstance(row, expected_type) for row in rows):
        raise SuiteStoreError(
            "normalized suite role produced an unexpected record type"
        )
    return cast(tuple[_ModelT, ...], rows)


def _load_suite_evidence(
    store: NormalizedSuiteStore,
    manifest: BenchmarkSuiteManifest,
    known_case_ids: set[str],
) -> _SuiteEvidence:
    evidence = _SuiteEvidence(
        outcomes=_load_optional(store, manifest, "outcomes", NormalizedOutcome),
        decisions=_load_optional(store, manifest, "decisions", NormalizedDecision),
        preferences=_load_optional(
            store, manifest, "preferences", NormalizedPreference
        ),
        trajectories=_load_optional(
            store, manifest, "trajectories", NormalizedTrajectoryStep
        ),
        multimodal=_load_optional(
            store,
            manifest,
            "multimodal_observations",
            NormalizedMultimodalObservation,
        ),
        safety=_load_optional(
            store, manifest, "safety_observations", NormalizedSafetyObservation
        ),
        capacity=_load_optional(
            store,
            manifest,
            "capacity_observations",
            NormalizedCapacityObservation,
        ),
    )
    for rows in (
        evidence.outcomes,
        evidence.decisions,
        evidence.preferences,
        evidence.trajectories,
        evidence.multimodal,
        evidence.safety,
        evidence.capacity,
    ):
        if rows and any(row.case_id not in known_case_ids for row in rows):
            raise SuiteStoreError("normalized observation references an unknown case")
    for role, rows in (
        ("routing decisions", evidence.decisions),
        ("multimodal observations", evidence.multimodal),
        ("safety observations", evidence.safety),
    ):
        if rows:
            case_ids = [row.case_id for row in rows]
            if len(case_ids) != len(set(case_ids)):
                raise SuiteStoreError(f"normalized {role} has duplicate case rows")
    return evidence


def _rows_for_case(rows: tuple[_ModelT, ...] | None, case_id: str) -> list[_ModelT]:
    if rows is None:
        return []
    return [row for row in rows if row.case_id == case_id]


def _one_for_case(
    rows: tuple[_ModelT, ...] | None,
    case_id: str,
    role: str,
) -> _ModelT | None:
    matches = _rows_for_case(rows, case_id)
    if len(matches) > 1:
        raise SuiteStoreError(f"normalized {role} has ambiguous case observations")
    return matches[0] if matches else None


def _unavailable(
    case: _SelectedCase,
    track_id: str,
    reason: str,
    *,
    suffix: str = "0",
) -> ExecutionRecord:
    return ExecutionRecord(
        id=f"{track_id}-{case.visible.id}-unavailable-{suffix}",
        track_id=track_id,
        case_id=case.visible.id,
        attempt_id=f"replay-{case.visible.id}-{track_id}-{suffix}",
        status="unavailable",
        evidence_kind=_evidence_kind(case),
        error=reason,
    )


def _selected_alias(case: _SelectedCase, decision: NormalizedDecision) -> str | None:
    if decision.selected_arm_id:
        return _arm_alias(case.manifest, decision.selected_arm_id)
    if decision.selected_action_id:
        return _action_alias(case.manifest, decision.selected_action_id)
    return None


def _routing_records(
    case: _SelectedCase, evidence: _SuiteEvidence
) -> list[ExecutionRecord]:
    if evidence.decisions is None:
        return [
            _unavailable(
                case,
                "routing",
                "normalized suite lacks routing decision observations",
            )
        ]
    decision = _one_for_case(
        evidence.decisions, case.source_visible.id, "routing decisions"
    )
    if decision is None:
        return [
            _unavailable(
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
    return [
        ExecutionRecord(
            id=f"routing-{case.visible.id}",
            track_id="routing",
            case_id=case.visible.id,
            attempt_id=f"replay-{case.visible.id}-routing",
            status="succeeded" if decision.success else "failed",
            selected_arm_id=_selected_alias(case, decision),
            selection_status=decision.selection_status,
            selection_method=_EXECUTOR_ID,
            success=decision.success,
            quality=quality,
            fallback=decision.fallback,
            latency_ms=decision.latency_ms,
            evidence_kind=_evidence_kind(case),
        )
    ]


_MIN_MODEL_POOL_ARMS = 2


def _qualified_outcome(outcome: NormalizedOutcome) -> bool:
    return outcome.success is not None or outcome.quality is not None


def _outcome_status(outcome: NormalizedOutcome) -> str:
    if not _qualified_outcome(outcome):
        return "unavailable"
    return "failed" if outcome.success is False else "succeeded"


def _model_pool_records(
    case: _SelectedCase, evidence: _SuiteEvidence
) -> list[ExecutionRecord]:
    if evidence.outcomes is None:
        return [
            _unavailable(
                case,
                "model_pool",
                "normalized suite lacks dense arm outcome observations",
            )
        ]
    outcomes = sorted(
        _rows_for_case(evidence.outcomes, case.source_visible.id),
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
            _unavailable(
                case,
                "model_pool",
                "normalized model-pool evidence requires at least two arms",
            )
        ]
    if declared and observed_arms != declared:
        return [
            _unavailable(
                case,
                "model_pool",
                "normalized dense matrix is incomplete for this case",
                suffix=str(index),
            ).model_copy(update={"arm_id": _arm_alias(case.manifest, arm_id)})
            for index, arm_id in enumerate(sorted(declared))
        ]
    records: list[ExecutionRecord] = []
    for index, outcome in enumerate(outcomes):
        if not _qualified_outcome(outcome):
            records.append(
                _unavailable(
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
                status=_outcome_status(outcome),
                arm_id=_arm_alias(case.manifest, outcome.arm_id),
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
                evidence_kind=_evidence_kind(case),
            )
        )
    if not records:
        records.append(
            _unavailable(
                case,
                "model_pool",
                "normalized suite has no arm outcome for this case",
            )
        )
    return records


def _decision_outcome(
    case: _SelectedCase,
    evidence: _SuiteEvidence,
    decision: NormalizedDecision,
) -> NormalizedOutcome | None:
    if evidence.outcomes is None:
        return None
    selected = decision.selected_action_id or decision.selected_arm_id
    if selected is None:
        return None
    matches = [
        row
        for row in _rows_for_case(evidence.outcomes, case.source_visible.id)
        if (row.action_id or row.arm_id) == selected
    ]
    if len(matches) > 1:
        raise SuiteStoreError("normalized joint outcome is ambiguous for its decision")
    return matches[0] if matches else None


def _joint_records(
    case: _SelectedCase, evidence: _SuiteEvidence
) -> list[ExecutionRecord]:
    if evidence.decisions is None or evidence.outcomes is None:
        return [
            _unavailable(
                case,
                "joint",
                "normalized joint evidence requires both decisions and outcomes",
            )
        ]
    decision = _one_for_case(
        evidence.decisions, case.source_visible.id, "routing decisions"
    )
    if decision is None:
        return [
            _unavailable(
                case,
                "joint",
                "normalized suite has no qualified joint decision for this case",
            )
        ]
    outcome = _decision_outcome(case, evidence, decision)
    if outcome is None or not _qualified_outcome(outcome):
        return [
            _unavailable(
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
            selected_arm_id=_selected_alias(case, decision),
            selection_status=decision.selection_status,
            selection_method=_EXECUTOR_ID,
            success=success,
            quality=outcome.quality,
            fallback=decision.fallback,
            latency_ms=sum(latencies) if latencies else None,
            input_tokens=outcome.input_tokens,
            output_tokens=outcome.output_tokens,
            runtime_cost=outcome.runtime_cost_usd,
            evidence_kind=_evidence_kind(case),
        )
    ]


def _agentic_records(
    case: _SelectedCase, evidence: _SuiteEvidence
) -> list[ExecutionRecord]:
    if evidence.trajectories is None:
        return [
            _unavailable(
                case,
                "agentic",
                "normalized suite lacks trajectory observations",
            )
        ]
    steps = sorted(
        _rows_for_case(evidence.trajectories, case.source_visible.id),
        key=lambda row: (row.sequence, row.step_id),
    )
    if not steps:
        return [
            _unavailable(
                case,
                "agentic",
                "normalized suite has no trajectory for this case",
            )
        ]
    if len({row.step_id for row in steps}) != len(steps) or len(
        {row.sequence for row in steps}
    ) != len(steps):
        raise SuiteStoreError("normalized trajectory contains duplicate steps")
    trajectory_ids = {row.trajectory_id for row in steps}
    if len(trajectory_ids) != 1:
        raise SuiteStoreError("normalized case spans multiple trajectory ids")
    if (
        case.source_visible.trajectory_id is not None
        and case.source_visible.trajectory_id not in trajectory_ids
    ):
        raise SuiteStoreError("normalized trajectory does not match its visible case")
    terminals = [row for row in steps if row.terminal]
    if len(terminals) != 1 or terminals[0] != steps[-1]:
        return [
            _unavailable(
                case,
                "agentic",
                "normalized trajectory lacks one qualified terminal observation",
            )
        ]
    terminal = terminals[0]
    if terminal.terminal_success is None:
        return [
            _unavailable(
                case,
                "agentic",
                "normalized trajectory terminal result is unavailable",
            )
        ]
    tool_steps = [
        row
        for row in steps
        if row.tool_name is not None or row.tool_call_valid is not None
    ]
    selected = next(
        (row.selected_action_id for row in reversed(steps) if row.selected_action_id),
        None,
    )
    return [
        ExecutionRecord(
            id=f"agentic-{case.visible.id}",
            track_id="agentic",
            case_id=case.visible.id,
            attempt_id=f"replay-{case.visible.id}-agentic",
            status="succeeded" if terminal.terminal_success else "failed",
            selected_arm_id=(
                _action_alias(case.manifest, selected) if selected else None
            ),
            success=terminal.terminal_success,
            quality=(
                terminal.task_score
                if terminal.task_score is not None
                else float(terminal.terminal_success)
            ),
            trajectory_steps=len(steps),
            tool_calls=len(tool_steps),
            invalid_tool_calls=sum(row.tool_call_valid is False for row in tool_steps),
            privacy_violations=sum(row.privacy_exposures or 0 for row in steps),
            evidence_kind=_evidence_kind(case),
        )
    ]


def _multimodal_records(
    case: _SelectedCase, evidence: _SuiteEvidence
) -> list[ExecutionRecord]:
    if case.source_visible.modality == "text":
        return []
    if evidence.multimodal is None:
        return [
            _unavailable(
                case,
                "multimodal",
                "normalized suite lacks multimodal outcome observations",
            )
        ]
    observation = _one_for_case(
        evidence.multimodal,
        case.source_visible.id,
        "multimodal observations",
    )
    if observation is None:
        return [
            _unavailable(
                case,
                "multimodal",
                "normalized suite has no multimodal observation for this case",
            )
        ]
    if observation.modality != case.source_visible.modality:
        raise SuiteStoreError("normalized multimodal observation has wrong modality")
    return [
        ExecutionRecord(
            id=f"multimodal-{case.visible.id}",
            track_id="multimodal",
            case_id=case.visible.id,
            attempt_id=f"replay-{case.visible.id}-multimodal",
            status="succeeded" if observation.supported else "failed",
            success=observation.supported,
            quality=observation.quality,
            modality=observation.modality,
            privacy_violations=observation.privacy_violations,
            evidence_kind=_evidence_kind(case),
        )
    ]


def _preference_records(
    case: _SelectedCase, evidence: _SuiteEvidence
) -> list[ExecutionRecord]:
    if evidence.preferences is None:
        return [
            _unavailable(
                case,
                "preference",
                "normalized suite lacks preference observations",
            )
        ]
    rows = sorted(
        _rows_for_case(evidence.preferences, case.source_visible.id),
        key=lambda row: (
            row.left_action_id,
            row.right_action_id,
            row.source_record_digest,
        ),
    )
    if not rows:
        return [
            _unavailable(
                case,
                "preference",
                "normalized suite has no preference observation for this case",
            )
        ]
    records: list[ExecutionRecord] = []
    for index, preference in enumerate(rows):
        if preference.chosen_action_id not in {
            preference.left_action_id,
            preference.right_action_id,
        }:
            records.append(
                _unavailable(
                    case,
                    "preference",
                    "normalized preference lacks a qualified chosen action",
                    suffix=str(index),
                )
            )
            continue
        preferred = {
            "left": preference.left_action_id,
            "right": preference.right_action_id,
        }.get(preference.preference)
        if preferred is None:
            records.append(
                _unavailable(
                    case,
                    "preference",
                    "tie or skipped preference cannot qualify agreement",
                    suffix=str(index),
                )
            )
            continue
        match = preference.chosen_action_id == preferred
        records.append(
            ExecutionRecord(
                id=f"preference-{case.visible.id}-{index}",
                track_id="preference",
                case_id=case.visible.id,
                attempt_id=f"replay-{case.visible.id}-preference-{index}",
                status="succeeded",
                selected_arm_id=_action_alias(
                    case.manifest, preference.chosen_action_id
                ),
                success=True,
                quality=(
                    preference.reward if preference.reward is not None else float(match)
                ),
                preference_match=match,
                behavior_propensity=preference.behavior_propensity,
                evidence_kind=_evidence_kind(case),
            )
        )
    return records


def _safety_records(
    case: _SelectedCase, evidence: _SuiteEvidence
) -> list[ExecutionRecord]:
    if evidence.safety is None:
        return [
            _unavailable(
                case,
                "safety",
                "normalized suite lacks safety enforcement observations",
            )
        ]
    observation = _one_for_case(
        evidence.safety, case.source_visible.id, "safety observations"
    )
    if observation is None:
        return [
            _unavailable(
                case,
                "safety",
                "normalized suite has no safety observation for this case",
            )
        ]
    return [
        ExecutionRecord(
            id=f"safety-{case.visible.id}",
            track_id="safety",
            case_id=case.visible.id,
            attempt_id=f"replay-{case.visible.id}-safety",
            status="succeeded",
            success=observation.violations == 0,
            safety_violations=observation.violations,
            should_block=case.source_grading.should_block,
            blocked=observation.blocked,
            evidence_kind=_evidence_kind(case),
        )
    ]


def _capacity_records(
    case: _SelectedCase, evidence: _SuiteEvidence
) -> list[ExecutionRecord]:
    if evidence.capacity is None:
        return [
            _unavailable(
                case,
                "capacity",
                "normalized suite lacks bounded capacity observations",
            )
        ]
    rows = sorted(
        _rows_for_case(evidence.capacity, case.source_visible.id),
        key=lambda row: (row.concurrency, row.source_record_digest),
    )
    if not rows:
        return [
            _unavailable(
                case,
                "capacity",
                "normalized suite has no capacity observation for this case",
            )
        ]
    return [
        ExecutionRecord(
            id=f"capacity-{case.visible.id}-{index}",
            track_id="capacity",
            case_id=case.visible.id,
            attempt_id=f"replay-{case.visible.id}-capacity-{index}",
            status="succeeded" if observation.success else "failed",
            success=observation.success,
            latency_ms=observation.latency_ms,
            runtime_cost=observation.runtime_cost_usd,
            capacity_tco=observation.capacity_tco_usd,
            concurrency=observation.concurrency,
            throughput_rps=observation.throughput_rps,
            gpu_seconds=observation.gpu_seconds,
            energy_kwh=observation.energy_kwh,
            load_elapsed_seconds=observation.elapsed_seconds,
            evidence_kind=_evidence_kind(case),
        )
        for index, observation in enumerate(rows)
    ]


_TRACK_EXECUTORS = {
    "routing": _routing_records,
    "model_pool": _model_pool_records,
    "joint": _joint_records,
    "agentic": _agentic_records,
    "multimodal": _multimodal_records,
    "preference": _preference_records,
    "safety": _safety_records,
    "capacity": _capacity_records,
}


def execute_normalized_suites(
    *,
    store: NormalizedSuiteStore,
    manifests: tuple[BenchmarkSuiteManifest, ...],
    track_ids: tuple[str, ...],
    sample_limit: int,
    seed: int,
) -> NormalizedSuiteExecution:
    """Replay exact installed suite revisions without loading upstream code."""

    if not manifests:
        raise SuiteStoreError("normalized suite execution requires at least one suite")
    manifests = tuple(sorted(manifests, key=lambda item: item.id))
    selected, known_case_ids = _load_selected_cases(
        store, manifests, sample_limit, seed
    )
    if not selected:
        raise SuiteStoreError("normalized suite sampling selected no cases")
    evidence_by_suite = {
        manifest.id: _load_suite_evidence(store, manifest, known_case_ids[manifest.id])
        for manifest in manifests
    }
    records: list[ExecutionRecord] = []
    for case in selected:
        evidence = evidence_by_suite[case.manifest.id]
        for track_id in track_ids:
            if track_id not in case.manifest.track_ids:
                if track_id != "multimodal" or case.source_visible.modality != "text":
                    records.append(
                        _unavailable(
                            case,
                            track_id,
                            "normalized suite does not declare this track for the case",
                        )
                    )
                continue
            records.extend(_TRACK_EXECUTORS[track_id](case, evidence))
    inputs = _build_inputs(manifests, selected, evidence_by_suite, track_ids)
    return NormalizedSuiteExecution(inputs=inputs, records=records)
