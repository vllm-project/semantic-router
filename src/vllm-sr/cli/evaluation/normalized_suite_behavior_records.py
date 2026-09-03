"""Build agentic, multimodal, and preference replay records."""

from __future__ import annotations

from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.normalized_suite_inputs import (
    SelectedCase,
    SuiteEvidence,
    evidence_kind,
    opaque_action_id,
)
from cli.evaluation.normalized_suite_record_helpers import (
    one_for_case,
    rows_for_case,
    unavailable_record,
)
from cli.evaluation.suite_store_error import SuiteStoreError


def agentic_records(
    case: SelectedCase, evidence: SuiteEvidence
) -> list[ExecutionRecord]:
    if evidence.trajectories is None:
        return [
            unavailable_record(
                case,
                "agentic",
                "normalized suite lacks trajectory observations",
            )
        ]
    steps = sorted(
        rows_for_case(evidence.trajectories, case.source_visible.id),
        key=lambda row: (row.sequence, row.step_id),
    )
    if not steps:
        return [
            unavailable_record(
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
            unavailable_record(
                case,
                "agentic",
                "normalized trajectory lacks one qualified terminal observation",
            )
        ]
    terminal = terminals[0]
    if terminal.terminal_success is None:
        return [
            unavailable_record(
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
    matching_faults = [
        row for row in (evidence.faults or ()) if row.trajectory_id in trajectory_ids
    ]
    if len(matching_faults) > 1:
        raise SuiteStoreError("normalized trajectory has multiple failover diagnostics")
    if matching_faults:
        fault = matching_faults[0]
        if fault.sequence not in {row.sequence for row in steps if not row.terminal}:
            raise SuiteStoreError(
                "normalized failover diagnostic is not a pre-terminal label"
            )
    return [
        ExecutionRecord(
            id=f"agentic-{case.visible.id}",
            track_id="agentic",
            case_id=case.visible.id,
            attempt_id=f"replay-{case.visible.id}-agentic",
            status="succeeded" if terminal.terminal_success else "failed",
            selected_arm_id=(
                opaque_action_id(case.manifest, selected) if selected else None
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
            evidence_kind=evidence_kind(case, "agentic"),
        )
    ]


def multimodal_records(
    case: SelectedCase, evidence: SuiteEvidence
) -> list[ExecutionRecord]:
    if case.source_visible.modality == "text":
        return []
    if evidence.multimodal is None:
        return [
            unavailable_record(
                case,
                "multimodal",
                "normalized suite lacks multimodal outcome observations",
            )
        ]
    observation = one_for_case(
        evidence.multimodal,
        case.source_visible.id,
        "multimodal observations",
    )
    if observation is None:
        return [
            unavailable_record(
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
            evidence_kind=evidence_kind(case, "multimodal"),
        )
    ]


def preference_records(
    case: SelectedCase, evidence: SuiteEvidence
) -> list[ExecutionRecord]:
    if evidence.preferences is None:
        return [
            unavailable_record(
                case,
                "preference",
                "normalized suite lacks preference observations",
            )
        ]
    rows = sorted(
        rows_for_case(evidence.preferences, case.source_visible.id),
        key=lambda row: (
            row.left_action_id,
            row.right_action_id,
            row.source_record_digest,
        ),
    )
    if not rows:
        return [
            unavailable_record(
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
                unavailable_record(
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
                unavailable_record(
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
                selected_arm_id=opaque_action_id(
                    case.manifest, preference.chosen_action_id
                ),
                success=True,
                quality=(
                    preference.reward if preference.reward is not None else float(match)
                ),
                preference_match=match,
                behavior_propensity=preference.behavior_propensity,
                evidence_kind=evidence_kind(case, "preference"),
            )
        )
    return records
