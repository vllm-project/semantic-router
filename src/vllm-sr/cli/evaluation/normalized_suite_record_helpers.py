"""Shared primitives for normalized-suite execution-record builders."""

from __future__ import annotations

from typing import TypeVar

from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.normalized_suite_inputs import (
    SelectedCase,
    evidence_kind,
    opaque_action_id,
    opaque_arm_id,
)
from cli.evaluation.suite_contract import NormalizedDecision, NormalizedOutcome
from cli.evaluation.suite_store_error import SuiteStoreError

_ModelT = TypeVar("_ModelT")


def rows_for_case(rows: tuple[_ModelT, ...] | None, case_id: str) -> list[_ModelT]:
    if rows is None:
        return []
    return [row for row in rows if row.case_id == case_id]


def one_for_case(
    rows: tuple[_ModelT, ...] | None,
    case_id: str,
    role: str,
) -> _ModelT | None:
    matches = rows_for_case(rows, case_id)
    if len(matches) > 1:
        raise SuiteStoreError(f"normalized {role} has ambiguous case observations")
    return matches[0] if matches else None


def unavailable_record(
    case: SelectedCase,
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
        evidence_kind=evidence_kind(case, track_id),
        error=reason,
    )


def selected_identity(case: SelectedCase, decision: NormalizedDecision) -> str | None:
    if decision.selected_arm_id:
        return opaque_arm_id(case.manifest, decision.selected_arm_id)
    if decision.selected_action_id:
        return opaque_action_id(case.manifest, decision.selected_action_id)
    return None


def is_qualified_outcome(outcome: NormalizedOutcome) -> bool:
    return outcome.success is not None or outcome.quality is not None


def outcome_status(outcome: NormalizedOutcome) -> str:
    if not is_qualified_outcome(outcome):
        return "unavailable"
    return "failed" if outcome.success is False else "succeeded"
