"""R2-Bench records owned by the compound model+budget method contract."""

from __future__ import annotations

from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.method_contract_v2 import COMPOUND_MODEL_BUDGET_METHOD_ID
from cli.evaluation.normalized_suite_inputs import (
    SelectedCase,
    SuiteEvidence,
    opaque_action_id,
)
from cli.evaluation.normalized_suite_record_helpers import (
    is_qualified_outcome,
    outcome_status,
    rows_for_case,
    unavailable_record,
)
from cli.evaluation.suite_store_error import SuiteStoreError


def r2_compound_model_budget_records(
    case: SelectedCase, evidence: SuiteEvidence
) -> list[ExecutionRecord]:
    """Keep R2's action x budget tensor out of generic model-pool reducers."""

    if evidence.outcomes is None:
        return [
            unavailable_record(
                case,
                "model_pool",
                "R2 suite lacks compound model-budget outcomes",
            )
        ]
    outcomes = sorted(
        rows_for_case(evidence.outcomes, case.source_visible.id),
        key=lambda row: (row.action_id or "", row.budget_tokens or 0),
    )
    if not outcomes:
        return [
            unavailable_record(
                case,
                "model_pool",
                "R2 suite has no compound model-budget outcomes for this case",
            )
        ]
    records: list[ExecutionRecord] = []
    for index, outcome in enumerate(outcomes):
        if (
            outcome.action_id is None
            or outcome.budget_tokens is None
            or not is_qualified_outcome(outcome)
            or outcome.quality is None
        ):
            raise SuiteStoreError("R2 compound outcome lacks action, budget, or score")
        action_id = opaque_action_id(case.manifest, outcome.action_id)
        records.append(
            ExecutionRecord(
                id=f"r2-compound-{case.visible.id}-{index}",
                track_id="model_pool",
                case_id=case.visible.id,
                attempt_id=f"replay-{case.visible.id}-r2-compound-{index}",
                status=outcome_status(outcome),
                arm_id=outcome.arm_id,
                method_id=COMPOUND_MODEL_BUDGET_METHOD_ID,
                action_id=action_id,
                budget_tokens=outcome.budget_tokens,
                slice_ids=("all",),
                success=outcome.success,
                quality=outcome.quality,
                output_tokens=outcome.output_tokens,
                grader="normalized-grader",
                evidence_kind=f"{case.executor_id};ceiling=E0",
            )
        )
    return records
