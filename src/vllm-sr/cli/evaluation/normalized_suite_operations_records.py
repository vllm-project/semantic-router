"""Build safety and capacity replay records."""

from __future__ import annotations

from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.normalized_suite_inputs import (
    SelectedCase,
    SuiteEvidence,
    evidence_kind,
)
from cli.evaluation.normalized_suite_record_helpers import (
    one_for_case,
    rows_for_case,
    unavailable_record,
)


def safety_records(
    case: SelectedCase, evidence: SuiteEvidence
) -> list[ExecutionRecord]:
    if evidence.safety is None:
        return [
            unavailable_record(
                case,
                "safety",
                "normalized suite lacks safety enforcement observations",
            )
        ]
    observation = one_for_case(
        evidence.safety, case.source_visible.id, "safety observations"
    )
    if observation is None:
        return [
            unavailable_record(
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
            evidence_kind=evidence_kind(case, "safety"),
        )
    ]


def capacity_records(
    case: SelectedCase, evidence: SuiteEvidence
) -> list[ExecutionRecord]:
    if evidence.capacity is None:
        return [
            unavailable_record(
                case,
                "capacity",
                "normalized suite lacks bounded capacity observations",
            )
        ]
    rows = sorted(
        rows_for_case(evidence.capacity, case.source_visible.id),
        key=lambda row: (row.concurrency, row.source_record_digest),
    )
    if not rows:
        return [
            unavailable_record(
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
            evidence_kind=evidence_kind(case, "capacity"),
        )
        for index, observation in enumerate(rows)
    ]
