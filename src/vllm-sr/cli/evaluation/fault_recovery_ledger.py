"""Strict ingestion of server-brokered live exact-step fault-recovery ledgers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from cli.evaluation.contract_primitives import Message, StrictModel
from cli.evaluation.contract_validation import validate_portable_id as _validate_id
from cli.evaluation.contracts import (
    CaseGrading,
    CaseVisible,
    GradingCaseSet,
    VisibleCaseSet,
)
from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.evidence_source_ids import (
    LIVE_FAULT_RECOVERY_EVIDENCE_SOURCE_ID,
)
from cli.evaluation.http_client import EvaluationHTTPClient
from cli.evaluation.method_evidence import (
    MINIMUM_RECOVERY_CLUSTER_COUNT,
    MINIMUM_RECOVERY_DISTINCT_SEED_COUNT,
    MINIMUM_RECOVERY_PAIR_COUNT,
    RecoveryMethodEvidence,
)
from cli.evaluation.method_ledger_identity import (
    MethodMixtureBinding,
    method_mixture_binding,
    validate_method_ledger_freshness,
)
from cli.evaluation.target_contracts import ManifestMixture

FAULT_RECOVERY_LEDGER_VERSION = "evaluation-fault-recovery-ledger.v1"


class FaultRecoveryLedger(StrictModel):
    """One sealed live window of paired baseline/treatment fault receipts."""

    contract_version: Literal[FAULT_RECOVERY_LEDGER_VERSION]
    ledger_id: str
    source_id: str
    environment: Literal["production"]
    policy_snapshot_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    config_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    target_id: str
    backend_topology_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    mixture: MethodMixtureBinding
    minimum_pair_count: int = Field(ge=MINIMUM_RECOVERY_PAIR_COUNT, strict=True)
    minimum_cluster_count: int = Field(ge=MINIMUM_RECOVERY_CLUSTER_COUNT, strict=True)
    minimum_distinct_seed_count: int = Field(
        ge=MINIMUM_RECOVERY_DISTINCT_SEED_COUNT, strict=True
    )
    maximum_recovery_latency_ms: float = Field(gt=0, allow_inf_nan=False)
    maximum_retry_amplification: float = Field(ge=1, allow_inf_nan=False)
    window_started_at: datetime
    window_ended_at: datetime
    sealed_at: datetime
    pairs: tuple[RecoveryMethodEvidence, ...] = Field(min_length=1)

    _ids = field_validator("ledger_id", "source_id", "target_id")(_validate_id)

    @model_validator(mode="after")
    def validate_window(self) -> FaultRecoveryLedger:
        if any(
            value.tzinfo is None
            for value in (self.window_started_at, self.window_ended_at, self.sealed_at)
        ) or not (self.window_started_at < self.window_ended_at <= self.sealed_at):
            raise ValueError("fault-recovery ledger window is invalid")
        total = len(self.pairs)
        faults: set[str] = set()
        repetitions: set[tuple[str, str]] = set()
        injection_receipts: set[str] = set()
        for row in self.pairs:
            if (
                row.ledger_id != self.ledger_id
                or row.source_id != self.source_id
                or row.policy_snapshot_digest != self.policy_snapshot_digest
                or row.config_digest != self.config_digest
                or row.target_id != self.target_id
                or row.backend_topology_digest != self.backend_topology_digest
                or row.mixture_snapshot_digest != self.mixture.snapshot_digest
                or row.ledger_total_pair_count != total
                or row.minimum_pair_count != self.minimum_pair_count
                or row.minimum_cluster_count != self.minimum_cluster_count
                or row.minimum_distinct_seed_count != self.minimum_distinct_seed_count
                or row.maximum_recovery_latency_ms != self.maximum_recovery_latency_ms
                or row.maximum_retry_amplification != self.maximum_retry_amplification
                or not self.window_started_at <= row.observed_at <= self.window_ended_at
            ):
                raise ValueError("fault-recovery pair does not bind the sealed ledger")
            repetition = (row.cohort_pair_id, row.repetition_id)
            if (
                row.fault_id in faults
                or repetition in repetitions
                or row.fault_injection_receipt_digest in injection_receipts
            ):
                raise ValueError("fault-recovery ledger identities must be unique")
            faults.add(row.fault_id)
            repetitions.add(repetition)
            injection_receipts.add(row.fault_injection_receipt_digest)
        return self


@dataclass(frozen=True)
class FaultRecoveryExecution:
    visible: VisibleCaseSet
    grading: GradingCaseSet
    records: list[ExecutionRecord]


def _case_id(ledger_id: str, fault_id: str) -> str:
    digest = hashlib.sha256(f"{ledger_id}\x00{fault_id}".encode()).hexdigest()[:24]
    return f"fault-recovery-{digest}"


def _pair_passed(row: RecoveryMethodEvidence) -> bool:
    retry_amplification = (row.treatment_retry_count + 1) / (
        row.baseline_retry_count + 1
    )
    return (
        row.injection_observed
        and row.recovered
        and row.state_preserved
        and row.treatment_terminal_success
        and row.duplicate_side_effect_count == 0
        and row.treatment_recovery_latency_ms <= row.maximum_recovery_latency_ms
        and retry_amplification <= row.maximum_retry_amplification
    )


def execute_fault_recovery_ledger(
    client: EvaluationHTTPClient,
    endpoint: str,
    *,
    policy_snapshot_digest: str,
    config_digest: str,
    target_id: str,
    backend_topology_digest: str,
    mixture: ManifestMixture,
    sample_limit: int,
    seed: int,
) -> FaultRecoveryExecution:
    """Fetch one complete sealed fault window through the server-owned broker."""

    result = client.get(
        endpoint,
        track_id="agentic",
        case_id="fault-recovery-ledger",
        attempt_id="ledger-fetch",
        broker_operation="fault-recovery.ledger",
    )
    if not result.success or result.payload is None or result.broker_receipt is None:
        raise ValueError("fault-recovery ledger could not be read")
    ledger = FaultRecoveryLedger.model_validate(result.payload)
    validate_method_ledger_freshness(ledger.sealed_at, result.fetched_at)
    expected_mixture = method_mixture_binding(mixture)
    if (
        ledger.policy_snapshot_digest != policy_snapshot_digest
        or ledger.config_digest != config_digest
        or ledger.target_id != target_id
        or ledger.backend_topology_digest != backend_topology_digest
        or ledger.mixture != expected_mixture
    ):
        raise ValueError(
            "fault-recovery ledger belongs to a different runtime snapshot"
        )
    if sample_limit < len(ledger.pairs):
        raise ValueError(
            "sample_limit must cover every pair in the sealed fault-recovery window"
        )
    selected = sorted(
        ledger.pairs,
        key=lambda row: (
            hashlib.sha256(
                f"{seed}\x00{ledger.ledger_id}\x00{row.fault_id}".encode()
            ).digest(),
            row.fault_id,
        ),
    )
    visible: list[CaseVisible] = []
    grading: list[CaseGrading] = []
    records: list[ExecutionRecord] = []
    for pair in selected:
        case_id = _case_id(ledger.ledger_id, pair.fault_id)
        passed = _pair_passed(pair)
        visible.append(
            CaseVisible(
                id=case_id,
                track_ids=("agentic",),
                messages=(
                    Message(
                        role="user",
                        content="Brokered paired exact-step fault-recovery receipt",
                    ),
                ),
                tags=("live-fault-recovery", pair.fault_kind, pair.cluster_id),
                trajectory_id=pair.repetition_id,
            )
        )
        grading.append(CaseGrading(case_id=case_id))
        records.append(
            ExecutionRecord(
                id=f"agentic-{case_id}",
                track_id="agentic",
                case_id=case_id,
                attempt_id=f"agentic-{case_id}",
                status="succeeded" if passed else "failed",
                success=passed,
                quality=float(passed),
                recovery=pair,
                evidence_kind=LIVE_FAULT_RECOVERY_EVIDENCE_SOURCE_ID,
                broker_receipt=result.broker_receipt,
            )
        )
    return FaultRecoveryExecution(
        visible=VisibleCaseSet(cases=tuple(visible)),
        grading=GradingCaseSet(cases=tuple(grading)),
        records=records,
    )
