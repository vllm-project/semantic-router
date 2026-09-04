"""Strict live hard-policy proof and dynamic safety evidence ingestion."""

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
from cli.evaluation.evidence_source_ids import LIVE_HARD_POLICY_EVIDENCE_SOURCE_ID
from cli.evaluation.http_client import EvaluationHTTPClient
from cli.evaluation.method_evidence import (
    HardPolicyMethodEvidence,
    HardPolicyStaticProof,
)
from cli.evaluation.method_ledger_identity import (
    MethodMixtureBinding,
    method_mixture_binding,
    validate_method_ledger_freshness,
)
from cli.evaluation.target_contracts import ManifestMixture

HARD_POLICY_LEDGER_VERSION = "evaluation-hard-policy-ledger.v1"


class HardPolicyLedger(StrictModel):
    """One sealed Router-owned policy proof plus brokered attack decisions."""

    contract_version: Literal[HARD_POLICY_LEDGER_VERSION]
    ledger_id: str
    source_id: str
    environment: Literal["production"]
    policy_snapshot_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    config_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    target_id: str
    backend_topology_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    mixture: MethodMixtureBinding
    proof: HardPolicyStaticProof
    window_started_at: datetime
    window_ended_at: datetime
    sealed_at: datetime
    observations: tuple[HardPolicyMethodEvidence, ...] = Field(min_length=1)

    _ids = field_validator("ledger_id", "source_id", "target_id")(_validate_id)

    @model_validator(mode="after")
    def validate_ledger(self) -> HardPolicyLedger:
        if any(
            value.tzinfo is None
            for value in (self.window_started_at, self.window_ended_at, self.sealed_at)
        ) or not (self.window_started_at < self.window_ended_at <= self.sealed_at):
            raise ValueError("hard-policy ledger window is invalid")
        if (
            self.proof.policy_snapshot_digest != self.policy_snapshot_digest
            or self.proof.config_digest != self.config_digest
            or self.proof.source_id != self.source_id
            or self.proof.target_id != self.target_id
            or self.proof.backend_topology_digest != self.backend_topology_digest
            or self.proof.mixture_snapshot_digest != self.mixture.snapshot_digest
            or self.proof.ledger_total_observation_count != len(self.observations)
        ):
            raise ValueError("hard-policy proof does not bind the ledger snapshot")
        observation_ids: set[str] = set()
        attack_ids: set[str] = set()
        receipts: set[str] = set()
        for row in self.observations:
            if row.proof != self.proof or not (
                self.window_started_at <= row.observed_at <= self.window_ended_at
            ):
                raise ValueError("hard-policy observation does not bind the ledger")
            if (
                row.observation_id in observation_ids
                or row.attack_id in attack_ids
                or row.decision_receipt_id in receipts
            ):
                raise ValueError("hard-policy dynamic identities must be unique")
            observation_ids.add(row.observation_id)
            attack_ids.add(row.attack_id)
            receipts.add(row.decision_receipt_id)
        required = {
            (binding.rule_id, binding.enforcement_point)
            for binding in self.proof.required_bindings
        }
        observed = {(row.rule_id, row.enforcement_point) for row in self.observations}
        if observed != required:
            raise ValueError(
                "hard-policy dynamic coverage must exactly cover proof bindings"
            )
        return self


@dataclass(frozen=True)
class HardPolicyExecution:
    visible: VisibleCaseSet
    grading: GradingCaseSet
    records: list[ExecutionRecord]


def _case_id(ledger_id: str, observation_id: str) -> str:
    digest = hashlib.sha256(f"{ledger_id}\x00{observation_id}".encode()).hexdigest()[
        :24
    ]
    return f"hard-policy-{digest}"


def execute_hard_policy_ledger(
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
) -> HardPolicyExecution:
    """Fetch a live Router policy proof and dynamic decisions through the broker."""

    result = client.get(
        endpoint,
        track_id="safety",
        case_id="hard-policy-ledger",
        attempt_id="ledger-fetch",
        broker_operation="hard-policy.ledger",
    )
    if not result.success or result.payload is None or result.broker_receipt is None:
        raise ValueError("hard-policy evidence ledger could not be read")
    ledger = HardPolicyLedger.model_validate(result.payload)
    validate_method_ledger_freshness(ledger.sealed_at, result.fetched_at)
    expected_mixture = method_mixture_binding(mixture)
    if (
        ledger.policy_snapshot_digest != policy_snapshot_digest
        or ledger.config_digest != config_digest
        or ledger.target_id != target_id
        or ledger.backend_topology_digest != backend_topology_digest
        or ledger.mixture != expected_mixture
    ):
        raise ValueError("hard-policy ledger belongs to a different runtime snapshot")
    ranked = sorted(
        ledger.observations,
        key=lambda row: (
            hashlib.sha256(
                f"{seed}\x00{ledger.ledger_id}\x00{row.observation_id}".encode()
            ).digest(),
            row.observation_id,
        ),
    )
    if sample_limit < len(ranked):
        raise ValueError(
            "sample_limit must cover every observation in the sealed hard-policy window"
        )
    selected = tuple(ranked)
    if not selected:
        raise ValueError("hard-policy sampling selected no observations")
    visible: list[CaseVisible] = []
    grading: list[CaseGrading] = []
    records: list[ExecutionRecord] = []
    for row in selected:
        case_id = _case_id(ledger.ledger_id, row.observation_id)
        visible.append(
            CaseVisible(
                id=case_id,
                track_ids=("safety",),
                messages=(
                    Message(role="user", content="Brokered hard-policy attack probe"),
                ),
                tags=("hard-policy", row.rule_id, row.enforcement_point),
            )
        )
        grading.append(CaseGrading(case_id=case_id, should_block=row.should_block))
        records.append(
            ExecutionRecord(
                id=f"safety-{case_id}",
                track_id="safety",
                case_id=case_id,
                attempt_id=f"safety-{case_id}",
                status="succeeded",
                success=True,
                quality=float(row.blocked == row.should_block and row.violations == 0),
                safety_violations=row.violations,
                should_block=row.should_block,
                blocked=row.blocked,
                hard_policy=row,
                evidence_kind=LIVE_HARD_POLICY_EVIDENCE_SOURCE_ID,
                broker_receipt=result.broker_receipt,
            )
        )
    return HardPolicyExecution(
        visible=VisibleCaseSet(cases=tuple(visible)),
        grading=GradingCaseSet(cases=tuple(grading)),
        records=records,
    )
