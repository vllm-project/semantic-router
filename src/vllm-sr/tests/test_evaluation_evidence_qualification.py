from __future__ import annotations

import pytest
from cli.evaluation.builtin_evidence_qualifications import (
    LIVE_RUNTIME_EVIDENCE_QUALIFICATIONS,
    NORMALIZED_LIVE_EVIDENCE_QUALIFICATIONS,
)
from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.evidence_level import track_evidence_level
from cli.evaluation.evidence_qualification import (
    EMPTY_EVIDENCE_QUALIFICATIONS,
    EvidenceAttestationRequirement,
    EvidenceQualificationContract,
    EvidenceQualificationRegistry,
    EvidenceReceiptRequirement,
    ReceiptScope,
    TypedEvidencePayloadRequirement,
)
from cli.evaluation.evidence_source_ids import (
    DECLARED_SHIFT_LIVE_EVIDENCE_SOURCE_ID,
    LIVE_AGENT_TASK_EVIDENCE_SOURCE_ID,
    LIVE_CAPACITY_EVIDENCE_SOURCE_ID,
    LIVE_FAULT_RECOVERY_EVIDENCE_SOURCE_ID,
    LIVE_HARD_POLICY_EVIDENCE_SOURCE_ID,
    LIVE_JOINT_EVIDENCE_SOURCE_ID,
    LIVE_MODEL_POOL_EVIDENCE_SOURCE_ID,
    LIVE_PRODUCTION_EXPERIMENT_EVIDENCE_SOURCE_ID,
    LIVE_ROUTING_EVIDENCE_SOURCE_ID,
    NORMALIZED_LIVE_MULTIMODAL_EVIDENCE_SOURCE_ID,
)
from cli.evaluation.executor_contracts import ExecutorContract
from cli.evaluation.reporting import EvidenceLevel

_SOURCE_ID = "provider-joint-outcome.v1"
_RECEIPT = "sha256:" + "1" * 64
_ATTESTATION = "sha256:" + "2" * 64


def _registered_contracts(
    registry: EvidenceQualificationRegistry,
) -> dict[str, tuple[tuple[str, ...], EvidenceLevel, EvidenceLevel, str]]:
    return {
        contract.source_id: (
            contract.allowed_tracks,
            contract.level,
            contract.ceiling,
            contract.receipt.scope,
        )
        for contract in registry.contracts
    }


def test_builtin_live_source_registry_is_exact_and_complete() -> None:
    assert _registered_contracts(LIVE_RUNTIME_EVIDENCE_QUALIFICATIONS) == {
        LIVE_ROUTING_EVIDENCE_SOURCE_ID: (("routing",), "E3", "E3", "record"),
        LIVE_MODEL_POOL_EVIDENCE_SOURCE_ID: (
            ("model_pool",),
            "E4",
            "E4",
            "record",
        ),
        LIVE_JOINT_EVIDENCE_SOURCE_ID: (("joint",), "E5", "E5", "record"),
        LIVE_AGENT_TASK_EVIDENCE_SOURCE_ID: (
            ("agentic",),
            "E5",
            "E5",
            "batch",
        ),
        LIVE_FAULT_RECOVERY_EVIDENCE_SOURCE_ID: (
            ("agentic",),
            "E5",
            "E5",
            "batch",
        ),
        LIVE_HARD_POLICY_EVIDENCE_SOURCE_ID: (
            ("safety",),
            "E4",
            "E4",
            "batch",
        ),
        LIVE_PRODUCTION_EXPERIMENT_EVIDENCE_SOURCE_ID: (
            ("preference",),
            "E5",
            "E5",
            "batch",
        ),
        LIVE_CAPACITY_EVIDENCE_SOURCE_ID: (
            ("capacity",),
            "E5",
            "E5",
            "record",
        ),
    }
    assert _registered_contracts(NORMALIZED_LIVE_EVIDENCE_QUALIFICATIONS) == {
        NORMALIZED_LIVE_MULTIMODAL_EVIDENCE_SOURCE_ID: (
            ("multimodal",),
            "E4",
            "E4",
            "record",
        ),
        DECLARED_SHIFT_LIVE_EVIDENCE_SOURCE_ID: (
            ("routing",),
            "E4",
            "E4",
            "record",
        ),
    }


def _payload_is_complete(record: ExecutionRecord) -> bool:
    return record.success is not None and record.selected_arm_id is not None


def _attestation_is_bound(record: ExecutionRecord) -> bool:
    return record.trace_digest is not None


def _broken_attestation(record: ExecutionRecord) -> bool:
    raise RuntimeError(f"provider validator failed for {record.id}")


def _qualification(
    *,
    level: EvidenceLevel = "E4",
    ceiling: EvidenceLevel = "E4",
    receipt_scope: ReceiptScope = "record",
) -> EvidenceQualificationContract:
    return EvidenceQualificationContract(
        source_id=_SOURCE_ID,
        allowed_tracks=("joint",),
        level=level,
        ceiling=ceiling,
        payload=TypedEvidencePayloadRequirement(
            field_name=None,
            payload_type=ExecutionRecord,
            validator=_payload_is_complete,
        ),
        receipt=EvidenceReceiptRequirement(scope=receipt_scope),
        attestations=(
            EvidenceAttestationRequirement(
                id="provider-execution-attestation",
                validator=_attestation_is_bound,
            ),
        ),
    )


def _executor(
    registry: EvidenceQualificationRegistry,
    *,
    ceiling: EvidenceLevel | None = None,
) -> ExecutorContract:
    return ExecutorContract(
        id="provider-live.v1",
        mode="live",
        suite_class="provider-suite",
        target_profile="brokered-runtime",
        lineage_profile="runtime",
        track_ids=("routing", "joint"),
        evidence_level_ceiling=ceiling,
        evidence_qualifications=registry,
    )


def _record(**updates: object) -> ExecutionRecord:
    record = ExecutionRecord(
        id="joint-case-a",
        track_id="joint",
        case_id="case-a",
        attempt_id="attempt-a",
        status="succeeded",
        selected_arm_id="arm-a",
        success=True,
        latency_ms=10,
        trace_digest=_ATTESTATION,
        evidence_kind=_SOURCE_ID,
        broker_receipt=_RECEIPT,
    )
    return record.model_copy(update=updates)


def test_registered_source_requires_exact_typed_receipt_and_attestation() -> None:
    executor = _executor(EvidenceQualificationRegistry((_qualification(),)))

    assert track_evidence_level("live", executor, "joint", [_record()]) == "E4"
    assert (
        track_evidence_level(
            "live",
            executor,
            "joint",
            [_record(evidence_kind="self-declared-live.v1;level=E5")],
        )
        == "E0"
    )
    assert (
        track_evidence_level("live", executor, "joint", [_record(broker_receipt=None)])
        == "E0"
    )
    assert (
        track_evidence_level("live", executor, "joint", [_record(selected_arm_id=None)])
        == "E0"
    )
    assert (
        track_evidence_level("live", executor, "joint", [_record(trace_digest=None)])
        == "E0"
    )


def test_source_cannot_cross_tracks_or_reuse_record_scoped_receipts() -> None:
    executor = _executor(EvidenceQualificationRegistry((_qualification(),)))

    assert (
        track_evidence_level("live", executor, "routing", [_record(track_id="routing")])
        == "E0"
    )
    duplicate_receipt = _record(
        id="joint-case-b",
        case_id="case-b",
        attempt_id="attempt-b",
    )
    assert (
        track_evidence_level("live", executor, "joint", [_record(), duplicate_receipt])
        == "E0"
    )


def test_batch_scoped_source_requires_one_shared_receipt() -> None:
    executor = _executor(
        EvidenceQualificationRegistry((_qualification(receipt_scope="batch"),))
    )
    second = _record(
        id="joint-case-b",
        case_id="case-b",
        attempt_id="attempt-b",
    )

    assert track_evidence_level("live", executor, "joint", [_record(), second]) == "E4"
    assert (
        track_evidence_level(
            "live",
            executor,
            "joint",
            [
                _record(),
                second.model_copy(update={"broker_receipt": "sha256:" + "3" * 64}),
            ],
        )
        == "E0"
    )


def test_unregistered_provider_and_source_fail_closed() -> None:
    executor = _executor(EMPTY_EVIDENCE_QUALIFICATIONS)

    assert track_evidence_level("live", executor, "joint", [_record()]) == "E0"


def test_provider_validator_errors_fail_closed() -> None:
    qualification = _qualification()
    broken = EvidenceQualificationContract(
        source_id=qualification.source_id,
        allowed_tracks=qualification.allowed_tracks,
        level=qualification.level,
        ceiling=qualification.ceiling,
        payload=qualification.payload,
        receipt=qualification.receipt,
        attestations=(
            EvidenceAttestationRequirement(
                id="broken-provider-attestation",
                validator=_broken_attestation,
            ),
        ),
    )
    executor = _executor(EvidenceQualificationRegistry((broken,)))

    assert track_evidence_level("live", executor, "joint", [_record()]) == "E0"


def test_registry_rejects_duplicates_and_levels_above_their_ceiling() -> None:
    qualification = _qualification()
    with pytest.raises(ValueError, match="duplicate evidence qualification source"):
        EvidenceQualificationRegistry((qualification, qualification))
    with pytest.raises(ValueError, match="invalid evidence qualification contract"):
        _qualification(level="E5", ceiling="E4")
    with pytest.raises(
        ValueError, match="evidence qualification exceeds executor ceiling"
    ):
        _executor(EvidenceQualificationRegistry((qualification,)), ceiling="E3")
