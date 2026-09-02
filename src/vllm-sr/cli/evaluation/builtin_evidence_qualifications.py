"""Source-specific qualification contracts owned by built-in live executors."""

from __future__ import annotations

from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.evidence_qualification import (
    EvidenceAttestationRequirement,
    EvidenceQualificationContract,
    EvidenceQualificationRegistry,
    EvidenceReceiptRequirement,
    EvidenceRecordValidator,
    TypedEvidencePayloadRequirement,
    status_matches_success,
)
from cli.evaluation.evidence_source_ids import (
    DECLARED_SHIFT_LIVE_EVIDENCE_SOURCE_ID,
    LIVE_CAPACITY_EVIDENCE_SOURCE_ID,
    LIVE_JOINT_EVIDENCE_SOURCE_ID,
    LIVE_MODEL_POOL_EVIDENCE_SOURCE_ID,
    LIVE_ROUTING_EVIDENCE_SOURCE_ID,
    NORMALIZED_LIVE_MULTIMODAL_EVIDENCE_SOURCE_ID,
)
from cli.evaluation.ledger_evidence_qualifications import (
    LIVE_LEDGER_EVIDENCE_QUALIFICATION_CONTRACTS,
)


def _broker_bound(record: ExecutionRecord) -> bool:
    return record.broker_receipt is not None


def _routing_payload(record: ExecutionRecord) -> bool:
    return (
        status_matches_success(record)
        and record.latency_ms is not None
        and (record.status != "succeeded" or record.trace_digest is not None)
    )


def _model_pool_payload(record: ExecutionRecord) -> bool:
    return (
        status_matches_success(record)
        and record.arm_id is not None
        and record.latency_ms is not None
    )


def _joint_payload(record: ExecutionRecord) -> bool:
    return (
        status_matches_success(record)
        and record.latency_ms is not None
        and (
            record.status != "succeeded"
            or (
                record.selected_arm_id is not None
                and record.selection_method is not None
                and record.recipe is not None
            )
        )
    )


def _capacity_payload(record: ExecutionRecord) -> bool:
    return (
        status_matches_success(record)
        and record.concurrency is not None
        and record.load_phase is not None
        and record.load_repetition is not None
        and record.load_request_index is not None
        and record.latency_ms is not None
        and record.throughput_rps is not None
        and record.load_elapsed_seconds is not None
    )


def _declared_shift_record_payload(record: ExecutionRecord) -> bool:
    method = record.robustness
    return (
        record.status == "succeeded"
        and record.success is True
        and record.selected_arm_id is not None
        and (
            method is None
            or (
                method.method_id == DECLARED_SHIFT_LIVE_EVIDENCE_SOURCE_ID
                and method.target_case_id == record.case_id
            )
        )
    )


def _declared_shift_attestation(record: ExecutionRecord) -> bool:
    method = record.robustness
    return bool(
        method is not None
        and method.suite_id
        and method.suite_revision
        and method.qualification_receipt_digest
        and method.perturbation_artifact_digest
    )


def _declared_shift_batch_attestation(records: list[ExecutionRecord]) -> bool:
    methods = tuple(
        record.robustness for record in records if record.robustness is not None
    )
    if not methods:
        return False
    native_counts = {method.native_pair_count for method in methods}
    if len(native_counts) != 1 or len(methods) != next(iter(native_counts)):
        return False
    by_case_id = {record.case_id: record for record in records}
    if len(by_case_id) != len(records):
        return False
    expected_case_ids: set[str] = set()
    for method in methods:
        expected_case_ids.update((method.source_case_id, method.target_case_id))
        target = by_case_id.get(method.target_case_id)
        if target is None or target.robustness != method:
            return False
        if not _declared_shift_attestation(target):
            return False
    return set(by_case_id) == expected_case_ids


def _normalized_multimodal_payload(record: ExecutionRecord) -> bool:
    return (
        record.status == "succeeded"
        and record.success is True
        and record.modality == "image"
        and record.quality is not None
        and record.grader == "normalized-suite-hidden-answer-exact.v1"
    )


def _unique_case_records(records: list[ExecutionRecord]) -> bool:
    return len({record.case_id for record in records}) == len(records)


_BROKER_ATTESTATION = EvidenceAttestationRequirement(
    id="broker-bound-execution",
    validator=_broker_bound,
)


def _record_payload(
    validator: EvidenceRecordValidator,
) -> TypedEvidencePayloadRequirement:
    return TypedEvidencePayloadRequirement(
        field_name=None,
        payload_type=ExecutionRecord,
        validator=validator,
    )


_LIVE_CAPACITY_QUALIFICATION = EvidenceQualificationContract(
    source_id=LIVE_CAPACITY_EVIDENCE_SOURCE_ID,
    allowed_tracks=("capacity",),
    level="E5",
    ceiling="E5",
    payload=_record_payload(_capacity_payload),
    receipt=EvidenceReceiptRequirement(scope="record"),
    attestations=(_BROKER_ATTESTATION,),
)


LIVE_RUNTIME_EVIDENCE_QUALIFICATIONS = EvidenceQualificationRegistry(
    (
        EvidenceQualificationContract(
            source_id=LIVE_ROUTING_EVIDENCE_SOURCE_ID,
            allowed_tracks=("routing",),
            level="E3",
            ceiling="E3",
            payload=_record_payload(_routing_payload),
            receipt=EvidenceReceiptRequirement(scope="record"),
            attestations=(_BROKER_ATTESTATION,),
            batch_validator=_unique_case_records,
        ),
        EvidenceQualificationContract(
            source_id=LIVE_MODEL_POOL_EVIDENCE_SOURCE_ID,
            allowed_tracks=("model_pool",),
            level="E4",
            ceiling="E4",
            payload=_record_payload(_model_pool_payload),
            receipt=EvidenceReceiptRequirement(scope="record"),
            attestations=(_BROKER_ATTESTATION,),
        ),
        EvidenceQualificationContract(
            source_id=LIVE_JOINT_EVIDENCE_SOURCE_ID,
            allowed_tracks=("joint",),
            level="E5",
            ceiling="E5",
            payload=_record_payload(_joint_payload),
            receipt=EvidenceReceiptRequirement(scope="record"),
            attestations=(_BROKER_ATTESTATION,),
            batch_validator=_unique_case_records,
        ),
        _LIVE_CAPACITY_QUALIFICATION,
        *LIVE_LEDGER_EVIDENCE_QUALIFICATION_CONTRACTS,
    )
)

NORMALIZED_LIVE_EVIDENCE_QUALIFICATIONS = EvidenceQualificationRegistry(
    (
        EvidenceQualificationContract(
            source_id=NORMALIZED_LIVE_MULTIMODAL_EVIDENCE_SOURCE_ID,
            allowed_tracks=("multimodal",),
            level="E4",
            ceiling="E4",
            payload=_record_payload(_normalized_multimodal_payload),
            receipt=EvidenceReceiptRequirement(scope="record"),
            attestations=(_BROKER_ATTESTATION,),
            batch_validator=_unique_case_records,
        ),
        EvidenceQualificationContract(
            source_id=DECLARED_SHIFT_LIVE_EVIDENCE_SOURCE_ID,
            allowed_tracks=("routing",),
            level="E4",
            ceiling="E4",
            payload=_record_payload(_declared_shift_record_payload),
            receipt=EvidenceReceiptRequirement(scope="record"),
            attestations=(_BROKER_ATTESTATION,),
            batch_validator=_declared_shift_batch_attestation,
        ),
    )
)
