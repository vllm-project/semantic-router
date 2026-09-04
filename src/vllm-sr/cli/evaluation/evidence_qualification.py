"""Immutable qualification contracts for exact, typed live evidence sources."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from cli.evaluation.constants import TRACK_IDS
from cli.evaluation.contract_validation import validate_portable_id
from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.reporting import EvidenceLevel, TrackID

EvidenceRecordValidator = Callable[[ExecutionRecord], bool]
EvidenceRecordBatchValidator = Callable[[list[ExecutionRecord]], bool]
ReceiptScope = Literal["record", "batch"]

_LEVEL_ORDER: tuple[EvidenceLevel, ...] = ("E0", "E1", "E2", "E3", "E4", "E5")


def status_matches_success(record: ExecutionRecord) -> bool:
    return record.success is not None and record.status == (
        "succeeded" if record.success else "failed"
    )


@dataclass(frozen=True)
class TypedEvidencePayloadRequirement:
    """Select and validate one concrete payload carried by an execution record."""

    field_name: str | None
    payload_type: type[object]
    validator: EvidenceRecordValidator

    def __post_init__(self) -> None:
        if not isinstance(self.payload_type, type) or not callable(self.validator):
            raise ValueError("invalid typed evidence payload requirement")
        if self.field_name is not None:
            validate_portable_id(self.field_name)

    def accepts(self, record: ExecutionRecord) -> bool:
        try:
            payload = (
                record if self.field_name is None else getattr(record, self.field_name)
            )
            return (
                isinstance(payload, self.payload_type)
                and self.validator(record) is True
            )
        except Exception:
            return False


@dataclass(frozen=True)
class EvidenceReceiptRequirement:
    """Require a broker-issued receipt with an explicit cardinality scope."""

    scope: ReceiptScope

    def __post_init__(self) -> None:
        if self.scope not in {"record", "batch"}:
            raise ValueError("invalid evidence receipt scope")


@dataclass(frozen=True)
class EvidenceAttestationRequirement:
    """Validate the source-specific facts that make a receipt meaningful."""

    id: str
    validator: EvidenceRecordValidator

    def __post_init__(self) -> None:
        validate_portable_id(self.id)
        if not callable(self.validator):
            raise ValueError("invalid evidence attestation validator")

    def accepts(self, record: ExecutionRecord) -> bool:
        try:
            return self.validator(record) is True
        except Exception:
            return False


@dataclass(frozen=True)
class EvidenceQualificationContract:
    """One provider-owned source admitted by exactly one executor contract."""

    source_id: str
    allowed_tracks: tuple[TrackID, ...]
    level: EvidenceLevel
    ceiling: EvidenceLevel
    payload: TypedEvidencePayloadRequirement
    receipt: EvidenceReceiptRequirement
    attestations: tuple[EvidenceAttestationRequirement, ...]
    batch_validator: EvidenceRecordBatchValidator | None = None

    def __post_init__(self) -> None:
        validate_portable_id(self.source_id)
        canonical_tracks = tuple(
            track_id for track_id in TRACK_IDS if track_id in self.allowed_tracks
        )
        if (
            not self.allowed_tracks
            or self.allowed_tracks != canonical_tracks
            or not self.attestations
            or not isinstance(self.payload, TypedEvidencePayloadRequirement)
            or not isinstance(self.receipt, EvidenceReceiptRequirement)
            or any(
                not isinstance(requirement, EvidenceAttestationRequirement)
                for requirement in self.attestations
            )
            or (self.batch_validator is not None and not callable(self.batch_validator))
            or self.level not in _LEVEL_ORDER
            or self.ceiling not in _LEVEL_ORDER
            or _LEVEL_ORDER.index(self.level) > _LEVEL_ORDER.index(self.ceiling)
        ):
            raise ValueError(
                f"invalid evidence qualification contract: {self.source_id}"
            )
        attestation_ids = tuple(requirement.id for requirement in self.attestations)
        if len(attestation_ids) != len(set(attestation_ids)):
            raise ValueError(
                f"duplicate evidence attestation requirement: {self.source_id}"
            )

    def accepts(self, record: ExecutionRecord) -> bool:
        return (
            record.evidence_kind == self.source_id
            and record.track_id in self.allowed_tracks
            and record.broker_receipt is not None
            and self.payload.accepts(record)
            and all(requirement.accepts(record) for requirement in self.attestations)
        )

    def accepts_batch(self, records: list[ExecutionRecord]) -> bool:
        if self.batch_validator is None:
            return True
        try:
            return self.batch_validator(records) is True
        except Exception:
            return False


@dataclass(frozen=True, init=False)
class EvidenceQualificationRegistry:
    """An immutable exact-source lookup; unknown sources always fail closed."""

    _by_source_id: Mapping[str, EvidenceQualificationContract]

    def __init__(self, contracts: Iterable[EvidenceQualificationContract]):
        by_source_id: dict[str, EvidenceQualificationContract] = {}
        for contract in contracts:
            if not isinstance(contract, EvidenceQualificationContract):
                raise ValueError("evidence qualification registry requires contracts")
            if contract.source_id in by_source_id:
                raise ValueError(
                    f"duplicate evidence qualification source: {contract.source_id}"
                )
            by_source_id[contract.source_id] = contract
        object.__setattr__(self, "_by_source_id", MappingProxyType(by_source_id))

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_source_id))

    @property
    def contracts(self) -> tuple[EvidenceQualificationContract, ...]:
        return tuple(self._by_source_id[source_id] for source_id in self.source_ids)

    def resolve(self, source_id: str | None) -> EvidenceQualificationContract | None:
        if source_id is None:
            return None
        return self._by_source_id.get(source_id)

    def qualify_records(
        self,
        track_id: str,
        records: list[ExecutionRecord],
    ) -> EvidenceLevel | None:
        contracts: dict[str, EvidenceQualificationContract] = {}
        records_by_source: dict[str, list[ExecutionRecord]] = {}
        for record in records:
            if record.track_id != track_id:
                return None
            contract = self.resolve(record.evidence_kind)
            if contract is None or not contract.accepts(record):
                return None
            contracts[contract.source_id] = contract
            records_by_source.setdefault(contract.source_id, []).append(record)
        if not contracts:
            return None
        for source_id, source_records in records_by_source.items():
            contract = contracts[source_id]
            if track_id not in contract.allowed_tracks:
                return None
            receipts = tuple(record.broker_receipt for record in source_records)
            unique_receipts = set(receipts)
            if contract.receipt.scope == "record" and len(receipts) != len(
                unique_receipts
            ):
                return None
            if contract.receipt.scope == "batch" and len(unique_receipts) != 1:
                return None
            if not contract.accepts_batch(source_records):
                return None
        return min(
            (contract.level for contract in contracts.values()),
            key=_LEVEL_ORDER.index,
        )


EMPTY_EVIDENCE_QUALIFICATIONS = EvidenceQualificationRegistry(())
