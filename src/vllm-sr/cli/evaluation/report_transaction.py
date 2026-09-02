"""Crash-recoverable publication for one immutable worker report bundle."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import Field, field_validator, model_validator

from cli.evaluation.bundle import (
    REPORT_TRANSACTION_ARTIFACT_NAMES,
    REPORT_TRANSACTION_REQUIRED_NAMES,
    checksum_bytes,
    private_receipt_names,
)
from cli.evaluation.canonical import (
    canonical_json_bytes,
    pretty_json_bytes,
    sha256_digest,
    strict_json_loads,
)
from cli.evaluation.contract_primitives import ArtifactRef, StrictModel
from cli.evaluation.contract_validation import validate_canonical_uuid
from cli.evaluation.errors import StoreError
from cli.evaluation.private_filesystem_publication import DurablePrivateFilesystem

_TRANSACTION_SCHEMA: Final = "evaluation-report-transaction.v1"
_PREPARING_DIRECTORY = ".report-preparing"
_TRANSACTION_DIRECTORY = ".report-transaction"
_TRANSACTION_RECORD = "transaction.json"

ArtifactRetainer = Callable[[str, bytes], ArtifactRef]


class ReportAlreadyPublishedError(StoreError):
    """A valid private report file already occupies the run commit point."""


class ReportTransactionArtifact(StrictModel):
    name: str
    digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if value not in REPORT_TRANSACTION_ARTIFACT_NAMES:
            raise ValueError("report transaction artifact name is invalid")
        return value


class ReportTransactionRecord(StrictModel):
    schema_version: Literal["evaluation-report-transaction.v1"] = _TRANSACTION_SCHEMA
    run_id: str
    manifest_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    artifacts: tuple[ReportTransactionArtifact, ...]

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        validate_canonical_uuid(value)
        return value

    @model_validator(mode="after")
    def validate_artifacts(self) -> ReportTransactionRecord:
        names = tuple(artifact.name for artifact in self.artifacts)
        if len(names) != len(
            set(names)
        ) or not REPORT_TRANSACTION_REQUIRED_NAMES.issubset(names):
            raise ValueError("report transaction artifact set is incomplete")
        if names != tuple(sorted(names)):
            raise ValueError("report transaction artifacts are not canonical")
        return self


def _decode_record(data: bytes) -> ReportTransactionRecord:
    try:
        value = strict_json_loads(data)
        return ReportTransactionRecord.model_validate(value)
    except (TypeError, ValueError) as exc:
        raise StoreError("report transaction record is invalid") from exc


def _verify_artifact(data: bytes, artifact: ReportTransactionArtifact) -> None:
    if len(data) != artifact.size_bytes or sha256_digest(data) != artifact.digest:
        raise StoreError("report transaction artifact digest is invalid")


def _require_private_receipt(
    receipt: bytes,
    expected_manifest_bytes: bytes,
    artifacts: Iterable[ReportTransactionArtifact],
) -> None:
    references = {
        artifact.name: ArtifactRef(
            digest=artifact.digest,
            media_type="application/octet-stream",
            size_bytes=artifact.size_bytes,
        )
        for artifact in artifacts
    }
    manifest_ref = ArtifactRef(
        digest=sha256_digest(expected_manifest_bytes),
        media_type="application/json",
        size_bytes=len(expected_manifest_bytes),
    )
    names = private_receipt_names(references)
    expected = checksum_bytes(
        [
            (
                name,
                manifest_ref if name == "run-manifest.json" else references[name],
            )
            for name in names
        ]
    )
    if receipt != expected:
        raise StoreError(
            "private checksum receipt does not bind the sealed report bundle"
        )


def _owned_atomic_temporary_name(name: str) -> bool:
    for target_name in REPORT_TRANSACTION_ARTIFACT_NAMES | {_TRANSACTION_RECORD}:
        prefix = f".{target_name}."
        if name.startswith(prefix) and re.fullmatch(
            r"[0-9a-f]{8}", name.removeprefix(prefix)
        ):
            return True
    return False


def _remove_staging_directory(
    filesystem: DurablePrivateFilesystem,
    run_dir: Path,
    path: Path,
) -> None:
    """Preflight the full owned shape before removing any staging entry."""

    if not filesystem.directory_exists(path):
        return
    entries = filesystem.directory_entries(path)
    contents: dict[str, bytes] = {}
    for name in entries:
        if name not in REPORT_TRANSACTION_ARTIFACT_NAMES | {
            _TRANSACTION_RECORD
        } and not _owned_atomic_temporary_name(name):
            raise StoreError("report transaction staging contains an unknown entry")
        data = filesystem.read_private_file(path / name)
        if data is None:
            raise StoreError("report transaction staging entry is unavailable")
        contents[name] = data
    if filesystem.directory_entries(path) != entries:
        raise StoreError("report transaction staging changed during preflight")
    for name in entries:
        filesystem.unlink_private_file(
            path / name,
            expected_data=contents[name],
        )
    filesystem.remove_private_directory(path)
    filesystem.sync_directory(run_dir)


def _sealed_preparation_exists(
    filesystem: DurablePrivateFilesystem,
    preparing: Path,
) -> bool:
    if not filesystem.directory_exists(preparing):
        return False
    return (
        filesystem.read_private_file(
            preparing / _TRANSACTION_RECORD,
            missing_ok=True,
        )
        is not None
    )


def _require_record_identity(
    record: ReportTransactionRecord,
    expected_run_id: str,
    expected_manifest_digest: str,
) -> None:
    if (
        record.run_id != expected_run_id
        or record.manifest_digest != expected_manifest_digest
    ):
        raise StoreError("report transaction belongs to another run manifest")


def _promote_sealed_preparation(
    filesystem: DurablePrivateFilesystem,
    run_dir: Path,
    expected_run_id: str,
    expected_manifest_digest: str,
    expected_manifest_bytes: bytes,
) -> bool:
    """Promote a fully sealed preparation interrupted before its rename."""

    preparing = filesystem.within_root(run_dir / _PREPARING_DIRECTORY)
    if not filesystem.directory_exists(preparing):
        return False
    record_data = filesystem.read_private_file(
        preparing / _TRANSACTION_RECORD,
        missing_ok=True,
    )
    if record_data is None:
        _remove_staging_directory(
            filesystem,
            run_dir,
            preparing,
        )
        return False
    record = _decode_record(record_data)
    _require_record_identity(record, expected_run_id, expected_manifest_digest)
    expected_names = {artifact.name for artifact in record.artifacts}
    actual_names = set(filesystem.directory_entries(preparing))
    if actual_names != expected_names | {_TRANSACTION_RECORD}:
        raise StoreError("sealed report preparation has an invalid artifact set")
    for artifact in record.artifacts:
        data = filesystem.read_private_file(preparing / artifact.name)
        if data is None:
            raise StoreError("sealed report preparation lost an artifact")
        _verify_artifact(data, artifact)
        if artifact.name == "private-checksums.sha256":
            _require_private_receipt(
                data,
                expected_manifest_bytes,
                record.artifacts,
            )
    transaction = filesystem.within_root(run_dir / _TRANSACTION_DIRECTORY)
    if filesystem.directory_exists(transaction):
        raise StoreError("multiple sealed report transactions are present")
    filesystem.rename_private_directory(preparing, transaction)
    filesystem.sync_directory(run_dir)
    return True


def _publish_transaction_artifact(
    filesystem: DurablePrivateFilesystem,
    run_dir: Path,
    transaction_dir: Path,
    artifact: ReportTransactionArtifact,
) -> None:
    staged = transaction_dir / artifact.name
    target = run_dir / artifact.name
    staged_data = filesystem.read_private_file(staged, missing_ok=True)
    target_data = filesystem.read_private_file(target, missing_ok=True)
    if staged_data is None and target_data is None:
        raise StoreError("report transaction lost an artifact during publication")
    if staged_data is not None:
        _verify_artifact(staged_data, artifact)
    if target_data is not None:
        _verify_artifact(target_data, artifact)
    if (
        staged_data is not None
        and target_data is not None
        and staged_data != target_data
    ):
        raise StoreError("report transaction conflicts with an immutable artifact")
    if staged_data is None:
        filesystem.sync_directory(run_dir)
        return
    moved = filesystem.replace_private_file(
        staged,
        target,
        expected_data=staged_data,
    )
    if moved:
        # Persist the destination before source removal. If the source name
        # reappears after a crash, equal-duplicate recovery removes it safely;
        # the reverse order can lose the only durable copy.
        filesystem.sync_directory(run_dir)
    filesystem.sync_directory(transaction_dir)
    published_data = filesystem.read_private_file(target)
    if published_data is None:
        raise StoreError("published report transaction artifact is unavailable")
    _verify_artifact(published_data, artifact)


def _publish_transaction(
    filesystem: DurablePrivateFilesystem,
    run_dir: Path,
    expected_run_id: str,
    expected_manifest_digest: str,
    expected_manifest_bytes: bytes,
) -> bool:
    """Resume one sealed transaction; report.json is always published last."""

    transaction_dir = filesystem.within_root(run_dir / _TRANSACTION_DIRECTORY)
    if not filesystem.directory_exists(transaction_dir):
        return False
    record_data = filesystem.read_private_file(
        transaction_dir / _TRANSACTION_RECORD,
        missing_ok=True,
    )
    if record_data is None:
        report_data = filesystem.read_private_file(
            run_dir / "report.json",
            missing_ok=True,
        )
        stranded = filesystem.directory_entries(transaction_dir)
        if report_data is not None and not stranded:
            filesystem.remove_private_directory(transaction_dir)
            filesystem.sync_directory(run_dir)
            return True
        if stranded:
            raise StoreError(
                "completed report transaction retains artifacts without its "
                "commit record"
            )
        raise StoreError("sealed report transaction omits its commit record")
    record = _decode_record(record_data)
    _require_record_identity(record, expected_run_id, expected_manifest_digest)
    expected_names = {artifact.name for artifact in record.artifacts}
    for name in filesystem.directory_entries(transaction_dir):
        if name not in expected_names | {_TRANSACTION_RECORD}:
            raise StoreError("report transaction contains an unknown artifact")
        if filesystem.read_private_file(transaction_dir / name) is None:
            raise StoreError("report transaction artifact is unavailable")
    receipt_artifact = next(
        artifact
        for artifact in record.artifacts
        if artifact.name == "private-checksums.sha256"
    )
    staged_receipt = filesystem.read_private_file(
        transaction_dir / receipt_artifact.name,
        missing_ok=True,
    )
    target_receipt = filesystem.read_private_file(
        run_dir / receipt_artifact.name,
        missing_ok=True,
    )
    if (
        staged_receipt is not None
        and target_receipt is not None
        and staged_receipt != target_receipt
    ):
        raise StoreError("report transaction checksum receipt conflicts")
    receipt = staged_receipt if staged_receipt is not None else target_receipt
    if receipt is None:
        raise StoreError("report transaction lost its private checksum receipt")
    _verify_artifact(receipt, receipt_artifact)
    _require_private_receipt(
        receipt,
        expected_manifest_bytes,
        record.artifacts,
    )
    ordered = sorted(
        record.artifacts,
        key=lambda artifact: (artifact.name == "report.json", artifact.name),
    )
    for artifact in ordered:
        _publish_transaction_artifact(
            filesystem,
            run_dir,
            transaction_dir,
            artifact,
        )
    filesystem.unlink_private_file(
        transaction_dir / _TRANSACTION_RECORD,
        expected_data=record_data,
    )
    filesystem.sync_directory(transaction_dir)
    filesystem.remove_private_directory(transaction_dir)
    filesystem.sync_directory(run_dir)
    return True


class ReportBundleTransaction:
    """Attempt-private builder promoted by one durable transaction record."""

    def __init__(
        self,
        filesystem: DurablePrivateFilesystem,
        retain_artifact: ArtifactRetainer,
        run_dir: Path,
        run_id: str,
        manifest_digest: str,
        expected_manifest_bytes: bytes,
    ) -> None:
        self._filesystem = filesystem
        self._retain_artifact = retain_artifact
        self._run_dir = filesystem.within_root(run_dir)
        self._run_id = run_id
        self._manifest_digest = manifest_digest
        self._expected_manifest_bytes = expected_manifest_bytes
        self._preparing = filesystem.within_root(run_dir / _PREPARING_DIRECTORY)
        self._transaction = filesystem.within_root(run_dir / _TRANSACTION_DIRECTORY)
        existing_report = filesystem.read_private_file(
            self._run_dir / "report.json",
            missing_ok=True,
        )
        if existing_report is not None:
            try:
                report_value = strict_json_loads(existing_report)
            except ValueError as exc:
                raise StoreError("existing report bundle is corrupt") from exc
            if not isinstance(report_value, dict):
                raise StoreError("existing report bundle is corrupt")
            raise ReportAlreadyPublishedError("report bundle is already published")
        if filesystem.directory_exists(self._transaction):
            raise StoreError("prepared report transaction must be recovered first")
        if _sealed_preparation_exists(filesystem, self._preparing):
            raise StoreError("sealed report preparation must be recovered first")
        _remove_staging_directory(
            filesystem,
            self._run_dir,
            self._preparing,
        )
        filesystem.ensure_private_directory(self._preparing, parents=False)
        self._artifacts: dict[str, ArtifactRef] = {}
        self._committed = False

    @property
    def committed(self) -> bool:
        return self._committed

    def write_bytes(self, name: str, data: bytes) -> ArtifactRef:
        if name not in REPORT_TRANSACTION_ARTIFACT_NAMES or name in self._artifacts:
            raise StoreError(
                "report transaction artifact name is invalid or duplicated"
            )
        self._filesystem.atomic_write(self._preparing / name, data)
        reference = self._retain_artifact(name, data)
        self._artifacts[name] = reference
        return reference

    def write_json(self, name: str, value: Any) -> ArtifactRef:
        return self.write_bytes(name, pretty_json_bytes(value))

    def write_jsonl(self, name: str, values: Iterable[Any]) -> ArtifactRef:
        data = b"".join(canonical_json_bytes(value) + b"\n" for value in values)
        return self.write_bytes(name, data)

    def commit(self) -> None:
        if self._committed or "report.json" not in self._artifacts:
            raise StoreError("report transaction is incomplete or already committed")
        record = ReportTransactionRecord(
            run_id=self._run_id,
            manifest_digest=self._manifest_digest,
            artifacts=tuple(
                ReportTransactionArtifact(
                    name=name,
                    digest=reference.digest,
                    size_bytes=reference.size_bytes,
                )
                for name, reference in sorted(self._artifacts.items())
            ),
        )
        private_receipt = self._filesystem.read_private_file(
            self._preparing / "private-checksums.sha256"
        )
        if private_receipt is None:
            raise StoreError("report transaction lost its private checksum receipt")
        _require_private_receipt(
            private_receipt,
            self._expected_manifest_bytes,
            record.artifacts,
        )
        self._filesystem.atomic_write(
            self._preparing / _TRANSACTION_RECORD,
            pretty_json_bytes(record),
        )
        self._filesystem.rename_private_directory(
            self._preparing,
            self._transaction,
        )
        self._filesystem.sync_directory(self._run_dir)
        _publish_transaction(
            self._filesystem,
            self._run_dir,
            self._run_id,
            self._manifest_digest,
            self._expected_manifest_bytes,
        )
        self._committed = True

    def close(self) -> None:
        if not self._committed and not _sealed_preparation_exists(
            self._filesystem,
            self._preparing,
        ):
            _remove_staging_directory(
                self._filesystem,
                self._run_dir,
                self._preparing,
            )


def recover_report_transaction(
    filesystem: DurablePrivateFilesystem,
    run_dir: Path,
    run_id: str,
    manifest_digest: str,
    expected_manifest_bytes: bytes,
) -> bool:
    _promote_sealed_preparation(
        filesystem,
        run_dir,
        run_id,
        manifest_digest,
        expected_manifest_bytes,
    )
    return _publish_transaction(
        filesystem,
        run_dir,
        run_id,
        manifest_digest,
        expected_manifest_bytes,
    )


def report_transaction_pending(
    filesystem: DurablePrivateFilesystem,
    run_dir: Path,
) -> bool:
    """Return whether recovery would inspect, promote, or remove staged state."""

    return filesystem.directory_exists(
        filesystem.within_root(run_dir / _PREPARING_DIRECTORY)
    ) or filesystem.directory_exists(
        filesystem.within_root(run_dir / _TRANSACTION_DIRECTORY)
    )
