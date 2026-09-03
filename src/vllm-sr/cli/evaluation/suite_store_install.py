"""Normalized benchmark suite installation and qualification workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from cli.evaluation.benchmark_normalization import (
    NormalizationError,
    verify_registered_normalization,
)
from cli.evaluation.benchmark_registry import (
    BenchmarkAdapterDescriptor,
    get_benchmark_adapter,
)
from cli.evaluation.benchmark_sources import (
    SourceVerificationError,
    require_verified_benchmark_source,
)
from cli.evaluation.canonical import digest_value
from cli.evaluation.execution_contract import NORMALIZED_REPLAY_EXECUTOR_ID
from cli.evaluation.suite_contract import (
    BenchmarkSourceReceipt,
    BenchmarkSuiteManifest,
    SuiteArtifactSet,
    qualification_manifest_subject_digest,
)
from cli.evaluation.suite_install_artifacts import stage_suite_artifacts
from cli.evaluation.suite_install_contract import (
    BenchmarkSuiteInstallRequest,
)
from cli.evaluation.suite_qualification import (
    BenchmarkQualificationReceipt,
    NormalizationOrigin,
    UnqualifiedBenchmarkEvidence,
)
from cli.evaluation.suite_store_cas import SuiteCAS
from cli.evaluation.suite_store_error import SuiteStoreError
from cli.evaluation.suite_store_index import SuiteManifestIndex, suite_identity
from cli.evaluation.suite_store_records import SuiteRecordReader


class SuiteInstaller:
    """Validates one source bundle before atomically publishing its manifest."""

    def __init__(
        self,
        cas: SuiteCAS,
        index: SuiteManifestIndex,
        records: SuiteRecordReader,
    ):
        self._cas = cas
        self._index = index
        self._records = records

    @staticmethod
    def _validate_source_receipt(
        descriptor: BenchmarkAdapterDescriptor, receipt: BenchmarkSourceReceipt
    ) -> None:
        source_exact = (
            receipt.adapter_id == descriptor.id
            and receipt.verified
            and receipt.source_clean
            and receipt.expected_source_revision == descriptor.source_revision
            and receipt.observed_source_revision == descriptor.source_revision
        )
        if descriptor.dataset_revision is None:
            dataset_exact = (
                receipt.expected_dataset_revision is None
                and receipt.observed_dataset_revision is None
                and receipt.dataset_clean is None
            )
        else:
            dataset_exact = (
                receipt.expected_dataset_revision == descriptor.dataset_revision
                and receipt.observed_dataset_revision == descriptor.dataset_revision
                and receipt.dataset_clean is True
            )
        if not source_exact or not dataset_exact:
            raise SuiteStoreError(
                "suite source receipt is dirty or does not match the registry exact pin"
            )

    @staticmethod
    def _validate_request_against_adapter(
        request: BenchmarkSuiteInstallRequest,
        descriptor: BenchmarkAdapterDescriptor,
    ) -> None:
        if request.decision_unit != descriptor.decision_unit:
            raise SuiteStoreError("suite decision unit does not match its adapter")
        if request.action_space != descriptor.action_space:
            raise SuiteStoreError("suite action space does not match its adapter")
        if not set(request.track_ids).issubset(descriptor.track_ids):
            raise SuiteStoreError("suite declares a track not supported by its adapter")

    @staticmethod
    def _qualification_receipt(
        origin: NormalizationOrigin,
        source_receipt: BenchmarkSourceReceipt,
        artifacts: SuiteArtifactSet,
        manifest_subject: dict[str, Any],
    ) -> BenchmarkQualificationReceipt:
        return BenchmarkQualificationReceipt(
            evidence_level="E0",
            manifest_subject_digest=qualification_manifest_subject_digest(
                manifest_subject
            ),
            source_receipt_digest=digest_value(source_receipt),
            artifact_set_digest=digest_value(artifacts),
            executor_id=NORMALIZED_REPLAY_EXECUTOR_ID,
            qualification=UnqualifiedBenchmarkEvidence(
                origin=origin,
                parser_verified=origin == "registered_parser_import",
            ),
        )

    def _publish_manifest(
        self,
        request: BenchmarkSuiteInstallRequest,
        artifacts: SuiteArtifactSet,
    ) -> BenchmarkSuiteManifest:
        manifest_fields: dict[str, Any] = {
            "id": request.id,
            "name": request.name,
            "adapter_id": request.adapter_id,
            "source_receipt": request.source_receipt,
            "decision_unit": request.decision_unit,
            "action_space": request.action_space,
            "track_ids": request.track_ids,
            "split_protocol": request.split_protocol,
            "case_count": request.case_count,
            "arm_ids": request.arm_ids,
            "data_classification": request.data_classification,
            "redistribution": request.redistribution,
            "artifacts": artifacts,
            "limitations": request.limitations,
        }
        manifest_fields["qualification_receipt"] = self._qualification_receipt(
            request.normalization_origin,
            request.source_receipt,
            artifacts,
            manifest_fields,
        )
        manifest_seed = BenchmarkSuiteManifest(
            revision="sha256:" + "0" * 64, **manifest_fields
        )
        revision = digest_value(suite_identity(manifest_seed))
        manifest = manifest_seed.model_copy(update={"revision": revision})
        if digest_value(suite_identity(manifest)) != manifest.revision:
            raise SuiteStoreError(
                "suite revision does not match its immutable identity"
            )
        return self._index.publish(manifest)

    def install(
        self,
        request: BenchmarkSuiteInstallRequest,
        bundle_root: str | Path,
        *,
        source_root: str | Path,
        native_export_root: str | Path | None = None,
    ) -> BenchmarkSuiteManifest:
        if self._cas.read_only:
            raise SuiteStoreError("read-only suite store cannot install suites")

        # model_copy(update=...) skips Pydantic validation. Re-parse at this
        # trust boundary so preconstructed objects cannot bypass the contract.
        try:
            request = BenchmarkSuiteInstallRequest.model_validate(
                request.model_dump(mode="json")
            )
        except (AttributeError, ValidationError) as exc:
            raise SuiteStoreError("invalid suite install request") from exc
        descriptor = get_benchmark_adapter(request.adapter_id)
        try:
            source_receipt = require_verified_benchmark_source(descriptor, source_root)
        except SourceVerificationError as exc:
            raise SuiteStoreError(str(exc)) from exc
        request = request.model_copy(update={"source_receipt": source_receipt})
        self._validate_source_receipt(descriptor, request.source_receipt)
        self._validate_request_against_adapter(request, descriptor)
        if request.normalization_origin == "registered_parser_import":
            try:
                verify_registered_normalization(
                    request,
                    source_root=source_root,
                    export_root=native_export_root,
                )
            except NormalizationError as exc:
                raise SuiteStoreError(str(exc)) from exc
        elif native_export_root is not None:
            raise SuiteStoreError(
                "user-provided import cannot claim parser verification through an export root"
            )
        artifact_set = stage_suite_artifacts(
            self._cas,
            self._records,
            request,
            bundle_root,
        )
        return self._publish_manifest(request, artifact_set)
