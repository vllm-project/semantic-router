"""Strict normalized record validation and streaming."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import TypeAlias

from cli.evaluation.canonical import strict_json_load, strict_json_loads
from cli.evaluation.contract_primitives import ArtifactRef, StrictModel
from cli.evaluation.contracts import CaseGrading, CaseVisible
from cli.evaluation.suite_contract import (
    BenchmarkSuiteManifest,
    NormalizedCapacityObservation,
    NormalizedDecision,
    NormalizedFault,
    NormalizedMultimodalObservation,
    NormalizedOutcome,
    NormalizedPerturbation,
    NormalizedPreference,
    NormalizedSafetyObservation,
    NormalizedTrajectoryStep,
)
from cli.evaluation.suite_install_contract import (
    ARTIFACT_ROLE_LAYOUT,
    NormalizedMediaEntry,
    SuiteArtifactRole,
    SuiteLicenseManifest,
)
from cli.evaluation.suite_store_cas import SuiteCAS
from cli.evaluation.suite_store_error import SuiteStoreError

_MAX_JSONL_LINE_BYTES = 16 * 1024 * 1024
_MAX_LICENSE_BYTES = 1024 * 1024

JSONLRecord: TypeAlias = (
    CaseVisible
    | CaseGrading
    | NormalizedCapacityObservation
    | NormalizedDecision
    | NormalizedOutcome
    | NormalizedMultimodalObservation
    | NormalizedPreference
    | NormalizedSafetyObservation
    | NormalizedTrajectoryStep
    | NormalizedPerturbation
    | NormalizedFault
    | NormalizedMediaEntry
)

_JSONL_MODELS: dict[SuiteArtifactRole, type[StrictModel]] = {
    "visible_cases": CaseVisible,
    "grading_cases": CaseGrading,
    "outcomes": NormalizedOutcome,
    "decisions": NormalizedDecision,
    "preferences": NormalizedPreference,
    "trajectories": NormalizedTrajectoryStep,
    "perturbations": NormalizedPerturbation,
    "faults": NormalizedFault,
    "multimodal_observations": NormalizedMultimodalObservation,
    "safety_observations": NormalizedSafetyObservation,
    "capacity_observations": NormalizedCapacityObservation,
    "media_manifest": NormalizedMediaEntry,
}


def validate_artifact_descriptor(
    descriptor: int, role: SuiteArtifactRole, size: int
) -> int:
    if role == "license_manifest":
        return _validate_license_descriptor(descriptor, size)
    return _validate_jsonl_descriptor(descriptor, role)


def _validate_jsonl_descriptor(descriptor: int, role: SuiteArtifactRole) -> int:
    model = _JSONL_MODELS[role]
    os.lseek(descriptor, 0, os.SEEK_SET)
    count = 0
    with os.fdopen(os.dup(descriptor), "rb") as handle:
        while True:
            line = handle.readline(_MAX_JSONL_LINE_BYTES + 1)
            if not line:
                break
            if len(line) > _MAX_JSONL_LINE_BYTES:
                raise SuiteStoreError("normalized JSONL record is too large")
            if not line.endswith(b"\n"):
                raise SuiteStoreError("normalized JSONL must end every record with LF")
            if not line.strip():
                raise SuiteStoreError("normalized JSONL cannot contain blank records")
            try:
                value = strict_json_loads(line)
                model.model_validate(value)
            except (UnicodeDecodeError, ValueError) as exc:
                raise SuiteStoreError(
                    f"invalid normalized {role} record at line {count + 1}"
                ) from exc
            count += 1
    if count == 0:
        raise SuiteStoreError(f"normalized {role} artifact cannot be empty")
    return count


def _validate_license_descriptor(descriptor: int, size: int) -> int:
    if size > _MAX_LICENSE_BYTES:
        raise SuiteStoreError("license manifest exceeds its fixed size limit")
    os.lseek(descriptor, 0, os.SEEK_SET)
    with os.fdopen(os.dup(descriptor), "rb") as handle:
        try:
            value = strict_json_load(handle)
            SuiteLicenseManifest.model_validate(value)
        except (UnicodeDecodeError, ValueError) as exc:
            raise SuiteStoreError("invalid normalized license manifest") from exc
    return 1


class SuiteRecordReader:
    """Streams typed records only after CAS digest and framing verification."""

    def __init__(self, cas: SuiteCAS):
        self._cas = cas

    def iter_ref(
        self, ref: ArtifactRef, role: SuiteArtifactRole
    ) -> Iterator[JSONLRecord]:
        path = self._cas.object_path(role, ref.digest)
        self._cas.verify_ref(path, ref)
        model = _JSONL_MODELS[role]
        descriptor = self._cas.open_readonly(path)
        count = 0
        try:
            with os.fdopen(os.dup(descriptor), "rb") as handle:
                while True:
                    line = handle.readline(_MAX_JSONL_LINE_BYTES + 1)
                    if not line:
                        break
                    if len(line) > _MAX_JSONL_LINE_BYTES or not line.endswith(b"\n"):
                        raise SuiteStoreError("corrupt normalized JSONL record framing")
                    try:
                        value = strict_json_loads(line)
                        yield model.model_validate(value)
                    except (UnicodeDecodeError, ValueError) as exc:
                        raise SuiteStoreError(
                            f"corrupt normalized {role} record at line {count + 1}"
                        ) from exc
                    count += 1
        finally:
            os.close(descriptor)

    def load(
        self, manifest: BenchmarkSuiteManifest, role: SuiteArtifactRole
    ) -> Iterator[JSONLRecord]:
        if role not in ARTIFACT_ROLE_LAYOUT:
            raise SuiteStoreError("unknown suite artifact role")
        if role == "license_manifest":
            raise SuiteStoreError("license_manifest is JSON, not JSONL")
        ref = getattr(manifest.artifacts, role)
        if ref is None:
            raise SuiteStoreError(f"suite has no {role!r} artifact")
        yield from self.iter_ref(ref, role)
