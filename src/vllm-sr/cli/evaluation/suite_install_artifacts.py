"""Artifact staging and case-plan validation for suite installation."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from cli.evaluation.case_plan import applicable_track_ids
from cli.evaluation.contract_primitives import ArtifactRef, StrictModel
from cli.evaluation.contracts import CaseVisible
from cli.evaluation.errors import SuiteStoreError
from cli.evaluation.suite_contract import SuiteArtifactSet
from cli.evaluation.suite_install_contract import (
    BenchmarkSuiteInstallRequest,
    SuiteArtifactRole,
)
from cli.evaluation.suite_store_cas import SuiteCAS
from cli.evaluation.suite_store_records import (
    SuiteRecordReader,
    validate_artifact_descriptor,
)


def _staged_records(
    records: SuiteRecordReader,
    refs: dict[SuiteArtifactRole, ArtifactRef],
    role: SuiteArtifactRole,
) -> Iterator[StrictModel]:
    ref = refs.get(role)
    if ref is None or role == "license_manifest":
        raise SuiteStoreError(f"normalized suite lacks required role {role!r}")
    yield from records.iter_ref(ref, role)


def _validate_visible_case_plan(
    records: SuiteRecordReader,
    refs: dict[SuiteArtifactRole, ArtifactRef],
    request: BenchmarkSuiteInstallRequest,
) -> None:
    planned_tracks: set[str] = set()
    for record in _staged_records(records, refs, "visible_cases"):
        if not isinstance(record, CaseVisible):
            raise SuiteStoreError("visible case artifact has an invalid type")
        allowed_tracks = applicable_track_ids(
            request.track_ids,
            modality=record.modality,
        )
        canonical_tracks = tuple(
            track_id for track_id in allowed_tracks if track_id in record.track_ids
        )
        if not canonical_tracks or record.track_ids != canonical_tracks:
            raise SuiteStoreError(
                "visible case track plan does not match the suite manifest"
            )
        planned_tracks.update(record.track_ids)
    if planned_tracks != set(request.track_ids):
        raise SuiteStoreError("visible case plans do not cover every suite track")


def stage_suite_artifacts(
    cas: SuiteCAS,
    records: SuiteRecordReader,
    request: BenchmarkSuiteInstallRequest,
    bundle_root: str | Path,
) -> SuiteArtifactSet:
    """Stage all artifacts and validate their cross-record invariants."""

    root = cas.safe_bundle_root(bundle_root)
    refs: dict[SuiteArtifactRole, ArtifactRef] = {}
    counts: dict[SuiteArtifactRole, int] = {}
    for artifact in sorted(request.artifacts, key=lambda item: item.role):
        ref, count = cas.stage_artifact(root, artifact, validate_artifact_descriptor)
        refs[artifact.role] = ref
        counts[artifact.role] = count
    if counts["visible_cases"] != request.case_count:
        raise SuiteStoreError("visible case count does not match install metadata")
    if counts["grading_cases"] != request.case_count:
        raise SuiteStoreError("grading case count does not match install metadata")
    artifacts = SuiteArtifactSet(**refs)
    _validate_visible_case_plan(records, refs, request)
    return artifacts
