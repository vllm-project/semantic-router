"""Trusted dispatch and atomic materialization for benchmark native exports."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from cli.evaluation.benchmark_normalization_io import (
    NormalizationError,
    safe_export_root,
)
from cli.evaluation.benchmark_normalization_registry import (
    get_benchmark_normalizer_definition,
)
from cli.evaluation.benchmark_normalization_types import (
    BenchmarkNormalizerDescriptor,
    NormalizedAdapterPayload,
)
from cli.evaluation.benchmark_registry import get_benchmark_adapter
from cli.evaluation.benchmark_sources import require_verified_benchmark_source
from cli.evaluation.canonical import canonical_json_bytes, sha256_digest
from cli.evaluation.case_plan import applicable_track_ids
from cli.evaluation.contract_primitives import StrictModel
from cli.evaluation.suite_install_contract import (
    ARTIFACT_ROLE_LAYOUT,
    BenchmarkSuiteInstallRequest,
    SuiteArtifactInstall,
    SuiteArtifactRole,
    SuiteLicenseEntry,
    SuiteLicenseManifest,
)


@dataclass(frozen=True)
class BenchmarkNormalizationResult:
    request: BenchmarkSuiteInstallRequest
    request_path: Path
    bundle_path: Path


@dataclass(frozen=True)
class DerivedBenchmarkNormalization:
    """Pure registered-normalizer output before private materialization."""

    request: BenchmarkSuiteInstallRequest
    artifacts: Mapping[SuiteArtifactRole, bytes]


def _validate_payload(
    payload: NormalizedAdapterPayload,
    descriptor: BenchmarkNormalizerDescriptor,
) -> None:
    if not payload.visible_cases or not payload.grading_cases:
        raise NormalizationError("normalizer produced no cases")
    visible_ids = tuple(case.id for case in payload.visible_cases)
    grading_ids = tuple(case.case_id for case in payload.grading_cases)
    if len(visible_ids) != len(set(visible_ids)) or len(grading_ids) != len(
        set(grading_ids)
    ):
        raise NormalizationError("normalizer produced duplicate case identities")
    if set(visible_ids) != set(grading_ids):
        raise NormalizationError("normalizer visible and grading cases do not align")
    if len(payload.arm_ids) != len(set(payload.arm_ids)):
        raise NormalizationError("normalizer produced duplicate arm identities")
    covered_tracks: set[str] = set()
    for case in payload.visible_cases:
        executable_tracks = applicable_track_ids(
            descriptor.track_ids,
            modality=case.modality,
        )
        if not case.track_ids or not set(case.track_ids).issubset(executable_tracks):
            raise NormalizationError(
                "normalizer case track plan exceeds its executable manifest"
            )
        covered_tracks.update(case.track_ids)
    if covered_tracks != set(descriptor.track_ids):
        raise NormalizationError(
            "normalizer cases do not cover every executable manifest track"
        )


def _jsonl(records: Iterable[StrictModel]) -> bytes:
    return b"".join(canonical_json_bytes(record) + b"\n" for record in records)


def _artifact_bytes(
    payload: NormalizedAdapterPayload,
    source_url: str,
    dataset_url: str | None,
) -> dict[SuiteArtifactRole, bytes]:
    records: dict[SuiteArtifactRole, tuple[StrictModel, ...]] = {
        "visible_cases": payload.visible_cases,
        "grading_cases": payload.grading_cases,
        "outcomes": payload.outcomes,
        "decisions": payload.decisions,
        "preferences": payload.preferences,
        "trajectories": payload.trajectories,
        "perturbations": payload.perturbations,
        "faults": payload.faults,
        "multimodal_observations": payload.multimodal_observations,
        "safety_observations": payload.safety_observations,
        "capacity_observations": payload.capacity_observations,
        "media_manifest": payload.media_manifest,
    }
    result = {role: _jsonl(rows) for role, rows in records.items() if rows}
    licenses = [
        SuiteLicenseEntry(
            id="benchmark-source",
            name="Pinned upstream benchmark source",
            source_url=source_url,
            redistribution="metadata_only",
            notice=(
                "Raw records remain private. Review the pinned upstream license "
                "before any redistribution."
            ),
        )
    ]
    if dataset_url is not None:
        licenses.append(
            SuiteLicenseEntry(
                id="benchmark-dataset",
                name="Pinned upstream benchmark dataset",
                source_url=dataset_url,
                redistribution="metadata_only",
                notice=(
                    "Dataset records remain private. Review the pinned dataset card "
                    "and license before any redistribution."
                ),
            )
        )
    result["license_manifest"] = canonical_json_bytes(
        SuiteLicenseManifest(licenses=tuple(licenses))
    )
    return result


def _artifact_installs(
    artifacts: Mapping[SuiteArtifactRole, bytes],
) -> tuple[SuiteArtifactInstall, ...]:
    installs: list[SuiteArtifactInstall] = []
    for role in ARTIFACT_ROLE_LAYOUT:
        data = artifacts.get(role)
        if data is None:
            continue
        relative_path, media_type, _ = ARTIFACT_ROLE_LAYOUT[role]
        installs.append(
            SuiteArtifactInstall(
                role=role,
                relative_path=relative_path,
                digest=sha256_digest(data),
                size_bytes=len(data),
                media_type=media_type,
            )
        )
    return tuple(installs)


def _write_bundle(
    temp_root: Path, artifacts: Mapping[SuiteArtifactRole, bytes]
) -> None:
    bundle = temp_root / "bundle"
    bundle.mkdir(mode=0o700)
    for role in ARTIFACT_ROLE_LAYOUT:
        data = artifacts.get(role)
        if data is None:
            continue
        relative_path, _, _ = ARTIFACT_ROLE_LAYOUT[role]
        path = bundle / relative_path
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path.parent, 0o700)
        path.write_bytes(data)
        os.chmod(path, 0o600)


def derive_benchmark_normalization(
    *,
    adapter_id: str,
    source_root: str | Path,
    export_root: str | Path,
    suite_id: str,
    suite_name: str | None = None,
) -> DerivedBenchmarkNormalization:
    """Re-run the registered parser and derive an exploratory E0 import.

    This proves parser determinism against the supplied export only. It does
    not attest that upstream benchmark code generated the export.
    """

    definition = get_benchmark_normalizer_definition(adapter_id)
    normalizer = definition.descriptor
    if not normalizer.executable:
        raise NormalizationError(
            f"benchmark adapter {adapter_id!r} is non-executable: {normalizer.blocker}"
        )
    parser = definition.parser
    if parser is None:
        raise NormalizationError(
            "executable benchmark adapter has no registered parser"
        )
    adapter = get_benchmark_adapter(adapter_id)
    receipt = require_verified_benchmark_source(adapter, source_root)
    native_root = safe_export_root(export_root)
    payload = parser(native_root, normalizer)
    _validate_payload(payload, normalizer)
    artifacts = MappingProxyType(
        _artifact_bytes(payload, adapter.source_url, adapter.dataset_url)
    )
    request = BenchmarkSuiteInstallRequest(
        id=suite_id,
        name=suite_name or f"{adapter.name} normalized replay",
        adapter_id=adapter.id,
        source_receipt=receipt,
        decision_unit=adapter.decision_unit,
        action_space=adapter.action_space,
        track_ids=normalizer.track_ids,
        normalization_origin="registered_parser_import",
        split_protocol=payload.split_protocol,
        case_count=payload.case_count,
        arm_ids=payload.arm_ids,
        data_classification="restricted",
        redistribution="metadata_only",
        artifacts=_artifact_installs(artifacts),
        limitations=adapter.limitations
        + normalizer.limitations
        + (
            f"Parsed deterministically from {normalizer.export_schema_id}; upstream benchmark execution was not attested.",
            "This imported suite is exploratory E0 evidence and cannot qualify promotion gates.",
        ),
    )
    return DerivedBenchmarkNormalization(request=request, artifacts=artifacts)


def verify_registered_normalization(
    request: BenchmarkSuiteInstallRequest,
    *,
    source_root: str | Path,
    export_root: str | Path | None,
) -> None:
    """Prove a parser-verified import is the exact current parser output."""

    if export_root is None:
        raise NormalizationError(
            "parser-verified import requires its frozen native export root"
        )
    derived = derive_benchmark_normalization(
        adapter_id=request.adapter_id,
        source_root=source_root,
        export_root=export_root,
        suite_id=request.id,
        suite_name=request.name,
    )
    if request != derived.request:
        raise NormalizationError(
            "parser-verified import is not the exact registered normalizer output"
        )


def normalize_benchmark_suite(
    *,
    adapter_id: str,
    source_root: str | Path,
    export_root: str | Path,
    output_root: str | Path,
    suite_id: str,
    suite_name: str | None = None,
) -> BenchmarkNormalizationResult:
    """Verify pins, parse one closed native schema, and atomically write a bundle."""

    derived = derive_benchmark_normalization(
        adapter_id=adapter_id,
        source_root=source_root,
        export_root=export_root,
        suite_id=suite_id,
        suite_name=suite_name,
    )

    output = Path(output_root).expanduser()
    if output.exists() or output.is_symlink():
        raise NormalizationError("normalization output is immutable and already exists")
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = Path(tempfile.mkdtemp(prefix=".suite-normalize-", dir=output.parent))
    os.chmod(temporary, 0o700)
    try:
        _write_bundle(temporary, derived.artifacts)
        request = derived.request
        request_path = temporary / "request.json"
        request_path.write_bytes(canonical_json_bytes(request))
        os.chmod(request_path, 0o600)
        os.rename(temporary, output)
    except Exception:
        # Temporary output contains only newly generated private files.
        for path in sorted(temporary.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        if temporary.exists():
            temporary.rmdir()
        raise
    return BenchmarkNormalizationResult(
        request=request,
        request_path=output / "request.json",
        bundle_path=output / "bundle",
    )
