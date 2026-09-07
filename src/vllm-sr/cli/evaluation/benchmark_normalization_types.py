"""Closed contracts for safe, source-native benchmark normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import Field, model_validator

from cli.evaluation.contract_primitives import StrictModel
from cli.evaluation.contracts import CaseGrading, CaseVisible
from cli.evaluation.reporting import TrackID
from cli.evaluation.suite_contract import (
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
from cli.evaluation.suite_install_contract import NormalizedMediaEntry

NORMALIZER_DESCRIPTOR_VERSION = "benchmark-normalizer.v1"


class NativeArtifactRequirement(StrictModel):
    """One exact safe export artifact; archives and executable inputs are absent."""

    id: str
    relative_path: str
    media_type: Literal["application/json", "application/x-ndjson", "text/csv"]
    max_bytes: int = Field(gt=0, strict=True)


class NativeMetricMapping(StrictModel):
    """Auditable mapping from one native field to the normalized IR."""

    native_field: str
    normalized_field: str
    interpretation: str


class BenchmarkNormalizerDescriptor(StrictModel):
    schema_version: Literal[NORMALIZER_DESCRIPTOR_VERSION] = (
        NORMALIZER_DESCRIPTOR_VERSION
    )
    adapter_id: str
    export_schema_id: str
    executable: bool
    track_ids: tuple[TrackID, ...]
    required_artifacts: tuple[NativeArtifactRequirement, ...]
    native_metric_mappings: tuple[NativeMetricMapping, ...]
    limitations: tuple[str, ...] = ()
    blocker: str | None = None

    @model_validator(mode="after")
    def executable_contract_is_complete(self) -> BenchmarkNormalizerDescriptor:
        if self.executable:
            if self.blocker is not None:
                raise ValueError("executable normalizer cannot declare a blocker")
            if not self.track_ids or not self.required_artifacts:
                raise ValueError("executable normalizer lacks tracks or artifacts")
            if not self.native_metric_mappings:
                raise ValueError("executable normalizer lacks native metric mappings")
            if not self.limitations:
                raise ValueError(
                    "executable normalizer must declare export limitations"
                )
        elif self.blocker is None or self.track_ids or self.required_artifacts:
            raise ValueError("non-executable normalizer must expose only a blocker")
        paths = tuple(item.relative_path for item in self.required_artifacts)
        if len(paths) != len(set(paths)):
            raise ValueError("native artifact paths must be unique")
        return self


@dataclass(frozen=True)
class NormalizedAdapterPayload:
    """Validated records returned by one explicit native parser."""

    visible_cases: tuple[CaseVisible, ...]
    grading_cases: tuple[CaseGrading, ...]
    split_protocol: str
    arm_ids: tuple[str, ...] = ()
    outcomes: tuple[NormalizedOutcome, ...] = ()
    decisions: tuple[NormalizedDecision, ...] = ()
    preferences: tuple[NormalizedPreference, ...] = ()
    trajectories: tuple[NormalizedTrajectoryStep, ...] = ()
    perturbations: tuple[NormalizedPerturbation, ...] = ()
    faults: tuple[NormalizedFault, ...] = ()
    multimodal_observations: tuple[NormalizedMultimodalObservation, ...] = ()
    safety_observations: tuple[NormalizedSafetyObservation, ...] = ()
    capacity_observations: tuple[NormalizedCapacityObservation, ...] = ()
    media_manifest: tuple[NormalizedMediaEntry, ...] = ()

    @property
    def case_count(self) -> int:
        return len(self.visible_cases)


def artifact(
    artifact_id: str,
    path: str,
    media_type: Literal["application/json", "application/x-ndjson", "text/csv"],
    *,
    max_mib: int = 512,
) -> NativeArtifactRequirement:
    return NativeArtifactRequirement(
        id=artifact_id,
        relative_path=path,
        media_type=media_type,
        max_bytes=max_mib * 1024 * 1024,
    )


def metric(native: str, normalized: str, meaning: str) -> NativeMetricMapping:
    return NativeMetricMapping(
        native_field=native,
        normalized_field=normalized,
        interpretation=meaning,
    )
