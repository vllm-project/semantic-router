"""Strict install contract for normalized external benchmark suites.

Adapters normalize upstream data into this fixed, non-executable bundle layout.
Callers select an artifact role; they cannot choose an arbitrary destination or
media type.  Raw source checkouts are never installed into the suite store.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

from pydantic import Field, field_validator, model_validator

from cli.evaluation.constants import TRACK_IDS
from cli.evaluation.contract_primitives import StrictModel
from cli.evaluation.reporting import TrackID
from cli.evaluation.suite_contract import (
    SUITE_CONTRACT_VERSION,
    BenchmarkSourceReceipt,
)
from cli.evaluation.suite_qualification import NormalizationOrigin

SUITE_INSTALL_CONTRACT_VERSION = "evaluation-suite-install.v1"
LICENSE_CONTRACT_VERSION = "evaluation-suite-license.v1"

SuiteArtifactRole = Literal[
    "visible_cases",
    "grading_cases",
    "outcomes",
    "decisions",
    "preferences",
    "trajectories",
    "perturbations",
    "faults",
    "multimodal_observations",
    "safety_observations",
    "capacity_observations",
    "media_manifest",
    "license_manifest",
]

ArtifactDomain = Literal["visible", "grading", "metadata"]

# The layout is deliberately closed.  In particular, an install request cannot
# smuggle an absolute path, executable, archive, or alternate media type into
# the private store.
ARTIFACT_ROLE_LAYOUT: Mapping[SuiteArtifactRole, tuple[str, str, ArtifactDomain]] = (
    MappingProxyType(
        {
            "visible_cases": (
                "visible/cases.jsonl",
                "application/x-ndjson",
                "visible",
            ),
            "grading_cases": (
                "grading/cases.jsonl",
                "application/x-ndjson",
                "grading",
            ),
            "outcomes": (
                "grading/outcomes.jsonl",
                "application/x-ndjson",
                "grading",
            ),
            "decisions": (
                "grading/decisions.jsonl",
                "application/x-ndjson",
                "grading",
            ),
            "preferences": (
                "grading/preferences.jsonl",
                "application/x-ndjson",
                "grading",
            ),
            "trajectories": (
                "grading/trajectories.jsonl",
                "application/x-ndjson",
                "grading",
            ),
            "perturbations": (
                "grading/perturbations.jsonl",
                "application/x-ndjson",
                "grading",
            ),
            "faults": (
                "grading/faults.jsonl",
                "application/x-ndjson",
                "grading",
            ),
            "multimodal_observations": (
                "grading/multimodal-observations.jsonl",
                "application/x-ndjson",
                "grading",
            ),
            "safety_observations": (
                "grading/safety-observations.jsonl",
                "application/x-ndjson",
                "grading",
            ),
            "capacity_observations": (
                "grading/capacity-observations.jsonl",
                "application/x-ndjson",
                "grading",
            ),
            "media_manifest": (
                "metadata/media.jsonl",
                "application/x-ndjson",
                "metadata",
            ),
            "license_manifest": (
                "metadata/licenses.json",
                "application/json",
                "metadata",
            ),
        }
    )
)

REQUIRED_ARTIFACT_ROLES = frozenset(
    {"visible_cases", "grading_cases", "license_manifest"}
)

_PORTABLE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class SuiteArtifactInstall(StrictModel):
    """One normalized file declared by role and exact content identity."""

    role: SuiteArtifactRole
    relative_path: str
    digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0, strict=True)
    media_type: str

    @model_validator(mode="after")
    def matches_closed_layout(self) -> SuiteArtifactInstall:
        expected_path, expected_media_type, _ = ARTIFACT_ROLE_LAYOUT[self.role]
        if self.relative_path != expected_path:
            raise ValueError(
                f"artifact role {self.role!r} requires relative path {expected_path!r}"
            )
        if self.media_type != expected_media_type:
            raise ValueError(
                f"artifact role {self.role!r} requires media type "
                f"{expected_media_type!r}"
            )
        return self


class BenchmarkSuiteInstallRequest(StrictModel):
    """Metadata plus fixed normalized artifacts used to build one manifest."""

    schema_version: Literal[SUITE_INSTALL_CONTRACT_VERSION] = (
        SUITE_INSTALL_CONTRACT_VERSION
    )
    id: str
    name: str = Field(min_length=1, max_length=160)
    adapter_id: str
    source_receipt: BenchmarkSourceReceipt
    decision_unit: str = Field(min_length=1, max_length=256)
    action_space: str = Field(min_length=1, max_length=256)
    track_ids: tuple[TrackID, ...] = Field(min_length=1)
    normalization_origin: NormalizationOrigin
    split_protocol: str = Field(min_length=1, max_length=1024)
    case_count: int = Field(gt=0, strict=True)
    arm_ids: tuple[str, ...] = ()
    data_classification: Literal["public", "internal", "confidential", "restricted"]
    redistribution: Literal["allowed", "metadata_only", "prohibited"]
    artifacts: tuple[SuiteArtifactInstall, ...] = Field(min_length=3)
    limitations: tuple[str, ...] = Field(min_length=1)

    @field_validator("id", "adapter_id")
    @classmethod
    def portable_identifier(cls, value: str) -> str:
        if not _PORTABLE_ID_RE.fullmatch(value):
            raise ValueError("must be a portable identifier")
        return value

    @field_validator("arm_ids")
    @classmethod
    def portable_unique_arm_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("arm ids must be unique")
        if any(not _PORTABLE_ID_RE.fullmatch(item) for item in value):
            raise ValueError("arm ids must be portable identifiers")
        return value

    @model_validator(mode="after")
    def coherent_bundle(self) -> BenchmarkSuiteInstallRequest:
        if self.source_receipt.adapter_id != self.adapter_id:
            raise ValueError("source receipt belongs to a different adapter")
        if (
            self.source_receipt.source_kind == "benchmark_pack"
            and self.normalization_origin != "user_provided_import"
        ):
            raise ValueError("benchmark packs cannot claim registered normalization")
        if (
            self.normalization_origin == "registered_parser_import"
            and self.source_receipt.source_kind != "registered_adapter"
        ):
            raise ValueError("registered normalization requires a registered adapter")
        if len(self.track_ids) != len(set(self.track_ids)):
            raise ValueError("track ids must be unique")
        canonical_tracks = tuple(
            track_id for track_id in TRACK_IDS if track_id in self.track_ids
        )
        if self.track_ids != canonical_tracks:
            raise ValueError("track ids must use canonical catalog order")
        roles = [artifact.role for artifact in self.artifacts]
        if len(roles) != len(set(roles)):
            raise ValueError("artifact roles must be unique")
        missing = REQUIRED_ARTIFACT_ROLES.difference(roles)
        if missing:
            raise ValueError(
                "missing required artifact roles: " + ", ".join(sorted(missing))
            )
        by_role = {artifact.role: artifact for artifact in self.artifacts}
        if by_role["visible_cases"].digest == by_role["grading_cases"].digest:
            raise ValueError("visible and grading artifacts must be separate")
        return self


class NormalizedMediaEntry(StrictModel):
    """Private media inventory; URLs and inline media are intentionally absent."""

    schema_version: Literal[SUITE_CONTRACT_VERSION] = SUITE_CONTRACT_VERSION
    id: str
    digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    media_type: str = Field(min_length=1, max_length=128)
    size_bytes: int = Field(ge=0, strict=True)
    modality: Literal["image", "document", "audio", "video"]
    license_id: str | None = None

    @field_validator("id", "license_id")
    @classmethod
    def portable_media_ids(cls, value: str | None) -> str | None:
        if value is not None and not _PORTABLE_ID_RE.fullmatch(value):
            raise ValueError("must be a portable identifier")
        return value


class SuiteLicenseEntry(StrictModel):
    id: str
    name: str = Field(min_length=1, max_length=256)
    source_url: str | None = Field(default=None, max_length=2048)
    redistribution: Literal["allowed", "metadata_only", "prohibited"]
    notice: str | None = Field(default=None, max_length=16384)

    @field_validator("id")
    @classmethod
    def portable_license_id(cls, value: str) -> str:
        if not _PORTABLE_ID_RE.fullmatch(value):
            raise ValueError("must be a portable identifier")
        return value


class SuiteLicenseManifest(StrictModel):
    schema_version: Literal[LICENSE_CONTRACT_VERSION] = LICENSE_CONTRACT_VERSION
    licenses: tuple[SuiteLicenseEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_license_ids(self) -> SuiteLicenseManifest:
        ids = [license_entry.id for license_entry in self.licenses]
        if len(ids) != len(set(ids)):
            raise ValueError("license ids must be unique")
        return self
