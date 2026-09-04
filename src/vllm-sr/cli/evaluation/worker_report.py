"""Strict worker-to-server report draft and stdout event contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from cli.evaluation.constants import SCHEMA_VERSION
from cli.evaluation.contract_primitives import StrictModel
from cli.evaluation.contract_validation import (
    validate_canonical_uuid,
    validate_run_description,
    validate_run_name,
)
from cli.evaluation.contracts import CapacityLoadProtocol, CapacitySLO, RunManifest
from cli.evaluation.gate_contract import GATE_DEFINITIONS, ChangeProfile
from cli.evaluation.reporting import (
    DecisionVerdict,
    EvaluationArtifact,
    EvaluationCostLedgers,
    EvaluationCoverage,
    EvaluationGate,
    EvaluationMetric,
    EvaluationProvenance,
    EvidenceLevel,
    TrackID,
)
from cli.evaluation.target_contracts import CatalogMixture

_MAX_EVENT_MESSAGE_BYTES = 512
_MIN_CAPACITY_CONCURRENCY = 2
_COMPLETE_PROGRESS_PERCENT = 100
WorkerRunStatus = Literal[
    "pending", "running", "sealing", "completed", "failed", "cancelled"
]


class WorkerRunProgress(StrictModel):
    percent: float = Field(ge=0, le=_COMPLETE_PROGRESS_PERCENT)
    completed: int = Field(ge=0)
    total: int = Field(ge=0)
    current_track_id: TrackID | None = None
    message: str | None = None

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str | None) -> str | None:
        if value is not None and (
            value.strip() != value
            or len(value.encode("utf-8")) > _MAX_EVENT_MESSAGE_BYTES
        ):
            raise ValueError("progress message must be trimmed and at most 512 bytes")
        return value


class WorkerRunState(StrictModel):
    """Worker lifecycle echo; server-derived evidence and pair fields are absent."""

    schema_version: Literal[SCHEMA_VERSION]
    id: str
    client_request_id: str
    name: str
    description: str
    status: WorkerRunStatus
    mode: Literal["replay", "live"]
    evidence_level: EvidenceLevel
    target_id: str
    mixture: CatalogMixture | None = None
    change_profile: ChangeProfile
    suite_ids: tuple[str, ...]
    track_ids: tuple[TrackID, ...]
    sample_limit: int
    concurrency: int
    capacity_slo: CapacitySLO | None = None
    capacity_load_protocol: CapacityLoadProtocol | None = None
    seed: int
    baseline_run_id: str | None = None
    progress: WorkerRunProgress
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None

    _id = field_validator("id", "client_request_id")(validate_canonical_uuid)
    _name = field_validator("name")(validate_run_name)
    _description = field_validator("description")(validate_run_description)

    @field_validator("baseline_run_id")
    @classmethod
    def validate_baseline_run_id(cls, value: str | None) -> str | None:
        return validate_canonical_uuid(value) if value is not None else None

    @model_validator(mode="after")
    def client_identity_matches_run(self) -> WorkerRunState:
        if self.client_request_id != self.id:
            raise ValueError("client_request_id must equal the run id")
        # Replay admission is bound to the manifest's registered executor; this
        # lifecycle echo intentionally carries no second executor identity.
        if self.mode == "live" and self.mixture is None:
            raise ValueError("live evaluation run requires its frozen mixture summary")
        capacity_selected = "capacity" in self.track_ids
        if self.mode == "live" and capacity_selected:
            if self.concurrency < _MIN_CAPACITY_CONCURRENCY:
                raise ValueError("live capacity run requires concurrency of at least 2")
            if self.capacity_slo is None:
                raise ValueError("live capacity run requires capacity_slo")
            if self.capacity_load_protocol is None:
                raise ValueError("live capacity run requires capacity_load_protocol")
            if self.capacity_slo.required_concurrency > self.concurrency:
                raise ValueError(
                    "capacity_slo required_concurrency cannot exceed run concurrency"
                )
            if self.capacity_load_protocol.concurrency_levels[-1] != self.concurrency:
                raise ValueError(
                    "capacity_load_protocol must terminate at run concurrency"
                )
        elif self.capacity_slo is not None or self.capacity_load_protocol is not None:
            raise ValueError(
                "capacity_slo and capacity_load_protocol are valid only for a "
                "live capacity run"
            )
        return self


def worker_run_state_from_manifest(
    manifest: RunManifest,
    *,
    status: WorkerRunStatus,
    progress: WorkerRunProgress,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    error: str | None = None,
    evidence_level: EvidenceLevel = "E0",
) -> WorkerRunState:
    """Project immutable manifest fields into one lifecycle state."""

    return WorkerRunState(
        schema_version=SCHEMA_VERSION,
        id=manifest.run_id,
        client_request_id=manifest.run_id,
        name=manifest.name,
        description=manifest.description,
        status=status,
        mode=manifest.mode,
        evidence_level=evidence_level,
        target_id=manifest.target.id,
        mixture=(
            manifest.target.mixture.public_summary()
            if manifest.target.mixture is not None
            else None
        ),
        change_profile=manifest.change_profile,
        suite_ids=manifest.suite_ids,
        track_ids=manifest.track_ids,
        sample_limit=manifest.sample_limit,
        concurrency=manifest.concurrency,
        capacity_slo=manifest.capacity_slo,
        capacity_load_protocol=manifest.capacity_load_protocol,
        seed=manifest.seed,
        baseline_run_id=manifest.baseline_run_id,
        progress=progress,
        created_at=manifest.created_at,
        started_at=started_at,
        completed_at=completed_at,
        error=error,
    )


def require_manifest_run_state(
    manifest: RunManifest,
    run: WorkerRunState,
) -> None:
    """Reject a report lifecycle echo that drifts from its immutable manifest."""

    expected = worker_run_state_from_manifest(
        manifest,
        status=run.status,
        progress=run.progress,
        started_at=run.started_at,
        completed_at=run.completed_at,
        error=run.error,
        evidence_level=run.evidence_level,
    )
    if run != expected:
        raise ValueError(
            "report run metadata does not match its immutable staged manifest"
        )


class WorkerTrackReport(StrictModel):
    track_id: TrackID
    status: Literal[
        "pending",
        "running",
        "completed",
        "failed",
        "cancelled",
        "unavailable",
        "skipped",
    ]
    evidence_level: EvidenceLevel
    summary: str
    coverage: EvaluationCoverage
    metrics: tuple[EvaluationMetric, ...]
    gates: tuple[EvaluationGate, ...]
    artifacts: tuple[EvaluationArtifact, ...] = ()
    error: str | None = None


class WorkerReportSummary(StrictModel):
    verdict: DecisionVerdict
    quality_score: float | None
    latency_p95_ms: float | None
    runtime_cost: float | None
    capacity_tco: float | None
    coverage: EvaluationCoverage
    passed_gates: int = Field(ge=0)
    failed_gates: int = Field(ge=0)
    unavailable_gates: int = Field(ge=0)


class WorkerReportDraft(StrictModel):
    """Untrusted report.json payload consumed and sealed by the Dashboard server."""

    schema_version: Literal[SCHEMA_VERSION]
    run: WorkerRunState
    summary: WorkerReportSummary
    tracks: tuple[WorkerTrackReport, ...]
    metrics: tuple[EvaluationMetric, ...]
    gates: tuple[EvaluationGate, ...]
    costs: EvaluationCostLedgers
    provenance: EvaluationProvenance
    artifacts: tuple[EvaluationArtifact, ...]

    @model_validator(mode="after")
    def coherent_final_run_contract(self) -> WorkerReportDraft:
        if (
            self.run.status != "completed"
            or self.run.started_at is None
            or self.run.completed_at is None
            or self.run.error is not None
            or self.run.progress.percent != _COMPLETE_PROGRESS_PERCENT
            or self.run.progress.completed != len(self.run.track_ids)
            or self.run.progress.total != len(self.run.track_ids)
            or self.run.progress.current_track_id is not None
        ):
            raise ValueError(
                "worker report draft requires one fully completed run state"
            )
        if self.run.completed_at < self.run.started_at:
            raise ValueError("worker report completion cannot precede its start")
        return self

    @model_validator(mode="after")
    def coherent_gate_contract(self) -> WorkerReportDraft:
        expected_ids = tuple(definition.id for definition in GATE_DEFINITIONS)
        if tuple(gate.id for gate in self.gates) != expected_ids:
            raise ValueError(
                "worker draft gates must match the canonical definitions in order"
            )
        if any(gate.change_profile != self.run.change_profile for gate in self.gates):
            raise ValueError("worker draft gates must match the run change profile")
        return self

    @model_validator(mode="after")
    def coherent_track_contract(self) -> WorkerReportDraft:
        track_ids = tuple(track.track_id for track in self.tracks)
        if track_ids != self.run.track_ids:
            raise ValueError(
                "worker draft tracks must exactly match the run track order"
            )
        expected_level = min(
            (track.evidence_level for track in self.tracks), default="E0"
        )
        if self.run.evidence_level != expected_level:
            raise ValueError(
                "worker run evidence_level must equal the weakest selected track"
            )
        return self


class TrackWorkerEventPayload(StrictModel):
    record_count: int = Field(ge=0, le=100_000_000)


class CompletedWorkerEventPayload(StrictModel):
    verdict: DecisionVerdict


WorkerEventPayload = TrackWorkerEventPayload | CompletedWorkerEventPayload
WorkerEventType = Literal[
    "snapshot",
    "progress",
    "track",
    "gate",
    "artifact",
    "completed",
    "failed",
    "cancelled",
]


class WorkerEvent(StrictModel):
    type: WorkerEventType
    message: str
    track_id: TrackID | None = None
    progress: WorkerRunProgress | None = None
    payload: WorkerEventPayload | None = None

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        if (
            not value
            or value.strip() != value
            or len(value.encode("utf-8")) > _MAX_EVENT_MESSAGE_BYTES
        ):
            raise ValueError("worker event message must be 1-512 trimmed bytes")
        return value

    @model_validator(mode="after")
    def payload_matches_event(self) -> WorkerEvent:
        if self.type == "track":
            if not isinstance(self.payload, TrackWorkerEventPayload):
                raise ValueError("track event requires only record_count payload")
        elif self.type == "completed":
            if not isinstance(self.payload, CompletedWorkerEventPayload):
                raise ValueError("completed event requires only verdict payload")
        elif self.payload is not None:
            raise ValueError(f"{self.type} event does not accept a payload")
        return self
