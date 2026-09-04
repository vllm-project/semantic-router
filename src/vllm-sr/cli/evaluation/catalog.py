"""Built-in evaluation catalog exposed to CLI and Dashboard."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from cli.evaluation.constants import SCHEMA_VERSION
from cli.evaluation.contracts import (
    EvaluationTarget,
    EvaluationTargetArm,
    StrictModel,
)
from cli.evaluation.gate_contract import (
    CHANGE_PROFILE_DEFINITIONS,
    GATE_CONTRACT_VERSION,
    ChangeProfile,
)
from cli.evaluation.reporting import EvidenceLevel, TrackID


class CatalogTrack(StrictModel):
    id: TrackID
    name: str
    description: str
    modes: tuple[Literal["replay", "live"], ...]
    metrics: tuple[str, ...]
    evidence_levels: tuple[EvidenceLevel, ...] = ()


class CatalogSuite(StrictModel):
    id: str
    name: str
    description: str
    track_ids: tuple[TrackID, ...]
    modes: tuple[Literal["replay", "live"], ...]
    evidence_level: EvidenceLevel
    case_count: int | None = None
    revision: str | None = None
    tags: tuple[str, ...] = ()


class CatalogTarget(StrictModel):
    id: str
    name: str
    description: str
    kind: str
    track_ids: tuple[TrackID, ...]
    modes: tuple[Literal["replay", "live"], ...]
    evidence_level: EvidenceLevel | None = None
    healthy: bool | None = None
    labels: dict[str, str] | None = None


class CatalogChangeProfile(StrictModel):
    id: ChangeProfile
    name: str
    description: str


class EvaluationCatalog(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    generated_at: datetime | None = None
    gate_contract_version: Literal[GATE_CONTRACT_VERSION] = GATE_CONTRACT_VERSION
    change_profiles: tuple[CatalogChangeProfile, ...]
    tracks: tuple[CatalogTrack, ...]
    suites: tuple[CatalogSuite, ...]
    targets: tuple[CatalogTarget, ...]


_TRACKS = (
    CatalogTrack(
        id="routing",
        name="Routing",
        description="Recipe decisions, coverage, abstention, fallback, and oracle regret.",
        modes=("replay", "live"),
        metrics=(
            "routing.coverage",
            "routing.accuracy",
            "routing.abstention_rate",
            "routing.fallback_rate",
            "routing.latency_p95_ms",
        ),
        evidence_levels=("E0", "E3"),
    ),
    CatalogTrack(
        id="model_pool",
        name="Model pool",
        description="Arm quality, complementarity, unique wins, and pool oracle quality.",
        modes=("replay", "live"),
        metrics=(
            "model_pool.best_single_quality",
            "model_pool.oracle_quality",
            "model_pool.oracle_gain",
            "model_pool.unique_win_rate",
            "model_pool.selection_entropy_bits",
            "model_pool.selection_arm_coverage",
        ),
        evidence_levels=("E0", "E4"),
    ),
    CatalogTrack(
        id="joint",
        name="Routing + pool",
        description="Realized system utility, oracle regret, latency, reliability, and cost.",
        modes=("replay", "live"),
        metrics=(
            "joint.realized_quality",
            "joint.oracle_regret",
            "joint.normalized_regret",
            "joint.reliability",
        ),
        evidence_levels=("E0", "E5"),
    ),
    CatalogTrack(
        id="agentic",
        name="Agentic",
        description="Trajectory success, tool validity, state continuity, and recovery.",
        modes=("replay",),
        metrics=("agentic.success_rate", "agentic.invalid_tool_rate"),
        evidence_levels=("E0",),
    ),
    CatalogTrack(
        id="multimodal",
        name="Multimodal",
        description="Capability-aware routing, grounding quality, and privacy signals.",
        modes=("replay", "live"),
        metrics=("multimodal.support_rate", "multimodal.quality"),
        evidence_levels=("E0", "E5"),
    ),
    CatalogTrack(
        id="preference",
        name="Preference",
        description="Offline preference agreement and propensity-qualified online evidence.",
        modes=("replay",),
        metrics=("preference.agreement", "preference.propensity_coverage"),
        evidence_levels=("E0",),
    ),
    CatalogTrack(
        id="safety",
        name="Safety",
        description="Policy adherence, blocking correctness, privacy, and unsafe regressions.",
        modes=("replay",),
        metrics=(
            "safety.violation_rate",
            "safety.violation_upper_95",
            "safety.block_accuracy",
        ),
        evidence_levels=("E0",),
    ),
    CatalogTrack(
        id="capacity",
        name="Capacity",
        description="Throughput, tail latency, success envelope, GPU efficiency, and TCO.",
        modes=("replay", "live"),
        metrics=(
            "capacity.throughput_rps",
            "capacity.latency_p95_ms",
            "capacity.success_rate",
            "capacity.cost_per_successful_request",
        ),
        evidence_levels=("E0", "E5"),
    ),
)

_ALL_TRACK_IDS = tuple(track.id for track in _TRACKS)

_SUITES = (
    CatalogSuite(
        id="evaluation-smoke",
        name="Evaluation smoke",
        description="Deterministic all-track vertical slice.",
        track_ids=_ALL_TRACK_IDS,
        modes=("replay",),
        evidence_level="E0",
        case_count=4,
        revision="builtin-v1",
        tags=("smoke", "deterministic"),
    ),
    CatalogSuite(
        id="live-routing-core",
        name="Live routing core",
        description="Diagnostic routing smoke using bounded live probes; no promotion-grade policy claim.",
        track_ids=("routing",),
        modes=("live",),
        evidence_level="E0",
        revision="executor-v1",
    ),
    CatalogSuite(
        id="live-model-pool",
        name="Live model pool",
        description="Requires an attested server-owned direct-arm matrix target; unavailable on the generic runtime target.",
        track_ids=("model_pool",),
        modes=("live",),
        evidence_level="E0",
        revision="executor-v1",
    ),
    CatalogSuite(
        id="live-joint",
        name="Live routing + pool",
        description="Requires attested route correlation and direct-arm execution; unavailable on the generic runtime target.",
        track_ids=("routing", "model_pool", "joint"),
        modes=("live",),
        evidence_level="E0",
        revision="executor-v1",
    ),
    CatalogSuite(
        id="live-multimodal",
        name="Live multimodal",
        description="Diagnostic single-probe multimodal smoke; no grounding, privacy, or robustness claim.",
        track_ids=("multimodal",),
        modes=("live",),
        evidence_level="E0",
        revision="executor-v1",
    ),
    CatalogSuite(
        id="live-capacity",
        name="Live capacity",
        description="Diagnostic bounded concurrency smoke without warmup, repeats, duration, or a declared SLO.",
        track_ids=("capacity",),
        modes=("live",),
        evidence_level="E0",
        revision="executor-v1",
    ),
)


def _runtime_tracks(target: EvaluationTarget) -> tuple[TrackID, ...]:
    available: set[TrackID] = set()
    if target.router_api_url:
        available.add("routing")
    if target.envoy_url:
        available.add("capacity")
        if any(
            modality != "text"
            for arm in target.model_arms
            for modality in arm.modalities
        ):
            available.add("multimodal")
    return tuple(track_id for track_id in _ALL_TRACK_IDS if track_id in available)


def get_catalog(
    *,
    generated_at: bool = True,
    router_api_url: str | None = None,
    envoy_url: str | None = None,
    model_arms: tuple[EvaluationTargetArm, ...] = (),
) -> EvaluationCatalog:
    runtime_target = EvaluationTarget(
        id="runtime",
        kind="runtime",
        router_api_url=router_api_url,
        envoy_url=envoy_url,
        model_arms=model_arms,
    )
    runtime_tracks = _runtime_tracks(runtime_target)
    return EvaluationCatalog(
        generated_at=datetime.now(timezone.utc) if generated_at else None,
        change_profiles=tuple(
            CatalogChangeProfile(
                id=profile.id,
                name=profile.name,
                description=profile.description,
            )
            for profile in CHANGE_PROFILE_DEFINITIONS
        ),
        tracks=_TRACKS,
        suites=_SUITES,
        targets=(
            CatalogTarget(
                id="fixture",
                name="Built-in replay fixture",
                description="Deterministic evidence for validating the complete evaluation plane.",
                kind="builtin-fixture",
                track_ids=_ALL_TRACK_IDS,
                modes=("replay",),
                evidence_level="E0",
                healthy=True,
                labels={"execution": "local", "network": "none"},
            ),
            CatalogTarget(
                id="runtime",
                name="Active vLLM-SR runtime",
                description=(
                    "Capabilities derived from server-owned endpoints; model-pool and "
                    "joint evaluation require an attested direct-arm target seam."
                ),
                kind="runtime",
                track_ids=runtime_tracks,
                modes=("live",),
                evidence_level=None,
                healthy=bool(runtime_tracks),
                labels={
                    "capabilities": "manifest-dependent",
                    "credentials": "environment-only",
                    "model_arms": "server-owned",
                    "direct_arms": "unavailable",
                },
            ),
        ),
    )
