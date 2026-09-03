"""Load and validate normalized-suite evidence from the private store."""

from __future__ import annotations

from typing import TypeVar, cast

from cli.evaluation.normalized_suite_inputs import SuiteEvidence
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
from cli.evaluation.suite_install_contract import SuiteArtifactRole
from cli.evaluation.suite_store import NormalizedSuiteStore
from cli.evaluation.suite_store_error import SuiteStoreError

_ModelT = TypeVar("_ModelT")


def _load_optional(
    store: NormalizedSuiteStore,
    manifest: BenchmarkSuiteManifest,
    role: SuiteArtifactRole,
    expected_type: type[_ModelT],
) -> tuple[_ModelT, ...] | None:
    if getattr(manifest.artifacts, role) is None:
        return None
    rows = tuple(store.load_jsonl(manifest.id, role))
    if not all(isinstance(row, expected_type) for row in rows):
        raise SuiteStoreError(
            "normalized suite role produced an unexpected record type"
        )
    return cast(tuple[_ModelT, ...], rows)


def load_suite_evidence(
    store: NormalizedSuiteStore,
    manifest: BenchmarkSuiteManifest,
    known_case_ids: set[str],
) -> SuiteEvidence:
    """Load one suite revision and reject cross-case or duplicate evidence."""

    evidence = SuiteEvidence(
        outcomes=_load_optional(store, manifest, "outcomes", NormalizedOutcome),
        decisions=_load_optional(store, manifest, "decisions", NormalizedDecision),
        preferences=_load_optional(
            store, manifest, "preferences", NormalizedPreference
        ),
        trajectories=_load_optional(
            store, manifest, "trajectories", NormalizedTrajectoryStep
        ),
        perturbations=_load_optional(
            store, manifest, "perturbations", NormalizedPerturbation
        ),
        faults=_load_optional(store, manifest, "faults", NormalizedFault),
        multimodal=_load_optional(
            store,
            manifest,
            "multimodal_observations",
            NormalizedMultimodalObservation,
        ),
        safety=_load_optional(
            store, manifest, "safety_observations", NormalizedSafetyObservation
        ),
        capacity=_load_optional(
            store,
            manifest,
            "capacity_observations",
            NormalizedCapacityObservation,
        ),
    )
    for rows in (
        evidence.outcomes,
        evidence.decisions,
        evidence.preferences,
        evidence.trajectories,
        evidence.multimodal,
        evidence.safety,
        evidence.capacity,
    ):
        if rows and any(row.case_id not in known_case_ids for row in rows):
            raise SuiteStoreError("normalized observation references an unknown case")
    if evidence.perturbations and any(
        row.source_case_id not in known_case_ids
        or row.perturbed_case_id not in known_case_ids
        for row in evidence.perturbations
    ):
        raise SuiteStoreError("normalized perturbation references an unknown case")
    if evidence.faults:
        known_trajectories = {
            row.trajectory_id for row in (evidence.trajectories or ())
        }
        if any(row.trajectory_id not in known_trajectories for row in evidence.faults):
            raise SuiteStoreError("normalized fault references an unknown trajectory")
    for role, rows in (
        ("routing decisions", evidence.decisions),
        ("multimodal observations", evidence.multimodal),
        ("safety observations", evidence.safety),
    ):
        if rows:
            case_ids = [row.case_id for row in rows]
            if len(case_ids) != len(set(case_ids)):
                raise SuiteStoreError(f"normalized {role} has duplicate case rows")
    return evidence
