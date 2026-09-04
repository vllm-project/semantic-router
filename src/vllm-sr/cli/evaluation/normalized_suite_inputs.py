"""Resolve normalized-suite inputs and opaque replay identities."""

from __future__ import annotations

import hashlib
import heapq
from dataclasses import dataclass
from typing import cast

from cli.evaluation.canonical import digest_value
from cli.evaluation.contracts import (
    BindingSnapshot,
    CaseGrading,
    CaseVisible,
    EvaluationTargetArm,
    GradingCaseSet,
    PolicySnapshot,
    PoolDefinition,
    RunEnvironment,
    VisibleCaseSet,
)
from cli.evaluation.suite_contract import (
    SUITE_CONTRACT_VERSION,
    BenchmarkSuiteManifest,
    NormalizedCapacityObservation,
    NormalizedDecision,
    NormalizedMultimodalObservation,
    NormalizedOutcome,
    NormalizedPreference,
    NormalizedSafetyObservation,
    NormalizedTrajectoryStep,
)
from cli.evaluation.suite_store import NormalizedSuiteStore, SuiteStoreError

EXECUTOR_ID = "normalized-suite-replay.v1"
_OPAQUE_ID_HEX_LENGTH = 24


@dataclass(frozen=True)
class NormalizedSuiteInputs:
    visible: VisibleCaseSet
    grading: GradingCaseSet
    policy: PolicySnapshot
    pool: PoolDefinition
    arms: tuple[EvaluationTargetArm, ...]
    binding: BindingSnapshot
    environment: RunEnvironment
    suite_revisions: dict[str, str]
    private_lineage: dict[str, object]
    executor_ids: dict[str, str]


@dataclass(frozen=True)
class SelectedCase:
    manifest: BenchmarkSuiteManifest
    source_visible: CaseVisible
    source_grading: CaseGrading
    visible: CaseVisible
    grading: CaseGrading


@dataclass(frozen=True)
class SuiteEvidence:
    outcomes: tuple[NormalizedOutcome, ...] | None
    decisions: tuple[NormalizedDecision, ...] | None
    preferences: tuple[NormalizedPreference, ...] | None
    trajectories: tuple[NormalizedTrajectoryStep, ...] | None
    multimodal: tuple[NormalizedMultimodalObservation, ...] | None
    safety: tuple[NormalizedSafetyObservation, ...] | None
    capacity: tuple[NormalizedCapacityObservation, ...] | None


def opaque_id(prefix: str, revision: str, kind: str, source_id: str) -> str:
    digest = hashlib.sha256(
        f"{revision}\x00{kind}\x00{source_id}".encode()
    ).hexdigest()[:_OPAQUE_ID_HEX_LENGTH]
    return f"{prefix}-{digest}"


def case_alias(manifest: BenchmarkSuiteManifest, source_id: str) -> str:
    return opaque_id("case", manifest.revision, "case", source_id)


def arm_alias(manifest: BenchmarkSuiteManifest, source_id: str) -> str:
    return opaque_id("arm", manifest.revision, "arm", source_id)


def action_alias(manifest: BenchmarkSuiteManifest, source_id: str) -> str:
    return opaque_id("action", manifest.revision, "action", source_id)


def trajectory_alias(manifest: BenchmarkSuiteManifest, source_id: str) -> str:
    return opaque_id("trajectory", manifest.revision, "trajectory", source_id)


def evidence_kind(case: SelectedCase) -> str:
    # Installation validates schema, content hashes, and source pins, but it
    # does not yet attest that a repository-owned adapter produced every row.
    # Imported recorded-policy evidence therefore remains contract-only E0.
    return f"{EXECUTOR_ID};ceiling=E0"


def _sample_score(seed: int, manifest: BenchmarkSuiteManifest, case_id: str) -> int:
    digest = hashlib.sha256(
        f"{seed}\x00{manifest.revision}\x00{case_id}".encode()
    ).digest()
    return int.from_bytes(digest, "big")


def _selected_case_keys(
    store: NormalizedSuiteStore,
    manifests: tuple[BenchmarkSuiteManifest, ...],
    sample_limit: int,
    seed: int,
) -> tuple[set[tuple[str, str]], dict[str, set[str]]]:
    """Select the globally smallest deterministic hashes with bounded memory."""

    selected: list[tuple[int, str, str]] = []
    seen: set[tuple[str, str]] = set()
    known_by_suite: dict[str, set[str]] = {}
    for manifest in manifests:
        observed = 0
        known_by_suite[manifest.id] = set()
        for record in store.load_jsonl(manifest.id, "visible_cases"):
            case = cast(CaseVisible, record)
            key = (manifest.id, case.id)
            if key in seen:
                raise SuiteStoreError("normalized visible cases contain a duplicate id")
            seen.add(key)
            known_by_suite[manifest.id].add(case.id)
            observed += 1
            score = _sample_score(seed, manifest, case.id)
            item = (-score, manifest.id, case.id)
            if len(selected) < sample_limit:
                heapq.heappush(selected, item)
            elif score < -selected[0][0]:
                heapq.heapreplace(selected, item)
        if observed != manifest.case_count:
            raise SuiteStoreError("normalized visible case count drifted after install")
    return (
        {(suite_id, case_id) for _, suite_id, case_id in selected},
        known_by_suite,
    )


def load_selected_cases(
    store: NormalizedSuiteStore,
    manifests: tuple[BenchmarkSuiteManifest, ...],
    sample_limit: int,
    seed: int,
) -> tuple[tuple[SelectedCase, ...], dict[str, set[str]]]:
    selected_keys, known_by_suite = _selected_case_keys(
        store, manifests, sample_limit, seed
    )
    manifest_by_id = {manifest.id: manifest for manifest in manifests}
    visible_by_key: dict[tuple[str, str], CaseVisible] = {}
    for manifest in manifests:
        for record in store.load_jsonl(manifest.id, "visible_cases"):
            case = cast(CaseVisible, record)
            key = (manifest.id, case.id)
            if key in selected_keys:
                visible_by_key[key] = case

    grading_by_key: dict[tuple[str, str], CaseGrading] = {}
    for manifest in manifests:
        seen_grading: set[str] = set()
        observed = 0
        for record in store.load_jsonl(manifest.id, "grading_cases"):
            grading = cast(CaseGrading, record)
            if grading.case_id in seen_grading:
                raise SuiteStoreError("normalized grading cases contain a duplicate id")
            seen_grading.add(grading.case_id)
            observed += 1
            key = (manifest.id, grading.case_id)
            if key in selected_keys:
                grading_by_key[key] = grading
        if observed != manifest.case_count:
            raise SuiteStoreError("normalized grading case count drifted after install")
        if seen_grading != known_by_suite[manifest.id]:
            raise SuiteStoreError("visible and grading case identities do not match")

    if set(visible_by_key) != selected_keys or set(grading_by_key) != selected_keys:
        raise SuiteStoreError("visible and grading case identities do not match")

    selected: list[SelectedCase] = []
    for key in sorted(selected_keys):
        manifest = manifest_by_id[key[0]]
        source_visible = visible_by_key[key]
        source_grading = grading_by_key[key]
        selected.append(
            _resolved_selected_case(manifest, source_visible, source_grading)
        )
    return tuple(selected), known_by_suite


def _resolved_selected_case(
    manifest: BenchmarkSuiteManifest,
    source_visible: CaseVisible,
    source_grading: CaseGrading,
) -> SelectedCase:
    resolved_case_id = case_alias(manifest, source_visible.id)
    resolved_trajectory_id = (
        trajectory_alias(manifest, source_visible.trajectory_id)
        if source_visible.trajectory_id
        else None
    )
    visible = source_visible.model_copy(
        update={"id": resolved_case_id, "trajectory_id": resolved_trajectory_id}
    )
    grading = source_grading.model_copy(
        update={
            "case_id": resolved_case_id,
            "expected_route": (
                arm_alias(manifest, source_grading.expected_route)
                if source_grading.expected_route
                else None
            ),
            "preferred_arm_id": (
                arm_alias(manifest, source_grading.preferred_arm_id)
                if source_grading.preferred_arm_id
                else None
            ),
        }
    )
    return SelectedCase(
        manifest=manifest,
        source_visible=source_visible,
        source_grading=source_grading,
        visible=visible,
        grading=grading,
    )


def _recorded_arm_ids(
    manifests: tuple[BenchmarkSuiteManifest, ...],
    evidence_by_suite: dict[str, SuiteEvidence],
) -> set[tuple[str, str]]:
    arm_ids: set[tuple[str, str]] = set()
    for manifest in manifests:
        arm_ids.update((manifest.id, arm_id) for arm_id in manifest.arm_ids)
        evidence = evidence_by_suite[manifest.id]
        if evidence.outcomes:
            arm_ids.update((manifest.id, row.arm_id) for row in evidence.outcomes)
        if evidence.decisions:
            arm_ids.update(
                (manifest.id, row.selected_arm_id)
                for row in evidence.decisions
                if row.selected_arm_id
            )
    return arm_ids


def _recorded_action_ids(
    manifests: tuple[BenchmarkSuiteManifest, ...],
    evidence_by_suite: dict[str, SuiteEvidence],
) -> set[tuple[str, str]]:
    action_ids: set[tuple[str, str]] = set()
    for manifest in manifests:
        evidence = evidence_by_suite[manifest.id]
        if evidence.outcomes:
            action_ids.update(
                (manifest.id, row.action_id)
                for row in evidence.outcomes
                if row.action_id
            )
        if evidence.decisions:
            action_ids.update(
                (manifest.id, row.selected_action_id)
                for row in evidence.decisions
                if row.selected_action_id
            )
        if evidence.preferences:
            _add_preference_action_ids(action_ids, manifest, evidence.preferences)
        if evidence.trajectories:
            action_ids.update(
                (manifest.id, row.selected_action_id)
                for row in evidence.trajectories
                if row.selected_action_id
            )
    return action_ids


def _add_preference_action_ids(
    action_ids: set[tuple[str, str]],
    manifest: BenchmarkSuiteManifest,
    preferences: tuple[NormalizedPreference, ...],
) -> None:
    for row in preferences:
        action_ids.add((manifest.id, row.left_action_id))
        action_ids.add((manifest.id, row.right_action_id))
        if row.chosen_action_id:
            action_ids.add((manifest.id, row.chosen_action_id))


def _normalized_arms(
    manifest_by_id: dict[str, BenchmarkSuiteManifest],
    recorded_arms: list[tuple[str, str]],
) -> tuple[EvaluationTargetArm, ...]:
    return tuple(
        EvaluationTargetArm(
            id=arm_alias(manifest_by_id[suite_id], source_id),
            model=f"normalized-replay-{arm_alias(manifest_by_id[suite_id], source_id)}",
            provider_model_id_digest=digest_value(
                {
                    "suite_revision": manifest_by_id[suite_id].revision,
                    "source_arm_id": source_id,
                }
            ),
            input_cost_per_million_tokens_usd=0.0,
            output_cost_per_million_tokens_usd=0.0,
            runtime_revision=manifest_by_id[suite_id].revision,
            config_digest=digest_value(
                {
                    "kind": "normalized-recorded-arm",
                    "suite_revision": manifest_by_id[suite_id].revision,
                    "source_arm_id": source_id,
                }
            ),
        )
        for suite_id, source_id in recorded_arms
    )


def _normalized_binding(
    revisions: dict[str, str],
    arms: tuple[EvaluationTargetArm, ...],
) -> tuple[PolicySnapshot, PoolDefinition, BindingSnapshot, RunEnvironment]:
    revision_identity = tuple(sorted(revisions.items()))
    identity_digest = digest_value(revision_identity)
    policy = PolicySnapshot(
        id=opaque_id("policy", identity_digest, "policy", "normalized-replay"),
        entrypoint_model="normalized-replay",
        recipe_digest=digest_value(
            {"kind": "normalized-replay-policy", "suite_revisions": revisions}
        ),
    )
    pool = PoolDefinition(
        id=opaque_id("pool", identity_digest, "pool", "normalized-replay"),
        arm_ids=tuple(arm.id for arm in arms),
    )
    binding = BindingSnapshot(
        id=opaque_id("binding", identity_digest, "binding", "normalized-replay"),
        policy_id=policy.id,
        pool_id=pool.id,
    )
    environment = RunEnvironment(
        id=opaque_id(
            "environment", identity_digest, "environment", "normalized-replay"
        ),
        target_id="fixture",
        platform="normalized-suite-replay",
        hardware_class="recorded-unqualified",
    )
    return policy, pool, binding, environment


def build_inputs(
    manifests: tuple[BenchmarkSuiteManifest, ...],
    selected: tuple[SelectedCase, ...],
    evidence_by_suite: dict[str, SuiteEvidence],
    track_ids: tuple[str, ...],
) -> NormalizedSuiteInputs:
    manifest_by_id = {manifest.id: manifest for manifest in manifests}
    revisions = {manifest.id: manifest.revision for manifest in manifests}
    recorded_arms = sorted(_recorded_arm_ids(manifests, evidence_by_suite))
    recorded_actions = sorted(_recorded_action_ids(manifests, evidence_by_suite))
    arms = _normalized_arms(manifest_by_id, recorded_arms)
    policy, pool, binding, environment = _normalized_binding(revisions, arms)
    return NormalizedSuiteInputs(
        visible=VisibleCaseSet(cases=tuple(case.visible for case in selected)),
        grading=GradingCaseSet(cases=tuple(case.grading for case in selected)),
        policy=policy,
        pool=pool,
        arms=arms,
        binding=binding,
        environment=environment,
        suite_revisions=revisions,
        private_lineage=_private_lineage(
            manifest_by_id, revisions, selected, recorded_arms, recorded_actions
        ),
        executor_ids=dict.fromkeys(track_ids, EXECUTOR_ID),
    )


def _private_lineage(
    manifest_by_id: dict[str, BenchmarkSuiteManifest],
    revisions: dict[str, str],
    selected: tuple[SelectedCase, ...],
    recorded_arms: list[tuple[str, str]],
    recorded_actions: list[tuple[str, str]],
) -> dict[str, object]:
    return {
        "schema_version": SUITE_CONTRACT_VERSION,
        "suite_revisions": revisions,
        "case_aliases": [
            {
                "suite_id": case.manifest.id,
                "alias": case.visible.id,
                "source_id": case.source_visible.id,
            }
            for case in selected
        ],
        "arm_aliases": [
            {
                "suite_id": suite_id,
                "alias": arm_alias(manifest_by_id[suite_id], source_id),
                "source_id": source_id,
            }
            for suite_id, source_id in recorded_arms
        ],
        "action_aliases": [
            {
                "suite_id": suite_id,
                "alias": action_alias(manifest_by_id[suite_id], source_id),
                "source_id": source_id,
            }
            for suite_id, source_id in recorded_actions
        ],
    }
