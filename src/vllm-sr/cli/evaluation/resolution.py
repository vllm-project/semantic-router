"""Resolve public worker manifests into immutable experiment snapshots."""

from __future__ import annotations

import random

from cli.evaluation.canonical import digest_value
from cli.evaluation.contract_primitives import ArtifactRef
from cli.evaluation.contracts import (
    ExecutorMetadata,
    GradingCaseSet,
    ResolvedRunSnapshot,
    RunManifest,
    VisibleCaseSet,
    WorkloadSnapshot,
)
from cli.evaluation.evidence import ReplayFixture
from cli.evaluation.execution_contract import EvaluationInputs
from cli.evaluation.runtime_factors import RunFactors, runtime_factors


def sample_case_sets(
    visible: VisibleCaseSet,
    grading: GradingCaseSet,
    limit: int,
    seed: int,
) -> tuple[VisibleCaseSet, GradingCaseSet]:
    """Sample aligned visible/hidden cases without requiring replay evidence."""

    visible_ids = tuple(case.id for case in visible.cases)
    grading_ids = tuple(case.case_id for case in grading.cases)
    if visible_ids != grading_ids:
        raise ValueError("visible and grading cases must have identical ordering")
    indices = list(range(len(visible_ids)))
    random.Random(seed).shuffle(indices)
    selected = sorted(indices[: min(limit, len(indices))])
    if not selected:
        raise ValueError("evaluation plan sampled no cases")
    return (
        VisibleCaseSet(cases=tuple(visible.cases[index] for index in selected)),
        GradingCaseSet(cases=tuple(grading.cases[index] for index in selected)),
    )


def sample_fixture(
    inputs: EvaluationInputs,
    limit: int,
    seed: int,
    *,
    eligible_case_ids: frozenset[str] | None = None,
    required_case_groups: tuple[frozenset[str], ...] = (),
) -> EvaluationInputs:
    if inputs.fixture is None:
        raise ValueError("fixture sampling requires replay evidence")
    indices = [
        index
        for index, case in enumerate(inputs.visible.cases)
        if eligible_case_ids is None or case.id in eligible_case_ids
    ]
    if not indices:
        raise ValueError("evaluation plan has no eligible fixture cases")
    count = min(limit, len(indices))
    random.Random(seed).shuffle(indices)
    selected: list[int] = []
    for required_ids in required_case_groups:
        candidates = [
            index for index in indices if inputs.visible.cases[index].id in required_ids
        ]
        if not candidates:
            raise ValueError("evaluation plan has no case for a required track")
        if any(index in selected for index in candidates):
            continue
        if len(selected) == count:
            raise ValueError(
                "sample limit cannot cover every required evaluation track"
            )
        selected.append(candidates[0])
    selected.extend(index for index in indices if index not in selected)
    selected = sorted(selected[:count])
    return EvaluationInputs(
        visible=VisibleCaseSet(
            cases=tuple(inputs.visible.cases[index] for index in selected)
        ),
        grading=GradingCaseSet(
            cases=tuple(inputs.grading.cases[index] for index in selected)
        ),
        fixture=ReplayFixture(
            cases=tuple(inputs.fixture.cases[index] for index in selected)
        ),
        policy=inputs.policy,
        pool=inputs.pool,
        arms=inputs.arms,
        binding=inputs.binding,
        environment=inputs.environment,
        suite_revisions=dict(inputs.suite_revisions),
        suite_executors=dict(inputs.suite_executors),
        executor_ids=dict(inputs.executor_ids),
        private_identity_map=inputs.private_identity_map,
    )


def live_grading(grading: GradingCaseSet) -> GradingCaseSet:
    """Preserve hidden truth for the post-execution grader join."""

    return grading.model_copy(deep=True)


def resolve_snapshot(
    manifest: RunManifest,
    inputs: EvaluationInputs,
    visible_ref: ArtifactRef,
    grading_ref: ArtifactRef,
    fixture_ref: ArtifactRef | None,
    discovered_entrypoints: tuple[str, ...],
) -> ResolvedRunSnapshot:
    workload_id = digest_value(
        {
            "visible_cases": visible_ref.digest,
            "grading_cases": grading_ref.digest,
        }
    ).removeprefix("sha256:")[:16]
    workload = WorkloadSnapshot(
        id=f"workload-{workload_id}",
        visible_cases=visible_ref,
        grading_cases=grading_ref,
    )
    if manifest.mode == "replay":
        factors = RunFactors(
            inputs.policy,
            inputs.arms,
            inputs.pool,
            inputs.binding,
            inputs.environment,
        )
    else:
        factors = runtime_factors(manifest)
        if (
            inputs.policy != factors.policy
            or inputs.arms != factors.arms
            or inputs.pool != factors.pool
            or inputs.binding != factors.binding
            or inputs.environment != factors.environment
        ):
            raise ValueError(
                "live executor inputs do not bind the resolved runtime factors"
            )
    if manifest.policy_snapshot_digest != factors.policy.recipe_digest:
        raise ValueError(
            "manifest policy_snapshot_digest does not match the executed policy"
        )
    if inputs.suite_executors != manifest.suite_executors:
        raise ValueError("executed suite identities do not match the frozen manifest")
    missing_executors = sorted(set(manifest.track_ids) - set(inputs.executor_ids))
    if missing_executors:
        raise ValueError(
            "executor identity is missing for tracks: " + ", ".join(missing_executors)
        )
    executors = tuple(
        ExecutorMetadata(
            track_id=track_id,
            executor_id=inputs.executor_ids[track_id],
            mode=manifest.mode,
        )
        for track_id in manifest.track_ids
    )
    return ResolvedRunSnapshot(
        manifest_digest=manifest.manifest_digest,
        workload=workload,
        policy=factors.policy,
        binding=factors.binding,
        pool=factors.pool,
        arms=factors.arms,
        environment=factors.environment,
        fixture_ref=fixture_ref,
        discovered_entrypoints=discovered_entrypoints,
        executors=executors,
    )
