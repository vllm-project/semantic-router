"""Resolve public worker manifests into immutable experiment snapshots."""

from __future__ import annotations

import random
from dataclasses import dataclass

from cli.evaluation.canonical import digest_value
from cli.evaluation.contracts import (
    ArtifactRef,
    BindingSnapshot,
    EvaluationTargetArm,
    ExecutorMetadata,
    GradingCaseSet,
    HTTPServiceEndpoint,
    PolicySnapshot,
    PoolDefinition,
    ResolvedRunSnapshot,
    RunEnvironment,
    RunManifest,
    VisibleCaseSet,
    WorkloadSnapshot,
)
from cli.evaluation.evidence import ReplayFixture
from cli.evaluation.fixtures import FixtureInputs
from cli.evaluation.normalized_suite_executor import NormalizedSuiteInputs


@dataclass(frozen=True)
class _RunFactors:
    policy: PolicySnapshot
    arms: tuple[EvaluationTargetArm, ...]
    pool: PoolDefinition
    binding: BindingSnapshot
    environment: RunEnvironment


def _stable_id(prefix: str, value: object) -> str:
    suffix = digest_value(value).removeprefix("sha256:")[:16]
    return f"{prefix}-{suffix}"


def sample_fixture(inputs: FixtureInputs, limit: int, seed: int) -> FixtureInputs:
    count = min(limit, len(inputs.visible.cases))
    indices = list(range(len(inputs.visible.cases)))
    random.Random(seed).shuffle(indices)
    selected = sorted(indices[:count])
    return FixtureInputs(
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
    )


def live_grading(grading: GradingCaseSet) -> GradingCaseSet:
    """Preserve hidden truth for the post-execution grader join."""

    return grading.model_copy(deep=True)


def _live_factors(
    manifest: RunManifest,
    discovered_entrypoints: tuple[str, ...],
) -> _RunFactors:
    recipe_digest = manifest.policy_snapshot_digest
    entrypoint_model = (
        discovered_entrypoints[0] if discovered_entrypoints else "vllm-sr/auto"
    )
    policy = PolicySnapshot(
        id=_stable_id(
            "policy",
            {
                "target_id": manifest.target.id,
                "entrypoint_model": entrypoint_model,
                "recipe_digest": recipe_digest,
            },
        ),
        entrypoint_model=entrypoint_model,
        recipe_digest=recipe_digest,
    )
    arms = manifest.target.model_arms
    pool = PoolDefinition(
        id=_stable_id("pool", arms), arm_ids=tuple(arm.id for arm in arms)
    )
    binding = BindingSnapshot(
        id=_stable_id("binding", {"policy_id": policy.id, "pool_id": pool.id}),
        policy_id=policy.id,
        pool_id=pool.id,
    )
    environment_content = {
        "target_id": manifest.target.id,
        "platform": "runtime",
        "hardware_class": "runtime-reported",
        "route_eval": manifest.target.router_api_url,
        "routed_chat": manifest.target.envoy_url,
        "backend_topology_digest": manifest.target.backend_topology_digest,
    }
    environment = RunEnvironment(
        id=_stable_id("environment", environment_content),
        target_id=manifest.target.id,
        platform="runtime",
        hardware_class="runtime-reported",
        backend_topology_digest=manifest.target.backend_topology_digest,
        route_eval=(
            HTTPServiceEndpoint(url=manifest.target.router_api_url)
            if manifest.target.router_api_url
            else None
        ),
        routed_chat=(
            HTTPServiceEndpoint(url=manifest.target.envoy_url)
            if manifest.target.envoy_url
            else None
        ),
    )
    return _RunFactors(policy, arms, pool, binding, environment)


def resolve_snapshot(
    manifest: RunManifest,
    inputs: FixtureInputs | NormalizedSuiteInputs,
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
        factors = _RunFactors(
            inputs.policy,
            inputs.arms,
            inputs.pool,
            inputs.binding,
            inputs.environment,
        )
    else:
        factors = _live_factors(manifest, discovered_entrypoints)
    if manifest.policy_snapshot_digest != factors.policy.recipe_digest:
        raise ValueError(
            "manifest policy_snapshot_digest does not match the executed policy"
        )
    executor_ids = getattr(inputs, "executor_ids", {})
    executors = tuple(
        ExecutorMetadata(
            track_id=track_id,
            executor_id=executor_ids.get(
                track_id,
                {
                    "routing": "route-eval.v1",
                    "model_pool": "openai-arm-matrix.v1",
                    "joint": "routed-chat.v1",
                    "agentic": "trajectory-replay.v1",
                    "multimodal": "openai-multimodal.v1",
                    "preference": "offline-preference.v1",
                    "safety": "safety-evidence.v1",
                    "capacity": "bounded-load.v1",
                }[track_id],
            ),
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
