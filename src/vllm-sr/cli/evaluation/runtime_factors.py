"""Resolve the immutable factor graph for one current-runtime execution."""

from __future__ import annotations

from dataclasses import dataclass

from cli.evaluation.canonical import digest_value
from cli.evaluation.contracts import RunManifest
from cli.evaluation.target_contracts import (
    BindingSnapshot,
    EvaluationTargetArm,
    HTTPServiceEndpoint,
    PolicySnapshot,
    PoolDefinition,
    RunEnvironment,
)


@dataclass(frozen=True)
class RunFactors:
    policy: PolicySnapshot
    arms: tuple[EvaluationTargetArm, ...]
    pool: PoolDefinition
    binding: BindingSnapshot
    environment: RunEnvironment


def _stable_id(prefix: str, value: object) -> str:
    suffix = digest_value(value).removeprefix("sha256:")[:16]
    return f"{prefix}-{suffix}"


def _digest_id(prefix: str, digest: str) -> str:
    return f"{prefix}-{digest.removeprefix('sha256:')[:16]}"


def runtime_factors(manifest: RunManifest) -> RunFactors:
    """Bind a factor graph directly to a server-frozen Mixture target."""

    mixture = manifest.target.mixture
    if mixture is None:
        raise ValueError("Mixture execution requires a frozen target mixture")
    policy = PolicySnapshot(
        id=_digest_id("policy", mixture.recipe_digest),
        entrypoint_model=mixture.entrypoint_model,
        recipe_digest=mixture.recipe_digest,
    )
    arms = mixture.model_arms
    pool = PoolDefinition(
        id=_digest_id("pool", mixture.pool_digest),
        arm_ids=tuple(arm.id for arm in arms),
    )
    binding = BindingSnapshot(
        id=_digest_id("binding", mixture.binding_digest),
        policy_id=policy.id,
        pool_id=pool.id,
    )
    environment_content = {
        "target_id": manifest.target.id,
        "platform": "runtime" if manifest.mode == "live" else "mixture-replay",
        "hardware_class": (
            "runtime-reported" if manifest.mode == "live" else "frozen-counterfactual"
        ),
        "route_eval": manifest.target.router_api_url,
        "routed_chat": manifest.target.envoy_url,
        "agent_task_ledger": manifest.target.agent_task_ledger,
        "fault_recovery_ledger": manifest.target.fault_recovery_ledger,
        "hard_policy_ledger": manifest.target.hard_policy_ledger,
        "production_experiment_ledger": manifest.target.production_experiment_ledger,
        "backend_topology_digest": manifest.target.backend_topology_digest,
    }
    environment = RunEnvironment(
        id=_stable_id("environment", environment_content),
        target_id=manifest.target.id,
        platform="runtime" if manifest.mode == "live" else "mixture-replay",
        hardware_class=(
            "runtime-reported" if manifest.mode == "live" else "frozen-counterfactual"
        ),
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
        agent_task_ledger=manifest.target.agent_task_ledger,
        fault_recovery_ledger=manifest.target.fault_recovery_ledger,
        hard_policy_ledger=manifest.target.hard_policy_ledger,
        production_experiment_ledger=manifest.target.production_experiment_ledger,
    )
    return RunFactors(policy, arms, pool, binding, environment)
