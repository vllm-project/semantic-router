"""Test-only builders for canonical evaluation contracts."""

import hashlib
import uuid
from dataclasses import replace
from datetime import datetime, timezone

from cli.evaluation.canonical import digest_value
from cli.evaluation.capacity_load_contract import (
    CAPACITY_LOAD_CONFIDENCE_LEVEL,
    CAPACITY_LOAD_KIND,
    CAPACITY_MAX_ERROR_RATE_CLUSTER_RANGE,
    MAX_CAPACITY_STABILITY_CV,
    MIN_CAPACITY_MEASUREMENT_CLUSTERS_PER_LEVEL,
    MIN_CAPACITY_MEASUREMENT_REQUESTS,
    MIN_CAPACITY_REPETITIONS,
    MIN_CAPACITY_WARMUP_MULTIPLIER,
    capacity_concurrency_levels,
)
from cli.evaluation.constants import TRACK_IDS
from cli.evaluation.contracts import CapacityLoadProtocol, RunManifest
from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.execution_contract import FIXTURE_REPLAY_EXECUTOR_ID
from cli.evaluation.executor_contracts import ExecutorContract
from cli.evaluation.fixtures import fixture_inputs
from cli.evaluation.manifest_identity import (
    mixture_target_id,
    model_pool_snapshot_digest,
    routing_recipe_plan_digest,
    routing_recipe_target_snapshot_digest,
    selector_snapshot_digest,
)
from cli.evaluation.resolution import resolve_snapshot
from cli.evaluation.routing_recipe_plan import (
    ROUTING_RECIPE_PLAN_CONTRACT_VERSION,
    RoutingRecipeInputSpec,
    RoutingRecipePlan,
    RoutingRecipeProjectionSpec,
    routing_recipe_top_k,
)
from cli.evaluation.runtime_factors import runtime_factors
from cli.evaluation.store import LocalArtifactStore
from cli.evaluation.target_contracts import (
    EvaluationTarget,
    EvaluationTargetArm,
    ManifestMixture,
    MixtureDecisionBinding,
)
from cli.evaluation.worker_report import WorkerReportDraft


def default_capacity_load_protocol(maximum: int) -> CapacityLoadProtocol:
    return CapacityLoadProtocol(
        kind=CAPACITY_LOAD_KIND,
        concurrency_levels=capacity_concurrency_levels(maximum),
        warmup_request_multiplier=MIN_CAPACITY_WARMUP_MULTIPLIER,
        measurement_requests_per_repetition=MIN_CAPACITY_MEASUREMENT_REQUESTS,
        repetitions_per_level=MIN_CAPACITY_REPETITIONS,
        minimum_measurement_clusters_per_level=(
            MIN_CAPACITY_MEASUREMENT_CLUSTERS_PER_LEVEL
        ),
        confidence_level=CAPACITY_LOAD_CONFIDENCE_LEVEL,
        max_error_rate_cluster_range=CAPACITY_MAX_ERROR_RATE_CLUSTER_RANGE,
        max_throughput_cv=MAX_CAPACITY_STABILITY_CV,
        max_latency_p95_cv=MAX_CAPACITY_STABILITY_CV,
    )


def build_routing_recipe_plan(
    *,
    recipe_digest: str,
    pool_digest: str,
    selector_policy_digest: str,
    selector_digest: str,
    adaptation_digest: str,
    binding_digest: str,
    arm_ids: tuple[str, ...],
    fallback_arm_id: str | None,
    signals: tuple[RoutingRecipeInputSpec, ...],
    projections: tuple[RoutingRecipeProjectionSpec, ...],
) -> RoutingRecipePlan:
    target_snapshot_digest = routing_recipe_target_snapshot_digest(
        {
            "recipe_digest": recipe_digest,
            "pool_digest": pool_digest,
            "selector_policy_digest": selector_policy_digest,
            "selector_digest": selector_digest,
            "adaptation_digest": adaptation_digest,
            "binding_digest": binding_digest,
        }
    )
    draft = {
        "contract_version": ROUTING_RECIPE_PLAN_CONTRACT_VERSION,
        "target_snapshot_digest": target_snapshot_digest,
        "arm_ids": tuple(sorted(arm_ids)),
        "fallback_arm_id": fallback_arm_id,
        "signals": tuple(sorted(signals, key=lambda spec: spec.id)),
        "projections": tuple(sorted(projections, key=lambda spec: spec.id)),
        "top_k": routing_recipe_top_k(len(arm_ids)),
    }
    return RoutingRecipePlan(
        **draft,
        plan_digest=routing_recipe_plan_digest(draft),
    )


class _ExecutorStub:
    def __init__(self, executor_id: str):
        self.contract = ExecutorContract(
            id=executor_id,
            mode="replay",
            suite_class="test-provider",
            target_profile="recorded-source",
            lineage_profile="fixture-replay",
            track_ids=TRACK_IDS,
            requires_fixture_ref=True,
        )

    def collect(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("registry contract test must not execute the stub")


def _uuid(name: str) -> str:
    try:
        if str(uuid.UUID(name)) == name:
            return name
    except ValueError:
        pass
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"vllm-sr-evaluation:{name}"))


def _manifest(
    run_id: str = "fixture-run",
    sample_limit: int = 4,
    *,
    baseline_run_id: str | None = None,
    code_revision: str = "sha256:" + "1" * 64,
) -> RunManifest:
    return RunManifest.from_semantic_fields(
        run_id=_uuid(run_id),
        name=f"Evaluation {run_id}",
        description="Engine contract fixture",
        mode="replay",
        target=EvaluationTarget(id="fixture", kind="builtin-fixture"),
        change_profile="schema_adapter",
        gate_contract_version="evaluation-release-gates.v2",
        suite_ids=("evaluation-smoke",),
        suite_revisions={"evaluation-smoke": "builtin-v1"},
        suite_executors={"evaluation-smoke": FIXTURE_REPLAY_EXECUTOR_ID},
        track_ids=TRACK_IDS,
        sample_limit=sample_limit,
        concurrency=2,
        seed=17,
        baseline_run_id=_uuid(baseline_run_id) if baseline_run_id else None,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        code_revision=code_revision,
        policy_snapshot_digest=fixture_inputs().policy.recipe_digest,
        config_digest="sha256:"
        + "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        redaction_policy="public-safe-v1",
    )


def _live_mixture(
    arms: tuple[EvaluationTargetArm, ...],
    *,
    entrypoint_model: str = "entrypoint-a",
) -> ManifestMixture:
    recipe_name = "fixture-recipe"
    recipe_digest = digest_value("live-policy")
    pool_digest = model_pool_snapshot_digest(arms)
    aliases = (entrypoint_model,)
    mixture_id = mixture_target_id(recipe_name)
    selector_policy_digest = digest_value("live-selector-policy")
    selector_digest = selector_snapshot_digest(selector_policy_digest, ())
    adaptation_digest = digest_value("live-adaptation")
    binding_digest = digest_value(f"live-binding:{entrypoint_model}")
    fallback_arm_id = arms[0].id
    return ManifestMixture(
        id=mixture_id,
        entrypoint_model=entrypoint_model,
        aliases=aliases,
        recipe_name=recipe_name,
        recipe_description="Live engine test recipe",
        recipe_digest=recipe_digest,
        pool_digest=pool_digest,
        selector_policy_digest=selector_policy_digest,
        selector_digest=selector_digest,
        adaptation_digest=adaptation_digest,
        binding_digest=binding_digest,
        model_arms=arms,
        support_models=(),
        fallback_arm_id=fallback_arm_id,
        decisions=(
            MixtureDecisionBinding(
                name="default",
                algorithm="static" if len(arms) > 1 else "single",
                arm_ids=tuple(sorted(arm.id for arm in arms)),
            ),
        ),
        routing_recipe_plan=build_routing_recipe_plan(
            recipe_digest=recipe_digest,
            pool_digest=pool_digest,
            selector_policy_digest=selector_policy_digest,
            selector_digest=selector_digest,
            adaptation_digest=adaptation_digest,
            binding_digest=binding_digest,
            arm_ids=tuple(arm.id for arm in arms),
            fallback_arm_id=fallback_arm_id,
            signals=(),
            projections=(),
        ),
    )


def _live_manifest(
    run_id: str,
    *,
    envoy_url: str = "http://envoy:8801",
    price_delta: float = 0,
    topology_digest: str = "sha256:" + "b" * 64,
) -> RunManifest:
    arms = fixture_inputs().arms
    if price_delta:
        first = arms[0].model_copy(
            update={
                "input_cost_per_million_tokens_usd": (
                    arms[0].input_cost_per_million_tokens_usd + price_delta
                )
            }
        )
        arms = (first, *arms[1:])
    mixture = _live_mixture(arms)
    return RunManifest.from_semantic_fields(
        run_id=_uuid(run_id),
        name=f"Live evaluation {run_id}",
        description="Live engine contract fixture",
        mode="live",
        target=EvaluationTarget(
            id=mixture.id,
            kind="mixture-of-models",
            router_api_url="http://router:8080",
            envoy_url=envoy_url,
            backend_topology_digest=topology_digest,
            mixture=mixture,
        ),
        change_profile="recipe",
        gate_contract_version="evaluation-release-gates.v2",
        suite_ids=("live-mom-core",),
        suite_revisions={"live-mom-core": "mom-campaign-cohort-v1"},
        suite_executors={"live-mom-core": "live-runtime.v1"},
        track_ids=("routing",),
        sample_limit=4,
        concurrency=2,
        seed=17,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        code_revision="sha256:" + "1" * 64,
        policy_snapshot_digest=digest_value("live-policy"),
        config_digest="sha256:"
        + "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        redaction_policy="public-safe-v1",
    )


def _resolved_live(manifest: RunManifest, store: LocalArtifactStore):
    factors = runtime_factors(manifest)
    inputs = replace(
        fixture_inputs(),
        policy=factors.policy,
        arms=factors.arms,
        pool=factors.pool,
        binding=factors.binding,
        environment=factors.environment,
        suite_revisions=dict(manifest.suite_revisions),
        suite_executors=dict(manifest.suite_executors),
        executor_ids=dict.fromkeys(manifest.track_ids, "live-runtime.v1"),
    )
    return resolve_snapshot(
        manifest,
        inputs,
        store.put_json(inputs.visible),
        store.put_json(inputs.grading),
        None,
        ("entrypoint-a",),
    )


def _records(store: LocalArtifactStore, run_id: str) -> list[ExecutionRecord]:
    return [
        ExecutionRecord.model_validate_json(line)
        for line in store.read_run_bytes(run_id, "records.jsonl").splitlines()
    ]


def _assert_fixture_run_summary(report: WorkerReportDraft) -> None:
    assert tuple(track.track_id for track in report.tracks) == TRACK_IDS
    assert all(track.status == "completed" for track in report.tracks)
    assert report.summary.coverage.evaluated == report.summary.coverage.total == 29
    assert report.summary.failed_gates == 0
    assert report.summary.quality_score is None
    assert report.summary.runtime_cost is None
    assert report.summary.capacity_tco is None
    assert report.costs.runtime.amount is not None
    assert report.costs.capacity_tco.amount is not None
    assert report.costs.evaluation_overhead.amount is not None
    verdicts = {gate.id: gate.verdict for gate in report.gates}
    assert verdicts["G8"] == "not_applicable"
    assert verdicts["G9"] == "not_applicable"


def _assert_fixture_metrics(report: WorkerReportDraft) -> None:
    metrics = {metric.id: metric.value for metric in report.metrics}
    assert {
        "routing.abstention_rate",
        "routing.fallback_rate",
        "routing.success_rate",
        "routing.selection_entropy_bits",
        "model_pool.best_single_quality",
        "model_pool.arm_count",
        "model_pool.oracle_gain",
        "model_pool.unique_win_rate",
        "model_pool.selection_entropy_bits",
        "model_pool.selection_arm_coverage",
        "model_pool.quality_dominated_arm_count",
        "model_pool.pareto_evaluable_arm_count",
        "model_pool.pareto_dominated_arm_count",
        "model_pool.mean_pairwise_failure_jaccard",
        "model_pool.worst_arm_reliability",
        "model_pool.all_arm_failure_rate",
        "joint.normalized_regret",
        "joint.reliability",
        "joint.oracle_capture_ratio",
        "joint.runtime_cost_per_success",
        "agentic.mean_trajectory_steps",
        "agentic.privacy_exposures_per_trajectory",
        "multimodal.image.support_rate",
        "multimodal.image.quality",
        "preference.effective_sample_size",
        "preference.effective_sample_ratio",
        "preference.self_normalized_ips_agreement",
        "safety.violation_upper_95",
        "safety.false_negative_rate",
        "safety.false_positive_rate",
        "capacity.cost_per_successful_request",
        "capacity.success_concurrency_upper_bound",
    } <= set(metrics)
    assert metrics["safety.violation_rate"] == 0
    assert metrics["safety.violation_upper_95"] > 0
    assert metrics["safety.false_negative_rate"] == 0
    assert metrics["safety.false_positive_rate"] == 0
    assert metrics["model_pool.arm_count"] == 2
    assert metrics["model_pool.quality_dominated_arm_count"] == 0
    assert metrics["model_pool.pareto_evaluable_arm_count"] == 2
    assert metrics["model_pool.pareto_dominated_arm_count"] == 1
    assert metrics["model_pool.mean_pairwise_failure_jaccard"] == 0
    assert metrics["model_pool.worst_arm_reliability"] == 0.75
    assert metrics["model_pool.all_arm_failure_rate"] == 0
    assert metrics["agentic.mean_trajectory_steps"] == 2.5
    assert metrics["preference.effective_sample_size"] == 1
    assert metrics["preference.effective_sample_ratio"] == 1
    assert metrics["preference.self_normalized_ips_agreement"] == 1
    assert metrics["capacity.success_concurrency_upper_bound"] == 8
    assert any(metric_id.endswith("marginal_contribution") for metric_id in metrics)


def _assert_fixture_bundle(
    report: WorkerReportDraft, store: LocalArtifactStore
) -> None:
    names = {artifact.name for artifact in report.artifacts}
    assert names == {
        "metrics.json",
        "gates.json",
        "provenance.json",
        "failure-summary.json",
        "checksums.sha256",
    }
    assert "report.json" not in names
    assert all("/" not in (artifact.uri or "") for artifact in report.artifacts)

    checksum_lines = (
        (store.runs / report.run.id / "checksums.sha256").read_text().splitlines()
    )
    checksums = dict(line.split("  ", 1)[::-1] for line in checksum_lines)
    assert set(checksums) == names - {"checksums.sha256"}
    for name, expected in checksums.items():
        actual = hashlib.sha256(
            (store.runs / report.run.id / name).read_bytes()
        ).hexdigest()
        assert actual == expected

    private_checksum_lines = (
        (store.runs / report.run.id / "private-checksums.sha256")
        .read_text()
        .splitlines()
    )
    private_checksums = dict(
        line.split("  ", 1)[::-1] for line in private_checksum_lines
    )
    assert {
        "run-manifest.json",
        "cases.jsonl",
        "grading-cases.jsonl",
        "records.jsonl",
        "lineage.json",
        "checksums.sha256",
    } <= set(private_checksums)
    assert "private-checksums.sha256" not in names
