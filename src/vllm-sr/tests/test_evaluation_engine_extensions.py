from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from cli.evaluation.artifact_store_error import StoreError
from cli.evaluation.builtin_executors import (
    LiveRuntimeExecutor,
)
from cli.evaluation.canonical import digest_value
from cli.evaluation.case_plan import project_visible_case_set
from cli.evaluation.catalog import get_catalog
from cli.evaluation.catalog_suites import CatalogSuite
from cli.evaluation.catalog_tracks import (
    CATALOG_METHOD_EVIDENCE_SOURCES,
    CatalogMethod,
    CatalogMethodEvidenceSource,
)
from cli.evaluation.compare import compare_worker_drafts
from cli.evaluation.constants import TRACK_IDS
from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.evidence_level import run_evidence_level, track_evidence_level
from cli.evaluation.evidence_source_ids import LIVE_JOINT_EVIDENCE_SOURCE_ID
from cli.evaluation.execution_plan import (
    SuiteRegistry,
)
from cli.evaluation.executor_contracts import ExecutorContract
from cli.evaluation.executor_registry import CollectedEvidence, ExecutorRegistry
from cli.evaluation.fixture_executor import execute_fixture
from cli.evaluation.fixtures import fixture_inputs
from cli.evaluation.gates import compute_gates
from cli.evaluation.metric_analysis_catalog import static_metric_ids_for_track
from cli.evaluation.metric_core import canonical_ordered_float_sum
from cli.evaluation.metrics import compute_metrics
from cli.evaluation.orchestrator import run_evaluation, validate_manifest
from cli.evaluation.report_builder import _track_plan_totals
from cli.evaluation.resolution import sample_fixture
from cli.evaluation.runtime_factors import runtime_factors
from cli.evaluation.store import LocalArtifactStore
from cli.evaluation.target_capabilities import TargetContract, TargetRegistry
from cli.evaluation.target_contracts import EvaluationTarget, HTTPServiceEndpoint
from evaluation_contract_test_support import (
    _ExecutorStub,
    _live_manifest,
    _live_mixture,
    _manifest,
    _records,
    _uuid,
)


class _ProviderAgenticExecutor:
    contract = ExecutorContract(
        id="provider-agentic-live.v1",
        mode="live",
        suite_class="runtime",
        target_profile="brokered-runtime",
        lineage_profile="runtime",
        track_ids=("agentic", "preference", "safety"),
        evidence_level_ceiling="E0",
    )

    def collect(self, manifest, store, plan, suite_store) -> CollectedEvidence:
        del suite_store
        source = sample_fixture(fixture_inputs(), manifest.sample_limit, manifest.seed)
        assert source.fixture is not None
        visible = project_visible_case_set(source.visible, manifest.track_ids)
        records = execute_fixture(
            visible, source.grading, source.fixture, manifest.track_ids
        )
        assert manifest.target.mixture is not None
        discovered_entrypoints = manifest.target.mixture.aliases
        factors = runtime_factors(manifest)
        inputs = replace(
            source,
            visible=visible,
            fixture=None,
            policy=factors.policy,
            arms=factors.arms,
            pool=factors.pool,
            binding=factors.binding,
            environment=factors.environment,
            suite_revisions=dict(plan.suite_revisions),
            suite_executors=dict(plan.suite_executors),
            executor_ids=dict.fromkeys(manifest.track_ids, self.contract.id),
        )
        return CollectedEvidence(
            inputs=inputs,
            visible_ref=store.put_json(inputs.visible),
            grading_ref=store.put_json(inputs.grading),
            fixture_ref=None,
            records=records,
            discovered_entrypoints=discovered_entrypoints,
            routing_traces=(),
        )


def test_suite_and_executor_registries_form_an_explicit_extension_boundary() -> None:
    manifest = _manifest().with_semantic_updates(
        target=EvaluationTarget(id="provider-source", kind="provider-recording"),
        suite_ids=("custom-routing",),
        suite_revisions={"custom-routing": "custom-revision"},
        suite_executors={"custom-routing": "custom-executor.v1"},
        track_ids=("routing",),
    )
    suites = SuiteRegistry(
        (
            CatalogSuite(
                id="custom-routing",
                name="Custom routing",
                description="Custom registry contract",
                track_ids=("routing",),
                modes=("replay",),
                evidence_level="E0",
                executors={"replay": "custom-executor.v1"},
                revision="custom-revision",
                methods=(
                    CatalogMethod(
                        id="custom.routing.v1",
                        track_id="routing",
                        qualified_gate_ids=(),
                        evidence_source="diagnostic_fixture",
                        status="configured",
                    ),
                ),
            ),
        )
    )
    executors = ExecutorRegistry((_ExecutorStub("custom-executor.v1"),))
    targets = TargetRegistry(
        (
            TargetContract(
                id="provider-source",
                name="Provider source",
                description="Custom provider recording.",
                kind="provider-recording",
                track_requirements={"routing": frozenset()},
                modes=("replay",),
                accepted_executors={"replay": ("custom-executor.v1",)},
                execution_profile="recorded-source",
                policy_snapshot_profile="fixture",
                health_profile="always",
            ),
        ),
        (executors.contract("custom-executor.v1"),),
    )

    validate_manifest(
        manifest,
        executor_registry=executors,
        suite_registry=suites,
        target_registry=targets,
    )


def test_catalog_method_evidence_source_inventory_is_strict_and_extensible() -> None:
    assert CATALOG_METHOD_EVIDENCE_SOURCES
    seen: set[str] = set()
    for source in CATALOG_METHOD_EVIDENCE_SOURCES:
        gate_ids = (
            ("G4",)
            if source is CatalogMethodEvidenceSource.SERVER_BROKERED_LIVE
            else ()
        )
        method = CatalogMethod(
            id=f"inventory.{source.value}",
            track_id="routing",
            qualified_gate_ids=gate_ids,
            evidence_source=source,
            status="configured",
        )
        assert method.evidence_source is source
        assert method.model_dump(mode="json")["evidence_source"] == source.value
        seen.add(source.value)
    assert len(seen) == len(CATALOG_METHOD_EVIDENCE_SOURCES)

    with pytest.raises(ValueError):
        CatalogMethod.model_validate(
            {
                "id": "inventory.unknown",
                "track_id": "routing",
                "qualified_gate_ids": [],
                "evidence_source": "unknown_source",
                "status": "configured",
            }
        )


def test_provider_agentic_capability_executes_with_duplicate_suite_class(
    tmp_path: Path,
) -> None:
    executor = _ProviderAgenticExecutor()
    mixture = _live_mixture(
        fixture_inputs().arms,
        entrypoint_model="provider-agent-entrypoint",
    )
    manifest = _live_manifest("provider-agentic-e2e").with_semantic_updates(
        target=EvaluationTarget(
            id=mixture.id,
            kind="mixture-of-models",
            envoy_url="http://provider.invalid",
            fault_recovery_ledger=HTTPServiceEndpoint(
                url="http://provider-recovery-ledger.invalid"
            ),
            hard_policy_ledger=HTTPServiceEndpoint(
                url="http://provider-policy-ledger.invalid"
            ),
            production_experiment_ledger=HTTPServiceEndpoint(
                url="http://provider-experiment-ledger.invalid"
            ),
            backend_topology_digest="sha256:" + "c" * 64,
            mixture=mixture,
        ),
        change_profile="agent_multimodal",
        suite_ids=("provider-agentic",),
        suite_revisions={"provider-agentic": "provider-v1"},
        suite_executors={"provider-agentic": executor.contract.id},
        track_ids=("agentic", "preference", "safety"),
    )
    suites = SuiteRegistry(
        (
            CatalogSuite(
                id="provider-agentic",
                name="Provider agentic",
                description="Provider-owned live agent trajectory evaluation.",
                track_ids=("agentic", "preference", "safety"),
                modes=("live",),
                evidence_level="E0",
                executors={"live": executor.contract.id},
                revision="provider-v1",
                methods=tuple(
                    CatalogMethod(
                        id=f"provider.{track_id}.v1",
                        track_id=track_id,
                        qualified_gate_ids=(),
                        evidence_source="live_runtime",
                        status="configured",
                    )
                    for track_id in ("agentic", "preference", "safety")
                ),
            ),
        )
    )
    executors = ExecutorRegistry((LiveRuntimeExecutor(), executor))
    targets = TargetRegistry(
        (
            TargetContract(
                id=mixture.id,
                name="Provider agent runtime",
                description="Provider-owned agent trajectory runtime.",
                kind="mixture-of-models",
                track_requirements={
                    "agentic": frozenset({"agent-runtime", "fault_recovery_ledger"}),
                    "preference": frozenset({"production_experiment_ledger"}),
                    "safety": frozenset({"hard_policy_ledger"}),
                },
                modes=("live",),
                accepted_executors={"live": (executor.contract.id,)},
                execution_profile="brokered-runtime",
                policy_snapshot_profile="runtime-config",
                health_profile="capabilities",
                provided_features=frozenset({"agent-runtime"}),
            ),
        ),
        (LiveRuntimeExecutor.contract, executor.contract),
    )

    report = run_evaluation(
        manifest,
        LocalArtifactStore(tmp_path / "store"),
        executor_registry=executors,
        suite_registry=suites,
        target_registry=targets,
    )

    assert report.run.status == "completed"
    assert report.run.target_id == mixture.id
    assert tuple(track.track_id for track in report.tracks) == (
        "agentic",
        "preference",
        "safety",
    )
    assert all(track.coverage.evaluated > 0 for track in report.tracks)
    assert report.provenance.benchmark_revisions == {"provider-agentic": "provider-v1"}


def test_mixture_target_does_not_advertise_unconfigured_ledgers() -> None:
    mixture = _live_mixture(fixture_inputs().arms)
    catalog = get_catalog(
        generated_at=False,
        router_api_url="http://router.invalid",
        envoy_url="http://envoy.invalid",
        backend_topology_digest="sha256:" + "d" * 64,
        mixture=mixture,
    )
    runtime = next(target for target in catalog.targets if target.id == mixture.id)

    assert "preference" not in runtime.track_ids
    assert "safety" not in runtime.track_ids
    assert "agentic" not in runtime.track_ids


def test_every_mixture_manifest_requires_at_least_two_frozen_arms() -> None:
    manifest = _live_manifest("single-arm-routing")
    mixture = _live_mixture((fixture_inputs().arms[0],))
    target = manifest.target.model_copy(update={"id": mixture.id, "mixture": mixture})

    with pytest.raises(ValueError, match="at least two arms"):
        manifest.with_semantic_updates(
            target=target,
            policy_snapshot_digest=mixture.recipe_digest,
        )


def test_runtime_environment_identity_excludes_mixture_factor_treatments() -> None:
    baseline = _live_manifest("factor-environment-baseline")
    baseline_factors = runtime_factors(baseline)
    baseline_mixture = baseline.target.mixture
    assert baseline_mixture is not None

    recipe_digest = digest_value("treated-recipe-policy")
    recipe_treatment = baseline.with_semantic_updates(
        policy_snapshot_digest=recipe_digest,
        target=baseline.target.model_copy(
            update={
                "mixture": baseline_mixture.model_copy(
                    update={"recipe_digest": recipe_digest}
                )
            }
        ),
    )
    pool_treatment = _live_manifest("factor-environment-pool", price_delta=0.25)
    binding_treatment = baseline.with_semantic_updates(
        target=baseline.target.model_copy(
            update={
                "mixture": baseline_mixture.model_copy(
                    update={"binding_digest": digest_value("treated-binding")}
                )
            }
        )
    )

    assert runtime_factors(recipe_treatment).environment == baseline_factors.environment
    assert runtime_factors(pool_treatment).environment == baseline_factors.environment
    assert (
        runtime_factors(binding_treatment).environment == baseline_factors.environment
    )
    assert runtime_factors(recipe_treatment).policy != baseline_factors.policy
    assert runtime_factors(pool_treatment).pool != baseline_factors.pool
    assert runtime_factors(binding_treatment).binding != baseline_factors.binding

    topology_treatment = _live_manifest(
        "factor-environment-topology",
        topology_digest=digest_value("treated-topology"),
    )
    assert (
        runtime_factors(topology_treatment).environment != baseline_factors.environment
    )


def test_normalized_regret_uses_canonical_record_order_sum() -> None:
    values = [-1e16, *([1.0] * 99_999)]

    # Python 3.12+ changed built-in sum() for floats. The report protocol uses
    # explicit binary64 additions so the Go seal-time reducer is bit-stable.
    assert canonical_ordered_float_sum(values) == -1e16


def test_heterogeneous_track_plan_totals_do_not_form_a_cartesian_product() -> None:
    manifest = _manifest().with_semantic_updates(track_ids=("routing", "capacity"))
    assert _track_plan_totals(
        manifest,
        {
            "routing": frozenset({"routing-only"}),
            "capacity": frozenset({"capacity-only"}),
        },
    ) == {"routing": 1, "capacity": 1}


def test_fixture_report_is_deterministic_and_coverage_is_case_based(
    tmp_path: Path,
) -> None:
    first = run_evaluation(_manifest("same-run", 2), LocalArtifactStore(tmp_path / "a"))
    second = run_evaluation(
        _manifest("same-run", 2), LocalArtifactStore(tmp_path / "b")
    )

    first_payload = first.model_dump(mode="json", exclude_none=False)
    second_payload = second.model_dump(mode="json", exclude_none=False)
    for payload in (first_payload, second_payload):
        payload["run"]["started_at"] = None
        payload["run"]["completed_at"] = None
        payload["provenance"]["generated_at"] = None
        for artifact in payload["artifacts"]:
            artifact["digest"] = None
            artifact["size_bytes"] = None
        for gate in payload["gates"]:
            gate["evaluated_at"] = None
        for track in payload["tracks"]:
            for gate in track["gates"]:
                gate["evaluated_at"] = None
    assert first_payload == second_payload
    assert first.summary.coverage.evaluated == 15
    assert first.summary.coverage.total == 15
    assert first.summary.coverage.fraction == 1.0


def test_existing_report_is_idempotent_only_for_the_same_frozen_manifest(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    manifest = _manifest("idempotent-run")
    first = run_evaluation(manifest, store)
    assert run_evaluation(manifest, store) == first

    changed_code = manifest.with_semantic_updates(code_revision="sha256:" + "2" * 64)
    with pytest.raises(StoreError, match="different run manifest"):
        run_evaluation(changed_code, store)

    changed_suite = manifest.with_semantic_updates(
        suite_revisions={"evaluation-smoke": "builtin-v2"}
    )
    with pytest.raises(ValueError, match="suite revisions"):
        run_evaluation(changed_suite, store)


def test_hidden_grading_is_not_written_to_policy_visible_cases(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    report = run_evaluation(_manifest(), store)
    visible_payload = (store.runs / report.run.id / "cases.jsonl").read_text()

    assert "expected_answer" not in visible_payload
    assert "expected_route" not in visible_payload
    assert "preferred_arm_id" not in visible_payload
    assert "should_block" not in visible_payload
    assert "grading-cases.jsonl" not in {artifact.name for artifact in report.artifacts}
    assert store.read_run_text(report.run.id, "grading-cases.jsonl")


def test_real_regression_remains_diagnostic_without_qualification() -> None:
    inputs = fixture_inputs()
    records = execute_fixture(inputs.visible, inputs.grading, inputs.fixture, TRACK_IDS)
    records = [
        (
            row.model_copy(update={"safety_violations": 1})
            if row.track_id == "safety" and row.case_id == "safety-1"
            else row
        )
        for row in records
    ]
    metrics = compute_metrics(records, capacity_profile=None)
    gates = compute_gates(metrics, has_records=True)
    hard_policy = next(gate for gate in gates if gate.id == "G2")
    violation_rate = next(
        metric for metric in metrics if metric.id == "safety.violation_rate"
    )
    assert hard_policy.verdict == "unavailable"
    assert hard_policy.observed is None
    assert hard_policy.threshold is None
    assert violation_rate.value is not None and violation_rate.value > 0


def test_catalog_track_capabilities_are_the_canonical_analysis_catalog() -> None:
    for track in get_catalog(generated_at=False).tracks:
        assert track.metrics == static_metric_ids_for_track(track.id)


def test_cost_per_success_never_uses_a_partial_track_ledger() -> None:
    records = [
        ExecutionRecord(
            id="joint-1",
            track_id="joint",
            case_id="case-1",
            attempt_id="attempt-1",
            status="succeeded",
            success=True,
            quality=1,
            runtime_cost=0.1,
        ),
        ExecutionRecord(
            id="joint-2",
            track_id="joint",
            case_id="case-2",
            attempt_id="attempt-2",
            status="succeeded",
            success=True,
            quality=1,
        ),
    ]

    metric = next(
        row
        for row in compute_metrics(records, capacity_profile=None)
        if row.id == "joint.runtime_cost_per_success"
    )
    assert metric.value is None


def test_pool_failure_overlap_uses_exact_common_cases() -> None:
    records = [
        ExecutionRecord(
            id=f"pool-{case_id}-{arm_id}",
            track_id="model_pool",
            case_id=case_id,
            attempt_id=f"attempt-{case_id}-{arm_id}",
            status="failed" if failed else "succeeded",
            arm_id=arm_id,
            success=not failed,
            quality=0 if failed else 1,
        )
        for case_id, arm_id, failed in (
            ("case-1", "arm-a", True),
            ("case-1", "arm-b", True),
            ("case-2", "arm-a", True),
            ("case-2", "arm-b", False),
        )
    ]

    metrics = {
        metric.id: metric.value
        for metric in compute_metrics(records, capacity_profile=None)
    }
    assert metrics["model_pool.mean_pairwise_failure_jaccard"] == 0.5
    assert metrics["model_pool.worst_arm_reliability"] == 0
    assert metrics["model_pool.all_arm_failure_rate"] == 0.5


def test_worst_arm_reliability_requires_a_dense_frozen_pool() -> None:
    records = [
        ExecutionRecord(
            id=f"pool-{case_id}-{arm_id}",
            track_id="model_pool",
            case_id=case_id,
            attempt_id=f"attempt-{case_id}-{arm_id}",
            status="succeeded",
            arm_id=arm_id,
            success=True,
            quality=1,
        )
        for case_id, arm_id in (
            ("case-1", "arm-a"),
            ("case-1", "arm-b"),
            ("case-2", "arm-a"),
        )
    ]

    metric = next(
        metric
        for metric in compute_metrics(records, capacity_profile=None)
        if metric.id == "model_pool.worst_arm_reliability"
    )

    assert metric.value is None


def test_pool_and_joint_quality_keep_failed_rows_in_the_dense_cohort() -> None:
    records = [
        ExecutionRecord(
            id=f"pool-{case_id}-{arm_id}",
            track_id="model_pool",
            case_id=case_id,
            attempt_id=f"attempt-{case_id}-{arm_id}",
            status="succeeded" if success else "failed",
            arm_id=arm_id,
            success=success,
            quality=quality,
        )
        for case_id, arm_id, success, quality in (
            ("case-1", "arm-a", False, None),
            ("case-1", "arm-b", True, 0.4),
            ("case-2", "arm-a", True, 1.0),
            ("case-2", "arm-b", True, 0.4),
        )
    ]
    records.extend(
        (
            ExecutionRecord(
                id="joint-case-1",
                track_id="joint",
                case_id="case-1",
                attempt_id="attempt-joint-case-1",
                status="failed",
                success=False,
            ),
            ExecutionRecord(
                id="joint-case-2",
                track_id="joint",
                case_id="case-2",
                attempt_id="attempt-joint-case-2",
                status="succeeded",
                success=True,
                quality=0.5,
            ),
        )
    )

    metrics = {
        metric.id: metric for metric in compute_metrics(records, capacity_profile=None)
    }
    assert metrics["model_pool.arm.arm-a.quality"].value == pytest.approx(0.5)
    assert metrics["model_pool.arm.arm-a.quality"].sample_count == 2
    assert metrics["model_pool.best_single_quality"].value == pytest.approx(0.5)
    assert metrics["model_pool.oracle_quality"].value == pytest.approx(0.7)
    assert metrics["model_pool.oracle_quality"].sample_count == 2
    assert metrics["model_pool.worst_arm_reliability"].value == pytest.approx(0.5)
    assert metrics["model_pool.all_arm_failure_rate"].value == 0
    assert metrics["joint.realized_quality"].value == pytest.approx(0.25)
    assert metrics["joint.realized_quality"].sample_count == 2
    assert metrics["joint.normalized_regret"].value == pytest.approx(0.75)
    assert metrics["joint.normalized_regret"].sample_count == 2


def test_compare_is_paired_by_metric_id(tmp_path: Path) -> None:
    baseline_store = LocalArtifactStore(tmp_path / "a")
    candidate_store = LocalArtifactStore(tmp_path / "b")
    baseline = run_evaluation(_manifest("baseline"), baseline_store)
    candidate = run_evaluation(
        _manifest(
            "candidate",
            baseline_run_id=_uuid("baseline"),
            code_revision="sha256:" + "2" * 64,
        ),
        candidate_store,
    )
    comparison = compare_worker_drafts(
        baseline,
        candidate,
        _records(baseline_store, baseline.run.id),
        _records(candidate_store, candidate.run.id),
    )

    assert comparison.baseline_run_id == _uuid("baseline")
    assert comparison.candidate_run_id == _uuid("candidate")
    assert all(
        metric.delta == 0 for metric in comparison.metrics if metric.value is not None
    )
    assert comparison.verdict == "unavailable"


def test_run_evidence_is_weakest_selected_track_and_unavailable_is_e0() -> None:
    joint = ExecutionRecord(
        id="joint-qualified",
        track_id="joint",
        case_id="case-a",
        attempt_id="joint-attempt",
        status="succeeded",
        selected_arm_id="arm-a",
        selection_method="weighted",
        recipe="recipe-a",
        latency_ms=10,
        evidence_kind=LIVE_JOINT_EVIDENCE_SOURCE_ID,
        broker_receipt="sha256:" + "1" * 64,
        success=True,
    )
    safety = ExecutionRecord(
        id="safety-unavailable",
        track_id="safety",
        case_id="case-a",
        attempt_id="safety-attempt",
        status="unavailable",
        error="no qualified safety observation",
    )

    executor = LiveRuntimeExecutor.contract
    assert track_evidence_level("live", executor, "joint", [joint]) == "E5"
    assert track_evidence_level("live", executor, "safety", [safety]) == "E0"
    assert (
        run_evidence_level("live", executor, ("joint", "safety"), [joint, safety])
        == "E0"
    )
