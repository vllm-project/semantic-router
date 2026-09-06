from __future__ import annotations

import json
from pathlib import Path

import pytest
from cli.evaluation.builtin_executors import FixtureReplayExecutor
from cli.evaluation.canonical import digest_value
from cli.evaluation.compare import compare_runs, compare_worker_drafts
from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.executor_contracts import ExecutorContract
from cli.evaluation.executor_registry import ExecutorRegistry
from cli.evaluation.fixtures import fixture_inputs
from cli.evaluation.manifest_identity import (
    selector_snapshot_digest,
)
from cli.evaluation.orchestrator import run_evaluation, validate_manifest
from cli.evaluation.paired_statistics import paired_statistic_results
from cli.evaluation.store import LocalArtifactStore
from cli.evaluation.target_contracts import CatalogMixture
from cli.evaluation.worker_report import WorkerReportDraft
from evaluation_contract_test_support import (
    _live_manifest,
    _live_mixture,
    _manifest,
    _resolved_live,
    _uuid,
)


class _RecordingExecutorRegistry(ExecutorRegistry):
    def __init__(self) -> None:
        self.requested_contracts: list[str] = []
        super().__init__((FixtureReplayExecutor(),))

    def contract(self, executor_id: str) -> ExecutorContract:
        self.requested_contracts.append(executor_id)
        return super().contract(executor_id)


def _qualified_report(current: WorkerReportDraft) -> WorkerReportDraft:
    gates = tuple(
        (
            gate.model_copy(update={"verdict": "pass"})
            if gate.disposition == "required"
            else gate
        )
        for gate in current.gates
    )
    tracks = tuple(
        track.model_copy(
            update={
                "status": "completed",
                "evidence_level": "E3",
                "coverage": track.coverage.model_copy(update={"unavailable": 0}),
            }
        )
        for track in current.tracks
    )
    return WorkerReportDraft.model_validate(
        current.model_copy(
            update={
                "run": current.run.model_copy(update={"evidence_level": "E3"}),
                "summary": current.summary.model_copy(
                    update={"verdict": "pass", "unavailable_gates": 0}
                ),
                "tracks": tracks,
                "gates": gates,
            }
        ).model_dump(mode="python")
    )


def _routing_quality_records() -> (
    tuple[list[ExecutionRecord], list[ExecutionRecord], list[ExecutionRecord]]
):
    baseline = [
        ExecutionRecord(
            id=f"routing-paired-{index}",
            track_id="routing",
            case_id=f"paired-{index}",
            attempt_id=f"paired-{index}",
            status="succeeded",
            success=True,
            quality=1.0,
        )
        for index in range(4)
    ]
    equal = [record.model_copy() for record in baseline]
    regressed = [record.model_copy(update={"quality": 0.0}) for record in baseline]
    return baseline, equal, regressed


def _regret_records() -> tuple[list[ExecutionRecord], list[ExecutionRecord]]:
    baseline: list[ExecutionRecord] = []
    candidate: list[ExecutionRecord] = []
    for index in range(20):
        case_id = f"regret-{index}"
        for arm_id, old_quality, new_quality in (
            ("arm-a", 0.5, 1.0),
            ("arm-b", 0.4, 0.6),
        ):
            baseline.append(
                ExecutionRecord(
                    id=f"pool-{case_id}-{arm_id}",
                    track_id="model_pool",
                    case_id=case_id,
                    attempt_id=f"pool-{case_id}-{arm_id}",
                    arm_id=arm_id,
                    status="succeeded",
                    success=True,
                    quality=old_quality,
                )
            )
            candidate.append(baseline[-1].model_copy(update={"quality": new_quality}))
        baseline.append(
            ExecutionRecord(
                id=f"joint-{case_id}",
                track_id="joint",
                case_id=case_id,
                attempt_id=f"joint-{case_id}",
                status="succeeded",
                success=True,
                quality=0.5,
            )
        )
        candidate.append(baseline[-1].model_copy(update={"quality": 0.8}))
    return baseline, candidate


def test_qualified_case_aligned_comparison_can_pass_and_fail(tmp_path: Path) -> None:
    baseline = run_evaluation(
        _manifest("baseline"), LocalArtifactStore(tmp_path / "baseline")
    )
    candidate = run_evaluation(
        _manifest(
            "candidate",
            baseline_run_id=_uuid("baseline"),
            code_revision="sha256:" + "2" * 64,
        ),
        LocalArtifactStore(tmp_path / "candidate"),
    )
    baseline = _qualified_report(baseline)
    candidate = _qualified_report(candidate)
    baseline_records, equal_records, regressed_records = _routing_quality_records()

    passed = compare_worker_drafts(baseline, candidate, baseline_records, equal_records)
    failed = compare_worker_drafts(
        baseline, candidate, baseline_records, regressed_records
    )

    assert passed.verdict == "pass"
    assert failed.verdict == "fail"
    paired_accuracy = next(
        metric for metric in failed.metrics if metric.id == "routing.accuracy"
    )
    assert paired_accuracy.confidence_interval == (-1.0, -1.0)
    assert paired_accuracy.sample_count == 4

    baseline_regret, candidate_regret = _regret_records()
    regret_failure = compare_worker_drafts(
        baseline, candidate, baseline_regret, candidate_regret
    )
    assert regret_failure.verdict == "fail"
    normalized_regret = next(
        metric
        for metric in regret_failure.metrics
        if metric.id == "joint.normalized_regret"
    )
    assert normalized_regret.confidence_interval == pytest.approx((0.2, 0.2))


def test_paired_pool_oracle_reduces_each_case_before_bootstrap() -> None:
    baseline: list[ExecutionRecord] = []
    candidate: list[ExecutionRecord] = []
    for index in range(100):
        case_id = f"oracle-{index}"
        for arm_id, old_quality, new_quality in (
            ("arm-a", 0.9, 0.8),
            ("arm-b", 0.1, 0.8),
        ):
            baseline.append(
                ExecutionRecord(
                    id=f"pool-{case_id}-{arm_id}",
                    track_id="model_pool",
                    case_id=case_id,
                    attempt_id=f"pool-{case_id}-{arm_id}",
                    arm_id=arm_id,
                    status="succeeded",
                    success=True,
                    quality=old_quality,
                )
            )
            candidate.append(baseline[-1].model_copy(update={"quality": new_quality}))

    results = paired_statistic_results(baseline, candidate, seed=17)
    oracle = next(
        row for row in results if row.metric_id == "model_pool.oracle_quality"
    )

    assert oracle.sample_count == 100
    assert oracle.delta == pytest.approx(-0.1)
    assert oracle.confidence_interval == pytest.approx((-0.1, -0.1))

    candidate[0] = candidate[0].model_copy(update={"case_id": "different-cluster"})
    with pytest.raises(ValueError, match="analysis identities"):
        paired_statistic_results(baseline, candidate, seed=17)


def test_paired_pool_and_joint_quality_include_failed_attempts_as_zero() -> None:
    baseline: list[ExecutionRecord] = []
    candidate: list[ExecutionRecord] = []
    for index in range(20):
        case_id = f"failed-{index}"
        for arm_id, quality in (("arm-a", 0.5), ("arm-b", 0.4)):
            baseline.append(
                ExecutionRecord(
                    id=f"pool-{case_id}-{arm_id}",
                    track_id="model_pool",
                    case_id=case_id,
                    attempt_id=f"pool-{case_id}-{arm_id}",
                    arm_id=arm_id,
                    status="succeeded",
                    success=True,
                    quality=quality,
                )
            )
            candidate.append(
                baseline[-1].model_copy(
                    update={
                        "status": "failed" if arm_id == "arm-a" else "succeeded",
                        "success": arm_id != "arm-a",
                        "quality": None if arm_id == "arm-a" else quality,
                    }
                )
            )
        baseline.append(
            ExecutionRecord(
                id=f"joint-{case_id}",
                track_id="joint",
                case_id=case_id,
                attempt_id=f"joint-{case_id}",
                status="succeeded",
                success=True,
                quality=0.5,
            )
        )
        candidate.append(
            baseline[-1].model_copy(
                update={"status": "failed", "success": False, "quality": None}
            )
        )

    results = {
        result.metric_id: result
        for result in paired_statistic_results(baseline, candidate, seed=17)
    }
    assert results["model_pool.oracle_quality"].candidate_value == pytest.approx(0.4)
    assert results["model_pool.oracle_quality"].delta == pytest.approx(-0.1)
    assert results["joint.realized_quality"].candidate_value == 0
    assert results["joint.realized_quality"].delta == pytest.approx(-0.5)
    assert results["joint.normalized_regret"].candidate_value == 1
    assert results["joint.normalized_regret"].delta == 1


def test_compare_rejects_different_workloads(tmp_path: Path) -> None:
    baseline = run_evaluation(
        _manifest("baseline", sample_limit=4), LocalArtifactStore(tmp_path / "a")
    )
    candidate = run_evaluation(
        _manifest(
            "candidate",
            sample_limit=2,
            baseline_run_id=_uuid("baseline"),
            code_revision="sha256:" + "2" * 64,
        ),
        LocalArtifactStore(tmp_path / "b"),
    )

    with pytest.raises(ValueError, match="workload_snapshot_digest"):
        compare_worker_drafts(baseline, candidate, [], [])


def test_compare_runs_uses_the_supplied_executor_registry(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    baseline = run_evaluation(_manifest("baseline"), store)
    candidate = run_evaluation(
        _manifest(
            "candidate",
            baseline_run_id="baseline",
            code_revision="sha256:" + "2" * 64,
        ),
        store,
    )
    executor_registry = _RecordingExecutorRegistry()

    comparison = compare_runs(
        store,
        baseline.run.id,
        candidate.run.id,
        executor_registry=executor_registry,
    )

    assert comparison.baseline_run_id == baseline.run.id
    assert comparison.candidate_run_id == candidate.run.id
    assert executor_registry.requested_contracts == [
        FixtureReplayExecutor.contract.id,
        FixtureReplayExecutor.contract.id,
    ]


def test_compare_rejects_self_and_unlinked_candidate(tmp_path: Path) -> None:
    baseline = run_evaluation(_manifest("baseline"), LocalArtifactStore(tmp_path / "a"))
    unlinked = run_evaluation(
        _manifest("candidate", code_revision="sha256:" + "2" * 64),
        LocalArtifactStore(tmp_path / "b"),
    )

    with pytest.raises(ValueError, match="cannot be compared with itself"):
        compare_worker_drafts(baseline, baseline, [], [])
    with pytest.raises(ValueError, match="baseline_run_id"):
        compare_worker_drafts(baseline, unlinked, [], [])


def test_live_factor_snapshot_ids_are_stable_and_detect_pool_environment_drift(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    first_manifest = _live_manifest("first")
    first = _resolved_live(first_manifest, store)
    second = _resolved_live(_live_manifest("second"), store)

    assert first.manifest_digest == first_manifest.manifest_digest
    assert first.manifest_digest != digest_value(first_manifest)
    assert first.policy.id == second.policy.id
    assert first.pool.id == second.pool.id
    assert first.binding.id == second.binding.id
    assert first.environment.id == second.environment.id

    changed_environment = _resolved_live(
        _live_manifest("changed-env", envoy_url="http://envoy:9901"), store
    )
    changed_pool = _resolved_live(
        _live_manifest("changed-pool", price_delta=0.5), store
    )
    changed_topology = _resolved_live(
        _live_manifest("changed-topology", topology_digest="sha256:" + "c" * 64),
        store,
    )
    assert changed_environment.environment.id != first.environment.id
    assert changed_environment.pool.id == first.pool.id
    assert changed_pool.pool.id != first.pool.id
    assert changed_pool.binding.id == first.binding.id
    assert changed_topology.environment.id != first.environment.id
    assert changed_topology.pool.id == first.pool.id


def test_live_manifest_rejects_unsupported_target_capabilities() -> None:
    manifest = _live_manifest("unsupported")
    with pytest.raises(
        ValueError, match="manifest target cannot execute selected tracks: routing"
    ):
        validate_manifest(
            manifest.with_semantic_updates(
                target=manifest.target.model_copy(update={"router_api_url": None})
            )
        )


@pytest.mark.parametrize(
    "field",
    ("pool_snapshot_digest", "environment_snapshot_digest"),
)
def test_compare_rejects_pool_or_environment_drift(
    tmp_path: Path,
    field: str,
) -> None:
    baseline = run_evaluation(_manifest("baseline"), LocalArtifactStore(tmp_path / "a"))
    candidate = run_evaluation(
        _manifest(
            "candidate",
            baseline_run_id=_uuid("baseline"),
            code_revision="sha256:" + "2" * 64,
        ),
        LocalArtifactStore(tmp_path / "b"),
    )
    changed_provenance = candidate.provenance.model_copy(
        update={field: "sha256:" + "f" * 64}
    )
    candidate = candidate.model_copy(update={"provenance": changed_provenance})

    with pytest.raises(ValueError, match=field):
        compare_worker_drafts(baseline, candidate, [], [])


def test_model_pool_comparison_allows_target_topology_environment_change(
    tmp_path: Path,
) -> None:
    baseline_manifest = _manifest("pool-baseline").with_semantic_updates(
        change_profile="model_pool"
    )
    candidate_manifest = _manifest(
        "pool-candidate",
        baseline_run_id=_uuid("pool-baseline"),
    ).with_semantic_updates(change_profile="model_pool")
    baseline = run_evaluation(
        baseline_manifest, LocalArtifactStore(tmp_path / "pool-a")
    )
    candidate = run_evaluation(
        candidate_manifest, LocalArtifactStore(tmp_path / "pool-b")
    )
    candidate = candidate.model_copy(
        update={
            "provenance": candidate.provenance.model_copy(
                update={
                    "pool_snapshot_digest": "sha256:" + "e" * 64,
                    "environment_snapshot_digest": "sha256:" + "f" * 64,
                }
            )
        }
    )

    comparison = compare_worker_drafts(baseline, candidate, [], [])
    assert comparison.baseline_run_id == baseline.run.id
    assert comparison.candidate_run_id == candidate.run.id


def _comparison_cohort(
    baseline: WorkerReportDraft,
    candidate: WorkerReportDraft,
    mixture: CatalogMixture,
    profile: str,
) -> tuple[WorkerReportDraft, WorkerReportDraft]:
    baseline_run = baseline.run.model_copy(
        update={
            "change_profile": profile,
            "target_id": mixture.id,
            "mixture": mixture,
        }
    )
    candidate_run = candidate.run.model_copy(
        update={
            "change_profile": profile,
            "target_id": mixture.id,
            "mixture": mixture,
        }
    )
    return (
        baseline.model_copy(update={"run": baseline_run}),
        candidate.model_copy(update={"run": candidate_run}),
    )


def _assert_comparison_accepts(
    baseline: WorkerReportDraft, candidate: WorkerReportDraft
) -> None:
    assert compare_worker_drafts(baseline, candidate, [], []).candidate_run_id


def test_comparison_profiles_require_exact_canonical_treatment(
    tmp_path: Path,
) -> None:
    baseline = run_evaluation(
        _manifest("factor-baseline"), LocalArtifactStore(tmp_path / "factor-a")
    )
    candidate = run_evaluation(
        _manifest("factor-candidate", baseline_run_id=_uuid("factor-baseline")),
        LocalArtifactStore(tmp_path / "factor-b"),
    )
    mixture = _live_mixture(fixture_inputs().arms).public_summary()

    recipe_baseline, recipe_candidate = _comparison_cohort(
        baseline, candidate, mixture, "recipe"
    )
    recipe_digest = digest_value("candidate-recipe")
    recipe_mixture = mixture.model_copy(update={"recipe_digest": recipe_digest})
    recipe_candidate = recipe_candidate.model_copy(
        update={
            "run": recipe_candidate.run.model_copy(update={"mixture": recipe_mixture}),
            "provenance": recipe_candidate.provenance.model_copy(
                update={"policy_snapshot_digest": recipe_digest}
            ),
        }
    )
    _assert_comparison_accepts(recipe_baseline, recipe_candidate)

    selector_policy = digest_value("candidate-selector-policy")
    selector_mixture = mixture.model_copy(
        update={
            "selector_policy_digest": selector_policy,
            "selector_digest": selector_snapshot_digest(selector_policy, ()),
        }
    )
    selector_baseline, selector_candidate = _comparison_cohort(
        baseline, candidate, mixture, "selector"
    )
    selector_candidate = selector_candidate.model_copy(
        update={
            "run": selector_candidate.run.model_copy(
                update={"mixture": selector_mixture}
            )
        }
    )
    _assert_comparison_accepts(selector_baseline, selector_candidate)

    disguised_recipe = selector_candidate.model_copy(
        update={
            "run": selector_candidate.run.model_copy(
                update={"change_profile": "recipe"}
            )
        }
    )
    with pytest.raises(ValueError, match="selector_digest"):
        compare_worker_drafts(recipe_baseline, disguised_recipe, [], [])

    mixed_selector = selector_candidate.model_copy(
        update={
            "run": selector_candidate.run.model_copy(
                update={
                    "mixture": selector_mixture.model_copy(
                        update={"recipe_digest": recipe_digest}
                    )
                }
            ),
            "provenance": selector_candidate.provenance.model_copy(
                update={"policy_snapshot_digest": recipe_digest}
            ),
        }
    )
    with pytest.raises(ValueError, match="policy_snapshot_digest"):
        compare_worker_drafts(selector_baseline, mixed_selector, [], [])

    adaptation_baseline, adaptation_candidate = _comparison_cohort(
        baseline, candidate, mixture, "online_adaptation"
    )
    adaptation_candidate = adaptation_candidate.model_copy(
        update={
            "run": adaptation_candidate.run.model_copy(
                update={
                    "mixture": mixture.model_copy(
                        update={
                            "adaptation_digest": digest_value("candidate-adaptation")
                        }
                    )
                }
            )
        }
    )
    _assert_comparison_accepts(adaptation_baseline, adaptation_candidate)

    agent_baseline, agent_candidate = _comparison_cohort(
        baseline, candidate, mixture, "agent_multimodal"
    )
    with pytest.raises(ValueError, match="no independent server-owned"):
        compare_worker_drafts(agent_baseline, agent_candidate, [], [])


def test_failure_summary_is_derived_without_case_identity(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    report = run_evaluation(_manifest(), store)
    records = [
        json.loads(line)
        for line in (store.runs / report.run.id / "records.jsonl")
        .read_text()
        .splitlines()
    ]
    failed = sum(row["status"] == "failed" for row in records)
    assert failed
    summary = store.read_run_json(report.run.id, "failure-summary.json")
    assert summary["total_records"] == len(records)
    assert summary["failed"] == failed
    assert "case_id" not in json.dumps(summary)
