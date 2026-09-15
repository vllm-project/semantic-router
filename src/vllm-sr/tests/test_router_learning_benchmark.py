from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections import defaultdict
from dataclasses import asdict
from math import sqrt
from pathlib import Path

import cli.evaluation.router_learning_executor as learning_executor
import pytest
from cli.evaluation.execution_contract import ROUTER_LEARNING_REPLAY_EXECUTOR_ID
from cli.evaluation.metric_analysis_catalog import resolve_metric_analysis
from cli.evaluation.metric_router_learning import _cluster_interval
from cli.evaluation.orchestrator import run_evaluation
from cli.evaluation.router_learning_corpus import (
    ROUTER_LEARNING_CASE_COUNT,
    ROUTER_LEARNING_CORPUS,
)
from cli.evaluation.router_learning_evidence import (
    ROUTER_LEARNING_CORPUS_REVISION,
    ROUTER_LEARNING_POLICY_IDS,
    RouterLearningMethodEvidence,
)
from cli.evaluation.store import LocalArtifactStore
from evaluation_contract_test_support import _manifest


def _learning_manifest(name: str = "router-learning-benchmark"):
    return _manifest(name).with_semantic_updates(
        suite_ids=("router-learning-core",),
        suite_revisions={"router-learning-core": ROUTER_LEARNING_CORPUS_REVISION},
        suite_executors={"router-learning-core": ROUTER_LEARNING_REPLAY_EXECUTOR_ID},
        track_ids=("joint",),
        sample_limit=ROUTER_LEARNING_CASE_COUNT,
    )


def test_router_learning_replay_is_paired_deterministic_and_guarded(
    tmp_path: Path,
) -> None:
    manifest = _learning_manifest()
    first = run_evaluation(manifest, LocalArtifactStore(tmp_path / "first"))
    second = run_evaluation(manifest, LocalArtifactStore(tmp_path / "second"))
    first_metrics = {
        metric.id: (metric.value, metric.sample_count, metric.confidence_interval)
        for metric in first.metrics
        if metric.id.startswith("joint.router_learning.")
    }
    second_metrics = {
        metric.id: (metric.value, metric.sample_count, metric.confidence_interval)
        for metric in second.metrics
        if metric.id.startswith("joint.router_learning.")
    }
    assert first_metrics == second_metrics
    assert len(first_metrics) == len(ROUTER_LEARNING_POLICY_IDS) * 8

    records = [
        json.loads(line)
        for line in LocalArtifactStore(tmp_path / "first")
        .read_run_text(manifest.run_id, "records.jsonl")
        .splitlines()
    ]
    learning = [row["router_learning"] for row in records]
    assert len(learning) == (
        len(ROUTER_LEARNING_CORPUS.trial_seeds)
        * len(ROUTER_LEARNING_POLICY_IDS)
        * ROUTER_LEARNING_CASE_COUNT
    )
    seeds_by_trial: dict[str, set[int]] = defaultdict(set)
    policies_by_trial: dict[str, set[str]] = defaultdict(set)
    for row in learning:
        seeds_by_trial[row["trial_id"]].add(row["trial_seed"])
        policies_by_trial[row["trial_id"]].add(row["policy_id"])
        assert row["selected_arm_id"] in row["eligible_arm_ids"]
        assert not row["hard_constraint_violation"]
        assert not row["protection_violation"]
        assert row["propensity_status"] == "unsupported"
    assert {row["policy_id"] for row in learning} == {
        "static-base",
        "routing-sampling",
        "beta-bernoulli",
    }
    production = next(row for row in learning if row["policy_id"] == "routing-sampling")
    with pytest.raises(ValueError, match="policy_id"):
        RouterLearningMethodEvidence.model_validate(
            {**production, "policy_id": "simplified-routing-sampling"}
        )
    assert "joint.router_learning.routing-sampling.solve_rate" in first_metrics
    assert not any(
        ".simplified-routing-sampling." in metric_id for metric_id in first_metrics
    )
    assert all(len(seeds) == 1 for seeds in seeds_by_trial.values())
    assert all(
        policies == set(ROUTER_LEARNING_POLICY_IDS)
        for policies in policies_by_trial.values()
    )


def test_router_learning_metric_denominators_and_uncertainty(tmp_path: Path) -> None:
    report = run_evaluation(
        _learning_manifest("router-learning-metrics"),
        LocalArtifactStore(tmp_path / "store"),
    )
    metrics = {metric.id: metric for metric in report.metrics}
    rounds = len(ROUTER_LEARNING_CORPUS.trial_seeds) * ROUTER_LEARNING_CASE_COUNT
    protected_rounds = len(ROUTER_LEARNING_CORPUS.trial_seeds) * sum(
        case.protected_arm_id is not None for case in ROUTER_LEARNING_CORPUS.cases
    )
    for policy_id in ROUTER_LEARNING_POLICY_IDS:
        prefix = f"joint.router_learning.{policy_id}."
        assert metrics[prefix + "solve_rate"].sample_count == rounds
        assert metrics[prefix + "solve_rate"].confidence_interval is not None
        assert metrics[prefix + "lifecycle_cost_mean_usd"].sample_count == rounds
        assert metrics[prefix + "latency_mean_ms"].sample_count == rounds
        assert metrics[prefix + "model_call_mean"].sample_count == rounds
        assert (
            metrics[prefix + "protection_violation_rate"].sample_count
            == protected_rounds
        )
        assert metrics[prefix + "hard_constraint_violation_rate"].sample_count == rounds
        assert metrics[prefix + "propensity_coverage"].value == 0.0
        assert metrics[prefix + "propensity_coverage"].sample_count == rounds
        assert metrics[prefix + "trial_count"].value == len(
            ROUTER_LEARNING_CORPUS.trial_seeds
        )


def test_feedback_is_delayed_and_censored(
    monkeypatch,
) -> None:
    observed_counts: list[int] = []

    def capture_state(policy_id, case, state, rng):
        del policy_id, case, rng
        observed_counts.append(state[ROUTER_LEARNING_CORPUS.base_arm_id].observations)
        return ROUTER_LEARNING_CORPUS.base_arm_id

    monkeypatch.setattr(learning_executor, "_proposal", capture_state)
    source = ROUTER_LEARNING_CORPUS.cases[0]
    delayed = source.model_copy(
        update={"feedback_delay_rounds": 1, "feedback_observed": True}
    )
    learning_executor._execute_policy_trial(
        policy_id="static-base",
        trial_index=0,
        trial_seed=11,
        cases=(delayed, delayed, delayed),
    )
    assert observed_counts == [0, 0, 1]

    observed_counts.clear()
    censored = source.model_copy(update={"feedback_observed": False})
    learning_executor._execute_policy_trial(
        policy_id="static-base",
        trial_index=0,
        trial_seed=11,
        cases=(censored, censored, censored),
    )
    assert observed_counts == [0, 0, 0]


def test_replay_observes_only_selected_telemetry_before_delayed_quality(
    monkeypatch,
) -> None:
    snapshots = []
    base = ROUTER_LEARNING_CORPUS.base_arm_id
    other = next(
        arm.id for arm in ROUTER_LEARNING_CORPUS.candidate_arms if arm.id != base
    )

    def capture_state(policy_id, case, state, rng):
        snapshots.append({key: asdict(value) for key, value in state.items()})
        return base

    monkeypatch.setattr(learning_executor, "_proposal", capture_state)
    source = ROUTER_LEARNING_CORPUS.cases[0]
    outcome = source.outcomes[base].model_copy(
        update={
            "feedback_verdict": "overprovisioned",
            "feedback_weight": 2.8,
            "latency_ms": 800.0,
            "cache_hit_ratio": 0.6,
            "input_cost_multiplier": 0.4,
            "provider_failed": True,
        }
    )
    case = source.model_copy(
        update={
            "outcomes": {**source.outcomes, base: outcome},
            "feedback_delay_rounds": 1,
        }
    )
    learning_executor._execute_policy_trial(
        policy_id="routing-sampling",
        trial_index=0,
        trial_seed=11,
        cases=(case, case, case),
    )
    assert not snapshots[0][base]["updated"]
    assert snapshots[1][base]["updated"]
    assert snapshots[1][base]["overprovisioned"] == 0
    assert snapshots[1][base]["failed"] == 1
    assert snapshots[1][base]["latency_ewma"] == 0.8
    assert snapshots[1][base]["cache_hit_ewma"] == 0.6
    assert snapshots[1][base]["input_cost_multiplier_ewma"] == 0.4
    assert snapshots[2][base]["overprovisioned"] == 2
    assert snapshots[2][base]["failed"] == 2
    assert all(snapshot[other] == snapshots[0][other] for snapshot in snapshots)


def test_corpus_exercises_every_production_adjustment(monkeypatch) -> None:
    observed = []
    original = learning_executor.score_candidate

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        observed.append((args[1].input_cost_multiplier_ewma, result))
        return result

    monkeypatch.setattr(learning_executor, "score_candidate", capture)
    for seed in ROUTER_LEARNING_CORPUS.trial_seeds:
        learning_executor._execute_policy_trial(
            policy_id="routing-sampling",
            trial_index=0,
            trial_seed=seed,
            cases=ROUTER_LEARNING_CORPUS.cases,
        )
    assert any(value > 0 for value, _ in observed)
    for field in ("overuse_penalty", "reliability_penalty", "cache_adjustment"):
        assert any(getattr(score, field) > 0 for _, score in observed), field
    assert any(score.latency_adjustment < 0 for _, score in observed)


@pytest.mark.parametrize("sample_limit,seed", [(12, 17), (5, 2**32 - 5)])
def test_python_replay_metrics_are_attested_by_go_reducer(
    tmp_path: Path, sample_limit: int, seed: int
) -> None:
    manifest = _learning_manifest(
        "router-learning-reducer-parity"
    ).with_semantic_updates(sample_limit=sample_limit, seed=seed)
    store = LocalArtifactStore(tmp_path / "store")
    report = run_evaluation(manifest, store)
    (tmp_path / "parity-records.jsonl").write_text(
        store.read_run_text(manifest.run_id, "records.jsonl")
    )
    (tmp_path / "parity-metrics.json").write_text(
        json.dumps(
            [
                metric.model_dump(mode="json", exclude_none=True)
                for metric in report.metrics
            ]
        )
    )
    inputs, _ = learning_executor.collect_router_learning_evidence(manifest)
    (tmp_path / "parity-plan.json").write_text(
        json.dumps(
            {
                "Seed": manifest.seed,
                "CaseIDs": [case.id for case in inputs.visible.cases],
            }
        )
    )
    go = shutil.which("go")
    assert go is not None, "Go is required to attest the Python replay metrics"
    backend = Path(__file__).resolve().parents[3] / "dashboard/backend"
    result = subprocess.run(
        [
            go,
            "test",
            "./evaluationplane",
            "-run",
            "^TestRouterLearningPythonEvidenceParity$",
            "-count=1",
        ],
        cwd=backend,
        env={**os.environ, "VLLM_SR_ROUTER_LEARNING_PARITY_BUNDLE": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("policy_id", ROUTER_LEARNING_POLICY_IDS)
@pytest.mark.parametrize(
    "statistic,estimator",
    [
        ("solve_rate", "round-cluster-rate"),
        ("hard_constraint_violation_rate", "round-cluster-rate"),
        ("protection_violation_rate", "protected-round-cluster-rate"),
        ("lifecycle_cost_mean_usd", "round-cluster-mean"),
        ("latency_mean_ms", "round-cluster-mean"),
        ("model_call_mean", "round-cluster-mean"),
    ],
)
def test_router_learning_estimator_method_contract(policy_id, statistic, estimator):
    specification = resolve_metric_analysis(
        f"joint.router_learning.{policy_id}.{statistic}"
    ).specification
    assert specification.estimator_id == (
        f"router-learning-{estimator}-student-t-95-df31"
    )
    assert specification.estimator_version == "v1"
    assert specification.cluster_unit == "trial_id"
    assert specification.weighting == "uniform_trial"
    assert len(ROUTER_LEARNING_CORPUS.trial_seeds) == 32

    # Known sample: mean 15.5, unbiased variance 88, 31 degrees of freedom.
    margin = 2.0395134463964077 * sqrt(88 / 32)
    assert _cluster_interval(list(range(32))) == pytest.approx(
        (15.5 - margin, 15.5 + margin)
    )
    assert _cluster_interval([0.0] * 32, bounds=(0, 1)) == (0.0, 0.0)
    with pytest.raises(ValueError, match="32 complete trials"):
        _cluster_interval([0.0] * 8)
