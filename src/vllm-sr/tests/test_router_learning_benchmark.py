from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cli.evaluation.router_learning_executor as learning_executor
from cli.evaluation.execution_contract import ROUTER_LEARNING_REPLAY_EXECUTOR_ID
from cli.evaluation.orchestrator import run_evaluation
from cli.evaluation.router_learning_corpus import (
    ROUTER_LEARNING_CASE_COUNT,
    ROUTER_LEARNING_CORPUS,
)
from cli.evaluation.router_learning_evidence import (
    ROUTER_LEARNING_CORPUS_REVISION,
    ROUTER_LEARNING_POLICY_IDS,
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
