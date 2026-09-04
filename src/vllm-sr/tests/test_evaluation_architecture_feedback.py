from __future__ import annotations

from cli.evaluation.architecture_feedback import architecture_recommendations
from cli.evaluation.reporting import EvaluationGate, EvaluationMetric


def _metric(metric_id: str, value: float, track_id: str) -> EvaluationMetric:
    return EvaluationMetric(
        id=metric_id,
        name=metric_id,
        track_id=track_id,
        value=value,
        unit="fraction",
        direction="higher_is_better",
        sample_count=20,
    )


def _gate(gate_id: str, verdict: str) -> EvaluationGate:
    return EvaluationGate(
        id=gate_id,
        name=gate_id,
        disposition="required",
        verdict=verdict,
        change_profile="online_adaptation",
        contract_version="evaluation-release-gates.v1",
        evidence_refs=("records.jsonl",),
        evidence_level="E5",
    )


def test_feedback_separates_recipe_pool_agent_and_serving_owners() -> None:
    recommendations = architecture_recommendations(
        [
            _metric("routing.coverage", 0.8, "routing"),
            _metric("model_pool.oracle_gain", 0.01, "model_pool"),
            _metric("joint.normalized_regret", 0.35, "joint"),
            _metric("agentic.success_rate", 0.6, "agentic"),
            _metric("capacity.saturation_concurrency", 8.0, "capacity"),
        ],
        [],
    )
    rendered = "\n".join(recommendations)
    assert "Owner=Router recipe owner" in rendered
    assert "Owner=Model-pool owner" in rendered
    assert "Owner=Agent and Router session owners" in rendered
    assert "Owner=Serving and placement owner" in rendered
    assert "hold the pool fixed" in rendered


def test_missing_online_evidence_produces_concrete_contract_actions() -> None:
    recommendations = architecture_recommendations(
        [],
        [
            _gate("G5", "unavailable"),
            _gate("G8", "unavailable"),
            _gate("G9", "unavailable"),
        ],
    )
    rendered = "\n".join(recommendations)
    assert "paired replay/live campaign" in rendered
    assert "sample-ratio checks" in rendered
    assert "effective sample size" in rendered


def test_healthy_metrics_do_not_create_speculative_actions() -> None:
    assert (
        architecture_recommendations(
            [
                _metric("routing.coverage", 1.0, "routing"),
                _metric("model_pool.oracle_gain", 0.2, "model_pool"),
                _metric("joint.normalized_regret", 0.05, "joint"),
                _metric("agentic.success_rate", 1.0, "agentic"),
                _metric("multimodal.support_rate", 1.0, "multimodal"),
                _metric("safety.violation_rate", 0.0, "safety"),
                _metric("preference.propensity_coverage", 1.0, "preference"),
            ],
            [],
        )
        == ()
    )
