import importlib.util
import sys
from pathlib import Path


def load_experiment_module():
    path = Path(__file__).with_name("agentic_routing_experiment.py")
    spec = importlib.util.spec_from_file_location("agentic_routing_experiment", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_remaining_turn_prior_blocks_early_low_history_switch():
    exp = load_experiment_module()

    current = exp.MODELS[0]
    stronger = exp.MODELS[1]
    full = exp.choose_agentic_model(
        base_choice=stronger,
        current=current,
        phase="user_turn",
        turn=1,
        switch_count=0,
        history_tokens=512,
        idle_expired=False,
        remaining_turn_prior=8.0,
        policy="acr-full",
    )
    no_prior = exp.choose_agentic_model(
        base_choice=stronger,
        current=current,
        phase="user_turn",
        turn=1,
        switch_count=0,
        history_tokens=512,
        idle_expired=False,
        remaining_turn_prior=8.0,
        policy="acr-no-remaining-prior",
    )

    assert full.model.name == current.name
    assert full.reason == "continuity_cost_blocks_switch"
    assert full.prior_mass > no_prior.prior_mass
    assert full.continuation_mass > no_prior.continuation_mass
    assert no_prior.model.name == stronger.name


def test_simulated_rows_include_continuation_evidence():
    exp = load_experiment_module()

    rows = exp.simulate_workload(
        session_count=2,
        turn_count=4,
        seed=20260530,
        scenario=exp.SCENARIOS["tool-heavy"],
        agentic_policy="acr-full",
    )
    summary = exp.summarize(rows)

    assert rows
    assert "remaining_turn_prior" in rows[0]
    assert "remaining_turns_estimate" in rows[0]
    assert "continuation_mass" in rows[0]
    assert summary["mean_continuation_mass"] is not None
    assert summary["mean_remaining_turns_estimate"] is not None


def test_ablation_outputs_include_remaining_prior_policy_and_svg(tmp_path):
    exp = load_experiment_module()

    summary = exp.write_ablation_outputs(
        scenarios=("balanced",),
        seeds=[20260530],
        sessions=2,
        turns=4,
        out_dir=tmp_path,
    )

    policies = {row["policy"] for row in summary["by_policy"]}
    assert "acr-no-remaining-prior" in policies
    assert (tmp_path / "ablation_summary.csv").exists()
    svg = (tmp_path / "summary.svg").read_text()
    assert "Agentic Routing Ablation" in svg
    assert "acr-no-remaining-prior" in svg
