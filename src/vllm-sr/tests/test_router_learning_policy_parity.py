from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest
from cli.evaluation.router_learning_policy import (
    ArmState,
    apply_outcome,
    apply_telemetry,
    cost_penalty,
    score_candidate,
    select_winner,
)
from router_learning_parity_support import (
    STATE_FIELDS,
    go_state,
    parity_cases,
    parity_events,
    python_state,
)


def _python_result(case: dict) -> dict:
    max_cost = max(arm["Cost"] for arm in case["Arms"])
    scores, diagnostics = [], []
    for arm in case["Arms"]:
        parameters = {"alpha": 0.0, "beta": 0.0}

        def sample(alpha, beta, parameters=parameters, arm=arm):
            parameters.update(alpha=alpha, beta=beta)
            return arm["Sample"]

        score = score_candidate(
            arm["Model"],
            python_state(arm["State"]),
            cost_penalty(arm["Cost"], max_cost, case["Scope"]),
            arm["Model"] == case["Base"],
            sample if case["Sampling"] else None,
        )
        scores.append(score)
        diagnostics.append({**asdict(score), **parameters})
    return {
        "Scores": diagnostics,
        "Winner": select_winner(
            scores, case["Base"], case["Scope"], case["Sampling"]
        ).model,
    }


def _python_states(events: list[dict]) -> list[dict]:
    state = ArmState()
    states = []
    for event in events:
        if "Telemetry" in event:
            obs = event["Telemetry"]
            apply_telemetry(
                state,
                latency_seconds=(
                    obs["LatencySeconds"] if obs.get("LatencyObserved") else None
                ),
                cache_hit_ratio=(
                    obs["CacheHitRatio"] if obs.get("CacheObserved") else None
                ),
                cache_write_pressure=obs.get("CacheWritePressure", 0.0),
                input_cost_multiplier=(
                    obs["InputCostMultiplier"] if obs.get("InputCostObserved") else None
                ),
                provider_failed=obs.get("ProviderFailureObserved", False),
            )
        else:
            apply_outcome(state, event["Verdict"], event["Weight"])
        states.append(go_state(state))
    return states


def test_complete_equation_and_feedback_match_production_go(tmp_path: Path) -> None:
    go = shutil.which("go")
    assert (
        go is not None
    ), "Go is required for the production Router Learning parity contract"
    source = Path(__file__).resolve().parents[2] / "semantic-router/pkg/extproc"
    request = {"Cases": parity_cases(), "Events": parity_events()}
    input_path, output_path = tmp_path / "input.json", tmp_path / "output.json"
    input_path.write_text(json.dumps(request))
    files = [
        "router_learning_sampling_score.go",
        "router_learning_experience.go",
        "router_learning_numeric.go",
        "router_learning_outcome.go",
        "router_learning_sampling_parity_test.go",
    ]
    completed = subprocess.run(
        [go, "test", *files, "-count=1"],
        cwd=source,
        env={
            **os.environ,
            "VLLM_SR_SAMPLING_PARITY_INPUT": str(input_path),
            "VLLM_SR_SAMPLING_PARITY_OUTPUT": str(output_path),
        },
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    actual = json.loads(output_path.read_text())
    for index, (case, result) in enumerate(
        zip(request["Cases"], actual["Cases"], strict=True)
    ):
        expected = _python_result(case)
        assert result["Winner"] == expected["Winner"], f"case {index}: {case}"
        for left, right in zip(result["Scores"], expected["Scores"], strict=True):
            assert left.keys() == right.keys()
            for key, value in right.items():
                if isinstance(value, float):
                    assert left[key] == pytest.approx(value, abs=1e-12, rel=1e-12), (
                        index,
                        key,
                    )
                else:
                    assert left[key] == value, (index, key)
    for left, right in zip(
        actual["States"], _python_states(request["Events"]), strict=True
    ):
        assert left["LastUpdated"] == right["LastUpdated"]
        for key in STATE_FIELDS.values():
            assert left[key] == pytest.approx(right[key], abs=1e-12, rel=1e-12), key


def test_cold_start_and_exact_tie_order_are_production_order() -> None:
    base = score_candidate("base", ArmState(updated=True), 0, True, lambda a, b: 0.9)
    cold = score_candidate("cold", ArmState(), 0.1, False, lambda a, b: 0.1)
    assert select_winner([base, cold], "base", "global", True).model == "cold"
    assert select_winner([base, cold], "base", "global", False).model == "base"
    assert _python_result(parity_cases()[-1])["Winner"] == "a-model"
