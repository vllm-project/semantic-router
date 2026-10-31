from pathlib import Path

import yaml
from cli.commands.recipe_learning import build_recipe_learning_artifact
from cli.commands.recipe_learning_findings import build_run_findings
from cli.commands.recipe_learning_runs import (
    build_run_model_choice,
    group_runs,
    wilson_interval,
)


def _record(conversation, turn, model, cost, decision="code", outcome=None, stamp="t"):
    record = {
        "id": f"{conversation}-{turn}",
        "session_id": "s-" + conversation,
        "conversation_id": conversation,
        "turn_index": turn,
        "timestamp": f"2026-10-20T00:00:{turn:02d}Z",
        "decision": decision,
        "selected_model": model,
        "actual_cost": cost,
        "outcomes": [],
    }
    if outcome is not None:
        record["outcomes"].append(
            {
                "timestamp": stamp,
                "target": "model",
                "target_ref": model,
                "verdict": outcome,
            }
        )
    return record


def _runs(model, n, successes, cost_per_turn, turns=3, decision="code", prefix=None):
    records = []
    for i in range(n):
        conv = f"{prefix or model}-{i}"
        verdict = "good_fit" if i < successes else "underpowered"
        for turn in range(turns):
            records.append(
                _record(
                    conv,
                    turn,
                    model,
                    cost_per_turn,
                    decision,
                    verdict if turn == 0 else None,
                )
            )
    return records


def test_run_cost_sums_every_turn_and_outcome_comes_from_any_record():
    records = [
        _record("a", 0, "cheap", 0.01),
        _record("a", 1, "cheap", 0.02),
        _record("a", 2, "cheap", 0.03, outcome="good_fit"),
    ]
    runs, unidentified = group_runs(records)
    assert unidentified == 0
    assert abs(runs["a"].cost - 0.06) < 1e-12
    assert runs["a"].model == "cheap"
    assert runs["a"].outcome[1] is True


def test_latest_outcome_wins_and_failed_counts_as_failure():
    records = [
        _record(
            "a", 0, "cheap", 0.01, outcome="good_fit", stamp="2026-10-20T01:00:00Z"
        ),
        _record("a", 1, "cheap", 0.01, outcome="failed", stamp="2026-10-20T02:00:00Z"),
    ]
    runs, _ = group_runs(records)
    assert runs["a"].outcome[1] is False


def test_switched_run_belongs_to_its_first_model_but_stays_out_of_the_statistics():
    records = [
        _record("a", 1, "strong", 0.5),
        _record("a", 0, "cheap", 0.01, outcome="good_fit"),
    ]
    payload = build_run_model_choice(records)
    models = payload["per_decision"]["code"]["models"]
    assert list(models) == ["cheap"]
    assert models["cheap"]["runs"] == 1
    assert models["cheap"]["switched_runs"] == 1
    assert models["cheap"]["labeled_runs"] == 0


def test_times_compare_as_times_not_strings():
    # replay timestamps drop trailing zeros: .1Z is earlier than .15Z
    early = _record(
        "a", 0, "cheap", 0.01, outcome="underpowered", stamp="2026-10-20T00:00:03.1Z"
    )
    late = _record(
        "a", 0, "cheap", 0.01, outcome="good_fit", stamp="2026-10-20T00:00:03.15Z"
    )
    late["id"] = "a-0b"
    runs, _ = group_runs([late, early])
    assert runs["a"].outcome[1] is True


def test_under_sampled_model_can_block_a_confident_call():
    # "rare" has only 10 runs, too few to be recommended, but its wide interval
    # still overlaps the leader's, so the leader is not confident.
    records = _runs("lead", 300, 180, 0.01, turns=1) + _runs(
        "rare", 10, 7, 0.01, turns=1
    )
    rec = build_run_model_choice(records, [1.0])["per_decision"]["code"][
        "recommendations"
    ][0]
    assert rec["model"] == "lead"
    assert rec["confident"] is False
    assert "rare" in rec["reason"]


def test_cost_uncertainty_can_block_a_confident_call():
    # Same success rates; a's runs cost 0.10 on average but vary widely.
    records = []
    for i in range(40):
        records.append(
            _record(f"a{i}", 0, "a", 0.0 if i % 2 else 0.2, outcome="good_fit")
        )
        records.append(_record(f"b{i}", 0, "b", 0.15, outcome="good_fit"))
    rec = build_run_model_choice(records, [1.0])["per_decision"]["code"][
        "recommendations"
    ][0]
    assert rec["model"] == "a"
    assert rec["confident"] is False


def test_records_without_identity_are_counted_not_grouped():
    record = _record("a", 0, "cheap", 0.01)
    record["conversation_id"] = ""
    record["session_id"] = ""
    payload = build_run_model_choice([record])
    assert payload["runs"] == 0
    assert payload["unidentified_records"] == 1


def test_wilson_interval_brackets_the_rate():
    low, high = wilson_interval(30, 50)
    assert low < 0.6 < high
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_recommendation_depends_on_value_per_success():
    # cheap: 60% at $0.05 per run; strong: 75% at $0.75 per run. At $10 per success
    # the strong model wins only once enough runs narrow both intervals.
    records = _runs("cheap", 1000, 600, 0.05, turns=1) + _runs(
        "strong", 1000, 750, 0.75, turns=1
    )
    payload = build_run_model_choice(records, [1.0, 10.0])
    low_value, high_value = payload["per_decision"]["code"]["recommendations"]
    assert low_value["model"] == "cheap" and low_value["confident"]
    assert high_value["model"] == "strong" and high_value["confident"]


def test_wide_intervals_are_not_confident():
    records = _runs("cheap", 200, 120, 0.05 / 3) + _runs("strong", 200, 150, 0.25)
    rec = build_run_model_choice(records, [10.0])["per_decision"]["code"][
        "recommendations"
    ][0]
    assert rec["model"] == "strong"
    assert rec["confident"] is False


def test_close_models_are_not_confident():
    records = _runs("a", 30, 15, 0.01) + _runs("b", 30, 16, 0.01)
    rec = build_run_model_choice(records, [1.0])["per_decision"]["code"][
        "recommendations"
    ][0]
    assert rec["model"] == "b"
    assert rec["confident"] is False


def test_too_few_labeled_runs_gives_no_recommendation():
    rec = build_run_model_choice(_runs("a", 5, 3, 0.01), [1.0])["per_decision"]["code"][
        "recommendations"
    ][0]
    assert rec["model"] is None


def test_confident_recommendation_becomes_default_model_patch():
    records = _runs("cheap", 200, 120, 0.05 / 3) + _runs("strong", 200, 150, 0.25)
    recipe = {
        "routing": {
            "decisions": [
                {"name": "code", "modelRefs": [{"model": "strong"}, {"model": "cheap"}]}
            ]
        }
    }
    artifact = build_recipe_learning_artifact(
        records, {}, recipe, values_per_success=[1.0]
    )
    findings = [f for f in artifact["findings"] if f["type"] == "default_model"]
    assert len(findings) == 1
    assert findings[0]["evidence"]["current_default"] == "strong"
    suggestion = next(
        s
        for s in artifact["recipe_patch"]["suggestions"]
        if s["finding_type"] == "default_model"
    )
    assert suggestion["path"] == "/routing/decisions/code/modelRefs"
    assert suggestion["action"] == "reorder"
    candidate = next(
        c
        for c in artifact["candidate_recipes"]
        if c["patch"]["suggestions"][0] == suggestion
    )
    refs = candidate["recipe"]["routing"]["decisions"][0]["modelRefs"]
    assert [r["model"] for r in refs] == ["cheap", "strong"]


def test_no_finding_when_recommendation_is_already_the_default():
    records = _runs("cheap", 200, 120, 0.05 / 3) + _runs("strong", 200, 150, 0.25)
    recipe = {
        "routing": {
            "decisions": [
                {"name": "code", "modelRefs": [{"model": "cheap"}, {"model": "strong"}]}
            ]
        }
    }
    artifact = build_recipe_learning_artifact(
        records, {}, recipe, values_per_success=[1.0]
    )
    assert not [f for f in artifact["findings"] if f["type"] == "default_model"]


def test_no_values_means_no_recommendations():
    payload = build_run_model_choice(_runs("a", 30, 15, 0.01))
    assert payload["per_decision"]["code"]["recommendations"] == []


def _typed_cascade_runs():
    records = _runs("native-kai", 200, 150, 0.05, decision="typed-cascade")
    records += _runs("native-vega", 200, 120, 0.25, decision="typed-cascade")
    for record in records:
        record["recipe"] = "native-handwritten"
    return records


def test_recipe_owned_decision_reads_its_recipe_default():
    config = yaml.safe_load(
        (Path(__file__).parents[3] / "config" / "config.yaml").read_text()
    )
    runs = build_run_model_choice(_typed_cascade_runs(), [1.0])
    (rec,) = runs["per_decision"]["native-handwritten/typed-cascade"]["recommendations"]
    assert rec["model"] == "native-kai" and rec["confident"]
    assert build_run_findings(runs, config) == []


def test_recipe_owned_decision_patch_lands_in_its_recipe():
    top_level = {"name": "typed-cascade", "modelRefs": [{"model": "native-vega"}]}
    recipe = {
        "routing": {"decisions": [top_level]},
        "recipes": [
            {
                "name": "native-handwritten",
                "routing": {
                    "decisions": [
                        {
                            "name": "typed-cascade",
                            "modelRefs": [
                                {"model": "native-vega"},
                                {"model": "native-kai"},
                            ],
                        }
                    ]
                },
            }
        ],
    }
    artifact = build_recipe_learning_artifact(
        _typed_cascade_runs(), {}, recipe, values_per_success=[1.0]
    )
    (candidate,) = [
        c
        for c in artifact["candidate_recipes"]
        if c["patch"]["suggestions"][0]["finding_type"] == "default_model"
    ]
    assert candidate["patch"]["suggestions"][0]["recipe"] == "native-handwritten"
    owned = candidate["recipe"]["recipes"][0]["routing"]["decisions"][0]
    assert [r["model"] for r in owned["modelRefs"]] == ["native-kai", "native-vega"]
    assert candidate["recipe"]["routing"]["decisions"][0] == top_level
