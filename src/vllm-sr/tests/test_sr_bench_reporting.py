"""No-inference checks for comparison eligibility and observed provenance."""

import copy
import time

import pytest
from cli.sr_bench.contracts import plan
from cli.sr_bench.engine import Engine
from cli.sr_bench.offline import replay
from cli.sr_bench.provenance import capture_runner
from cli.sr_bench.report import (
    compare,
    make_report,
    metric,
    paired_conservative_interval,
)
from cli.sr_bench.store import Store
from cli.sr_bench.transport import CallFailure


def _manifest():
    return {
        "version": "sr-bench-1.0",
        "targets": [
            {
                "id": "flash",
                "kind": "single",
                "model": "flash",
                "base_url": "http://127.0.0.1:1/v1",
                "prices": {
                    "flash": {
                        "input": 1,
                        "cached_input": 1,
                        "cache_write": 1,
                        "output": 1,
                    }
                },
            }
        ],
        "cases": [
            {
                "id": "one",
                "benchmark": "mmlu-pro",
                "messages": [{"role": "user", "content": "A"}],
                "answer": "A",
            }
        ],
        "benchmark_options": {"simpleqa-verified": {"judge": "flash"}},
    }


def _record(store, manifest, provenance=None):
    frozen = plan(manifest)
    run, _ = store.create(frozen, provenance=provenance)
    for target in frozen["targets"]:
        store.result(
            run["id"],
            "one",
            target["id"],
            "completed",
            {
                "benchmark": "mmlu-pro",
                "correct": True,
                "score": 1,
                "answer": "A",
                "latency_s": 1,
            },
        )
        call = store.start_call(
            run["id"],
            "one",
            target["id"],
            "subject",
            {"request": {"messages": frozen["cases"][0]["messages"]}},
        )
        store.finish_call(
            call, "completed", {"final": "A", "cost_usd": 0.001, "latency_s": 1}
        )
    store.status(run["id"], "completed")
    return run["id"]


def test_comparison_resolves_equivalent_auxiliary_placement(tmp_path):
    store = Store(tmp_path)
    baseline = _manifest()
    candidate = copy.deepcopy(baseline)
    candidate["auxiliary_targets"] = {"flash": candidate["targets"][0]}
    candidate["targets"] = [
        {
            **candidate["targets"][0],
            "id": "balance",
            "kind": "mom",
            "model": "balance",
            "config_hash": "frozen",
            "max_inference_calls": 1,
        }
    ]
    result = compare(store, _record(store, baseline), _record(store, candidate))
    assert result["comparisons"][0]["paired_cases"] == 1
    assert result["comparisons"][0]["cost_saving_percent"] == 0


def _weighted_matrix(store, order, costs, correct, candidate=False):
    manifest = _manifest()
    prototype = manifest["targets"][0]
    manifest["benchmark_options"] = {}
    manifest["targets"] = [
        {
            **prototype,
            "id": target,
            "model": target,
            "prices": {target: prototype["prices"]["flash"]},
            **(
                {"kind": "mom", "config_hash": "frozen", "max_inference_calls": 1}
                if candidate
                else {}
            ),
        }
        for target in order
    ]
    manifest["cases"] = [
        {**manifest["cases"][0], "id": str(i), "benchmark": benchmark}
        for i, benchmark in enumerate(
            ["mmlu-pro", "mmlu-pro", "gpqa-diamond", "gpqa-diamond", "gpqa-diamond"]
        )
    ]
    # One MMLU success and one GPQA success have exactly equal weighted value,
    # despite different binary float arithmetic paths for 0.1*(1/2) and 0.15*(1/3).
    run, _ = store.create(plan(manifest))
    for target in order:
        for case in manifest["cases"]:
            passed = case["id"] in correct[target]
            store.result(
                run["id"],
                case["id"],
                target,
                "completed",
                {
                    "correct": passed,
                    "score": int(passed),
                    "benchmark": case["benchmark"],
                },
            )
            call = store.start_call(run["id"], case["id"], target, "subject", {})
            store.finish_call(
                call,
                "completed",
                {
                    "cost_usd": (
                        None
                        if costs[target] is None
                        else costs[target] / len(manifest["cases"])
                    )
                },
            )
    store.status(run["id"], "completed")
    return run["id"]


@pytest.mark.parametrize("reverse", [False, True])
def test_best_single_quality_tie_uses_cheapest_independent_of_manifest_order(
    tmp_path, reverse
):
    store = Store(tmp_path)
    order = ["expensive", "cheap"]
    if reverse:
        order.reverse()
    baseline = _weighted_matrix(
        store,
        order,
        {"expensive": 8, "cheap": 2},
        {"expensive": {"3"}, "cheap": {"0"}},
    )
    candidate = _weighted_matrix(
        store, ["balance"], {"balance": 1}, {"balance": {"3"}}, candidate=True
    )
    result = compare(store, baseline, candidate)
    assert result["baseline_tied_best_target_ids"] == ["cheap", "expensive"]
    assert result["baseline_selected_target_id"] == "cheap"
    assert result["baseline_cost_comparison_eligible"] is True
    assert result["baseline_cost_comparison_reason"] is None
    paired = result["comparisons"][0]
    assert paired["baseline_target_id"] == "cheap"
    assert paired["quality_delta"] == 0
    assert paired["cost_saving_percent"] == 50


@pytest.mark.parametrize(
    "baseline_correct,candidate_correct,candidate_cost,quality_delta,saving",
    [
        ({"0", "1", "2", "3", "4"}, set(), 3, -1, -50),
        (set(), {"0", "1", "2", "3", "4"}, 1, 1, 50),
        ({"0"}, {"0"}, 2, 0, 0),
    ],
)
def test_comparison_retains_negative_positive_and_zero_changes(
    tmp_path,
    baseline_correct,
    candidate_correct,
    candidate_cost,
    quality_delta,
    saving,
):
    store = Store(tmp_path)
    baseline = _weighted_matrix(
        store, ["flash"], {"flash": 2}, {"flash": baseline_correct}
    )
    candidate = _weighted_matrix(
        store,
        ["balance"],
        {"balance": candidate_cost},
        {"balance": candidate_correct},
        candidate=True,
    )
    row = compare(store, baseline, candidate)["comparisons"][0]
    assert row["quality_delta"] == pytest.approx(quality_delta)
    assert row["cost_saving_percent"] == pytest.approx(saving)


@pytest.mark.parametrize(
    "costs,selected,eligible",
    [
        ({"a": 2, "z": 2}, "a", True),
        ({"a": None, "z": 2}, "z", False),
        ({"a": None, "z": None}, "a", False),
    ],
)
def test_tied_baseline_has_stable_id_and_explicit_unknown_cost_policy(
    tmp_path, costs, selected, eligible
):
    store = Store(tmp_path)
    baseline = _weighted_matrix(store, ["z", "a"], costs, {"a": {"0"}, "z": {"0"}})
    candidate = _weighted_matrix(
        store, ["balance"], {"balance": 1}, {"balance": {"0"}}, candidate=True
    )
    result = compare(store, baseline, candidate)
    assert result["baseline_selected_target_id"] == selected
    assert result["baseline_cost_comparison_eligible"] is eligible
    assert result["baseline_tied_best_target_ids"] == ["a", "z"]
    if not eligible:
        assert "incomplete cost" in result["baseline_cost_comparison_reason"]
        assert result["comparisons"][0]["cost_saving_percent"] is None


def test_stronger_quality_is_not_replaced_by_cheaper_lower_quality(tmp_path):
    store = Store(tmp_path)
    baseline = _weighted_matrix(
        store,
        ["cheap", "best"],
        {"cheap": 1, "best": 8},
        {"cheap": {"0"}, "best": {"0", "1"}},
    )
    candidate = _weighted_matrix(
        store, ["balance"], {"balance": 2}, {"balance": {"0", "1"}}, candidate=True
    )
    result = compare(store, baseline, candidate)
    assert result["baseline_selected_target_id"] == "best"
    assert result["baseline_tied_best_target_ids"] == ["best"]


@pytest.mark.parametrize("successes", [set(), {"0", "1", "2", "3", "4"}])
def test_identical_or_all_wrong_pairs_do_not_prove_equivalence(tmp_path, successes):
    store = Store(tmp_path)
    baseline = _weighted_matrix(store, ["single"], {"single": 1}, {"single": successes})
    candidate = _weighted_matrix(
        store, ["balance"], {"balance": 1}, {"balance": successes}, candidate=True
    )
    result = compare(store, baseline, candidate)["comparisons"][0]
    assert result["quality_delta"] == 0
    assert result["quality_delta_bootstrap_ci95"] == [0, 0]
    assert result["quality_delta_ci95"][0] < 0 < result["quality_delta_ci95"][1]
    assert result["quality_delta_ci95_method"] == "weighted-paired-hoeffding"


def test_conservative_paired_bound_accounts_for_strata_and_sample_size():
    small = paired_conservative_interval(0, {"a": [0] * 25}, {"a": 1})
    larger = paired_conservative_interval(0, {"a": [0] * 100}, {"a": 1})
    assert small[1] == pytest.approx(2 * larger[1])
    stratified = paired_conservative_interval(
        0, {"a": [0] * 10, "b": [0] * 90}, {"a": 0.5, "b": 0.5}
    )
    assert stratified[1] > larger[1]
    assert paired_conservative_interval(1, {"a": [1]}, {"a": 1}) == [-1, 1]


def test_pending_model_alias_is_not_an_observed_backend():
    calls = [
        {"role": "subject", "status": "sent", "model": "vllm-sr/mom-v1-blend"},
        {"role": "subject", "status": "completed", "model": "actual-backend"},
        {
            "role": "subject",
            "status": "failed",
            "model": "vllm-sr/mom-v1-blend",
            "selected_model": "acknowledged-backend",
        },
    ]
    result = metric("balance", [], calls, 3)
    assert result["selected_models"] == {"actual-backend": 1, "acknowledged-backend": 1}
    assert result["pending_selection_count"] == 1


@pytest.mark.parametrize("change", ["prices", "limits", "native-profile", "judge"])
def test_comparison_rejects_incompatible_protocol(tmp_path, change):
    store = Store(tmp_path)
    baseline, candidate = _manifest(), _manifest()
    candidate["benchmark_options"] = baseline["benchmark_options"] = {}
    if change == "prices":
        candidate["targets"][0]["prices"]["flash"]["input"] = 0.01
    elif change == "limits":
        candidate["limits"] = {"max_output_chars": 16}
    elif change == "native-profile":
        candidate["targets"][0]["request_params"] = {"reasoning_effort": "high"}
    else:
        baseline["benchmark_options"] = candidate["benchmark_options"] = {
            "simpleqa-verified": {"judge": "flash"}
        }
        candidate["targets"][0]["model"] = "other"
        candidate["targets"][0]["prices"]["other"] = candidate["targets"][0]["prices"][
            "flash"
        ]
    with pytest.raises(ValueError, match="Cannot compare"):
        compare(store, _record(store, baseline), _record(store, candidate))


def test_server_provenance_is_observed_and_old_rows_stay_unknown(tmp_path):
    store = Store(tmp_path)
    manifest = _manifest()
    observed = capture_runner(plan(manifest))
    assert len(observed["source_sha256"]) == 64
    assert len(observed["environment_sha256"]) == 64
    assert "transport.py" in observed["source_files"]
    old = _record(store, manifest)
    new = _record(store, manifest, observed)
    assert make_report(store, old)["provenance"]["runner"] is None
    assert make_report(store, new)["provenance"]["runner"] == observed
    with pytest.raises(ValueError, match="captured by the service"):
        plan({**manifest, "runner_provenance": {"source_sha256": "forged"}})


def test_learning_preview_distribution_and_replay_rejection(tmp_path):
    store = Store(tmp_path)
    baseline = _record(store, _manifest())
    manifest = _manifest()
    manifest["mode"] = "preview"
    manifest["targets"] = [
        {
            **manifest["targets"][0],
            "id": "balance",
            "kind": "mom",
            "preview_url": "http://127.0.0.1:1/api/v1/routing/preview",
        }
    ]
    frozen = plan(manifest)
    assert frozen["preview_context"]["sampling_seed"] == frozen["seed"]
    preview, _ = store.create(frozen)
    routing = {
        "selection_status": "execution_required",
        "selection_reason": "runtime learning state",
        "selection_method": "static",
        "selected_model": "flash",
        "decision_result": {"plugins": []},
        "selection_provenance": {"mode": "read_only_snapshot", "state_dependent": True},
    }
    store.result(
        preview["id"], "one", "balance", "completed", {"details": {"routing": routing}}
    )
    store.status(preview["id"], "completed")
    report = make_report(store, preview["id"])
    assert report["summary"]["targets"][0]["selection_statuses"] == {
        "execution_required": 1
    }
    assert report["benchmarks"][0]["selection_reasons"] == {"runtime learning state": 1}
    with pytest.raises(ValueError, match="State-dependent"):
        replay(store, baseline, preview["id"])


def test_failed_run_explains_first_failure_and_legacy_report_is_read_only(
    tmp_path, monkeypatch
):
    calls = []

    def fail(*args, **kwargs):
        calls.append(True)
        raise CallFailure("total request deadline exceeded; private transport detail")

    monkeypatch.setattr("cli.sr_bench.engine.chat", fail)
    store = Store(tmp_path)
    run = Engine(store).start(_manifest())
    deadline = time.monotonic() + 5
    while store.get(run["id"])["status"] in {"queued", "running"}:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    completed = store.get(run["id"])
    assert completed["status"] == "failed"
    assert "Case one, target flash" in completed["error"]
    assert "deadline exceeded" in completed["error"]
    assert "private" not in completed["error"]
    assert calls == [True]
    report = make_report(store, run["id"])
    assert report["failure"]["inferred_from_saved_results"] is False
    assert store.events(run["id"])[-1]["data"]["error"] == completed["error"]

    # A historical failed row stays untouched: inference is confined to reporting.
    old, _ = store.create(plan(_manifest()))
    store.result(old["id"], "one", "flash", "failed", {"error": "ReadTimeout"})
    store.status(old["id"], "failed")
    old_report = make_report(store, old["id"])
    assert old_report["failure"]["reason"] == "Target idle/read timeout"
    assert old_report["failure"]["inferred_from_saved_results"] is True
    assert store.get(old["id"])["error"] is None
