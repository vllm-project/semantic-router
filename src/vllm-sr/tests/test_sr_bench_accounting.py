"""Raw-stream reconciliation preserves receipts and never repeats generation."""

import copy
import json
import threading
from http import HTTPStatus

import pytest
import requests
from cli.sr_bench.accounting import reconcile_usage
from cli.sr_bench.contracts import digest, plan
from cli.sr_bench.engine import Engine
from cli.sr_bench.recovery import RecoveryPlanError, recover, recovery_plan
from cli.sr_bench.report import compare, make_report
from cli.sr_bench.service import PREFIX, Server
from cli.sr_bench.store import Store
from cli.sr_bench.transport import CallFailure, cost_for, normalize_usage


@pytest.mark.parametrize("field", ["created_cache_tokens", "cache_creation_tokens"])
def test_four_exclusive_buckets_include_provider_cache_writes(field):
    raw = {
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "prompt_tokens_details": {"cached_tokens": 20, field: 30},
    }
    usage = normalize_usage(raw)
    assert usage == {
        "input_tokens": 50,
        "cached_input_tokens": 20,
        "cache_write_tokens": 30,
        "output_tokens": 10,
    }
    assert sum(usage.values()) == 110
    assert cost_for(usage, "model", {"model": _prices()}) == pytest.approx(0.0001195)


def test_actual_provider_usage_receipt_is_not_priced_as_fresh_input():
    usage = normalize_usage(
        {
            "prompt_tokens": 4175,
            "completion_tokens": 4096,
            "prompt_tokens_details": {"cached_tokens": 0, "created_cache_tokens": 3136},
        }
    )
    price = {
        "input": 0.65,
        "cached_input": 0.065,
        "cache_write": 0.8125,
        "output": 1.95,
    }
    assert usage["input_tokens"] == 1039
    assert cost_for(usage, "flash", {"flash": price}) == pytest.approx(0.01121055)


@pytest.mark.parametrize(
    "details",
    [
        {"created_cache_tokens": 30, "cache_creation_tokens": 31},
        {"created_cache_tokens": True},
        {"created_cache_tokens": -1},
        {"created_cache_tokens": 90, "cached_tokens": 20},
    ],
)
def test_invalid_or_conflicting_cache_writes_fail_closed(details):
    with pytest.raises(CallFailure):
        normalize_usage(
            {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "prompt_tokens_details": details,
            }
        )


def _prices():
    return {"input": 1, "cached_input": 0.1, "cache_write": 1.25, "output": 3}


def _saved(store, *, kind="single", owner="alice", cache_write=30, cached=20):
    target = {
        "id": "model" if kind == "single" else "balance",
        "kind": kind,
        "model": "model" if kind == "single" else "balance",
        "base_url": "http://127.0.0.1:1/v1",
        "prices": {"model": _prices()},
    }
    if kind == "mom":
        target.update(config_hash="frozen", max_inference_calls=1)
    frozen = plan(
        {
            "version": "sr-bench-1.0",
            "targets": [target],
            "cases": [
                {
                    "id": "one",
                    "benchmark": "mmlu-pro",
                    "messages": [{"role": "user", "content": "A"}],
                    "answer": "A",
                }
            ],
        }
    )
    run, _ = store.create(frozen, owner=owner)
    raw = {
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "prompt_tokens_details": {
            "cached_tokens": cached,
            "created_cache_tokens": cache_write,
        },
    }
    old_usage = {
        "input_tokens": 100 - cached,
        "cached_input_tokens": cached,
        "cache_write_tokens": 0,
        "output_tokens": 10,
    }
    cost = cost_for(old_usage, "model", target["prices"])
    call = store.start_call(run["id"], "one", target["id"], "subject", {})
    store.finish_call(
        call,
        "completed",
        {
            "model": "model",
            "selected_model": "model",
            "raw_usage": raw,
            "usage": old_usage,
            "cost_usd": cost,
            "finish_reason": "stop",
            "output_complete": True,
            "final": "A",
            "inference_call_count": 1,
        },
    )
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
            "usage": old_usage,
            "cost_usd": cost,
        },
    )
    store.status(run["id"], "completed")
    path = (
        store.root
        / "runs"
        / run["id"]
        / digest(["one", target["id"]])[:24]
        / (call + ".sse")
    )
    path.parent.mkdir(parents=True)
    event = {
        "model": "model",
        "usage": raw,
        "choices": [{"index": 0, "delta": {"content": "A"}, "finish_reason": "stop"}],
    }
    path.write_text("data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n")
    return run["id"], call, path


def test_reconcile_is_idempotent_durable_and_preserves_all_original_receipts(
    tmp_path, monkeypatch
):
    store = Store(tmp_path)
    run, call, path = _saved(store)
    original = copy.deepcopy(store.call(run, call))
    original_run = store.get(run)
    original_results = store.results(run)
    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **kw: pytest.fail(
            "Offline reconciliation attempted a model request"
        ),
    )
    first = reconcile_usage(store, run)
    assert first == reconcile_usage(store, run)
    assert first["qualified"] and first["corrected_call_count"] == 1
    assert first["model_requests"] == 0
    assert store.call(run, call) == original
    assert store.get(run) == original_run
    assert store.results(run) == original_results
    assert path.exists()
    assert (
        len([e for e in store.events(run) if e["kind"] == "accounting_reconciled"]) == 1
    )
    store.db.close()
    reopened = Store(tmp_path)
    report = make_report(reopened, run)
    target = report["summary"]["targets"][0]
    assert target["cost_usd"] == pytest.approx(0.0001195)
    assert target["tokens"]["cache_write_tokens"] == 30
    assert report["provenance"]["accounting_correction"]["id"] == first["id"]
    assert "calls" not in report["provenance"]["accounting_correction"]
    assert reopened.call(run, call) == original


@pytest.mark.parametrize("kind", ["single", "mom"])
def test_report_and_comparison_use_correction_and_cache_neutral_cost(tmp_path, kind):
    store = Store(tmp_path)
    baseline, _, _ = _saved(store, cache_write=30, cached=0)
    candidate, _, _ = _saved(store, kind=kind, cache_write=0, cached=80)
    reconcile_usage(store, baseline)
    reconcile_usage(store, candidate)
    result = compare(store, baseline, candidate)["comparisons"][0]
    assert result["baseline_subject_cost_usd"] == pytest.approx(0.0001375)
    assert result["candidate_subject_cost_usd"] == pytest.approx(0.000058)
    assert result["subject_cost_saving_percent"] > 50
    assert result["baseline_total_cost_usd"] == result["baseline_subject_cost_usd"]
    assert result["candidate_total_cost_usd"] == result["candidate_subject_cost_usd"]
    assert result["total_cost_saving_percent"] == result["subject_cost_saving_percent"]
    assert result["cache_neutral_baseline_cost_usd"] == pytest.approx(0.00013)
    assert result["cache_neutral_candidate_cost_usd"] == pytest.approx(0.00013)
    assert result["cache_neutral_cost_saving_percent"] == 0


@pytest.mark.parametrize("problem", ["missing", "mismatch", "incomplete", "multi_call"])
def test_unverifiable_usage_stays_unknown_in_corrected_reports(tmp_path, problem):
    store = Store(tmp_path)
    run, call, path = _saved(store, kind="mom" if problem == "multi_call" else "single")
    if problem == "missing":
        path.unlink()
    elif problem == "mismatch":
        path.write_text(
            path.read_text().replace(
                '"created_cache_tokens": 30', '"created_cache_tokens": 31'
            )
        )
    elif problem == "incomplete":
        path.write_text(path.read_text().replace("data: [DONE]", ""))
    else:
        store.finish_call(call, "completed", {"inference_call_count": 2})
    artifact = reconcile_usage(store, run)
    assert not artifact["qualified"]
    assert artifact["unverifiable_call_count"] == 1
    target = make_report(store, run)["summary"]["targets"][0]
    assert target["cost_usd"] is None
    assert target["tokens"] is None
    assert not target["cost_complete"]


def test_reconcile_rejects_active_run(tmp_path):
    store = Store(tmp_path)
    run, _, _ = _saved(store)
    store.status(run, "running")
    with pytest.raises(ValueError, match="terminal live"):
        reconcile_usage(store, run)
    assert store.accounting_correction(run) is None


def _failed_scoring_parent(store):
    run, call, path = _saved(store)
    store.result(
        run,
        "one",
        "model",
        "failed",
        {"benchmark": "mmlu-pro", "correct": None, "error": "Scoring failed"},
    )
    store.status(run, "failed", "Scoring failed")
    return run, call, path


def test_recovery_snapshots_corrected_parent_spend_without_rewriting_history(
    tmp_path, monkeypatch
):
    store = Store(tmp_path)
    parent, call, _ = _failed_scoring_parent(store)
    original_run = store.get(parent)
    original_call = store.call(parent, call)
    original_results = store.results(parent)
    stale = recovery_plan(store, parent, "failed")
    correction = reconcile_usage(store, parent)
    fresh = recovery_plan(store, parent, "failed")
    assert fresh["eligible_cells"] == stale["eligible_cells"]
    assert fresh["plan_sha256"] != stale["plan_sha256"]
    assert fresh["parent"]["known_spend_usd"] == pytest.approx(0.0001195)
    assert fresh["parent"]["spend_complete"] is True
    assert fresh["parent"]["accounting_correction"]["id"] == correction["id"]
    assert "calls" not in fresh["parent"]["accounting_correction"]
    assert make_report(store, parent)["summary"]["total_spend_usd"] == pytest.approx(
        fresh["parent"]["known_spend_usd"]
    )
    engine = Engine(store)

    def no_dispatch(*_args, **_kwargs):
        pytest.fail("Recovery unexpectedly dispatched a model request")

    monkeypatch.setattr(engine, "start", no_dispatch)
    body = {
        "mode": "failed",
        "cells": fresh["eligible_cells"],
        "acknowledge_new_attempt": True,
        "idempotency_key": "new-recovery-attempt",
        "plan_sha256": stale["plan_sha256"],
    }
    with pytest.raises(RecoveryPlanError, match="eligibility changed"):
        recover(engine, parent, body, owner="alice")

    def save_child(manifest, owner, key, *, recovery, actor_role):
        assert recovery is True
        assert actor_role == "local"
        return store.create(plan(manifest), owner, key)[0]

    monkeypatch.setattr(engine, "start", save_child)
    child = recover(
        engine,
        parent,
        {**body, "plan_sha256": fresh["plan_sha256"]},
        owner="alice",
    )
    assert child["manifest"]["recovery"]["parent_snapshot"] == fresh["parent"]
    assert store.get(parent) == original_run
    assert store.call(parent, call) == original_call
    assert store.results(parent) == original_results
    store.db.close()
    reopened = Store(tmp_path)
    report = make_report(reopened, child["id"])
    assert report["recovery"]["parent_snapshot"] == fresh["parent"]
    assert report["summary"]["total_spend_usd"] is None
    assert reopened.call(parent, call) == original_call


def test_failed_recovery_excludes_accounting_reconciliation_unknowns(tmp_path):
    store = Store(tmp_path)
    parent, call, path = _failed_scoring_parent(store)
    original = store.call(parent, call)
    assert recovery_plan(store, parent, "failed")["counts"]["eligible"] == 1
    path.unlink()
    correction = reconcile_usage(store, parent)
    assert not correction["qualified"]
    proposed = recovery_plan(store, parent, "failed")
    assert proposed["counts"]["eligible"] == 0
    assert proposed["excluded"][0]["reason"] == (
        "unfinished_or_unknown_accounting_requires_reconciliation"
    )
    assert proposed["parent"]["known_spend_usd"] == 0
    assert proposed["parent"]["spend_complete"] is False
    assert proposed["parent"]["accounting_correction"]["id"] == correction["id"]
    report = make_report(store, parent)
    assert report["summary"]["targets"][0]["known_cost_usd"] == 0
    assert report["summary"]["total_spend_usd"] is None
    assert store.call(parent, call) == original


def test_reconcile_api_owner_and_write_scope(tmp_path):
    store = Store(tmp_path)
    run, _, _ = _saved(store)
    service = Server(("127.0.0.1", 0), store, "test-token")
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{service.server_port}{PREFIX}/runs/{run}/reconcile-usage"
    headers = {
        "Authorization": "Bearer test-token",
        "X-SR-Bench-Actor-ID": "alice",
        "X-SR-Bench-Actor-Role": "write",
    }
    try:
        assert (
            requests.post(
                url,
                json={},
                headers={**headers, "X-SR-Bench-Actor-ID": "bob"},
                timeout=2,
            ).status_code
            == HTTPStatus.NOT_FOUND
        )
        assert (
            requests.post(
                url,
                json={},
                headers={**headers, "X-SR-Bench-Actor-Role": "read"},
                timeout=2,
            ).status_code
            == HTTPStatus.FORBIDDEN
        )
        response = requests.post(url, json={}, headers=headers, timeout=2)
        assert response.status_code == HTTPStatus.OK
        assert response.json()["qualified"]
    finally:
        service.shutdown()
        service.server_close()
