"""Recovery creates explicit child attempts and never duplicates saved dispatches."""

import copy
import threading
import time
from http import HTTPStatus

import pytest
import requests
from cli.sr_bench.contracts import plan
from cli.sr_bench.engine import Engine
from cli.sr_bench.recovery import recover, recovery_plan
from cli.sr_bench.report import make_report
from cli.sr_bench.service import PREFIX, Server
from cli.sr_bench.store import Store


def _parent(store, owner="local"):
    target = {
        "kind": "single",
        "model": "model",
        "base_url": "http://127.0.0.1:1/v1",
        "prices": {
            "model": {"input": 1, "cached_input": 1, "cache_write": 1, "output": 1}
        },
    }
    manifest = plan(
        {
            "version": "sr-bench-1.0",
            "targets": [{**target, "id": "a"}, {**target, "id": "b"}],
            "cases": [
                {
                    "id": f"q{i}",
                    "benchmark": "mmlu-pro",
                    "messages": [{"role": "user", "content": f"q{i}"}],
                    "answer": "A",
                }
                for i in range(4)
            ],
        }
    )
    parent, _ = store.create(manifest, owner)
    identity = parent["id"]
    for case, target_id, status, known in [
        ("q0", "a", "completed", True),
        ("q1", "a", "sent_unknown", False),
        ("q1", "b", "failed", False),
        ("q3", "a", "completed", True),
        ("q3", "b", "failed", True),
    ]:
        call = store.start_call(identity, case, target_id, "subject", {})
        response = (
            {
                "usage": {
                    "input_tokens": 1,
                    "cached_input_tokens": 0,
                    "cache_write_tokens": 0,
                    "output_tokens": 1,
                },
                "cost_usd": 0.001,
                "finish_reason": "stop",
                "output_complete": True,
                "final": "A",
            }
            if known
            else {}
        )
        store.finish_call(call, status, response)
        result_status = "completed" if case == "q0" else "failed"
        store.result(
            identity,
            case,
            target_id,
            result_status,
            {
                "correct": case == "q0",
                "score": int(case == "q0"),
                "latency_s": 1,
                "error": None if case == "q0" else "Harness failed",
            },
        )
    store.status(identity, "failed", "Saved parent failure")
    return identity


def _body(proposed, key, selected=None):
    return {
        "mode": proposed["mode"],
        "plan_sha256": proposed["plan_sha256"],
        "cells": proposed["eligible_cells"] if selected is None else selected,
        "idempotency_key": key,
    }


def _wait(store, identity):
    deadline = time.monotonic() + 5
    while store.get(identity)["status"] in {"queued", "running"}:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    return store.get(identity)


def _transport(monkeypatch):
    dispatched = []

    def chat(target, messages, *_args, **_kwargs):
        dispatched.append((messages[0]["content"], target["id"]))
        return {
            "final": "A",
            "usage": {
                "input_tokens": 1,
                "cached_input_tokens": 0,
                "cache_write_tokens": 0,
                "output_tokens": 1,
            },
            "cost_usd": 0.000002,
            "latency_s": 0.01,
            "finish_reason": "stop",
            "output_complete": True,
            "inference_call_count": 1,
        }

    monkeypatch.setattr("cli.sr_bench.engine.chat", chat)
    return dispatched


def test_continue_only_undispatched_cells_and_keep_parent_immutable(
    tmp_path, monkeypatch
):
    store = Store(tmp_path)
    parent = _parent(store)
    original = copy.deepcopy(store.get(parent))
    original_results, original_calls = store.results(parent), store.calls(parent)
    proposed = recovery_plan(store, parent)
    assert {
        (cell["case_id"], cell["target_id"]) for cell in proposed["eligible_cells"]
    } == {("q0", "b"), ("q2", "a"), ("q2", "b")}
    dispatched = _transport(monkeypatch)
    engine = Engine(store)
    body = _body(proposed, "safe-once")
    child = recover(engine, parent, body)
    assert _wait(store, child["id"])["status"] == "completed"
    assert set(dispatched) == {("q0", "b"), ("q2", "a"), ("q2", "b")}
    assert recover(engine, parent, body)["id"] == child["id"]
    assert len(dispatched) == 3
    assert store.get(parent) == original
    assert (
        store.results(parent) == original_results
        and store.calls(parent) == original_calls
    )
    assert recovery_plan(store, parent)["counts"]["eligible"] == 0
    with pytest.raises(ValueError, match="eligibility changed"):
        recover(engine, parent, {**body, "idempotency_key": "another"})
    report = make_report(store, child["id"])
    assert {row["id"]: row["total"] for row in report["summary"]["targets"]} == {
        "a": 1,
        "b": 2,
    }
    assert report["summary"]["total_spend_usd"] == pytest.approx(0.000006)
    assert report["recovery"]["parent_snapshot"]["known_spend_usd"] == 0.003
    assert report["recovery"]["parent_snapshot"]["spend_complete"] is False
    assert all(row["sr_bench_score"] is None for row in report["summary"]["targets"])
    assert make_report(store, parent)["child_attempts"][0]["id"] == child["id"]


def test_failed_retry_requires_known_final_accounting_and_exact_acknowledgment(
    tmp_path, monkeypatch
):
    store = Store(tmp_path)
    parent = _parent(store)
    proposed = recovery_plan(store, parent, "failed")
    assert {
        (cell["case_id"], cell["target_id"]) for cell in proposed["eligible_cells"]
    } == {("q3", "a"), ("q3", "b")}
    assert any(
        cell["reason"] == "ambiguous_dispatch_requires_reconciliation"
        for cell in proposed["excluded"]
    )
    assert any(
        cell["reason"] == "unfinished_or_unknown_accounting_requires_reconciliation"
        for cell in proposed["excluded"]
    )
    dispatched = _transport(monkeypatch)
    engine = Engine(store)
    body = _body(proposed, "explicit-retry", [{"case_id": "q3", "target_id": "b"}])
    with pytest.raises(ValueError, match="acknowledgment"):
        recover(engine, parent, body)
    assert not dispatched
    child = recover(engine, parent, {**body, "acknowledge_new_attempt": True})
    assert _wait(store, child["id"])["status"] == "completed"
    assert dispatched == [("q3", "b")]
    assert child["manifest"]["recovery"]["new_attempt_acknowledged"] is True
    assert len(child["manifest"]["targets"]) == 1


def test_recovery_requires_terminal_parent_and_rejects_forged_lineage(tmp_path):
    store = Store(tmp_path)
    parent = _parent(store)
    store.status(parent, "running")
    with pytest.raises(ValueError, match="terminal"):
        recovery_plan(store, parent)
    store.status(parent, "failed")
    manifest = copy.deepcopy(store.get(parent)["manifest"])
    manifest["recovery"] = {"parent_run_id": parent}
    with pytest.raises(ValueError, match="explicit recovery endpoint"):
        Engine(store).start(manifest)


def test_claims_are_atomic_across_independent_child_starts(tmp_path):
    store = Store(tmp_path)
    parent = _parent(store)
    manifest = copy.deepcopy(store.get(parent)["manifest"])
    manifest["execution_cells"] = [{"case_id": "q2", "target_id": "a"}]
    manifest["targets"] = [manifest["targets"][0]]
    manifest["recovery"] = {"parent_run_id": parent}
    frozen = plan(manifest)
    store.create(frozen, request_key="one")
    before = len(store.list())
    with pytest.raises(ValueError, match="already claimed"):
        store.create(frozen, request_key="two")
    assert len(store.list()) == before


def test_recovery_api_scopes_parent_and_reconciles_idempotent_submission(
    tmp_path, monkeypatch
):
    store = Store(tmp_path)
    parent = _parent(store, "alice")
    dispatched = _transport(monkeypatch)
    service = Server(("127.0.0.1", 0), store, "service-test-token")
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{service.server_port}{PREFIX}/runs/{parent}"
    headers = {
        "Authorization": "Bearer service-test-token",
        "X-SR-Bench-Actor-ID": "alice",
        "X-SR-Bench-Actor-Role": "write",
    }
    try:
        bob = {**headers, "X-SR-Bench-Actor-ID": "bob"}
        viewer = {**headers, "X-SR-Bench-Actor-Role": "read"}
        for actor, expected in [
            (bob, HTTPStatus.NOT_FOUND),
            (viewer, HTTPStatus.FORBIDDEN),
        ]:
            response = requests.post(
                url + "/recover-plan", json={}, headers=actor, timeout=2
            )
            assert response.status_code == expected
        response = requests.post(
            url + "/recover-plan", json={}, headers=headers, timeout=2
        )
        assert response.status_code == HTTPStatus.OK
        body = _body(response.json(), "recover-http-once")
        response = requests.post(
            url + "/recover", json=body, headers=headers, timeout=2
        )
        assert response.status_code == HTTPStatus.CREATED
        child = response.json()
        assert child["owner"] == "alice" and child["progress"]["total"] == 3
        _wait(store, child["id"])
        repeated = requests.post(
            url + "/recover", json=body, headers=headers, timeout=2
        )
        assert repeated.status_code == HTTPStatus.CREATED
        assert repeated.json()["id"] == child["id"] and len(dispatched) == 3
        stale = requests.post(
            url + "/recover",
            json={**body, "idempotency_key": "stale-other-key"},
            headers=headers,
            timeout=2,
        )
        assert stale.status_code == HTTPStatus.BAD_REQUEST
        assert stale.json()["code"] == "recovery_plan_required"
        assert stale.json()["dispatch_started"] is False
        assert len(dispatched) == 3 and len(store.list()) == 2
    finally:
        service.shutdown()
        service.server_close()


def test_existing_run_receipt_does_not_recapture_changed_environment(
    tmp_path, monkeypatch
):
    store = Store(tmp_path)
    parent = _parent(store)
    manifest = store.get(parent)["manifest"]
    _transport(monkeypatch)
    engine = Engine(store)
    monkeypatch.setattr(
        "cli.sr_bench.engine.capture_runner", lambda _: {"observed": True}
    )
    run = engine.start(manifest, request_key="first")
    _wait(store, run["id"])

    def changed_environment(_manifest):
        raise ValueError("Configuration changed after the completed run")

    monkeypatch.setattr("cli.sr_bench.engine.capture_runner", changed_environment)
    assert engine.start(manifest, request_key="first")["id"] == run["id"]
    changed = plan({**manifest, "name": "another frozen protocol"})
    with pytest.raises(ValueError, match="different plan"):
        engine.start(changed, request_key="first")
