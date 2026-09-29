"""Shutdown fences new attempts while preserving durable reconciliation."""

import concurrent.futures
import copy
import threading
from http import HTTPStatus

import pytest
import requests
from cli.sr_bench.contracts import plan
from cli.sr_bench.engine import Engine, EngineClosedError
from cli.sr_bench.recovery import recovery_plan
from cli.sr_bench.service import PREFIX, Server
from cli.sr_bench.store import Store
from test_sr_bench_replay import manifest, record


def _manifest():
    return {
        "version": "sr-bench-1.0",
        "targets": [
            {
                "id": "single",
                "kind": "single",
                "model": "model",
                "base_url": "http://127.0.0.1:1/v1",
                "prices": {
                    "model": {
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
                "id": "question",
                "benchmark": "mmlu-pro",
                "messages": [{"role": "user", "content": "Return A"}],
                "answer": "A",
            }
        ],
    }


def _pause_capture(monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def capture(_manifest):
        entered.set()
        assert release.wait(5), "test did not release provenance capture"
        return {}

    monkeypatch.setattr("cli.sr_bench.engine.capture_runner", capture)
    return entered, release


def test_close_rejects_preparing_start_without_waiting_for_capture(
    tmp_path, monkeypatch
):
    engine = Engine(Store(tmp_path))
    entered, release = _pause_capture(monkeypatch)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        pending = executor.submit(engine.start, _manifest())
        try:
            assert entered.wait(5)
            executor.submit(engine.close).result(timeout=5)
        finally:
            release.set()
        with pytest.raises(EngineClosedError):
            pending.result(timeout=5)
    assert engine.store.list() == []
    assert engine.cancels == {} and engine.threads == {}
    assert not (tmp_path / "runs").exists()


def test_close_cannot_miss_persisted_run_awaiting_registration(tmp_path, monkeypatch):
    engine = Engine(Store(tmp_path))
    persisted, release_create = threading.Event(), threading.Event()
    closing, release_worker = threading.Event(), threading.Event()
    observed = []
    create = engine.store.create
    admission = engine._admission

    class ObservedAdmission:
        def __enter__(self):
            if threading.current_thread().name.startswith("shutdown"):
                closing.set()
            admission.acquire()

        def __exit__(self, *_exc):
            admission.release()

    def paused_create(*args, **kwargs):
        result = create(*args, **kwargs)
        persisted.set()
        assert release_create.wait(5), "test did not release durable creation"
        return result

    def worker(_run_id, _frozen, cancel):
        assert release_worker.wait(5), "test did not release registered worker"
        observed.append(cancel.is_set())

    monkeypatch.setattr(engine, "_admission", ObservedAdmission())
    monkeypatch.setattr(engine.store, "create", paused_create)
    monkeypatch.setattr(engine, "_run", worker)
    monkeypatch.setattr("cli.sr_bench.engine.capture_runner", lambda _: {})
    with (
        concurrent.futures.ThreadPoolExecutor() as starts,
        concurrent.futures.ThreadPoolExecutor(thread_name_prefix="shutdown") as stops,
    ):
        pending = starts.submit(engine.start, _manifest())
        try:
            assert persisted.wait(5)
            stopping = stops.submit(engine.close)
            assert closing.wait(5)
            assert not stopping.done()
            release_create.set()
            run = pending.result(timeout=5)
            stopping.result(timeout=5)
        finally:
            release_create.set()
            release_worker.set()
    engine.threads[run["id"]].join(timeout=5)
    assert not engine.threads[run["id"]].is_alive()
    assert observed == [True]
    assert engine.cancels[run["id"]].is_set()
    assert engine.store.calls(run["id"]) == []
    engine.close()  # Repeated shutdown keeps admission closed.
    with pytest.raises(EngineClosedError):
        engine.start(_manifest())
    assert len(engine.store.list()) == 1


def test_close_preserves_existing_idempotency_and_rejects_new_keys(
    tmp_path, monkeypatch
):
    engine = Engine(Store(tmp_path))
    manifest = _manifest()
    existing, _ = engine.store.create(plan(manifest), request_key="accepted")
    engine.store.status(existing["id"], "completed")
    engine.close()

    def unexpected_capture(_manifest):
        pytest.fail("a closed engine must not capture a new runner")

    monkeypatch.setattr("cli.sr_bench.engine.capture_runner", unexpected_capture)
    assert engine.start(manifest, request_key="accepted")["id"] == existing["id"]
    with pytest.raises(ValueError, match="idempotency key"):
        engine.start({**manifest, "name": "changed"}, request_key="accepted")
    with pytest.raises(EngineClosedError):
        engine.start(manifest, request_key="late")
    assert len(engine.store.list()) == 1
    assert engine.threads == {}


def test_concurrent_idempotent_starts_register_only_one_worker(tmp_path, monkeypatch):
    engine = Engine(Store(tmp_path))
    capture_barrier = threading.Barrier(2)
    dispatched = []

    def capture(_manifest):
        capture_barrier.wait(timeout=5)
        return {}

    monkeypatch.setattr("cli.sr_bench.engine.capture_runner", capture)
    monkeypatch.setattr(engine, "_run", lambda run_id, *_: dispatched.append(run_id))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(engine.start, _manifest(), request_key="same")
            for _ in range(2)
        ]
        runs = [future.result(timeout=5) for future in futures]
    engine.close()
    for worker in engine.threads.values():
        worker.join(timeout=5)
    assert runs[0]["id"] == runs[1]["id"]
    assert dispatched == [runs[0]["id"]]
    assert len(engine.store.list()) == 1


def _recovery_request(store):
    manifest = _manifest()
    manifest["cases"].append({**manifest["cases"][0], "id": "unknown"})
    parent, _ = store.create(plan(manifest))
    identity = parent["id"]
    call = store.start_call(identity, "unknown", "single", "subject", {})
    store.finish_call(call, "sent_unknown", {})
    store.status(identity, "failed", "Saved interrupted attempt")
    proposed = recovery_plan(store, identity)
    assert proposed["eligible_cells"] == [
        {"case_id": "question", "target_id": "single"}
    ]
    return identity, {
        "mode": proposed["mode"],
        "plan_sha256": proposed["plan_sha256"],
        "cells": proposed["eligible_cells"],
        "idempotency_key": "recovery-once",
    }


@pytest.mark.parametrize("recovery", [False, True], ids=["new", "recovery"])
def test_http_shutdown_rejects_already_accepted_preparing_handler(
    tmp_path, monkeypatch, recovery
):
    store = Store(tmp_path)
    server = Server(("127.0.0.1", 0), store)
    path, body = "/runs", {"manifest": _manifest(), "idempotency_key": "new-once"}
    parent = None
    if recovery:
        parent, body = _recovery_request(store)
        path = f"/runs/{parent}/recover"
    before = copy.deepcopy(store.list())
    calls = store.calls(parent) if parent else []
    entered, release = _pause_capture(monkeypatch)
    loop = threading.Thread(target=server.serve_forever, daemon=True)
    loop.start()
    clients = concurrent.futures.ThreadPoolExecutor()
    try:
        pending = clients.submit(
            requests.post,
            f"http://127.0.0.1:{server.server_port}{PREFIX}{path}",
            json=body,
            timeout=5,
        )
        assert entered.wait(5)
        server.shutdown()
        release.set()
        response = pending.result(timeout=5)
        assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
        assert response.json() == {
            "error": "Evaluation service is shutting down",
            "code": "service_stopping",
            "dispatch_started": False,
            "model_requests": 0,
        }
        assert store.list() == before
        assert server.engine.threads == {}
        if parent:
            assert store.calls(parent) == calls
            assert store.recovery_claims(parent) == {}
            assert recovery_plan(store, parent)["counts"] == {
                "eligible": 1,
                "excluded": 1,
            }
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        loop.join(timeout=5)
        clients.shutdown()


def test_closed_http_engine_reconciles_existing_recovery_and_preserves_unknowns(
    tmp_path, monkeypatch
):
    store = Store(tmp_path)
    server = Server(("127.0.0.1", 0), store)
    parent, body = _recovery_request(store)
    registered = []

    def worker(run_id, *_args):
        registered.append(run_id)
        store.status(run_id, "completed")

    monkeypatch.setattr(server.engine, "_run", worker)
    monkeypatch.setattr("cli.sr_bench.engine.capture_runner", lambda _: {})
    loop = threading.Thread(target=server.serve_forever, daemon=True)
    loop.start()
    url = f"http://127.0.0.1:{server.server_port}{PREFIX}/runs/{parent}/recover"
    try:
        first = requests.post(url, json=body, timeout=5)
        assert first.status_code == HTTPStatus.CREATED
        child = first.json()["id"]
        server.engine.threads[child].join(timeout=5)
        server.engine.close()
        again = requests.post(url, json=body, timeout=5)
        assert again.status_code == HTTPStatus.CREATED
        assert again.json()["id"] == child
        assert registered == [child]
        proposed = recovery_plan(store, parent)
        ambiguous = requests.post(
            url,
            json={
                **body,
                "idempotency_key": "ambiguous",
                "plan_sha256": proposed["plan_sha256"],
                "cells": [{"case_id": "unknown", "target_id": "single"}],
            },
            timeout=5,
        )
        assert ambiguous.status_code == HTTPStatus.BAD_REQUEST
        assert ambiguous.json()["code"] == "recovery_plan_required"
        assert len(store.list()) == 2
        assert store.calls(parent)[0]["status"] == "sent_unknown"
    finally:
        server.shutdown()
        server.server_close()
        loop.join(timeout=5)


def test_closed_http_engine_rejects_a_new_replay_and_reconciles_a_known_one(tmp_path):
    store = Store(tmp_path)
    baseline = record(store)
    document = manifest(True)
    document["cases"].reverse()
    document["targets"][0]["capture_recipe"] = True
    preview = record(store, document, preview=True)
    server = Server(("127.0.0.1", 0), store)
    loop = threading.Thread(target=server.serve_forever, daemon=True)
    loop.start()
    url = f"http://127.0.0.1:{server.server_port}{PREFIX}/replays"
    body = {"baseline_run_id": baseline["id"], "preview_run_id": preview["id"]}
    once = {"idempotency_key": "replay-once"}
    try:
        replayed = requests.post(url, json={**body, **once}, timeout=5)
        assert replayed.status_code == HTTPStatus.CREATED
        server.engine.close()
        before = store.list()
        known = requests.post(url, json={**body, **once}, timeout=5)
        assert known.status_code == HTTPStatus.CREATED
        assert known.json()["id"] == replayed.json()["id"]
        for request in (body, {**body, "idempotency_key": "new-after-close"}):
            late = requests.post(url, json=request, timeout=5)
            assert late.status_code == HTTPStatus.SERVICE_UNAVAILABLE
            assert late.json()["code"] == "service_stopping"
        assert store.list() == before
    finally:
        server.shutdown()
        server.server_close()
        loop.join(timeout=5)
