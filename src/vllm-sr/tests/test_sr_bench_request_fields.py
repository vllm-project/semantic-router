"""Every sr-bench write route rejects the request fields it does not consume."""

import threading

import pytest
import requests
from cli.sr_bench.contracts import plan
from cli.sr_bench.service import PREFIX, Server
from cli.sr_bench.store import Store


def manifest():
    return {
        "version": "sr-bench-1.0",
        "mode": "live",
        "targets": [
            {
                "id": "single",
                "kind": "single",
                "model": "model",
                "base_url": "http://127.0.0.1:1/v1",
                "prices": {
                    "model": {
                        "input": 1,
                        "cached_input": 2,
                        "cache_write": 3,
                        "output": 4,
                    }
                },
            }
        ],
        "cases": [
            {
                "id": "one",
                "benchmark": "mmlu-pro",
                "messages": [{"role": "user", "content": "private-question-one"}],
                "answer": "A",
            }
        ],
    }


# route, consumed body, one field the route does not consume, expected error
CASES = [
    (
        "/experiments",
        {"name": "study", "idempotency_key": "once"},
        "naame",
        "Unsupported request fields",
    ),
    (
        "/experiments/study/runs",
        {"run_id": "run-1", "role": "candidate"},
        "roles",
        "Unsupported request fields",
    ),
    (
        "/datasets/compose",
        {"dataset_ids": ["set-one"], "benchmarks": ["mmlu-pro"]},
        "benchmark",
        "Unsupported dataset compose fields",
    ),
    ("/plans", {"manifest": {}}, "documents", "Unsupported plan request fields"),
    ("/runs", {"manifest": {}}, "idempotency_keey", "Unsupported request fields"),
    (
        "/replays",
        {"baseline_run_id": "run-1", "preview_run_id": "run-2"},
        "idempotency_keey",
        "Unsupported request fields",
    ),
    (
        "/comparisons",
        {"baseline_run_id": "run-1", "candidate_run_id": "run-2"},
        "candidate_run_idd",
        "Unsupported request fields",
    ),
    (
        "/runs/{run}/candidate-plan",
        {"target_ids": ["single"]},
        "targot_ids",
        "Unsupported candidate plan fields",
    ),
    (
        "/runs/{run}/recover-plan",
        {"mode": "failed"},
        "modes",
        "Unsupported request fields",
    ),
    (
        "/runs/{run}/recover",
        {"cells": [{"case_id": "one", "target_id": "single"}]},
        "cell",
        "Unsupported request fields",
    ),
    ("/runs/{run}/reconcile-usage", {}, "dry_run", "Unsupported request fields"),
    ("/runs/{run}/regrade", {}, "outputs", "Unsupported request fields"),
    ("/runs/{run}/export", {}, "formats", "Unsupported request fields"),
    ("/runs/{run}/cancel", {}, "force_cancel", "Unsupported request fields"),
]


@pytest.mark.parametrize("case", CASES)
def test_write_routes_reject_unconsumed_request_fields(tmp_path, case):
    route, body, unsupported, message = case
    store = Store(tmp_path)
    run, created = store.create(plan(manifest()), "alice", None)
    assert created
    service = Server(("127.0.0.1", 0), store, "fixture-token")
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{service.server_port}{PREFIX}{route.format(run=run['id'])}"
    headers = {
        "Authorization": "Bearer fixture-token",
        "X-SR-Bench-Actor-ID": "alice",
        "X-SR-Bench-Actor-Role": "write",
    }
    try:
        before = store.db.total_changes
        response = requests.post(
            url, headers=headers, json={**body, unsupported: "value"}, timeout=2
        )
        assert response.status_code == 400, (route, response.status_code)
        assert response.json()["error"] == message
        assert store.db.total_changes == before
    finally:
        service.shutdown()
        service.server_close()
        thread.join(timeout=1)
