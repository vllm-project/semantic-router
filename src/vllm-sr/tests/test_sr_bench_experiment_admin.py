"""Trusted Dashboard administration works across CLI-created experiments."""

import json
import threading

import requests
from cli.sr_bench.contracts import plan
from cli.sr_bench.engine import Engine
from cli.sr_bench.experiments import Experiments
from cli.sr_bench.recovery import recover, recovery_plan
from cli.sr_bench.service import PREFIX, Server
from cli.sr_bench.store import Store
from test_sr_bench_replay import manifest, record


def test_admin_can_continue_cli_experiment_without_reassigning_evidence(
    tmp_path, monkeypatch
):
    store = Store(tmp_path)
    experiments = Experiments(store)
    experiment = experiments.create("CLI experiment", "local")
    baseline = record(store)
    target = {
        **manifest(True)["targets"][0],
        "config_hash": "a" * 64,
        "max_inference_calls": 1,
    }
    (tmp_path / "targets.json").write_text(json.dumps([target]))
    service = Server(("127.0.0.1", 0), store, "fixture-token")
    monkeypatch.setattr(service.engine, "_run", lambda *_: None)
    threading.Thread(target=service.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{service.server_port}{PREFIX}"
    headers = {
        "Authorization": "Bearer fixture-token",
        "X-SR-Bench-Actor-ID": "dashboard-admin",
        "X-SR-Bench-Actor-Role": "admin",
    }
    try:
        response = requests.post(
            url + f'/runs/{baseline["id"]}/candidate-plan',
            headers=headers,
            json={
                "target_ids": [target["id"]],
                "mode": "preview",
                "experiment": {"id": experiment["id"], "role": "preview"},
            },
            timeout=2,
        )
        assert response.status_code == 200, response.text
        body = {"manifest": response.json()["manifest"], "idempotency_key": "candidate"}
        # An untrusted manifest field cannot elevate a writer's experiment access.
        writer = {**headers, "X-SR-Bench-Actor-Role": "write"}
        forged = {**body, "manifest": {**body["manifest"], "actor_role": "admin"}}
        rejected = requests.post(url + "/runs", headers=writer, json=forged, timeout=2)
        assert rejected.status_code in {400, 403}, rejected.text
        rejected = requests.post(url + "/runs", headers=writer, json=body, timeout=2)
        assert rejected.status_code == 403, rejected.text
        assert len(store.list()) == 1
        assert experiments.get(experiment["id"], "local")["run_count"] == 0
        created = requests.post(url + "/runs", headers=headers, json=body, timeout=2)
        assert created.status_code == 201, created.text
        run = created.json()
        assert run["owner"] == "dashboard-admin"
        assert store.get(baseline["id"]) == baseline
        assert (
            experiments.runs(experiment["id"], "local")["members"][0]["run_id"]
            == run["id"]
        )
        same = requests.post(url + "/runs", headers=headers, json=body, timeout=2)
        assert same.status_code == 201 and same.json()["id"] == run["id"]
        assert len(store.calls(run["id"])) == 0
    finally:
        service.shutdown()
        service.server_close()


def test_recovery_of_mixed_baseline_retains_experiment_without_claiming_baseline(
    tmp_path, monkeypatch
):
    store = Store(tmp_path)
    experiments = Experiments(store)
    experiment = experiments.create("Mixed baseline", "local")
    document = manifest()
    document["targets"].append(
        {
            **manifest(True)["targets"][0],
            "config_hash": "a" * 64,
            "max_inference_calls": 1,
        }
    )
    document["experiment"] = {"id": experiment["id"], "role": "baseline"}
    parent, _ = store.create(plan(document))
    store.status(parent["id"], "cancelled")
    parent = store.get(parent["id"])
    proposal = recovery_plan(store, parent["id"])
    chosen = [
        cell for cell in proposal["eligible_cells"] if cell["target_id"] == "balance"
    ]
    engine = Engine(store)
    monkeypatch.setattr(engine, "_run", lambda *_: None)
    body = {
        "mode": "undispatched",
        "plan_sha256": proposal["plan_sha256"],
        "cells": chosen,
        "idempotency_key": "recover-mom-only",
    }
    child = recover(engine, parent["id"], body, "dashboard-admin", actor_role="admin")
    assert child["manifest"]["experiment"]["role"] == "recovery"
    assert child["manifest"]["execution_cells"] == chosen
    assert len(child["manifest"]["targets"]) == 1
    assert [link["role"] for link in experiments.runs(experiment["id"])["members"]] == [
        "baseline",
        "recovery",
    ]
    assert store.get(parent["id"]) == parent
    assert store.calls(child["id"]) == []
    assert (
        recover(engine, parent["id"], body, "dashboard-admin", actor_role="admin")["id"]
        == child["id"]
    )
