"""Deleting organization preserves evidence and cannot resurrect a creation key."""

import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests
from cli.commands.benchmark import benchmark
from cli.sr_bench.contracts import plan
from cli.sr_bench.engine import Engine
from cli.sr_bench.experiments import (
    ActiveExperimentError,
    ExperimentDeletedError,
    Experiments,
)
from cli.sr_bench.offline import replay
from cli.sr_bench.recovery import recover, recovery_plan
from cli.sr_bench.service import PREFIX, Server
from cli.sr_bench.store import Store
from click.testing import CliRunner
from test_sr_bench_replay import evidence, manifest, record


def _saved_rows(store):
    tables = (
        "runs",
        "calls",
        "results",
        "events",
        "run_provenance",
        "accounting_corrections",
        "recovery_claims",
    )
    return {
        table: store.db.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
        for table in tables
    }


def test_delete_is_durable_idempotent_and_preserves_all_evidence(tmp_path):
    store = Store(tmp_path)
    experiments = Experiments(store)
    experiment = experiments.create("Saved study", "alice", "create-once")
    document = manifest()
    document["experiment"] = {"id": experiment["id"], "role": "baseline"}
    run = record(store, document, owner="alice")
    other = experiments.create("Other study", "alice")
    experiments.attach(other["id"], run["id"], "baseline", owner="alice")
    artifact = tmp_path / "report.json"
    artifact.write_bytes(b'{"receipt":"original evidence"}\n')
    original_bytes = artifact.read_bytes()
    rows = _saved_rows(store)
    other_before = experiments.runs(other["id"], "alice")

    receipt = experiments.delete(experiment["id"], "alice")
    assert receipt == {
        "id": experiment["id"],
        "deleted": True,
        "unlinked_runs": 1,
        "deleted_at": receipt["deleted_at"],
        "runs_deleted": 0,
        "model_requests": 0,
    }
    assert _saved_rows(store) == rows
    assert artifact.read_bytes() == original_bytes
    assert experiments.runs(other["id"], "alice") == other_before
    assert store.get(run["id"])["manifest"]["experiment"]["id"] == experiment["id"]
    assert [item["id"] for item in experiments.list("alice")["experiments"]] == [
        other["id"]
    ]
    with pytest.raises(KeyError):
        experiments.get(experiment["id"], "alice")
    with pytest.raises(KeyError):
        experiments.delete(experiment["id"], "bob")
    changes = store.db.total_changes
    assert experiments.delete(experiment["id"], "alice") == receipt
    assert experiments.delete(experiment["id"]) == receipt
    assert store.db.total_changes == changes
    store.db.close()

    reopened = Store(tmp_path)
    experiments = Experiments(reopened)
    assert experiments.delete(experiment["id"], "alice") == receipt
    assert _saved_rows(reopened) == rows
    with pytest.raises(ExperimentDeletedError) as error:
        experiments.create("Saved study", "alice", "create-once")
    assert error.value.identifier == experiment["id"]
    assert (
        experiments.create("Saved study", "alice", "new-key")["id"] != experiment["id"]
    )
    assert (
        experiments.create("Saved study", "bob", "create-once")["id"]
        != experiment["id"]
    )


@pytest.mark.parametrize("status", ["queued", "running", "cancelling", "unrecognized"])
def test_active_linked_runs_block_deletion_without_any_write(tmp_path, status):
    store = Store(tmp_path)
    experiments = Experiments(store)
    experiment = experiments.create("Active study", "alice")
    run = record(store, owner="alice")
    experiments.attach(experiment["id"], run["id"], "baseline", owner="alice")
    store.status(run["id"], status)
    before = list(store.db.iterdump())
    assert experiments.get(experiment["id"], "alice")["active_run_count"] == 1
    with pytest.raises(ActiveExperimentError) as error:
        experiments.delete(experiment["id"], "alice")
    assert error.value.active_run_count == 1
    assert list(store.db.iterdump()) == before
    store.status(run["id"], "cancelled")
    assert experiments.delete(experiment["id"], "alice")["unlinked_runs"] == 1


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled", "interrupted"])
def test_all_terminal_statuses_can_be_unlinked(tmp_path, status):
    store = Store(tmp_path)
    experiments = Experiments(store)
    experiment = experiments.create("Terminal study", "alice")
    run = record(store, owner="alice")
    experiments.attach(experiment["id"], run["id"], "baseline", owner="alice")
    store.status(run["id"], status)
    rows = _saved_rows(store)
    assert experiments.get(experiment["id"], "alice")["active_run_count"] == 0
    assert experiments.delete(experiment["id"], "alice")["unlinked_runs"] == 1
    assert _saved_rows(store) == rows


def test_deletion_receipt_and_unlinks_roll_back_together(tmp_path):
    store = Store(tmp_path)
    experiments = Experiments(store)
    experiment = experiments.create("Atomic study", "alice", "once")
    run = record(store, owner="alice")
    experiments.attach(experiment["id"], run["id"], "baseline", owner="alice")
    store.db.execute(
        "CREATE TRIGGER fixture_reject_delete BEFORE DELETE ON experiments "
        "BEGIN SELECT RAISE(ABORT,'fixture failure'); END"
    )
    before = list(store.db.iterdump())
    with pytest.raises(sqlite3.IntegrityError, match="fixture failure"):
        experiments.delete(experiment["id"], "alice")
    assert list(store.db.iterdump()) == before


def test_delete_serializes_with_new_run_on_another_connection(tmp_path, monkeypatch):
    store = Store(tmp_path)
    other = Store(tmp_path)
    experiments = Experiments(store)
    experiment = experiments.create("Concurrent study", "alice")
    document = manifest()
    document["experiment"] = {"id": experiment["id"], "role": "baseline"}
    frozen = plan(document)
    checked, resume, inserting = threading.Event(), threading.Event(), threading.Event()
    original_get = experiments.get

    def pause_after_active_check(*args):
        value = original_get(*args)
        checked.set()
        assert resume.wait(3)
        return value

    def trace(statement):
        if statement.startswith("INSERT INTO runs"):
            inserting.set()

    monkeypatch.setattr(experiments, "get", pause_after_active_check)
    other.db.set_trace_callback(trace)
    with ThreadPoolExecutor(max_workers=2) as pool:
        deleting = pool.submit(experiments.delete, experiment["id"], "alice")
        assert checked.wait(3)
        creating = pool.submit(other.create, frozen, "alice")
        assert inserting.wait(3)
        resume.set()
        assert deleting.result(timeout=3)["deleted"]
        with pytest.raises(PermissionError, match="cannot manage"):
            creating.result(timeout=3)
    assert store.list() == []
    assert store.db.execute("SELECT * FROM experiment_runs").fetchall() == []


def test_derived_attempts_keep_historical_reference_but_do_not_recreate_group(
    tmp_path, monkeypatch
):
    store = Store(tmp_path)
    experiments = Experiments(store)
    experiment = experiments.create("Derived study", "local", "study-once")
    baseline = record(store)
    preview_document = manifest(True)
    preview_document["experiment"] = {"id": experiment["id"], "role": "preview"}
    preview = record(store, preview_document, preview=True)
    parent_document = manifest()
    parent_document["experiment"] = {"id": experiment["id"], "role": "baseline"}
    parent, _ = store.create(plan(parent_document))
    store.status(parent["id"], "cancelled")
    sources = [evidence(store, run["id"]) for run in (baseline, preview, parent)]
    experiments.delete(experiment["id"], "local")

    estimate = replay(store, baseline["id"], preview["id"], request_key="estimate-once")
    assert "experiment" not in estimate["manifest"]
    proposal = recovery_plan(store, parent["id"])
    engine = Engine(store)
    monkeypatch.setattr(engine, "_run", lambda *_: None)
    body = {
        "mode": "undispatched",
        "plan_sha256": proposal["plan_sha256"],
        "cells": proposal["eligible_cells"],
        "idempotency_key": "recovery-once",
    }
    child = recover(engine, parent["id"], body)
    assert "experiment" not in child["manifest"]
    assert store.calls(child["id"]) == []
    assert [
        evidence(store, run["id"]) for run in (baseline, preview, parent)
    ] == sources
    assert experiments.list()["experiments"] == []
    assert store.db.execute("SELECT * FROM experiment_runs").fetchall() == []
    assert recover(engine, parent["id"], body)["id"] == child["id"]
    assert (
        replay(store, baseline["id"], preview["id"], request_key="estimate-once")["id"]
        == estimate["id"]
    )


def test_delete_http_and_cli_authorization_and_conflict_contract(tmp_path, monkeypatch):
    store = Store(tmp_path)
    experiments = Experiments(store)
    created = experiments.create("Service study", "alice", "once")
    run = record(store, owner="alice")
    experiments.attach(created["id"], run["id"], "baseline", owner="alice")
    service = Server(("127.0.0.1", 0), store, "fixture-token")
    monkeypatch.setattr(service.engine, "start", lambda *_: pytest.fail("No dispatch"))
    threading.Thread(target=service.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{service.server_port}"
    path = f"{PREFIX}/experiments/{created['id']}"
    headers = {
        "Authorization": "Bearer fixture-token",
        "X-SR-Bench-Actor-ID": "alice",
        "X-SR-Bench-Actor-Role": "write",
    }
    try:
        rows = _saved_rows(store)
        for overrides, expected in (
            ({"Authorization": "Bearer invalid"}, 403),
            ({"X-SR-Bench-Actor-Role": "read"}, 403),
            ({"X-SR-Bench-Actor-ID": "bob"}, 404),
        ):
            response = requests.delete(
                url + path, headers={**headers, **overrides}, timeout=2
            )
            assert response.status_code == expected
        for suffix, body in (("?cascade=true", None), ("", {})):
            response = requests.delete(
                url + path + suffix, headers=headers, json=body, timeout=2
            )
            assert response.status_code == 400
        store.status(run["id"], "running")
        blocked = requests.delete(url + path, headers=headers, timeout=2)
        assert blocked.status_code == 409
        assert blocked.json()["code"] == "experiment_active_runs"
        assert blocked.json()["active_run_count"] == 1
        store.status(run["id"], "completed")
        rows = _saved_rows(store)
        response = requests.delete(url + path, headers=headers, timeout=2)
        assert response.status_code == 200
        receipt = response.json()
        assert requests.delete(url + path, headers=headers, timeout=2).json() == receipt
        admin = {
            **headers,
            "X-SR-Bench-Actor-ID": "admin",
            "X-SR-Bench-Actor-Role": "admin",
        }
        assert requests.delete(url + path, headers=admin, timeout=2).json() == receipt
        foreign = experiments.create("Administrator-managed study", "bob")
        removed = requests.delete(
            url + PREFIX + f"/experiments/{foreign['id']}", headers=admin, timeout=2
        )
        assert removed.status_code == 200 and removed.json()["deleted"]
        assert requests.get(url + path, headers=headers, timeout=2).status_code == 404
        rejected = requests.post(
            url + PREFIX + "/experiments",
            headers=headers,
            json={"name": "Service study", "idempotency_key": "once"},
            timeout=2,
        )
        assert (
            rejected.status_code == 409
            and rejected.json()["code"] == "experiment_deleted"
        )
        assert _saved_rows(store) == rows

        # The real CLI uses the same authenticated HTTP DELETE, without a request body.
        local = experiments.create("CLI study", "local")
        monkeypatch.setenv("SR_BENCH_TOKEN", "fixture-token")
        result = CliRunner().invoke(
            benchmark,
            ["--url", url, "--no-autostart", "experiment", "delete", local["id"]],
        )
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["deleted"] is True
    finally:
        service.shutdown()
        service.server_close()


def test_cli_rejects_noncanonical_experiment_ids_before_request(monkeypatch):
    monkeypatch.setattr(
        "cli.sr_bench.client.Client.request",
        lambda *_: pytest.fail("Invalid ID contacted service"),
    )
    result = CliRunner().invoke(
        benchmark,
        [
            "--url",
            "http://127.0.0.1:1",
            "--no-autostart",
            "experiment",
            "delete",
            "exp-../../runs",
        ],
    )
    assert result.exit_code != 0
    assert "Invalid experiment identity" in result.output
