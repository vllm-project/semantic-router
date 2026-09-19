"""Durable experiment links never rewrite or dispatch saved evidence."""

import copy
import hashlib
import json
import threading

import pytest
import requests
from cli.sr_bench.candidate_plans import candidate_manifest, validate_candidate_protocol
from cli.sr_bench.contracts import plan
from cli.sr_bench.experiments import Experiments
from cli.sr_bench.service import PREFIX, Server
from cli.sr_bench.sources import _write_dataset
from cli.sr_bench.store import Store
from cli.sr_bench.target_contracts import (
    effective_auxiliary_targets,
    resolve_auxiliary_target,
)
from test_sr_bench_replay import manifest, record


def test_experiment_links_persist_paginate_and_preserve_runs(tmp_path):
    store = Store(tmp_path)
    experiments = Experiments(store)
    created = experiments.create("Recipe search", "alice", "once")
    assert experiments.create("Recipe search", "alice", "once") == created
    with pytest.raises(ValueError, match="another name"):
        experiments.create("Changed", "alice", "once")
    originals = []
    for index in range(3):
        run = record(store, owner="alice")
        originals.append(copy.deepcopy(run))
        experiments.attach(
            created["id"], run["id"], "baseline", f"Baseline {index}", "alice"
        )
        experiments.attach(
            created["id"], run["id"], "baseline", f"Baseline {index}", "alice"
        )
    first = experiments.runs(created["id"], "alice", limit=2)
    assert len(first["members"]) == 2 and first["has_more"]
    second = experiments.runs(
        created["id"], "alice", after=first["next_cursor"], limit=2
    )
    assert len(second["members"]) == 1 and not second["has_more"]
    assert [store.get(r["id"], "alice") for r in originals] == originals
    assert "messages" not in json.dumps(first)
    reopened = Store(tmp_path)
    assert Experiments(reopened).get(created["id"], "alice")["run_count"] == 3
    with pytest.raises(KeyError):
        experiments.get(created["id"], "bob")
    foreign = record(store, owner="bob")
    with pytest.raises(KeyError):
        experiments.attach(created["id"], foreign["id"], "baseline", owner="alice")
    experiments.attach(created["id"], foreign["id"], "baseline")
    assert experiments.get(created["id"], "alice")["run_count"] == 4
    with pytest.raises(ValueError, match="different role"):
        experiments.attach(
            created["id"], originals[0]["id"], "baseline", "Changed", "alice"
        )


def test_experiment_binding_is_atomic_and_roles_are_validated(tmp_path):
    store = Store(tmp_path)
    experiments = Experiments(store)
    experiment = experiments.create("One", "alice")
    document = manifest(True)
    document["experiment"] = {
        "id": experiment["id"],
        "role": "preview",
        "hypothesis": "Try a cheaper route",
    }
    frozen = plan(document)
    run, _ = store.create(frozen, "alice")
    assert (
        experiments.runs(experiment["id"], "alice")["members"][0]["run_id"] == run["id"]
    )
    count = len(store.list())
    with pytest.raises(PermissionError):
        store.create(frozen, "bob")
    document["experiment"]["role"] = "baseline"
    with pytest.raises(ValueError, match="role"):
        store.create(plan(document), "alice")
    assert len(store.list()) == count
    with pytest.raises(ValueError, match="identity"):
        plan({**document, "experiment": {"id": "exp-../../x", "role": "baseline"}})


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled", "interrupted"])
def test_candidate_inherits_exact_protocol_without_mutating_baseline(tmp_path, status):
    store = Store(tmp_path)
    baseline = record(store)
    store.status(baseline["id"], status)
    baseline = store.get(baseline["id"])
    original = copy.deepcopy(baseline)
    target = {
        **manifest(True)["targets"][0],
        "config_hash": "a" * 64,
        "max_inference_calls": 1,
    }
    candidate = plan(candidate_manifest(baseline, [target]))
    for key in ("cases", "sampling", "limits", "profile", "cost_policy", "case_sha256"):
        assert candidate[key] == baseline["manifest"][key]
    assert candidate["baseline_run_id"] == baseline["id"]
    assert baseline == original
    with pytest.raises(ValueError, match="MoM"):
        candidate_manifest(baseline, manifest()["targets"])
    with pytest.raises(ValueError, match="recovery subset"):
        candidate_manifest(
            {**baseline, "manifest": {**baseline["manifest"], "execution_cells": []}},
            [target],
        )
    with pytest.raises(ValueError, match="terminal"):
        candidate_manifest({**baseline, "status": "running"}, [target])
    with pytest.raises(ValueError, match="recovery subset"):
        candidate_manifest(
            {
                **baseline,
                "manifest": {
                    **baseline["manifest"],
                    "recovery": {"parent_run_id": "parent"},
                },
            },
            [target],
        )


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_experiment_and_candidate_service_authorization_and_no_dispatch(
    tmp_path, monkeypatch, status
):
    store = Store(tmp_path)
    baseline = record(store, owner="alice")
    store.status(baseline["id"], status)
    target = {
        **manifest(True)["targets"][0],
        "config_hash": "a" * 64,
        "max_inference_calls": 1,
    }
    (tmp_path / "targets.json").write_text(json.dumps([target]))
    service = Server(("127.0.0.1", 0), store, "fixture-token")
    monkeypatch.setattr(
        service.engine,
        "start",
        lambda *_: pytest.fail("Plan and organization cannot dispatch"),
    )
    threading.Thread(target=service.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{service.server_port}{PREFIX}"
    headers = {
        "Authorization": "Bearer fixture-token",
        "X-SR-Bench-Actor-ID": "alice",
        "X-SR-Bench-Actor-Role": "write",
    }
    try:
        response = requests.post(
            url + "/experiments",
            headers=headers,
            json={"name": "Recipe search", "idempotency_key": "once"},
            timeout=2,
        )
        assert response.status_code == 201
        experiment = response.json()
        candidate = requests.post(
            url + f'/runs/{baseline["id"]}/candidate-plan',
            headers=headers,
            json={"target_ids": [target["id"]]},
            timeout=2,
        )
        assert candidate.status_code == 200, candidate.text
        assert candidate.json()["model_requests"] == 0
        assert (
            candidate.json()["manifest"]["case_sha256"]
            == baseline["manifest"]["case_sha256"]
        )
        linked = requests.post(
            url + f'/experiments/{experiment["id"]}/runs',
            headers=headers,
            json={"run_id": baseline["id"], "role": "baseline"},
            timeout=2,
        )
        assert linked.status_code == 200
        viewer = {**headers, "X-SR-Bench-Actor-Role": "read"}
        assert (
            requests.post(
                url + "/experiments", headers=viewer, json={"name": "No"}, timeout=2
            ).status_code
            == 403
        )
        assert (
            requests.get(
                url + "/experiments?limit=0", headers=headers, timeout=2
            ).status_code
            == 400
        )
        foreign = {**headers, "X-SR-Bench-Actor-ID": "bob"}
        assert (
            requests.get(
                url + f'/experiments/{experiment["id"]}', headers=foreign, timeout=2
            ).status_code
            == 404
        )
        assert (
            requests.get(url + "/experiments", headers=foreign, timeout=2).json()[
                "experiments"
            ]
            == []
        )
        assert len(store.list()) == 1
    finally:
        service.shutdown()
        service.server_close()


def test_candidate_rejects_operator_protocol_drift_before_dispatch(tmp_path):
    store = Store(tmp_path)
    baseline = record(store)
    target = {
        **manifest(True)["targets"][0],
        "config_hash": "a" * 64,
        "max_inference_calls": 1,
    }
    frozen = plan(candidate_manifest(baseline, [target]))
    validate_candidate_protocol(baseline, frozen)
    frozen["benchmark_options"] = {"mmlu-pro": {"changed": True}}
    with pytest.raises(ValueError, match="Baseline protocol changed"):
        validate_candidate_protocol(baseline, frozen)
    assert len(store.list()) == 1


def auxiliary_baseline(placement):
    document = manifest()
    prototype = document["targets"][0]
    auxiliary = {
        role: {
            **copy.deepcopy(prototype),
            "id": role,
            "model": role,
            "prices": {role: copy.deepcopy(prototype["prices"]["model"])},
            "api_key_env": f"FIXTURE_{role.upper()}_TOKEN",
            "request_params": {"temperature": 0.2, "n": 1},
        }
        for role in ("judge", "simulator")
    }
    for case in document["cases"]:
        case["benchmark"] = "simpleqa-verified"
    document["benchmark_options"] = {
        "simpleqa-verified": {
            "judge": "judge",
            "grader_version": "sr-bench-reference-judge-v1",
        },
        "tau3": {"judge": "judge", "simulator": "simulator", "release": "1.0.1"},
    }
    if placement == "subjects":
        document["targets"].extend(auxiliary.values())
    else:
        document["auxiliary_targets"] = auxiliary
    return document


def test_candidate_keeps_identical_effective_auxiliaries_from_either_placement():
    target = {
        **manifest(True)["targets"][0],
        "config_hash": "a" * 64,
        "max_inference_calls": 1,
    }
    candidates = []
    for placement in ("subjects", "auxiliary"):
        source = plan(auxiliary_baseline(placement))
        baseline = {"id": "baseline", "status": "completed", "manifest": source}
        original = copy.deepcopy(baseline)
        candidate = plan(candidate_manifest(baseline, [target]))
        validate_candidate_protocol(baseline, candidate)
        assert effective_auxiliary_targets(candidate) == effective_auxiliary_targets(
            source
        )
        assert set(candidate["auxiliary_targets"]) == {"judge", "simulator"}
        for role in ("judge", "simulator"):
            assert (
                resolve_auxiliary_target(
                    source["benchmark_options"]["tau3"], role, source
                )
                == candidate["auxiliary_targets"][role]
            )
        assert baseline == original
        candidates.append(candidate)
    assert candidates[0] == candidates[1]


@pytest.mark.parametrize("role", ["judge", "simulator"])
@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("model", "changed-model"),
        ("base_url", "http://127.0.0.1:2/v1"),
        ("api_key_env", "CHANGED_CREDENTIAL_REFERENCE"),
        ("request_params", {"temperature": 0.8, "n": 1}),
        ("prices", {}),
        ("expected_response_model", "changed-identity"),
        ("cost_mode", "changed-cost-mode"),
    ],
)
def test_candidate_rejects_full_effective_auxiliary_definition_drift(
    role, field, replacement
):
    baseline = {
        "id": "baseline",
        "status": "completed",
        "manifest": plan(auxiliary_baseline("subjects")),
    }
    target = {
        **manifest(True)["targets"][0],
        "config_hash": "a" * 64,
        "max_inference_calls": 1,
    }
    candidate = plan(candidate_manifest(baseline, [target]))
    candidate["auxiliary_targets"][role][field] = replacement
    with pytest.raises(ValueError, match="effective auxiliary targets"):
        validate_candidate_protocol(baseline, candidate)


@pytest.mark.parametrize("placement", ["subjects", "auxiliary"])
@pytest.mark.parametrize("role", ["local", "write"])
def test_candidate_http_preserves_fixed_roles_without_dispatch(
    tmp_path, monkeypatch, placement, role
):
    store = Store(tmp_path)
    baseline = record(store, auxiliary_baseline(placement), owner="local")
    original = copy.deepcopy(store.get(baseline["id"]))
    calls = store.calls(baseline["id"])
    target = {
        **manifest(True)["targets"][0],
        "config_hash": "a" * 64,
        "max_inference_calls": 1,
    }
    inventory = auxiliary_baseline("auxiliary")["auxiliary_targets"]
    (tmp_path / "targets.json").write_text(json.dumps([target, *inventory.values()]))
    (tmp_path / "benchmark-options.json").write_text(
        json.dumps(baseline["manifest"]["benchmark_options"])
    )
    service = Server(("127.0.0.1", 0), store, "fixture-token")
    monkeypatch.setattr(
        service.engine,
        "start",
        lambda *_args, **_kwargs: pytest.fail("Candidate planning cannot dispatch"),
    )
    threading.Thread(target=service.serve_forever, daemon=True).start()
    headers = {"Authorization": "Bearer fixture-token"}
    if role != "local":
        headers.update({"X-SR-Bench-Actor-ID": "local", "X-SR-Bench-Actor-Role": role})
    try:
        response = requests.post(
            f"http://127.0.0.1:{service.server_port}{PREFIX}/runs/{baseline['id']}/candidate-plan",
            headers=headers,
            json={"target_ids": [target["id"]]},
            timeout=2,
        )
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["status"] == "validated" and result["model_requests"] == 0
        assert result["total"] == len(baseline["manifest"]["cases"])
        validate_candidate_protocol(baseline, result["manifest"])
        assert result["manifest"]["auxiliary_targets"] == inventory
        assert len(store.list()) == 1
        assert store.get(baseline["id"]) == original
        assert store.calls(baseline["id"]) == calls
    finally:
        service.shutdown()
        service.server_close()


def test_service_accepts_only_canonical_dashboard_roles(tmp_path, monkeypatch):
    store = Store(tmp_path)
    own_run = record(store, owner="alice")
    record(store, owner="bob")
    service = Server(("127.0.0.1", 0), store, "fixture-token")
    monkeypatch.setattr(
        service.engine,
        "start",
        lambda *_: pytest.fail("Role checks cannot dispatch"),
    )
    threading.Thread(target=service.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{service.server_port}{PREFIX}"
    try:
        for role in ("admin", "write", "read", "editor", "viewer"):
            headers = {
                "Authorization": "Bearer fixture-token",
                "X-SR-Bench-Actor-ID": "alice",
                "X-SR-Bench-Actor-Role": role,
            }
            response = requests.get(url + "/runs", headers=headers, timeout=2)
            if role in {"editor", "viewer"}:
                assert response.status_code == 403
                continue
            assert response.status_code == 200
            identifiers = {run["id"] for run in response.json()["runs"]}
            assert len(identifiers) == (2 if role == "admin" else 1)
            assert own_run["id"] in identifiers
            response = requests.post(
                url + "/experiments",
                headers=headers,
                json={"name": f"Role {role}"},
                timeout=2,
            )
            assert response.status_code == (403 if role == "read" else 201)
        assert len(store.list()) == 2
    finally:
        service.shutdown()
        service.server_close()


def test_read_role_can_compare_owned_evidence_without_writes(tmp_path, monkeypatch):
    store = Store(tmp_path)
    baseline = record(store, owner="alice")
    candidate = record(store, owner="alice")
    foreign = record(store, owner="bob")
    service = Server(("127.0.0.1", 0), store, "fixture-token")
    monkeypatch.setattr(
        service.engine,
        "start",
        lambda *_: pytest.fail("Read-only comparisons cannot dispatch"),
    )
    threading.Thread(target=service.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{service.server_port}{PREFIX}"
    headers = {
        "Authorization": "Bearer fixture-token",
        "X-SR-Bench-Actor-ID": "alice",
        "X-SR-Bench-Actor-Role": "read",
    }
    before = list(store.db.iterdump())
    try:
        response = requests.post(
            url + "/comparisons",
            headers=headers,
            json={
                "baseline_run_id": baseline["id"],
                "candidate_run_id": candidate["id"],
            },
            timeout=2,
        )
        assert response.status_code == 200
        assert response.json()["comparisons"][0]["paired_cases"] == 2
        for source, target in ((foreign, candidate), (baseline, foreign)):
            response = requests.post(
                url + "/comparisons",
                headers=headers,
                json={
                    "baseline_run_id": source["id"],
                    "candidate_run_id": target["id"],
                    "actor_role": "admin",
                },
                timeout=2,
            )
            assert response.status_code == 404
        for path in (
            "/experiments",
            "/runs",
            "/plans",
            "/replays",
            "/datasets/compose",
        ):
            response = requests.post(url + path, headers=headers, json={}, timeout=2)
            assert response.status_code == 403
        assert list(store.db.iterdump()) == before
    finally:
        service.shutdown()
        service.server_close()


def test_prepared_plan_roundtrip_uses_verified_dataset_reference(tmp_path, monkeypatch):
    store = Store(tmp_path)
    document = manifest()
    cases = document.pop("cases")
    cases[0].update(
        tools=[{"type": "function", "function": {"name": "fixture_lookup"}}],
        tool_choice="auto",
        response_format={"type": "json_object"},
        request_metadata={"session": "fixture-only"},
    )
    dataset = _write_dataset(
        tmp_path, cases, "quick", 20260918, {"mmlu-pro": {"revision": "fixture"}}
    )
    other_case = {**copy.deepcopy(cases[1]), "id": "three", "benchmark": "gpqa-diamond"}
    other = _write_dataset(
        tmp_path,
        [other_case],
        "quick",
        20260918,
        {"gpqa-diamond": {"revision": "fixture"}},
    )
    service = Server(("127.0.0.1", 0), store)
    monkeypatch.setattr(service.engine, "_run", lambda *_: None)
    threading.Thread(target=service.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{service.server_port}{PREFIX}"
    try:
        composed = requests.post(
            url + "/datasets/compose",
            json={
                "dataset_ids": [dataset["id"], other["id"]],
                "benchmarks": ["mmlu-pro", "gpqa-diamond"],
            },
            timeout=2,
        )
        assert composed.status_code == 200, composed.text
        composite = composed.json()["dataset"]
        for index, (source, expected_cases) in enumerate(
            ((dataset, cases), (composite, [other_case, cases[1], cases[0]]))
        ):
            document["dataset"] = {"path": source["path"], "sha256": source["sha256"]}
            result = requests.post(
                url + "/plans", json={"manifest": document}, timeout=2
            )
            assert result.status_code == 200, result.text
            reviewed = result.json()
            assert "cases" not in reviewed["manifest"]
            assert "private-question-" not in result.text
            restored = plan(reviewed["manifest"])
            assert restored["cases"] == expected_cases
            assert restored["plan_sha256"] == reviewed["plan_sha256"]
            assert restored["case_sha256"] == reviewed["manifest"]["case_sha256"]
            with pytest.raises(ValueError, match="inline cases do not match dataset"):
                plan({**restored, "cases": list(reversed(expected_cases))})
            changed = copy.deepcopy(restored)
            next(case for case in changed["cases"] if case["id"] == "two")[
                "request_metadata"
            ]["session"] = "changed"
            with pytest.raises(ValueError, match="inline cases do not match dataset"):
                plan(changed)
            submitted = requests.post(
                url + "/runs",
                json={
                    "manifest": reviewed["manifest"],
                    "idempotency_key": f"compact-plan-{index}",
                },
                timeout=2,
            )
            assert submitted.status_code == 201, submitted.text
            persisted = store.get(submitted.json()["id"])["manifest"]
            assert persisted["cases"] == expected_cases
            assert persisted["plan_sha256"] == reviewed["plan_sha256"]
        assert store.db.execute("SELECT count(*) FROM calls").fetchone()[0] == 0
    finally:
        service.shutdown()
        service.server_close()


@pytest.mark.parametrize("dataset_reference", [False, True])
def test_selected_benchmark_policy_matches_cli_candidate_review_and_submit(
    tmp_path, monkeypatch, dataset_reference
):
    """Other benchmarks' judges cannot change an MMLU-only frozen protocol."""
    document = manifest()
    if dataset_reference:
        case_file = tmp_path / "cases.json"
        case_file.write_text(json.dumps(document.pop("cases")))
        document["dataset"] = {
            "path": str(case_file),
            "sha256": hashlib.sha256(case_file.read_bytes()).hexdigest(),
        }
    store = Store(tmp_path)
    baseline = record(store, document)
    store.status(baseline["id"], "failed")
    before = copy.deepcopy(store.get(baseline["id"]))
    target = {
        **manifest(True)["targets"][0],
        "config_hash": "a" * 64,
        "max_inference_calls": 1,
    }
    judge = {**document["targets"][0], "id": "judge"}
    (tmp_path / "targets.json").write_text(
        json.dumps([*document["targets"], target, judge])
    )
    (tmp_path / "benchmark-options.json").write_text(
        json.dumps(
            {
                "simpleqa-verified": {
                    "judge": "judge",
                    "grader_version": "sr-bench-reference-judge-v1",
                },
                "tau3": {"judge": "judge", "simulator": "judge"},
            }
        )
    )
    service = Server(("127.0.0.1", 0), store, "fixture-token")
    # Exercise actual HTTP/Engine review and durable submission; this regression
    # never needs a model call or a live recipe snapshot.
    monkeypatch.setattr("cli.sr_bench.engine.capture_runner", lambda *_: {})
    monkeypatch.setattr(
        service.engine,
        "_run",
        lambda run_id, *_: store.status(run_id, "completed"),
    )
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{service.server_port}{PREFIX}"
    local = {"Authorization": "Bearer fixture-token"}
    admin = {
        **local,
        "X-SR-Bench-Actor-ID": "admin",
        "X-SR-Bench-Actor-Role": "admin",
    }
    try:
        reviewed = []
        candidates = []
        for headers in (local, admin):
            response = requests.post(
                url + "/plans", headers=headers, json={"manifest": document}, timeout=2
            )
            assert response.status_code == 200, response.text
            reviewed.append(response.json())
            response = requests.post(
                url + f'/runs/{baseline["id"]}/candidate-plan',
                headers=headers,
                json={"target_ids": [target["id"]]},
                timeout=2,
            )
            assert response.status_code == 200, response.text
            candidates.append(response.json())
        assert reviewed[0] == reviewed[1]
        assert candidates[0] == candidates[1]
        candidate = candidates[1]["manifest"]
        assert "benchmark_options" not in candidate
        assert "auxiliary_targets" not in candidate
        assert candidate["case_sha256"] == before["manifest"]["case_sha256"]
        if dataset_reference:
            assert "cases" not in candidate
        review = requests.post(
            url + "/plans", headers=admin, json={"manifest": candidate}, timeout=2
        )
        assert review.status_code == 200, review.text
        assert review.json() == candidates[1]
        response = requests.post(
            url + "/runs",
            headers=admin,
            json={"manifest": candidate, "idempotency_key": "reviewed-candidate"},
            timeout=2,
        )
        assert response.status_code == 201, response.text
        started = response.json()
        service.engine.threads[started["id"]].join(timeout=1)
        assert started["manifest"]["plan_sha256"] == candidate["plan_sha256"]
        assert store.calls(started["id"]) == []
        assert store.get(baseline["id"]) == before
        selected = manifest()
        for case in selected["cases"]:
            case["benchmark"] = "simpleqa-verified"
        response = requests.post(
            url + "/plans", headers=admin, json={"manifest": selected}, timeout=2
        )
        assert response.status_code == 200, response.text
        resolved = response.json()["manifest"]
        assert set(resolved["benchmark_options"]) == {"simpleqa-verified"}
        assert resolved["auxiliary_targets"] == {"judge": judge}
    finally:
        service.shutdown()
        service.server_close()
        thread.join(timeout=1)


@pytest.mark.parametrize("change", ["new-selected-options", "changed-selected-options"])
def test_candidate_http_rejects_selected_operator_protocol_changes(
    tmp_path, monkeypatch, change
):
    document = manifest()
    if change == "changed-selected-options":
        document["benchmark_options"] = {"mmlu-pro": {"protocol_note": "frozen"}}
    store = Store(tmp_path)
    baseline = record(store, document)
    before = copy.deepcopy(store.get(baseline["id"]))
    target = {
        **manifest(True)["targets"][0],
        "config_hash": "a" * 64,
        "max_inference_calls": 1,
    }
    (tmp_path / "targets.json").write_text(json.dumps([target]))
    (tmp_path / "benchmark-options.json").write_text(
        json.dumps({"mmlu-pro": {"protocol_note": "changed"}})
    )
    service = Server(("127.0.0.1", 0), store, "fixture-token")
    monkeypatch.setattr(
        service.engine, "start", lambda *_args, **_kwargs: pytest.fail("No dispatch")
    )
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    headers = {
        "Authorization": "Bearer fixture-token",
        "X-SR-Bench-Actor-ID": "admin",
        "X-SR-Bench-Actor-Role": "admin",
    }
    try:
        response = requests.post(
            f"http://127.0.0.1:{service.server_port}{PREFIX}/runs/{baseline['id']}/candidate-plan",
            headers=headers,
            json={"target_ids": [target["id"]]},
            timeout=2,
        )
        assert response.status_code == (
            400 if change == "new-selected-options" else 403
        )
        assert store.get(baseline["id"]) == before and len(store.list()) == 1
    finally:
        service.shutdown()
        service.server_close()
        thread.join(timeout=1)


@pytest.mark.parametrize("route", ["plans", "runs"])
@pytest.mark.parametrize(
    "override", ["target", "selected-option", "other-option", "auxiliary"]
)
def test_benchmark_projection_does_not_authorize_client_overrides(
    tmp_path, route, override
):
    document = manifest()
    target = document["targets"][0]
    judge = {**target, "id": "judge"}
    options = {
        "mmlu-pro": {"protocol_note": "fixed"},
        "simpleqa-verified": {"judge": "judge"},
    }
    (tmp_path / "targets.json").write_text(json.dumps([target, judge]))
    (tmp_path / "benchmark-options.json").write_text(json.dumps(options))
    if override == "target":
        document["targets"][0] = {**target, "base_url": "http://127.0.0.1:2/v1"}
    elif override == "auxiliary":
        document["auxiliary_targets"] = {"judge": {**judge, "model": "unregistered"}}
    else:
        key = "mmlu-pro" if override == "selected-option" else "simpleqa-verified"
        document["benchmark_options"] = {key: {"client_override": True}}
    service = Server(("127.0.0.1", 0), Store(tmp_path), "fixture-token")
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    headers = {
        "Authorization": "Bearer fixture-token",
        "X-SR-Bench-Actor-ID": "alice",
        "X-SR-Bench-Actor-Role": "write",
    }
    try:
        response = requests.post(
            f"http://127.0.0.1:{service.server_port}{PREFIX}/{route}",
            headers=headers,
            json={"manifest": document},
            timeout=2,
        )
        assert response.status_code == 403, response.text
        assert service.store.list() == []
    finally:
        service.shutdown()
        service.server_close()
        thread.join(timeout=1)
