"""Reviewed controls survive real JavaScript JSON transport without stale bypasses."""

import copy
import json
import shutil
import subprocess
import threading

import pytest
import requests
from cli.sr_bench.candidate_plans import candidate_manifest, validate_candidate_protocol
from cli.sr_bench.contracts import digest, plan, plan_digest, protocol_canonical
from cli.sr_bench.offline import replay
from cli.sr_bench.recovery import recovery_plan
from cli.sr_bench.replay_validation import ReplayValidator
from cli.sr_bench.report import compare
from cli.sr_bench.service import PREFIX, Server
from cli.sr_bench.sources import _write_dataset
from cli.sr_bench.store import Store
from test_sr_bench_replay import manifest, record


def javascript_roundtrip(value):
    node = shutil.which("node")
    assert (
        node is not None
    ), "Node.js is required for the browser JSON contract regression"
    result = subprocess.run(
        [
            node,
            "-e",
            "let text=''; process.stdin.setEncoding('utf8');"
            "process.stdin.on('data', chunk => text += chunk);"
            "process.stdin.on('end', () => "
            "process.stdout.write(JSON.stringify(JSON.parse(text))));",
        ],
        input=json.dumps(value),
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    return json.loads(result.stdout)


def numeric_manifest():
    value = manifest()
    value["sampling"] = {
        "temperature": 1.0,
        "top_p": 1.0,
        "presence_penalty": -0.0,
        "chat_template_kwargs": {"settings": [1.0, -0.0, True, "1", 0.125]},
    }
    value["limits"] = {"max_cost_usd": 1.0, "total_timeout_s": 120.0}
    value["targets"][0]["request_params"] = {"temperature": 1.0}
    value["targets"][0]["prices"]["model"]["input"] = 1.0
    return value


def test_plan_hash_survives_javascript_without_rewriting_raw_evidence():
    frozen = plan(numeric_manifest())
    original = copy.deepcopy(frozen)
    transported = javascript_roundtrip(frozen)
    assert isinstance(frozen["sampling"]["temperature"], float)
    assert isinstance(transported["sampling"]["temperature"], int)
    assert digest(frozen) != digest(transported)
    assert plan(transported)["plan_sha256"] == frozen["plan_sha256"]
    assert frozen == original
    assert frozen["case_sha256"] == digest(frozen["cases"])
    # Case identity remains exact evidence, even where protocol numbers are equal.
    assert digest({"answer": 1.0}) != digest({"answer": 1})


def test_raw_inline_case_identity_is_not_redefined_by_plan_hashing():
    document = numeric_manifest()
    document["cases"][0]["metadata"]["source_measurement"] = 1.0
    frozen = plan(document)
    transported = plan(javascript_roundtrip(frozen))
    assert frozen["case_sha256"] == digest(document["cases"])
    assert transported["case_sha256"] != frozen["case_sha256"]
    assert transported["plan_sha256"] != frozen["plan_sha256"]


@pytest.mark.parametrize(
    "left,right",
    [(True, 1), (False, 0), ("1", 1), (0.125, 0.126), ([1, 2], [2, 1])],
)
def test_numeric_protocol_identity_preserves_real_differences(left, right):
    assert protocol_canonical(left) != protocol_canonical(right)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_numeric_protocol_rejects_nonfinite_values(value):
    with pytest.raises(ValueError):
        protocol_canonical({"nested": [value]})


@pytest.mark.parametrize("value", [True, 1.0])
def test_integer_control_types_remain_strict(value):
    with pytest.raises(ValueError, match="seed must be an integer"):
        plan({**manifest(), "seed": value})
    with pytest.raises(ValueError, match="concurrency"):
        plan({**manifest(), "limits": {"concurrency": value}})


def test_javascript_precision_loss_is_not_normalized_away():
    frozen = plan(numeric_manifest())
    frozen["sampling"]["chat_template_kwargs"]["integer"] = 2**53 + 1
    assert plan_digest(frozen) != plan_digest(javascript_roundtrip(frozen))


@pytest.mark.parametrize("prepared_source", [False, True])
def test_candidate_http_javascript_review_and_submit_share_one_identity(
    tmp_path, monkeypatch, prepared_source
):
    store = Store(tmp_path)
    document = numeric_manifest()
    if prepared_source:
        source = _write_dataset(
            tmp_path,
            document.pop("cases"),
            "quick",
            20260918,
            {"mmlu-pro": {"revision": "fixture"}},
        )
        document["dataset"] = {"path": source["path"], "sha256": source["sha256"]}
    # Retain a pre-normalization hash on saved evidence; never rewrite that row.
    baseline = record(store, document, edit=lambda _manifest: None)
    store.status(baseline["id"], "failed")
    original = copy.deepcopy(store.get(baseline["id"]))
    target = {
        **manifest(True)["targets"][0],
        "config_hash": "a" * 64,
        "max_inference_calls": 1,
        "request_params": {"temperature": 1.0},
    }
    (tmp_path / "targets.json").write_text(json.dumps([target]))
    service = Server(("127.0.0.1", 0), store, "fixture-token")
    captures = []
    monkeypatch.setattr(
        "cli.sr_bench.engine.capture_runner",
        lambda frozen: captures.append(frozen["plan_sha256"]) or {},
    )
    monkeypatch.setattr(
        service.engine, "_run", lambda run_id, *_: store.status(run_id, "completed")
    )
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{service.server_port}{PREFIX}"
    headers = {
        "Authorization": "Bearer fixture-token",
        "X-SR-Bench-Actor-ID": "admin",
        "X-SR-Bench-Actor-Role": "admin",
    }
    try:
        response = requests.post(
            url + f'/runs/{baseline["id"]}/candidate-plan',
            headers=headers,
            json={"target_ids": [target["id"]]},
            timeout=2,
        )
        assert response.status_code == 200, response.text
        reviewed = response.json()
        transported = javascript_roundtrip(reviewed)["manifest"]
        response = requests.post(
            url + "/plans", headers=headers, json={"manifest": transported}, timeout=2
        )
        assert response.status_code == 200, response.text
        assert response.json()["plan_sha256"] == reviewed["plan_sha256"]
        validate_candidate_protocol(original, plan(transported))
        before = list(store.db.iterdump())
        for section, key, value in (
            ("sampling", "temperature", 0.5),
            ("limits", "max_cost_usd", 1.25),
        ):
            changed = copy.deepcopy(transported)
            changed[section][key] = value
            response = requests.post(
                url + "/runs", headers=headers, json={"manifest": changed}, timeout=2
            )
            assert response.status_code == 400, response.text
            assert response.json()["code"] == "reviewed_plan_changed"
            assert response.json()["dispatch_started"] is False
            assert list(store.db.iterdump()) == before
            assert not captures
        changed = copy.deepcopy(transported)
        changed["targets"][0]["prices"]["model"]["input"] = 9
        response = requests.post(
            url + "/runs", headers=headers, json={"manifest": changed}, timeout=2
        )
        assert response.status_code == 403
        assert list(store.db.iterdump()) == before
        assert not captures
        response = requests.post(
            url + "/runs",
            headers=headers,
            json={"manifest": transported, "idempotency_key": "browser-reviewed"},
            timeout=2,
        )
        assert response.status_code == 201, response.text
        run = response.json()
        service.engine.threads[run["id"]].join(timeout=1)
        assert run["manifest"]["plan_sha256"] == reviewed["plan_sha256"]
        assert captures == [reviewed["plan_sha256"]]
        assert store.get(baseline["id"]) == original
        assert not store.calls(run["id"])
    finally:
        service.shutdown()
        service.server_close()
        thread.join(timeout=1)


def test_saved_float_protocol_compares_replays_and_recovers_without_rewriting(tmp_path):
    store = Store(tmp_path)
    baseline = record(store, numeric_manifest(), edit=lambda _manifest: None)
    original = copy.deepcopy(store.get(baseline["id"]))
    transformed = javascript_roundtrip(numeric_manifest())
    candidate = record(store, transformed)
    compared = compare(store, baseline["id"], candidate["id"])
    assert compared["comparisons"][0]["quality_delta"] == 0
    target = {**manifest(True)["targets"][0], "request_params": {"temperature": 1}}
    preview = record(
        store, candidate_manifest(baseline, [target], mode="preview"), preview=True
    )
    assert ReplayValidator(store, baseline).validate(preview)["eligible"]
    replayed = replay(store, baseline["id"], preview["id"])
    assert replayed["manifest"]["plan_sha256"] == plan_digest(replayed["manifest"])
    recovery = recovery_plan(store, baseline["id"])
    assert recovery["plan_sha256"] == plan_digest(javascript_roundtrip(recovery))
    assert store.get(baseline["id"]) == original
