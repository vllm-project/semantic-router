"""Actual loopback HTTP exercises trace/resume and paired replay without models."""

from __future__ import annotations

import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from systemone_auto.artifacts import digest, read_json, read_jsonl, write_json
from systemone_auto.collection import collect
from systemone_auto.dataset import synthetic_edges
from systemone_auto.replay import load_matrix, replay
from test_contracts import response


def pilot():
    rows = synthetic_edges()[::4]
    for index, row in enumerate(rows):
        row["cohort"] = "public"
        row["split"] = (
            "train" if index < 4 else "calibration" if index < 6 else "held_out"
        )
    return {
        "schema_version": "systemone-pilot/v1",
        "records": rows,
        "records_sha256": digest(rows),
    }


@pytest.fixture
def collection_fixture(tmp_path, monkeypatch):
    data = pilot()
    requests = []
    by_state = {
        json.dumps(row["request"]["state"], sort_keys=True): row
        for row in data["records"]
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(self.headers.get("Authorization"))
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            row = by_state[json.dumps(payload["state"], sort_keys=True)]
            raw = response(row)
            raw.update(
                model=payload["model"],
                meta={
                    "revision": "a" * 40,
                    "model_sha256": "b" * 64,
                    "engine": "native",
                    "profile": "exact",
                    "numerics": "exact",
                    "accelerator": "cpu",
                    "compute_ms": (
                        10.0 if payload["model"].endswith("Kai-0.6B") else 20.0
                    ),
                },
            )
            if payload["model"].endswith("Kai-0.6B"):
                raw["answers"]["exceeds"]["noul"] = 0.49
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(raw).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1/systemone"
    targets = [
        {
            "name": name,
            "model_id": model,
            "protocol": "systemone",
            "revision": "a" * 40,
            "endpoint": endpoint,
            "api_key_env": "SYSTEMONE_TEST_KEY",
        }
        for name, model in (
            ("kai", "vllm-sr/Decision-2.0-Kai-0.6B"),
            ("eos", "vllm-sr/Decision-2.0-Eos-0.8B"),
        )
    ]
    monkeypatch.setenv("SYSTEMONE_TEST_KEY", "secret-not-for-artifacts")
    data_path, targets_path, output = (
        tmp_path / "pilot.json",
        tmp_path / "targets.json",
        tmp_path / "collection",
    )
    write_json(data_path, data)
    write_json(targets_path, targets)
    yield data_path, targets_path, output, requests, data
    server.shutdown()
    worker.join(timeout=5)
    server.server_close()


def test_rotating_models_resume_preserves_full_matrix_and_no_secrets(
    collection_fixture,
):
    data_path, target_path, output, requests, data = collection_fixture
    first = collect(data_path, target_path, output, limit=3, target_names=["kai"])
    assert first["observation_count"] == 3 and not first["complete"]
    with pytest.raises(ValueError, match="complete"):
        load_matrix(data, output)
    collect(data_path, target_path, output, target_names=["kai"])
    final = collect(data_path, target_path, output, target_names=["eos"])
    assert final["complete"] and final["observation_count"] == 16
    assert len(requests) == 16
    collect(data_path, target_path, output)
    assert len(requests) == 16
    contents = "".join(path.read_text() for path in output.iterdir())
    assert "secret-not-for-artifacts" not in contents and "127.0.0.1" not in contents
    assert all(value == "Bearer secret-not-for-artifacts" for value in requests)
    _, matrix = load_matrix(data, output)
    assert len(matrix) == 8
    assert all(row["kai"]["client_elapsed_ms"] > 0 for row in matrix.values())
    trace = read_jsonl(output / "observations.jsonl")
    (output / "observations.jsonl").write_text(
        "\n".join(json.dumps(row) for row in trace[:-1]) + "\n"
    )
    with pytest.raises(ValueError, match="missing"):
        load_matrix(data, output)


def test_held_out_labels_do_not_fit_heads_or_select_operating_points(
    collection_fixture, tmp_path
):
    data_path, target_path, output, _, data = collection_fixture
    collect(data_path, target_path, output)
    report = replay(data_path, output, tmp_path / "first", "kai")
    for point in report["curves"]:
        for result in point["calibration"].values():
            assert result["mean_calls"] <= 2
            assert (
                result["mean_policy_cost_ms"] <= point["calibration_budget_ms"] + 1e-9
            )
    changed = copy.deepcopy(data)
    for row in changed["records"]:
        if row["split"] == "held_out":
            for label in row["labels"].values():
                label["label"] = {
                    "true": "false",
                    "false": "true",
                    "above": "below",
                    "below": "above",
                    "equal": "below",
                    "0": "2",
                    "1": "0",
                    "2": "0",
                }[label["label"]]
    changed["records_sha256"] = digest(changed["records"])
    write_json(data_path, changed)
    manifest = read_json(output / "collection.json")
    manifest["dataset_sha256"] = digest(changed)
    write_json(output / "collection.json", manifest)
    second = replay(data_path, output, tmp_path / "second", "kai")
    assert (
        read_json(tmp_path / "first/policy.json")["heads"]
        == read_json(tmp_path / "second/policy.json")["heads"]
    )
    assert [point["settings"] for point in report["curves"]] == [
        point["settings"] for point in second["curves"]
    ]
    assert report["single_model_held_out"] != second["single_model_held_out"]


def test_resume_rejects_changed_model_revision(collection_fixture):
    data_path, target_path, output, _, _ = collection_fixture
    collect(data_path, target_path, output, limit=1)
    targets = read_json(target_path)
    targets[0]["revision"] = "b" * 40
    write_json(target_path, targets)
    with pytest.raises(ValueError, match="resume identity"):
        collect(data_path, target_path, output)


def test_restricted_pool_is_a_separate_bound_experiment(collection_fixture, tmp_path):
    data_path, target_path, output, _, data = collection_fixture
    collect(data_path, target_path, output)
    protocol = {
        "schema_version": "systemone-replay-protocol/v1",
        "base": "kai",
        "native_pool": ["kai", "eos"],
        "max_model_calls": 2,
        "operating_point_budget_fractions": [0, 0.25, 0.5, 0.75, 1],
        "dataset_sha256": digest(data),
    }
    path = tmp_path / "protocol.json"
    write_json(path, protocol)
    report = replay(
        data_path, output, tmp_path / "restricted", "kai", protocol_path=path
    )
    assert report["native_pool"] == ["kai", "eos"]
    assert report["protocol_sha256"]
    policy = read_json(tmp_path / "restricted/policy.json")
    assert set(policy["actions"]) == {"kai", "eos"}
    assert policy["training"]["protocol_sha256"] == report["protocol_sha256"]
    protocol["native_pool"] = ["kai", "missing"]
    write_json(path, protocol)
    with pytest.raises(ValueError, match="protocol"):
        replay(data_path, output, tmp_path / "invalid", "kai", protocol_path=path)
