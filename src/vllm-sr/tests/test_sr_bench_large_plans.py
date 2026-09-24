"""Dataset plans retain needed assets without full-file text or duplicate resolution."""

import copy
import gc
import hashlib
import json
import threading
import tracemalloc
from pathlib import Path

import pytest
import requests
from cli.commands.benchmark import benchmark
from cli.sr_bench import contracts
from cli.sr_bench.contracts import (
    canonical,
    digest,
    plan,
    plan_digest,
    protocol_canonical,
)
from cli.sr_bench.service import PREFIX, Server
from cli.sr_bench.sources import _write_dataset
from cli.sr_bench.store import Store
from click.testing import CliRunner

MIB = 1024 * 1024


def _cases(count, asset_bytes):
    for index in range(count):
        yield {
            "id": f"case-{index}",
            "benchmark": "mmlu-pro",
            "messages": [{"role": "user", "content": f"Question {index}"}],
            "answer": "A",
            "metadata": {"split": "dev", "grading_asset": "x" * asset_bytes},
        }


def _document(root, *, count=3, asset_bytes=1024):
    dataset = _write_dataset(
        root,
        _cases(count, asset_bytes),
        "quick",
        7,
        {"mmlu-pro": {"revision": "synthetic-v1"}},
    )
    target = {
        "id": "single",
        "kind": "single",
        "model": "fixture",
        "base_url": "http://127.0.0.1:1/v1",
        "prices": {
            "fixture": {
                "input": 1,
                "cached_input": 0.1,
                "cache_write": 1.25,
                "output": 3,
            }
        },
    }
    (root / "targets.json").write_text(json.dumps([target]))
    return {
        "version": "sr-bench-1.0",
        "profile": "quick",
        "seed": 7,
        "dataset": dataset,
        "targets": [target],
    }


@pytest.fixture
def api(tmp_path, monkeypatch):
    server = Server(("127.0.0.1", 0), Store(tmp_path), "fixture-token")
    monkeypatch.setattr(
        server.engine, "_run", lambda *_: pytest.fail("No run dispatch")
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    headers = {
        "Authorization": "Bearer fixture-token",
        "X-SR-Bench-Actor-ID": "alice",
        "X-SR-Bench-Actor-Role": "write",
    }
    try:
        yield server, f"http://127.0.0.1:{server.server_port}{PREFIX}", headers
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _measure(action):
    gc.collect()
    tracemalloc.start()
    try:
        result = action()
        retained, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, retained, peak


def test_actual_large_dataset_plans_retain_cases_once_and_use_compact_http(
    tmp_path, monkeypatch, api, record_property
):
    document = _document(tmp_path, count=34, asset_bytes=4 * MIB)
    path = Path(document["dataset"]["path"])
    source_size = path.stat().st_size
    assert source_size > 128 * MIB
    original_bytes, original_text = Path.read_bytes, Path.read_text

    def guard_bytes(file):
        if file == path:
            pytest.fail("Dataset resolution must not read all source bytes")
        return original_bytes(file)

    def guard_text(file, *args, **kwargs):
        if file == path:
            pytest.fail("Dataset resolution must not read all source text")
        return original_text(file, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", guard_bytes)
    monkeypatch.setattr(Path, "read_text", guard_text)
    resolved = []
    original_resolve = contracts.resolve_dataset

    def count_resolutions(manifest):
        resolved.append(manifest["dataset"]["sha256"])
        return original_resolve(manifest)

    monkeypatch.setattr(contracts, "resolve_dataset", count_resolutions)
    frozen, retained, peak = _measure(lambda: plan(document))
    assert len(frozen["cases"]) == 34
    assert all(
        len(case["metadata"]["grading_asset"]) == 4 * MIB for case in frozen["cases"]
    )
    assert retained >= 136 * MIB
    assert peak - retained < 32 * MIB
    assert peak < source_size + 40 * MIB
    record_property("dataset_bytes", source_size)
    record_property("plan_retained_bytes", retained)
    record_property("plan_peak_bytes", peak)
    expected_plan, expected_cases = frozen["plan_sha256"], frozen["case_sha256"]
    del frozen
    assert resolved == [document["dataset"]["sha256"]]
    resolved.clear()
    server, base, headers = api
    submitted = {**document, "targets": [{"id": "single"}]}
    response, _, http_peak = _measure(
        lambda: requests.post(
            base + "/plans",
            headers=headers,
            json={"manifest": submitted},
            timeout=30,
        )
    )
    assert response.status_code == 200, response.text
    reviewed = response.json()
    assert reviewed["total"] == 34 and reviewed["model_requests"] == 0
    assert reviewed["plan_sha256"] == expected_plan
    assert reviewed["manifest"]["case_sha256"] == expected_cases
    assert "cases" not in reviewed["manifest"]
    assert "grading_asset" not in response.text
    assert len(response.content) < 8192
    record_property("http_peak_bytes", http_peak)
    assert http_peak < source_size + 40 * MIB
    assert resolved == [document["dataset"]["sha256"]]
    assert server.store.list() == []


def test_cli_and_http_plans_have_identical_frozen_hashes_without_mutation(
    tmp_path, api
):
    document = _document(tmp_path)
    source = tmp_path / "request.json"
    source.write_text(json.dumps(document))
    source_before = source.read_bytes()
    command = CliRunner().invoke(
        benchmark,
        ["--store", str(tmp_path), "--no-autostart", "plan", "--manifest", str(source)],
    )
    assert command.exit_code == 0, command.output
    cli = json.loads(command.output)
    server, base, headers = api
    response = requests.post(
        base + "/plans", headers=headers, json={"manifest": document}, timeout=5
    )
    assert response.status_code == 200, response.text
    http = response.json()
    assert cli["plan_sha256"] == http["plan_sha256"]
    assert cli["total"] == http["total"] == 3
    assert cli["manifest"]["case_sha256"] == http["manifest"]["case_sha256"]
    assert {k: v for k, v in cli["manifest"].items() if k != "cases"} == http[
        "manifest"
    ]
    assert cli["manifest"]["cases"] == list(_cases(3, 1024))
    assert source.read_bytes() == source_before and server.store.list() == []


@pytest.mark.parametrize("bad_inline", [None, [], [{"id": "wrong"}]])
def test_inline_cases_do_not_bypass_dataset_verification(tmp_path, bad_inline):
    document = _document(tmp_path)
    with pytest.raises(ValueError, match="inline cases"):
        plan({**document, "cases": bad_inline})


@pytest.mark.parametrize("inline_number", [1.0, True])
def test_inline_equal_numbers_keep_file_types_hashes_and_original_input(
    tmp_path, inline_number
):
    document = _document(tmp_path, count=2)
    rows = list(_cases(2, 1024))
    rows[0]["metadata"]["numeric"] = 1
    document["dataset"] = _write_dataset(
        tmp_path, rows, "quick", 7, {"mmlu-pro": {"revision": "synthetic-v1"}}
    )
    expected = plan(document)
    inline = copy.deepcopy(rows)
    inline[0]["metadata"]["numeric"] = inline_number
    supplied = {**document, "cases": inline}
    original = canonical(supplied)

    resolved = contracts.resolve_dataset(supplied)
    assert resolved["cases"] is not inline
    assert resolved["cases"][0] is not inline[0]
    assert resolved["cases"][1] is inline[1]
    assert type(resolved["cases"][0]["metadata"]["numeric"]) is int
    assert digest(resolved["cases"]) == expected["case_sha256"]
    frozen = plan(supplied)
    assert frozen["case_sha256"] == expected["case_sha256"]
    assert frozen["plan_sha256"] == expected["plan_sha256"]
    assert canonical(supplied) == original
    assert type(inline[0]["metadata"]["numeric"]) is type(inline_number)


@pytest.mark.parametrize("manifest", [["invalid"], "invalid", 1, True])
def test_http_run_rejects_non_object_manifest(api, manifest):
    server, base, headers = api
    response = requests.post(
        base + "/runs", headers=headers, json={"manifest": manifest}, timeout=5
    )
    assert response.status_code == 400, response.text
    assert "manifest must be an object" in response.text
    assert server.store.list() == []


def test_plan_rejects_tampered_source_and_changed_digest_before_policy(
    tmp_path, monkeypatch
):
    document = _document(tmp_path)
    original = Path(document["dataset"]["path"]).read_bytes()

    def policy(_):
        pytest.fail("Unverified source reached policy")

    wrong_digest = copy.deepcopy(document)
    wrong_digest["dataset"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        plan(wrong_digest, policy=policy)
    path = Path(document["dataset"]["path"])
    path.write_bytes(original.replace(b'"answer":"A"', b'"answer":"B"'))
    with pytest.raises(ValueError, match="digest"):
        plan(document, policy=policy)
    path.write_bytes(original)
    lines = contracts.verified_lines

    def change_during_scan(*args, **kwargs):
        for index, line in enumerate(lines(*args, **kwargs)):
            yield line
            if index == 0:
                # Buffered reads can retain the original bytes, and some
                # filesystems coalesce timestamps for same-size writes. Append
                # JSONL whitespace to change size without shifting row boundaries.
                path.write_bytes(
                    original.replace(b'"answer":"A"', b'"answer":"B"') + b"\n"
                )

    monkeypatch.setattr(contracts, "verified_lines", change_during_scan)
    with pytest.raises(ValueError, match=r"changed|digest"):
        plan(document, policy=policy)


@pytest.mark.parametrize("route", ["/plans", "/runs"])
def test_dataset_reference_cannot_override_registered_target_policy(
    tmp_path, monkeypatch, api, route
):
    document = _document(tmp_path)
    document["targets"][0]["model"] = "forbidden-override"
    # A JSON field cannot choose or disable the service's in-process policy.
    document["policy"] = None
    monkeypatch.setattr(
        "cli.sr_bench.engine.capture_runner",
        lambda *_: pytest.fail("Invalid policy must fail before capture"),
    )
    server, base, headers = api
    response = requests.post(
        base + route, headers=headers, json={"manifest": document}, timeout=5
    )
    assert response.status_code == 403, response.text
    assert server.store.list() == []


def test_streamed_hashes_match_original_canonical_serialization():
    fixtures = [
        {
            "unicode": "中文\u2028\u2029",
            "values": [True, None, -0.0, 1.0, 1e30, 0.125, '\\"\n'],
        },
        [
            {"row": index, "numeric": [float(index), index / 7], "empty": {}}
            for index in range(20)
        ],
    ]
    for fixture in fixtures:
        serialized = json.dumps(
            fixture,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        assert canonical(fixture) == serialized
        assert digest(fixture) == hashlib.sha256(serialized.encode()).hexdigest()
        manifest = {
            "cases": fixture,
            "plan_sha256": "ignored",
            "request": {"seed": 42.0},
        }
        content = {
            key: value for key, value in manifest.items() if key != "plan_sha256"
        }
        assert (
            plan_digest(manifest)
            == hashlib.sha256(protocol_canonical(content).encode()).hexdigest()
        )
    with pytest.raises(ValueError):
        digest({"invalid": float("nan")})
