"""Eligible-only cascades share submission checks and bound each discovery page."""

import base64
import json
import threading
from urllib.parse import parse_qs, urlsplit

import pytest
import requests
from cli.commands.benchmark import benchmark
from cli.sr_bench import run_options as options_module
from cli.sr_bench.client import Client
from cli.sr_bench.offline import replay
from cli.sr_bench.report import compare
from cli.sr_bench.run_options import run_options
from cli.sr_bench.service import PREFIX, Server
from cli.sr_bench.store import Store
from click.testing import CliRunner
from test_sr_bench_replay import manifest, record


def candidate(store, owner="local", edit=None):
    document = manifest(True)
    document["mode"] = "live"
    document["targets"][0].update(config_hash="a" * 64, max_inference_calls=1)
    return record(store, document, owner=owner, edit=edit)


def pages(store, kind, baseline=None, owner=None, limit=1):
    cursor = None
    values = []
    for _ in range(100):
        page = run_options(store, kind, baseline, owner, cursor, limit)
        values.append(page)
        if not page["has_more"]:
            assert page["next_cursor"] is None
            return values
        assert page["next_cursor"] and page["next_cursor"] != cursor
        cursor = page["next_cursor"]
    pytest.fail("Discovery pagination did not terminate")


@pytest.mark.parametrize("kind", ["replay", "comparison"])
def test_only_baselines_with_real_compatible_children_and_no_writes(tmp_path, kind):
    store = Store(tmp_path)
    baseline = record(store)
    good = record(store, preview=True) if kind == "replay" else candidate(store)
    orphan = record(store, edit=lambda m: m["cases"][0].update(id="orphan"))
    changes = store.db.total_changes
    rows = run_options(store, kind)
    assert [r["run_id"] for r in rows["baselines"]] == [baseline["id"]]
    assert orphan["id"] not in json.dumps(rows)
    assert rows["options"] == [] and rows["baseline"] is None
    result = run_options(store, kind, baseline["id"])
    assert [r["run_id"] for r in result["options"]] == [good["id"]]
    assert result["baseline"]["run_id"] == baseline["id"]
    assert not result["has_more"]
    assert "private-question" not in json.dumps([rows, result])
    assert store.db.total_changes == changes
    if kind == "comparison":
        assert compare(store, baseline["id"], good["id"])["comparisons"]
    else:
        assert replay(store, baseline["id"], good["id"])["status"] == "completed"


@pytest.mark.parametrize("kind", ["replay", "comparison"])
def test_no_self_pair_can_make_a_lone_baseline_eligible(tmp_path, kind):
    store = Store(tmp_path)
    baseline = record(store)
    assert run_options(store, kind)["baselines"] == []
    assert not run_options(store, kind)["has_more"]
    with pytest.raises(ValueError, match="distinct"):
        compare(store, baseline["id"], baseline["id"])


@pytest.mark.parametrize("kind", ["replay", "comparison"])
def test_actor_filter_applies_to_both_tiers_and_cursor_scope(tmp_path, kind):
    store = Store(tmp_path)
    baseline = record(store, owner="alice")
    child = (
        record(store, preview=True, owner="bob")
        if kind == "replay"
        else candidate(store, "bob")
    )
    assert run_options(store, kind, owner="alice")["baselines"] == []
    assert run_options(store, kind, baseline["id"], "alice")["options"] == []
    assert run_options(store, kind)["baselines"][0]["run_id"] == baseline["id"]
    with pytest.raises(KeyError):
        run_options(store, kind, child["id"], "alice")
    (
        record(store, preview=True, owner="alice")
        if kind == "replay"
        else candidate(store, "alice")
    )
    (
        record(store, preview=True, owner="alice")
        if kind == "replay"
        else candidate(store, "alice")
    )
    page = run_options(store, kind, baseline["id"], "alice", limit=1)
    for other_kind, other_baseline, owner in (
        (kind, baseline["id"], "bob"),
        (kind, None, "alice"),
        ("comparison" if kind == "replay" else "replay", baseline["id"], "alice"),
    ):
        with pytest.raises(ValueError, match="cursor"):
            run_options(store, other_kind, other_baseline, owner, page["next_cursor"])


@pytest.mark.parametrize("kind", ["replay", "comparison"])
def test_empty_scan_pages_resume_without_skipping_or_repeating(
    tmp_path, monkeypatch, kind
):
    store = Store(tmp_path)
    baseline = record(store)
    good = record(store, preview=True) if kind == "replay" else candidate(store)
    for _ in range(5):
        if kind == "replay":
            record(store, preview=True, routing={"selection_method": "multi_factor"})
        else:
            candidate(store, edit=lambda m: m["sampling"].update(temperature=0.9))
    monkeypatch.setattr(options_module, "MAX_SCAN_STEPS", 2)
    listing = pages(store, kind)
    assert listing[0]["baselines"] == [] and listing[0]["has_more"]
    assert [r["run_id"] for p in listing for r in p["baselines"]] == [baseline["id"]]
    children = pages(store, kind, baseline["id"])
    assert children[0]["options"] == [] and children[0]["has_more"]
    assert [r["run_id"] for p in children for r in p["options"]] == [good["id"]]
    assert all(page["scanned_pairs"] <= 2 for page in listing + children)


@pytest.mark.parametrize("kind", ["replay", "comparison"])
def test_pages_return_all_distinct_children_and_invalidate_changed_state(
    tmp_path, kind
):
    store = Store(tmp_path)
    baseline = record(store)
    children = [
        record(store, preview=True) if kind == "replay" else candidate(store)
        for _ in range(3)
    ]
    listing = pages(store, kind, baseline["id"])
    assert [r["run_id"] for p in listing for r in p["options"]] == [
        r["id"] for r in reversed(children)
    ]
    first = run_options(store, kind, baseline["id"], limit=1)
    store.status(children[0]["id"], "failed")
    with pytest.raises(ValueError, match="Saved runs changed"):
        run_options(store, kind, baseline["id"], after=first["next_cursor"])


def test_authoritative_quality_and_replay_rules_are_not_relaxed(tmp_path):
    store = Store(tmp_path)
    baseline = record(store)
    preview = record(
        store, preview=True, routing={"selection_provenance": {"state_dependent": True}}
    )
    other = candidate(store)
    row = store.results(other["id"])[0]
    store.result(
        other["id"], row["case_id"], row["target_id"], "completed", {"correct": None}
    )
    assert run_options(store, "replay")["baselines"] == []
    assert run_options(store, "comparison")["baselines"] == []
    with pytest.raises(ValueError, match="quality results"):
        compare(store, baseline["id"], other["id"])
    with pytest.raises(ValueError, match="State-dependent"):
        replay(store, baseline["id"], preview["id"])


def test_failed_outcome_comparison_discovery_and_submit_have_identical_scope(tmp_path):
    store = Store(tmp_path)
    baseline = record(store, owner="alice")
    child = candidate(store, "alice")
    for run in (baseline, child):
        row = store.results(run["id"])[0]
        store.result(
            run["id"],
            row["case_id"],
            row["target_id"],
            "failed",
            {"error": "Request deadline exceeded", "benchmark": row["benchmark"]},
        )
        store.status(run["id"], "failed")
    original = list(store.db.iterdump())
    assert (
        run_options(store, "comparison", owner="alice")["baselines"][0]["run_id"]
        == baseline["id"]
    )
    assert (
        run_options(store, "comparison", baseline["id"], "alice")["options"][0][
            "run_id"
        ]
        == child["id"]
    )
    assert run_options(store, "comparison", owner="bob")["baselines"] == []
    assert run_options(store, "replay", owner="alice")["baselines"] == []
    comparison = compare(store, baseline["id"], child["id"])
    assert comparison["comparisons"][0]["paired_cases"] == 2
    assert comparison["comparisons"][0]["quality_delta"] == 0
    assert list(store.db.iterdump()) == original


@pytest.mark.parametrize("status", ["running", "queued", "cancelled", "interrupted"])
def test_comparison_discovery_does_not_admit_noncomparable_terminal_or_active_runs(
    tmp_path, status
):
    store = Store(tmp_path)
    baseline = record(store)
    child = candidate(store)
    store.status(baseline["id"], status)
    assert run_options(store, "comparison")["baselines"] == []
    assert run_options(store, "comparison", baseline["id"])["baseline"] is None
    with pytest.raises(ValueError, match="completed or failed"):
        compare(store, baseline["id"], child["id"])


def test_malformed_cursor_and_ineligible_active_baseline_reject_before_loading(
    tmp_path, monkeypatch
):
    store = Store(tmp_path)
    record(store)
    preview = record(store, preview=True)
    for value in ([], None, "cursor", {"active": preview["id"]}):
        encoded = base64.urlsafe_b64encode(json.dumps(value).encode()).decode()
        with pytest.raises(ValueError, match="Invalid run options cursor"):
            run_options(store, "replay", after=encoded)
    monkeypatch.setattr(options_module, "MAX_SCAN_STEPS", 1)
    first = run_options(store, "replay")
    cursor = json.loads(base64.urlsafe_b64decode(first["next_cursor"]))
    cursor["active"] = preview["id"]
    monkeypatch.setattr(
        store, "get", lambda *_: pytest.fail("Invalid baseline was loaded")
    )
    with pytest.raises(ValueError, match="baseline no longer qualifies"):
        run_options(store, "replay", after=options_module._cursor(cursor))


def test_oversized_evidence_is_explicit_without_poisoning_small_pairs(
    tmp_path, monkeypatch
):
    store = Store(tmp_path)
    small = record(store)
    oversized = record(store)
    record(store, preview=True)
    original = options_module._evidence_size

    def size(storage, run_id, include_calls):
        return (
            options_module.MAX_EVIDENCE_BYTES + 1
            if run_id == oversized["id"]
            else original(storage, run_id, include_calls)
        )

    monkeypatch.setattr(options_module, "_evidence_size", size)
    result = run_options(store, "replay")
    assert [r["run_id"] for r in result["baselines"]] == [small["id"]]
    assert result["scan_limited"] and result["unverified_baselines"] == 1
    direct = run_options(store, "replay", oversized["id"])
    assert direct["baseline"] is None and direct["options"] == []
    assert direct["scan_limited"] and direct["empty_reason"] == "evidence_size_limit"


def test_oversized_child_warning_survives_cursor_and_other_child_still_returns(
    tmp_path, monkeypatch
):
    store = Store(tmp_path)
    baseline = record(store)
    good = record(store, preview=True)
    huge = record(store, preview=True)
    original = options_module._evidence_size

    def size(storage, run_id, include_calls):
        return (
            options_module.MAX_EVIDENCE_BYTES + 1
            if run_id == huge["id"]
            else original(storage, run_id, include_calls)
        )

    monkeypatch.setattr(options_module, "_evidence_size", size)
    monkeypatch.setattr(options_module, "MAX_SCAN_STEPS", 1)
    result = pages(store, "replay", baseline["id"])
    assert result[0]["options"] == [] and result[0]["has_more"]
    assert result[-1]["scan_limited"] and result[-1]["unverified_pairs"] == 1
    assert [r["run_id"] for page in result for r in page["options"]] == [good["id"]]


@pytest.mark.parametrize("kind", ["replay", "comparison"])
def test_multiple_eligible_baselines_paginate_once_each(tmp_path, kind):
    store = Store(tmp_path)
    baselines = [record(store) for _ in range(3)]
    record(store, preview=True) if kind == "replay" else candidate(store)
    result = pages(store, kind)
    assert [r["run_id"] for p in result for r in p["baselines"]] == [
        r["id"] for r in reversed(baselines)
    ]


def test_baseline_evidence_is_loaded_once_per_page_and_active_updates_do_not_invalidate(
    tmp_path, monkeypatch
):
    store = Store(tmp_path)
    baseline = record(store, owner="alice")
    for _ in range(3):
        record(store, preview=True, owner="alice")
    original = options_module.ReplayValidator.__init__
    initialized = []

    def count(self, storage, source):
        initialized.append(source["id"])
        original(self, storage, source)

    monkeypatch.setattr(options_module.ReplayValidator, "__init__", count)
    result = run_options(store, "replay", baseline["id"], "alice", limit=2)
    assert initialized == [baseline["id"]]
    pending, _ = store.create(baseline["manifest"], owner="alice")
    store.status(pending["id"], "running")
    record(store, preview=True, owner="bob")
    second = run_options(
        store, "replay", baseline["id"], "alice", result["next_cursor"]
    )
    assert len(second["options"]) == 1


def test_http_canonical_get_only_read_role_and_no_legacy_alias(tmp_path, monkeypatch):
    store = Store(tmp_path)
    baseline = record(store, owner="alice")
    record(store, preview=True, owner="alice")
    candidate(store, owner="alice")
    service = Server(("127.0.0.1", 0), store, "fixture-token")
    monkeypatch.setattr(
        service.engine, "start", lambda *_: pytest.fail("Discovery cannot dispatch")
    )
    threading.Thread(target=service.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{service.server_port}{PREFIX}"
    headers = {
        "Authorization": "Bearer fixture-token",
        "X-SR-Bench-Actor-ID": "alice",
        "X-SR-Bench-Actor-Role": "read",
    }
    changes = store.db.total_changes
    try:
        for route in ("replay-options", "comparison-options"):
            response = requests.get(url + "/" + route, headers=headers, timeout=2)
            assert response.status_code == 200, response.text
            assert response.json()["baselines"][0]["run_id"] == baseline["id"]
            assert requests.get(url + "/" + route, timeout=2).status_code == 403
            assert (
                requests.post(
                    url + "/" + route, headers=headers, json={}, timeout=2
                ).status_code
                == 403
            )
            assert (
                requests.get(
                    url + "/" + route + "?extra=1", headers=headers, timeout=2
                ).status_code
                == 400
            )
        assert (
            requests.get(
                url + f"/runs/{baseline['id']}/replay-options",
                headers=headers,
                timeout=2,
            ).status_code
            == 404
        )
        assert store.db.total_changes == changes
    finally:
        service.shutdown()
        service.server_close()


@pytest.mark.parametrize("command", ["replay-options", "comparison-options"])
@pytest.mark.parametrize("baseline", [None, "run-saved"])
def test_cli_cascade_uses_canonical_endpoint_and_opaque_cursor(
    tmp_path, monkeypatch, command, baseline
):
    observed = []

    def request(_self, method, path):
        observed.append((method, urlsplit(path)))
        return {"model_requests": 0}

    monkeypatch.setattr(Client, "request", request)
    arguments = ["--url", "http://127.0.0.1:1", "--store", str(tmp_path), command]
    if baseline:
        arguments.append(baseline)
    arguments.extend(["--limit", "2", "--after", "opaque+/="])
    result = CliRunner().invoke(benchmark, arguments)
    assert result.exit_code == 0, result.output
    assert len(observed) == 1
    method, url = observed[0]
    assert method == "GET" and url.path == "/" + command
    expected = {"limit": ["2"], "after": ["opaque+/="]}
    if baseline:
        expected["baseline_run_id"] = [baseline]
    assert parse_qs(url.query) == expected
    assert json.loads(result.output) == {"model_requests": 0}
