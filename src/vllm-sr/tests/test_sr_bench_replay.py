"""Replay discovery and execution share immutable, no-inference eligibility."""

import copy
import json
import threading

import pytest
import requests
from cli.commands.benchmark import benchmark
from cli.routing_preview import case_request_fields
from cli.sr_bench.contracts import digest, plan
from cli.sr_bench.offline import replay
from cli.sr_bench.replay_validation import (
    ReplayEligibilityError,
    ReplayValidator,
)
from cli.sr_bench.service import PREFIX, Server
from cli.sr_bench.store import Store
from click.testing import CliRunner


def manifest(preview=False):
    return {
        "version": "sr-bench-1.0",
        "mode": "preview" if preview else "live",
        "targets": [
            {
                "id": "balance" if preview else "single",
                "kind": "mom" if preview else "single",
                "model": "balance" if preview else "model",
                "base_url": "http://127.0.0.1:1/v1",
                "preview_url": "http://127.0.0.1:1/api/v1/routing/preview",
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
                "id": name,
                "benchmark": "mmlu-pro",
                "messages": [{"role": "user", "content": "private-question-" + name}],
                "answer": "A",
                "metadata": {"split": "dev", "reference": "private-answer"},
            }
            for name in ("two", "one")
        ],
    }


def record(store, document=None, preview=False, routing=None, owner="local", edit=None):
    frozen = plan(document or manifest(preview))
    if edit:
        edit(frozen)
        frozen["case_sha256"] = digest(frozen["cases"])
        frozen["plan_sha256"] = digest(
            {k: v for k, v in frozen.items() if k != "plan_sha256"}
        )
    run, _ = store.create(frozen, owner)
    for case in frozen["cases"]:
        target = frozen["targets"][0]
        data = {
            "correct": True,
            "score": 1,
            "answer": "A",
            "benchmark": case["benchmark"],
        }
        if preview:
            data = {
                "benchmark": case["benchmark"],
                "details": {
                    "routing": {
                        "selection_status": "selected",
                        "selection_method": "static",
                        "selected_model": "model",
                        "decision_result": {"plugins": []},
                        **(routing or {}),
                    }
                },
            }
        store.result(run["id"], case["id"], target["id"], "completed", data)
        if not preview:
            call = store.start_call(
                run["id"],
                case["id"],
                target["id"],
                "subject",
                {
                    "request": {
                        "messages": case["messages"],
                        "sampling": frozen["sampling"],
                        "request_params": target.get("request_params", {}),
                        "effective_body": {
                            "model": target["model"],
                            "messages": case["messages"],
                            **frozen["sampling"],
                            **target.get("request_params", {}),
                            **case_request_fields(case),
                        },
                    },
                },
            )
            store.finish_call(
                call, "completed", {"final": "A", "model": "model", "cost_usd": 0.01}
            )
    store.status(run["id"], "completed")
    return store.get(run["id"])


def evidence(store, run_id):
    return copy.deepcopy(
        (
            store.get(run_id),
            store.calls(run_id),
            store.results(run_id),
            store.events(run_id),
        )
    )


def test_order_only_replay_preserves_sources_and_idempotency(tmp_path, monkeypatch):
    store = Store(tmp_path)
    baseline = record(store)
    document = manifest(True)
    document["cases"].reverse()
    document["targets"][0]["capture_recipe"] = True
    preview = record(store, document, preview=True)
    sources = [evidence(store, run["id"]) for run in (baseline, preview)]
    changes = store.db.total_changes
    monkeypatch.setattr(
        "cli.sr_bench.provenance.capture_recipes",
        lambda *_: pytest.fail("offline replay contacted live configuration"),
    )
    options = ReplayValidator(store, baseline).validate(preview)
    assert options["eligible"] is True
    assert store.db.total_changes == changes
    created = replay(store, baseline["id"], preview["id"], request_key="once")
    receipt = created["manifest"]["replay_compatibility"]
    assert receipt["order_only_difference"] is True
    assert receipt["baseline_case_sha256"] != receipt["preview_case_sha256"]
    assert receipt["case_matching"] == "stable_id_full_content"
    assert created["progress"]["completed"] == 2
    assert all(call["status"] == "replayed" for call in store.calls(created["id"]))
    assert [evidence(store, run["id"]) for run in (baseline, preview)] == sources
    count = store.db.total_changes
    monkeypatch.setattr(
        ReplayValidator,
        "validate",
        lambda *_: pytest.fail("same-key receipt must be returned before revalidation"),
    )
    assert replay(store, baseline["id"], preview["id"], request_key="once") == created
    assert store.db.total_changes == count
    with pytest.raises(ValueError, match="different replay"):
        replay(store, baseline["id"], baseline["id"], request_key="once")


@pytest.mark.parametrize(
    "change,code",
    [
        (lambda m: m["cases"][0].update(id="new"), "case_set_mismatch"),
        (lambda m: m["cases"][0].update(answer="B"), "case_content_mismatch"),
        (
            lambda m: m["cases"][0]["metadata"].update(reference="changed"),
            "case_content_mismatch",
        ),
        (
            lambda m: m["cases"].append(copy.deepcopy(m["cases"][0])),
            "duplicate_case_ids",
        ),
        (lambda m: m["sampling"].update(temperature=0.7), "sampling_mismatch"),
        (
            lambda m: m["adapter_versions"].update({"mmlu-pro": "different"}),
            "grader_protocol_mismatch",
        ),
        (lambda m: m.update(version="different"), "version_mismatch"),
    ],
)
def test_content_and_protocol_mismatches_reject_without_writes(tmp_path, change, code):
    store = Store(tmp_path)
    baseline = record(store)
    preview = record(store, preview=True, edit=change)
    changes = store.db.total_changes
    option = ReplayValidator(store, baseline).validate(preview)
    assert not option["eligible"]
    assert code in {reason["code"] for reason in option["reasons"]}
    with pytest.raises(ReplayEligibilityError) as caught:
        replay(store, baseline["id"], preview["id"])
    assert caught.value.reasons == option["reasons"]
    assert store.db.total_changes == changes


def test_all_route_failures_are_reported_together(tmp_path):
    store = Store(tmp_path)
    baseline = record(store)
    preview = record(
        store,
        preview=True,
        routing={
            "selection_provenance": {
                "mode": "read_only_snapshot",
                "state_dependent": True,
            },
            "selection_status": "execution_required",
            "selection_method": "multi_factor",
            "selected_model": "absent",
            "decision_result": {"plugins": ["request_params"]},
        },
    )
    codes = {
        r["code"] for r in ReplayValidator(store, baseline).validate(preview)["reasons"]
    }
    assert codes == {
        "state_dependent_preview",
        "route_not_deterministic",
        "model_missing",
        "plugins_not_replayable",
    }


@pytest.mark.parametrize("problem", ["missing", "duplicate", "failed", "prompt"])
def test_saved_generation_must_be_complete_unique_and_same_prompt(tmp_path, problem):
    store = Store(tmp_path)
    baseline = record(store)
    preview = record(store, preview=True)
    call = store.calls(baseline["id"])[0]
    if problem == "missing":
        store.db.execute("DELETE FROM calls WHERE id=?", (call["id"],))
        store.db.commit()
    elif problem == "duplicate":
        extra = store.start_call(
            baseline["id"], call["case_id"], call["target_id"], "subject", {}
        )
        store.finish_call(extra, "completed", {})
    elif problem == "failed":
        store.finish_call(call["id"], "failed", {})
    else:
        store.finish_call(call["id"], "completed", {"request": {"messages": []}})
    changes = store.db.total_changes
    with pytest.raises(ReplayEligibilityError) as caught:
        replay(store, baseline["id"], preview["id"])
    expected = (
        "saved_prompt_mismatch"
        if problem == "prompt"
        else "saved_generation_incomplete"
    )
    assert expected in {r["code"] for r in caught.value.reasons}
    assert store.db.total_changes == changes


def test_effective_sampling_not_redundant_override_placement(tmp_path):
    store = Store(tmp_path)
    baseline = record(store)
    document = manifest(True)
    document["sampling"] = {"temperature": 0.7}
    document["targets"][0]["request_params"] = {
        "temperature": baseline["manifest"]["sampling"]["temperature"]
    }
    preview = record(store, document, preview=True)
    assert ReplayValidator(store, baseline).validate(preview)["eligible"] is True


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("temperature", 0.9, "saved_request_parameters_mismatch"),
        ("model", "another-model", "saved_model_mismatch"),
        ("metadata", {"unrecorded": "input"}, "saved_prompt_mismatch"),
        ("tools", [{"type": "function"}], "saved_prompt_mismatch"),
    ],
)
def test_wire_body_must_match_frozen_protocol(tmp_path, field, value, code):
    store = Store(tmp_path)
    baseline = record(store)
    preview = record(store, preview=True)
    call = store.calls(baseline["id"])[0]
    request = copy.deepcopy(call["request"])
    request["effective_body"][field] = value
    store.finish_call(call["id"], "completed", {"request": request})
    changes = store.db.total_changes
    option = ReplayValidator(store, baseline).validate(preview)
    assert code in {reason["code"] for reason in option["reasons"]}
    with pytest.raises(ReplayEligibilityError):
        replay(store, baseline["id"], preview["id"])
    assert store.db.total_changes == changes


def test_http_options_owner_pagination_errors_and_cli(tmp_path, monkeypatch):
    store = Store(tmp_path)
    baseline = record(store, owner="alice")
    previews = [record(store, preview=True, owner="alice") for _ in range(3)]
    foreign = record(store, preview=True, owner="bob")
    service = Server(("127.0.0.1", 0), store, "fixture-token")
    threading.Thread(target=service.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{service.server_port}"
    headers = {
        "Authorization": "Bearer fixture-token",
        "X-SR-Bench-Actor-ID": "alice",
        "X-SR-Bench-Actor-Role": "read",
    }
    url = origin + PREFIX + "/replay-options?baseline_run_id=" + baseline["id"]
    changes = store.db.total_changes
    try:
        first = requests.get(url, headers=headers, params={"limit": 2}, timeout=2)
        assert first.status_code == 200
        page = first.json()
        assert len(page["options"]) == 2 and page["has_more"]
        assert (
            "private-question" not in first.text and "private-answer" not in first.text
        )
        second = requests.get(
            url,
            headers=headers,
            params={"limit": 2, "after": page["next_cursor"]},
            timeout=2,
        ).json()
        assert second["next_cursor"] is None and not second["has_more"]
        assert {o["run_id"] for p in (page, second) for o in p["options"]} == {
            p["id"] for p in previews
        }
        for query in ({"limit": 26}, {"after": "bad"}, {"limit": 0}, {"extra": "x"}):
            assert (
                requests.get(url, headers=headers, params=query, timeout=2).status_code
                == 400
            )
        assert requests.get(url, timeout=2).status_code == 403
        denied = origin + PREFIX + "/replay-options?baseline_run_id=" + foreign["id"]
        assert requests.get(denied, headers=headers, timeout=2).status_code == 404
        body = {"baseline_run_id": baseline["id"], "preview_run_id": previews[0]["id"]}
        assert (
            requests.post(
                origin + PREFIX + "/replays", headers=headers, json=body, timeout=2
            ).status_code
            == 403
        )
        assert store.db.total_changes == changes
        incompatible = record(
            store, preview=True, owner="alice", routing={"selected_model": "absent"}
        )
        changes = store.db.total_changes
        response = requests.post(
            origin + PREFIX + "/replays",
            headers={**headers, "X-SR-Bench-Actor-Role": "write"},
            json={**body, "preview_run_id": incompatible["id"]},
            timeout=2,
        )
        assert response.status_code == 400
        assert response.json()["code"] == "replay_ineligible"
        assert response.json()["dispatch_started"] is False
        assert response.json()["model_requests"] == 0
        assert store.db.total_changes == changes
        monkeypatch.setenv("SR_BENCH_TOKEN", "fixture-token")
        monkeypatch.delenv("SR_BENCH_TOKEN_ENV", raising=False)
        local_baseline = record(store)
        record(store, preview=True)
        changes = store.db.total_changes
        result = CliRunner().invoke(
            benchmark,
            [
                "--url",
                origin,
                "--no-autostart",
                "replay-options",
                local_baseline["id"],
                "--limit",
                "1",
            ],
        )
        assert result.exit_code == 0, result.output
        assert len(json.loads(result.output)["options"]) == 1
        assert store.db.total_changes == changes
    finally:
        service.shutdown()
        service.server_close()
