"""Hermetic collection/scoring fixtures; no external service or model calls."""

from __future__ import annotations

import argparse
import copy
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from systemone_auto import __main__ as entry
from systemone_auto import jevbench as live
from systemone_auto.artifacts import read_json, read_jsonl, write_json
from systemone_auto.jevbench_suite import (
    REVISIONS,
    request_for,
    schedule_for,
    score_response,
    validate_schedule,
)


@pytest.fixture
def suite():
    questions = {
        "choice": {
            "type": "choice",
            "instructions": "Choose.",
            "criteria": {"b": "Second", "a": "First"},
        },
        "noul": {
            "type": "noul",
            "instructions": "Is it true?",
            "criteria": {"true": "Yes", "false": "No"},
        },
        "score": {
            "type": "score",
            "instructions": "Rate.",
            "criteria": ["low", "middle", "high"],
        },
    }
    tasks = {
        kind: SimpleNamespace(
            id=kind,
            group=None,
            state="Unlabelled input " + kind,
            question=question,
            expected={"choice": "a", "noul": "yes", "score": 0}[kind],
        )
        for kind, question in questions.items()
    }

    class Adapter:
        def __init__(self, model, key_env):
            self.model = model

        def build_request(self, task):
            return {
                "state": task.state,
                "model": self.model,
                "questions": {"decision": task.question},
            }

    def score(probabilities, task):
        predicted = max(probabilities, key=probabilities.get)
        return {"valid": True, "correct": predicted == str(task.expected)}

    return SimpleNamespace(
        tasks=tasks, adapter=Adapter, score=score, source={"commit": "fixture"}
    )


def arguments(tmp_path, **updates):
    parser = argparse.ArgumentParser()
    live.add_parser(parser.add_subparsers(dest="command", required=True))
    config = tmp_path / "config.yaml"
    config.write_text("version: v0.3\n")
    args = parser.parse_args(
        [
            "jevbench",
            "--jevbench-checkout",
            str(tmp_path),
            "--endpoint",
            "http://private-test.invalid",
            "--output-dir",
            str(tmp_path / "out"),
            "--config",
            str(config),
            "--key-env",
            "FIXTURE_API_KEY",
        ]
    )
    for key, value in updates.items():
        setattr(args, key, value)
    return args


def response(payload, *, no_coverage=False):
    kind = payload["questions"]["decision"]["type"]
    value = {
        "choice": {
            "type": "choice",
            "choice": "a",
            "probabilities": {"b": 0.1, "a": 0.9},
        },
        "noul": {"type": "noul", "noul": 0.8},
        # Expected-point rounding would choose level 1, while modal scoring is0.
        "score": {
            "type": "score",
            "score": 0.9,
            "probabilities": {"0": 0.6, "1": 0.1, "2": 0.3},
        },
    }[kind]
    if not no_coverage:
        value["input_coverage"] = "complete"
    model = "vega" if payload["model"] == "vega" else "kai"
    return {
        "model": payload["model"],
        "answers": {"decision": value},
        "meta": {
            "revision": REVISIONS[model],
            "model_sha256": ("a" if model == "kai" else "b") * 64,
            "profile": "exact",
            "numerics": "exact",
            "engine": "native",
            "accelerator": "cpu",
            "device": "private-device",
        },
        "private_server": "not-for-artifacts",
    }


def setup(monkeypatch, suite):
    monkeypatch.setattr(live, "load_suite", lambda *_: suite)
    monkeypatch.setenv("FIXTURE_API_KEY", "a-secret-never-recorded")


def test_plan_only_is_label_free_and_never_requests(tmp_path, monkeypatch, suite):
    setup(monkeypatch, suite)
    monkeypatch.delenv("FIXTURE_API_KEY")
    monkeypatch.setattr(
        live, "request_once", lambda *_: pytest.fail("no network in plan-only")
    )
    args = arguments(tmp_path, plan_only=True)
    result = live.run(args)
    assert result["expected_count"] == 27 and result["observation_count"] == 0
    plan = read_json(args.output_dir / "plan.json")
    assert plan["declared_gate_threshold"] == 0.6059704079536342
    assert plan["warmup_order"] == ["choice", "noul", "score"]
    assert "expected" not in str(plan) and "private-test" not in str(plan)
    assert plan["concurrency"] == 1 and plan["retries"] == 0
    assert not (args.output_dir / "observations.jsonl").exists()


def test_collect_all_passes_preserves_errors_and_unknown_metrics(
    tmp_path, monkeypatch, suite
):
    setup(monkeypatch, suite)
    requests = []

    def request(endpoint, key, payload, timeout):
        requests.append(copy.deepcopy(payload))
        assert key == "a-secret-never-recorded"
        assert payload["questions"]["decision"]["require_full_input"] is True
        assert payload["options"]["return_meta"] is True
        assert "expected" not in payload and "labels" not in payload
        if len(requests) > 9 and payload["model"] == "vega":
            return 503, {"error": {"message": "private-host"}}, 10.0
        return 200, response(payload), 4.0

    monkeypatch.setattr(live, "request_once", request)
    args = arguments(tmp_path)
    result = live.run(args)
    assert result["complete"] and result["observation_count"] == 27
    assert len(requests) == 36 and result["warmup_count"] == 9
    rows = read_jsonl(args.output_dir / "observations.jsonl")
    assert len(rows) == 27 and sum(r["http_status"] == 503 for r in rows) == 9
    summary = read_json(args.output_dir / "summary.json")
    assert summary["quality"]["direct_vega"] == {
        "planned": 3,
        "observed": 3,
        "correct": 0,
        "unresolved": 3,
        "accuracy": 0,
        "kai_coverage": 0,
    }
    assert summary["quality"]["auto"]["correct"] == 3
    assert summary["passes"]["1"]["direct_vega"]["elapsed_ms"]["mean"] == 10
    assert summary["passes"]["1"]["auto"]["physical_calls"] is None
    contents = "".join(f.read_text() for f in args.output_dir.iterdir())
    for private in (
        "private-test",
        "private-host",
        "private-device",
        "not-for-artifacts",
        "a-secret-never-recorded",
    ):
        assert private not in contents
    with pytest.raises(ValueError, match="empty"):
        live.run(args)
    assert len(requests) == 36


def test_upstream_scoring_uses_modal_score_and_noul_yes_no_mapping(suite):
    for task in suite.tasks.values():
        payload = request_for(suite, task, "kai")
        assert "require_full_input" not in task.question
        assert score_response(suite, task, payload, 200, response(payload))["correct"]
        assert not score_response(
            suite, task, payload, 200, response(payload, no_coverage=True)
        )["valid"]
        assert not score_response(suite, task, payload, 200, {"answers": None})["valid"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reordered_passes"])
def test_schedule_rejects_missing_duplicate_or_nonsequential_passes(suite, mutation):
    schedule = schedule_for(suite.tasks, 3, 42)
    validate_schedule(schedule, suite.tasks, 3)
    if mutation == "missing":
        schedule.pop()
    elif mutation == "duplicate":
        schedule[-1] = schedule[0]
    else:
        schedule.reverse()
    with pytest.raises(ValueError, match="schedule"):
        validate_schedule(schedule, suite.tasks, 3)


def test_frozen_schedule_and_warmups_are_used_verbatim(tmp_path, monkeypatch, suite):
    setup(monkeypatch, suite)
    plan = schedule_for(suite.tasks, 1, 12)
    path = tmp_path / "schedule.json"
    write_json(path, plan)
    warmups = {
        kind: request_for(suite, task, "ignored") for kind, task in suite.tasks.items()
    }
    warmup_path = tmp_path / "warmups.json"
    write_json(warmup_path, warmups)
    seen = []

    def request(_endpoint, _key, payload, _timeout):
        seen.append(payload)
        return 200, response(payload), 2.5

    monkeypatch.setattr(live, "request_once", request)
    args = arguments(tmp_path, passes=1, schedule=path, warmups=warmup_path)
    live.run(args)
    rows = read_jsonl(args.output_dir / "observations.jsonl")
    assert [{k: r[k] for k in plan[0]} for r in rows] == plan
    assert len(seen) == 18 and {row["pass"] for row in rows} == {0}


def test_identity_drift_and_wrong_direct_model_are_unresolved(
    tmp_path, monkeypatch, suite
):
    setup(monkeypatch, suite)
    count = 0

    def request(_endpoint, _key, payload, _timeout):
        nonlocal count
        count += 1
        raw = response(payload)
        if count > 9:
            raw["meta"]["model_sha256"] = "f" * 64
        return 200, raw, 1.0

    monkeypatch.setattr(live, "request_once", request)
    args = arguments(tmp_path, passes=1)
    live.run(args)
    summary = read_json(args.output_dir / "summary.json")
    assert all(value["unresolved"] == 3 for value in summary["quality"].values())


def test_interrupt_keeps_partial_receipt_and_planned_denominator(
    tmp_path, monkeypatch, suite
):
    setup(monkeypatch, suite)
    count = 0

    def request(_endpoint, _key, payload, _timeout):
        nonlocal count
        count += 1
        if count == 11:
            raise KeyboardInterrupt
        return 200, response(payload), 1.0

    monkeypatch.setattr(live, "request_once", request)
    args = arguments(tmp_path, passes=1)
    with pytest.raises(KeyboardInterrupt):
        live.run(args)
    receipt = read_json(args.output_dir / "collection.json")
    assert not receipt["complete"] and receipt["observation_count"] == 1
    summary = read_json(args.output_dir / "summary.json")
    assert all(value["planned"] == 3 for value in summary["quality"].values())
    assert sum(value["unresolved"] for value in summary["quality"].values()) == 8


def test_counter_unknown_reset_and_success():
    before = {
        model: {"physical_calls": 10.0, "forward_seconds": 3.0} for model in REVISIONS
    }
    after = {
        model: {"physical_calls": 11.0, "forward_seconds": 3.25} for model in REVISIONS
    }
    assert live.metric_delta(before, after)["kai"]["physical_calls"] == 1
    assert live.metric_delta(None, after) is None
    assert live.metric_delta(after, before) is None


def test_cli_registration_dispatches_without_network(
    tmp_path, monkeypatch, suite, capsys
):
    setup(monkeypatch, suite)
    args = arguments(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "systemone_auto",
            "jevbench",
            "--jevbench-checkout",
            str(tmp_path),
            "--endpoint",
            "http://unused.invalid",
            "--output-dir",
            str(args.output_dir),
            "--config",
            str(args.config),
            "--plan-only",
        ],
    )
    monkeypatch.setattr(
        live, "request_once", lambda *_: pytest.fail("no network in CLI plan")
    )
    entry.main()
    assert '"plan_only":true' in capsys.readouterr().out


@pytest.mark.parametrize(
    "status,raw",
    [
        (0, {}),
        (500, {}),
        (200, {"answers": {"decision": []}}),
        (200, {"answers": {"decision": {"type": "choice", "error": "secret"}}}),
    ],
)
def test_malformed_responses_remain_unresolved(suite, status, raw):
    task = suite.tasks["choice"]
    result = score_response(suite, task, request_for(suite, task, "kai"), status, raw)
    assert result == {"valid": False, "correct": False, "top_probability": None}


def test_metrics_are_outside_http_sample_and_failure_is_unknown(
    tmp_path, monkeypatch, suite
):
    setup(monkeypatch, suite)
    events = []
    counter = 0

    def metrics(*_):
        events.append("metrics")
        return {
            name: {"physical_calls": float(counter), "forward_seconds": counter / 1000}
            for name in REVISIONS
        }

    def request(_endpoint, _key, payload, _timeout):
        nonlocal counter
        events.append("request")
        counter += 1
        return 200, response(payload), 7.0

    monkeypatch.setattr(live, "metric_snapshot", metrics)
    monkeypatch.setattr(live, "request_once", request)
    args = arguments(
        tmp_path,
        passes=1,
        kai_metrics="http://kai.invalid/metrics",
        vega_metrics="http://vega.invalid/metrics",
    )
    live.run(args)
    assert events == ["metrics", "request", "metrics"] * 18
    rows = read_jsonl(args.output_dir / "observations.jsonl")
    assert all(row["client_elapsed_ms"] == 7 for row in rows)
    assert all(
        row["native_metrics_delta"]["kai"]["physical_calls"] == 1 for row in rows
    )


def test_response_projection_excludes_private_diagnostics(suite):
    question = suite.tasks["choice"].question
    raw = {
        "answers": {
            "decision": {
                "type": [],
                "choice": "private.invalid",
                "input_coverage": [],
                "probabilities": {"private.invalid": 1.0},
                "error": "password",
            }
        },
        "meta": {
            "revision": "private.invalid",
            "model_sha256": "private.invalid",
            "profile": "private.invalid",
            "accelerator": ["private.invalid"],
            "device": "private-device",
        },
        "error": "password",
    }
    clean = live.safe_response(raw, question)
    assert "private" not in str(clean) and "password" not in str(clean)
    assert clean["error"]["code"] == "native_error"


def test_metrics_parser_filters_endpoint_and_rejects_nonfinite(monkeypatch):
    body = b"""vllm_srun_requests_total{endpoint="/v1/systemone"} 12
vllm_srun_requests_total{endpoint="/v1/models"} 100
vllm_srun_forward_duration_seconds_sum 0.25
"""
    monkeypatch.setattr(
        live.urllib.request,
        "build_opener",
        lambda *_: SimpleNamespace(
            open=lambda *_args, **_kwargs: nullcontext(
                SimpleNamespace(read=lambda _limit: body)
            )
        ),
    )
    assert live._counters("http://unused.invalid/metrics", None, 1) == {
        "physical_calls": 12,
        "forward_seconds": 0.25,
    }
    body = body.replace(b"0.25", b"NaN")
    assert (
        live.metric_snapshot({"kai": "http://unused.invalid/metrics"}, None, 1) is None
    )


def test_metrics_count_signal_and_algorithm_exchanges(monkeypatch):
    body = b"""vllm_srun_requests_total{endpoint="/v1/systemone",status="200"} 12
vllm_srun_requests_total{status="200",endpoint="/v1/bundle"} 7
vllm_srun_requests_total{endpoint="/v1/decisions",status="500"} 3
vllm_srun_requests_total{endpoint="/v1/models",status="200"} 100
vllm_srun_requests_total{not_endpoint="/v1/systemone"} 100
vllm_srun_forward_duration_seconds_sum 0.25
"""
    monkeypatch.setattr(
        live.urllib.request,
        "build_opener",
        lambda *_: SimpleNamespace(
            open=lambda *_args, **_kwargs: nullcontext(
                SimpleNamespace(read=lambda _limit: body)
            )
        ),
    )
    endpoints = {"kai": "http://unused.invalid/metrics"}
    before = live.metric_snapshot(endpoints, None, 1)
    assert before["kai"]["physical_calls"] == 22
    # One signal bundle, one answer and one failed native exchange all count.
    body = body.replace(b"} 12\n", b"} 13\n")
    body = body.replace(b"} 7\n", b"} 8\n")
    body = body.replace(b"} 3\n", b"} 4\n")
    after = live.metric_snapshot(endpoints, None, 1)
    assert live.metric_delta(before, after)["kai"]["physical_calls"] == 3


@pytest.mark.parametrize("endpoint", ["classify", "embeddings", "rerank"])
def test_metrics_include_other_native_inference_surfaces(monkeypatch, endpoint):
    body = (
        f'vllm_srun_requests_total{{endpoint="/v1/{endpoint}",status="200"}} 2\n'
        "vllm_srun_forward_duration_seconds_sum 0.25\n"
    ).encode()
    monkeypatch.setattr(
        live.urllib.request,
        "build_opener",
        lambda *_: SimpleNamespace(
            open=lambda *_args, **_kwargs: nullcontext(
                SimpleNamespace(read=lambda _limit: body)
            )
        ),
    )
    assert live._counters("http://unused.invalid/metrics", None, 1) == {
        "physical_calls": 2,
        "forward_seconds": 0.25,
    }


@pytest.mark.parametrize(
    "body",
    [
        b'vllm_srun_requests_total{endpoint="/v1/models"} 100\n'
        b"vllm_srun_forward_duration_seconds_sum 0.25\n",
        b'vllm_srun_requests_total{endpoint="/v1/bundle"} 2\n',
        b"",
    ],
)
def test_metrics_missing_inference_counters_stay_unknown(monkeypatch, body):
    monkeypatch.setattr(
        live.urllib.request,
        "build_opener",
        lambda *_: SimpleNamespace(
            open=lambda *_args, **_kwargs: nullcontext(
                SimpleNamespace(read=lambda _limit: body)
            )
        ),
    )
    assert (
        live.metric_snapshot({"kai": "http://unused.invalid/metrics"}, None, 1) is None
    )


def test_wrong_public_alias_is_unresolved_without_recording_it(
    tmp_path, monkeypatch, suite
):
    setup(monkeypatch, suite)

    def request(_endpoint, _key, payload, _timeout):
        raw = response(payload)
        raw["model"] = "private-wrong-alias"
        return 200, raw, 1.0

    monkeypatch.setattr(live, "request_once", request)
    args = arguments(tmp_path, passes=1)
    live.run(args)
    rows = read_jsonl(args.output_dir / "observations.jsonl")
    assert all(
        not row["public_alias_valid"] and not row["score"]["valid"] for row in rows
    )
    assert "private-wrong-alias" not in str(rows)
