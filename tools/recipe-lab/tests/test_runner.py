"""Runner unit tests (pure functions; stack tests run as integration)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml
from recipe_lab.contract import Constraint, ContractError, OraclePolicy
from recipe_lab.runner import (
    RunnerEnv,
    RunnerError,
    RunnerTimeoutError,
    _patch_ports,
    attribution_delta,
    evaluate_candidate,
    replay_constraint_evidence,
    replay_metric_value,
    routed_cost_index,
)


def test_attribution_delta_counts_fixed_errors():
    baseline = {
        "class_counts": {
            "E1_signal": 3,
            "E2_policy": 2,
            "E3_capability": 5,
            "decision_mismatch": 4,
        }
    }
    candidate = {
        "class_counts": {
            "E1_signal": 1,
            "E2_policy": 2,
            "E3_capability": 6,
            "decision_mismatch": 1,
        }
    }
    assert attribution_delta(baseline, candidate) == {
        "e1_fixed": 2,
        "e2_fixed": 0,
        "decision_mismatch_fixed": 3,
        "e3_remaining": 6,
    }


def test_attribution_delta_never_negative():
    baseline = {"class_counts": {"E1_signal": 1}}
    candidate = {"class_counts": {"E1_signal": 4}}
    assert attribution_delta(baseline, candidate) == {
        "e1_fixed": 0,
        "e2_fixed": 0,
        "decision_mismatch_fixed": 0,
        "e3_remaining": 0,
    }


def test_replay_metric_value_rounds_and_rejects_unknown():
    # A1: the contracted replay metric maps to the runner's summary key
    assert (
        replay_metric_value({"accuracy": 0.8333333333}, "routed_accuracy") == 0.833333
    )
    assert replay_metric_value({"accuracy": None}, "routed_accuracy") is None
    assert replay_metric_value({}, "routed_accuracy") is None
    with pytest.raises(ContractError, match="unknown replay metric"):
        replay_metric_value({"accuracy": 1.0}, "capability.score")


# ---- E2: replay cost proxy ----


def test_routed_cost_index_averages_answered_samples_only():
    rows = [
        {"case_id": "a", "routed": "local/model-large", "answer_ok": True},
        {"case_id": "b", "routed": "local/model-small", "answer_ok": False},
        {
            "case_id": "c",
            "routed": "local/model-large",
            "answer_ok": None,
        },  # unanswered: excluded
        {
            "case_id": "d",
            "routed": "local/model-small",
            "answer_ok": None,
        },  # no_bank_entry: excluded
    ]
    index, unknown = routed_cost_index(
        rows, {"local/model-large": 10, "local/model-small": 1}
    )
    assert index == 5.5
    assert unknown == []


def test_routed_cost_index_unknown_model_is_never_zero():
    # sr-bench rule: unknown usage must not count as zero. A routed model
    # missing from cost_model makes the whole index incomputable.
    rows = [
        {"case_id": "a", "routed": "local/model-large", "answer_ok": True},
        {"case_id": "b", "routed": "local/model-x", "answer_ok": True},
        {
            "case_id": "c",
            "routed": "local/model-mystery",
            "answer_ok": None,
        },  # unanswered: ignored
    ]
    index, unknown = routed_cost_index(rows, {"local/model-large": 10})
    assert index is None
    assert unknown == ["local/model-x"]


def test_routed_cost_index_without_answered_rows():
    index, unknown = routed_cost_index(
        [{"routed": "local/model-large", "answer_ok": None}], {"local/model-large": 10}
    )
    assert index is None
    assert unknown == []


def _cost_constraint(**overrides):
    raw = {
        "name": "cost-not-worse",
        "metric": "total_spend_usd",
        "op": "lte",
        "bound": "baseline",
        "tolerance_pct": 0.0,
    }
    raw.update(overrides)
    return Constraint.from_dict(raw, 0)


def test_replay_constraint_evidence_flags_violation():
    evidence, warning = replay_constraint_evidence(
        (_cost_constraint(),), candidate_index=5.5, baseline_index=5.0
    )
    assert warning is True
    assert len(evidence) == 1
    entry = evidence[0]
    assert entry["status"] == "violated"
    assert entry["proxy"] == "routed_cost_index"  # the substitution is explicit
    assert entry["baseline"] == 5.0 and entry["candidate"] == 5.5
    assert entry["bound_value"] == 5.0
    assert entry["reason"] == "constraint cost-not-worse violated (replay cost proxy)"


def test_replay_constraint_evidence_numeric_bound_pass():
    evidence, warning = replay_constraint_evidence(
        (_cost_constraint(bound=12.0),), candidate_index=5.5, baseline_index=None
    )
    assert warning is False
    assert evidence[0]["status"] == "pass"
    assert evidence[0]["bound_value"] == 12.0


def test_replay_constraint_evidence_baseline_bound_without_baseline():
    # no baseline arm evaluated -> not_evaluated, never a guess
    evidence, warning = replay_constraint_evidence(
        (_cost_constraint(),), candidate_index=5.5, baseline_index=None
    )
    assert warning is False
    assert evidence[0]["status"] == "not_evaluated"
    assert evidence[0]["bound_value"] is None


# ---- A5: per-trial budget ----


def test_evaluate_candidate_trial_timeout(tmp_path):
    """A stack that never becomes ready must surface as RunnerTimeoutError within
    the budget deadline, not as a generic not-ready failure."""
    router = Path(tmp_path) / "router.sh"
    router.write_text("#!/bin/sh\nsleep 60\n")
    router.chmod(0o755)
    mock = Path(tmp_path) / "mock.py"
    mock.write_text("import time\nmock_port = 0\ntime.sleep(60)\n")
    cfg = Path(tmp_path) / "candidate.yaml"
    cfg.write_text(
        "version: v0.2\n"
        "listeners:\n"
        "  - name: http\n"
        "    address: 0.0.0.0\n"
        "    port: 8899\n"
    )
    env = RunnerEnv(
        router_bin=str(router),
        mock_backend_script=str(mock),
        python_bin=sys.executable,
        work_dir=Path(tmp_path) / "work",
        ready_timeout_s=30,
    )
    with pytest.raises(RunnerTimeoutError, match="trial exceeded"):
        evaluate_candidate(
            candidate_config=cfg,
            cases=[{"case_id": "c1", "prompt": "p", "truth": "math"}],
            oracle_policy=OraclePolicy.from_dict({"math": "local/model-math"}),
            bank={},
            env=env,
            trial_timeout_s=2.0,
        )


# ---- per-trial port allocation ----


def test_rejected_case_records_exact_evidence_and_no_partial_score(
    tmp_path, monkeypatch
):
    import io
    import json
    import urllib.error
    import urllib.request

    from recipe_lab import runner

    cfg = tmp_path / "candidate.yaml"
    cfg.write_text("routing: {}")
    cases = [{"case_id": i, "prompt": "same", "truth": "test"} for i in ("good", "bad")]
    code = "no_eligible_model"
    cause = "no assigned model satisfies the effective request capabilities and budget"

    def fake_post(request, timeout):
        body = json.dumps(
            {"error": {"code": code, "message": "no model is eligible"}}
        ).encode()
        raise urllib.error.HTTPError(
            request.full_url, 503, "unavailable", {}, io.BytesIO(body)
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_post)

    def probe(env, config, cases, *, request_id_prefix, http_errors, **kwargs):
        good, bad = [request_id_prefix + c["case_id"] for c in cases]
        status = runner._send_prompt(
            "http://localhost", "auto", "same", 1, bad, http_errors
        )
        (tmp_path / "router.log").write_text(
            "\n".join(
                json.dumps(
                    {
                        "event": "selection_rejected",
                        "request_id": rid,
                        "code": code,
                        "error": message,
                    }
                )
                for rid, message in [(bad, cause), ("another-request", "unrelated")]
            )
        )
        return (
            [
                {
                    "request_id": bad,
                    "decision": "hard",
                    "response_status": 503,
                    "terminal_reason": "selection_rejected",
                    "lifecycle_state": "failed",
                },
                {"request_id": good, "decision": "fast", "selected_model": "model-a"},
            ],
            {"bad": status, "good": 200},
            {},
        )

    monkeypatch.setattr(runner, "probe_records", probe)
    env = RunnerEnv(
        router_bin="unused",
        mock_backend_script="unused",
        python_bin="unused",
        work_dir=tmp_path,
    )
    result = evaluate_candidate(
        candidate_config=cfg,
        env=env,
        cases=cases,
        oracle_policy=OraclePolicy.from_dict({"test": "model-a"}),
        bank={},
    )
    assert result["incomplete"] and "summary" not in result
    [failure] = result["replay_failures"]
    assert result["candidate_tag"] == runner.candidate_tag(cfg.read_text())
    rid = runner.request_id_for_case(result["request_tag"], "bad")
    assert failure["case_id"] == "bad" and failure["request_id"] == rid
    assert failure["http_status"] == 503 and failure["http_error"]["code"] == code
    assert failure["replays"][0]["terminal_reason"] == "selection_rejected"
    assert failure["router_rejections"] == [
        {"request_id": rid, "code": code, "error": cause}
    ]


@pytest.mark.parametrize(
    "backend,rid,eligible",
    [
        ("memory", "rl-foreign-c1", False),
        ("redis", "755aaf97-a582-408d-87ff-b897a6c4d43b", False),
        ("memory", "755aaf97-a582-408d-87ff-b897a6c4d43b", True),
    ],
)
def test_restart_prompt_fallback_only_accepts_legacy_dedicated_records(
    tmp_path, monkeypatch, backend, rid, eligible
):
    from recipe_lab import runner

    cfg = tmp_path / "candidate.yaml"
    cfg.write_text(
        f"global: {{services: {{router_replay: {{store_backend: {backend}}}}}}}"
    )
    case = {"case_id": "c1", "prompt": "same", "truth": "test"}
    record = {
        "request_id": rid,
        "prompt": "same",
        "decision": "test",
        "selected_model": "a",
    }
    monkeypatch.setattr(
        runner, "probe_records", lambda *a, **kw: ([record], {"c1": 200}, {})
    )
    result = evaluate_candidate(
        candidate_config=cfg,
        env=RunnerEnv(
            router_bin="x", mock_backend_script="x", python_bin="x", work_dir=tmp_path
        ),
        cases=[case],
        oracle_policy=OraclePolicy.from_dict({"test": "a"}),
        bank={},
    )
    if eligible:
        assert "summary" in result
    else:
        assert result["incomplete"]
        assert result["replay_failures"][0]["replay_count"] == 0


@pytest.mark.parametrize("backend", ["redis", None])
def test_restart_shared_and_unknown_stores_isolate_repeated_trials(
    tmp_path, monkeypatch, backend
):
    from recipe_lab import runner

    cfg = tmp_path / "candidate.yaml"
    cfg.write_text(
        "routing: {}"
        if backend is None
        else "global: {services: {router_replay: {store_backend: redis}}}"
    )
    previous = []

    def probe(*a, request_id_prefix, **kw):
        record = {
            "request_id": request_id_prefix + "id",
            "prompt": "p",
            "decision": "test",
            "selected_model": "a",
        }
        if not previous:
            previous.append(record)
        # On retry only the previous trial's record is visible.
        return list(previous), {"id": 200}, {}

    monkeypatch.setattr(runner, "probe_records", probe)
    args = dict(
        candidate_config=cfg,
        env=RunnerEnv(
            router_bin="x", mock_backend_script="x", python_bin="x", work_dir=tmp_path
        ),
        cases=[{"case_id": "id", "prompt": "p", "truth": "test"}],
        oracle_policy=OraclePolicy.from_dict({"test": "a"}),
        bank={},
    )
    first = evaluate_candidate(**args)
    stale = evaluate_candidate(**args)
    assert "summary" in first and stale["incomplete"]
    assert first["candidate_tag"] == stale["candidate_tag"]
    assert first["request_tag"] != stale["request_tag"]
    assert stale["replay_failures"][0]["replay_count"] == 0


def test_patch_ports_assigns_per_trial_ports(tmp_path):
    """Gateway + management ports are re-assigned per trial; the original
    candidate file stays untouched."""
    cfg = Path(tmp_path) / "candidate.yaml"
    cfg.write_text(
        "version: v0.2\n"
        "listeners:\n"
        "  - name: http\n"
        "    address: 0.0.0.0\n"
        "    port: 8899\n"
    )
    patched, router_port, management_port = _patch_ports(cfg, Path(tmp_path) / "work")
    patched_config = yaml.safe_load(patched.read_text(encoding="utf-8"))
    assert patched_config["listeners"][0]["port"] == router_port != 8899
    assert 1024 <= management_port <= 65535
    assert (
        yaml.safe_load(cfg.read_text(encoding="utf-8"))["listeners"][0]["port"] == 8899
    )


@pytest.mark.parametrize(
    "connection", [{}, {"database": "lab", "user": "router"}, None]
)
def test_patch_ports_preserves_authored_postgres(tmp_path, connection):
    replay = {"store_backend": "postgres"}
    if connection is not None:
        replay["postgres"] = connection
    cfg = tmp_path / "candidate.yaml"
    original = yaml.safe_dump(
        {
            "listeners": [{"port": 8899}],
            "global": {"services": {"router_replay": replay}},
        }
    )
    cfg.write_text(original, encoding="utf-8")
    patched, _, _ = _patch_ports(cfg, tmp_path / "work")
    actual = yaml.safe_load(patched.read_text(encoding="utf-8"))["global"]["services"][
        "router_replay"
    ]
    assert actual == replay
    assert cfg.read_text(encoding="utf-8") == original


def test_patch_ports_rejects_config_without_listeners(tmp_path):
    cfg = Path(tmp_path) / "candidate.yaml"
    cfg.write_text("version: v0.2\n")
    with pytest.raises(RunnerError, match="no gateway listeners"):
        _patch_ports(cfg, Path(tmp_path) / "work")
