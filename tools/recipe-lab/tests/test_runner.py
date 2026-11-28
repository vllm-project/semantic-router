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


def test_patch_ports_rejects_config_without_listeners(tmp_path):
    cfg = Path(tmp_path) / "candidate.yaml"
    cfg.write_text("version: v0.2\n")
    with pytest.raises(RunnerError, match="no gateway listeners"):
        _patch_ports(cfg, Path(tmp_path) / "work")
