"""Threshold-search proposer tests: determinism, clamping, one-variable rule."""

from __future__ import annotations

import copy

import pytest
import yaml
from recipe_lab.contract import ContractError
from recipe_lab.proposers.threshold_search import propose_thresholds, write_candidates

from tests.fixtures import VALID_SPEC  # noqa: F401  (keeps import sanity)

BASE_CONFIG = {
    "version": "v0.3",
    "routing": {
        "signals": {
            "decision": [
                {
                    "name": "depth",
                    "predicate": {"gte": 0.6},
                }
            ]
        },
        "decisions": [{"name": "cloud_frontier_reasoning"}],
    },
}


def test_propose_generates_one_variable_candidates():
    candidates = propose_thresholds(BASE_CONFIG, "depth", [-0.1, 0.05, 0.1])
    assert [c.new_value for c in candidates] == [0.5, 0.65, 0.7]
    for candidate in candidates:
        assert candidate.old_value == 0.6
        # exactly one variable changed per candidate
        assert (
            candidate.config["routing"]["signals"]["decision"][0]["predicate"]["gte"]
            == candidate.new_value
        )
        assert (
            candidate.config["routing"]["decisions"]
            == BASE_CONFIG["routing"]["decisions"]
        )


def test_propose_deterministic():
    a = propose_thresholds(BASE_CONFIG, "depth", [-0.1, 0.05])
    b = propose_thresholds(BASE_CONFIG, "depth", [-0.1, 0.05])
    assert [c.new_value for c in a] == [c.new_value for c in b]
    assert yaml.safe_dump(a[0].config) == yaml.safe_dump(b[0].config)


def test_propose_clamps_and_drops_baseline():
    # -0.6 and +0.6 clamp to 0.0 / 1.0; a delta landing on the baseline is dropped
    candidates = propose_thresholds(BASE_CONFIG, "depth", [-0.6, 0.0, 0.4, 0.6])
    values = [c.new_value for c in candidates]
    assert values == [0.0, 1.0]


def test_propose_unknown_signal_rejected():
    with pytest.raises(ContractError, match="not found"):
        propose_thresholds(BASE_CONFIG, "no-such-signal", [0.05])


def test_propose_signal_without_gte_rejected():
    config = copy.deepcopy(BASE_CONFIG)
    config["routing"]["signals"]["decision"][0] = {
        "name": "depth",
        "predicate": {"lt": 0.6},
    }
    with pytest.raises(ContractError, match="gte"):
        propose_thresholds(config, "depth", [0.05])


def test_write_candidates_writes_yaml(tmp_path):
    candidates = propose_thresholds(BASE_CONFIG, "depth", [-0.1])
    paths = write_candidates(candidates, tmp_path)
    assert len(paths) == 1
    loaded = yaml.safe_load(paths[0].read_text(encoding="utf-8"))
    assert loaded["routing"]["signals"]["decision"][0]["predicate"]["gte"] == 0.5
