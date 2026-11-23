"""Contract tests: eager validation with actionable errors."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest
import yaml
from recipe_lab.contract import (
    ContractError,
    ExperimentSpec,
    OraclePolicy,
    dataset_content_hash,
    matches_condition,
)

from tests.fixtures import POLICY, RULES_POLICY, VALID_SPEC, VALID_SPLITS_SPEC


def test_valid_spec_roundtrip():
    spec = ExperimentSpec.from_dict(copy.deepcopy(VALID_SPEC))
    assert spec.objective.metric_key == "capability.score"
    assert spec.objective.direction == "maximize"
    assert spec.objective.replay_metric == "routed_accuracy"
    assert spec.workload.profile == "quick"
    assert spec.oracle_policy.expected("math") == "local/model-math"
    # to_dict -> from_dict must round-trip
    spec2 = ExperimentSpec.from_dict(spec.to_dict())
    assert spec2.to_dict() == spec.to_dict()


def test_missing_api_version_rejected():
    raw = copy.deepcopy(VALID_SPEC)
    raw["api_version"] = "recipe-lab/v1beta9"
    with pytest.raises(ContractError, match="api_version"):
        ExperimentSpec.from_dict(raw)


def test_bad_direction_rejected():
    raw = copy.deepcopy(VALID_SPEC)
    raw["objective"]["direction"] = "sideways"
    with pytest.raises(ContractError, match="direction"):
        ExperimentSpec.from_dict(raw)


def test_empty_metric_key_rejected():
    raw = copy.deepcopy(VALID_SPEC)
    raw["objective"]["metric_key"] = "  "
    with pytest.raises(ContractError, match="metric_key"):
        ExperimentSpec.from_dict(raw)


def test_missing_content_hash_rejected():
    raw = copy.deepcopy(VALID_SPEC)
    del raw["workload"]["content_hash"]
    with pytest.raises(ContractError, match="content_hash"):
        ExperimentSpec.from_dict(raw)


def test_negative_min_improvement_rejected():
    raw = copy.deepcopy(VALID_SPEC)
    raw["objective"]["min_improvement_pct"] = -0.5
    with pytest.raises(ContractError, match="min_improvement_pct"):
        ExperimentSpec.from_dict(raw)


def test_replay_metric_default_and_reject():
    # A1: replay_metric defaults to routed_accuracy; unknown values rejected
    raw = copy.deepcopy(VALID_SPEC)
    del raw["objective"]["replay_metric"]
    spec = ExperimentSpec.from_dict(raw)
    assert spec.objective.replay_metric == "routed_accuracy"
    raw["objective"]["replay_metric"] = "latency_p99"
    with pytest.raises(ContractError, match="replay_metric"):
        ExperimentSpec.from_dict(raw)


def test_evaluation_section_is_retired():
    # A5: replay and live are separate commands; the contract refuses the key
    raw = copy.deepcopy(VALID_SPEC)
    raw["evaluation"] = {"search_tier": "replay", "final_tier": "live"}
    with pytest.raises(ContractError, match="'evaluation' is retired"):
        ExperimentSpec.from_dict(raw)
    assert (
        "evaluation"
        not in ExperimentSpec.from_dict(copy.deepcopy(VALID_SPEC)).to_dict()
    )


# ---- B: three-way dataset splits ----


def test_legacy_workload_roundtrips_and_is_dev_only():
    spec = ExperimentSpec.from_dict(copy.deepcopy(VALID_SPEC))
    assert spec.workload.legacy
    assert spec.workload.split_hash("dev") == VALID_SPEC["workload"]["content_hash"]
    assert spec.workload.to_dict() == VALID_SPEC["workload"]


def test_splits_workload_roundtrip():
    spec = ExperimentSpec.from_dict(copy.deepcopy(VALID_SPLITS_SPEC))
    assert not spec.workload.legacy
    assert spec.workload.split_rule == "group by source_id; seed 42; 40/30/30"
    assert sorted(spec.workload.splits) == ["calib", "dev", "frozen_test"]
    assert spec.workload.split_hash("frozen_test").startswith("sha256:3")
    out = spec.to_dict()["workload"]
    assert set(out["splits"]) == {"calib", "dev", "frozen_test"}
    assert "manifest" not in out  # legacy keys gone


def test_splits_require_split_rule():
    raw = copy.deepcopy(VALID_SPLITS_SPEC)
    del raw["workload"]["split_rule"]
    with pytest.raises(ContractError, match="split_rule"):
        ExperimentSpec.from_dict(raw)


def test_splits_require_all_three():
    raw = copy.deepcopy(VALID_SPLITS_SPEC)
    del raw["workload"]["splits"]["frozen_test"]
    with pytest.raises(
        ContractError, match="exactly \\['calib', 'dev', 'frozen_test'\\]"
    ):
        ExperimentSpec.from_dict(raw)


def test_empty_oracle_policy_rejected():
    raw = copy.deepcopy(VALID_SPEC)
    raw["attribution"]["oracle_policy"] = {}
    with pytest.raises(ContractError, match="oracle_policy"):
        ExperimentSpec.from_dict(raw)


def test_policy_coverage_gaps():
    assert POLICY.covers(["math", "code"]) == []
    assert POLICY.covers(["math", "history"]) == ["history"]


def test_policy_rejects_empty_model():
    with pytest.raises(ContractError, match="model"):
        OraclePolicy.from_dict({"math": ""})


# ---- A4: frozen dataset identity ----


def _write_frozen_spec(
    tmp_path, content: bytes = b'{"case_id":"c1"}\n{"case_id":"c2"}\n'
):
    """A spec + dataset pair: workload.content_hash covers the exact bytes."""
    dataset = tmp_path / "datasets" / "dev.jsonl"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_bytes(content)
    spec = copy.deepcopy(VALID_SPEC)
    spec["workload"] = {
        "manifest": "datasets/dev.jsonl",
        "content_hash": dataset_content_hash(dataset),
        "profile": "quick",
    }
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return spec_path, dataset


def test_dataset_content_hash_is_sha256_of_raw_bytes(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_bytes(b"abc\n")
    assert (
        dataset_content_hash(path) == "sha256:" + hashlib.sha256(b"abc\n").hexdigest()
    )


def test_rules_policy_roundtrip():
    raw = RULES_POLICY.to_dict()
    assert raw["default"] == {"decision": "simple_general", "models": ["m/general"]}
    assert OraclePolicy.from_dict(raw).to_dict() == raw


def test_rules_policy_first_match_wins():
    complex_route = RULES_POLICY.evaluate({"domain": "code", "difficulty": 4})
    assert complex_route.decision == "domain_code_complex"
    assert complex_route.models == ("m/pro", "m/max")
    # first rule no longer matches (difficulty below 3): second rule wins
    assert (
        RULES_POLICY.evaluate({"domain": "code", "difficulty": 1}).decision
        == "domain_code"
    )
    # no rule matches: default
    assert RULES_POLICY.evaluate({"domain": "math"}).decision == "simple_general"
    # a missing fact never satisfies a condition
    assert RULES_POLICY.evaluate({}).decision == "simple_general"


def test_rules_policy_without_default_is_gap():
    policy = OraclePolicy.from_dict(
        {"rules": [{"when": {"domain": "code"}, "expect": {"decision": "domain_code"}}]}
    )
    assert policy.evaluate({"domain": "math"}) is None
    # string truths cannot be resolved by a rules-form policy at all
    assert policy.covers(["math"]) == ["math"]
    assert policy.expected("domain_code") is None


def test_rules_policy_rejects_mixed_forms():
    with pytest.raises(ContractError, match="not both"):
        OraclePolicy.from_dict(
            {
                "math": "local/model-math",
                "rules": [
                    {"when": {"domain": "code"}, "expect": {"decision": "domain_code"}}
                ],
            }
        )


def test_rules_policy_rejects_bad_condition():
    with pytest.raises(ContractError, match="predicate"):
        OraclePolicy.from_dict(
            {
                "rules": [
                    {
                        "when": {"difficulty": {"gte": "high"}},
                        "expect": {"decision": "d"},
                    }
                ]
            }
        )
    with pytest.raises(ContractError, match="predicate op"):
        OraclePolicy.from_dict(
            {
                "rules": [
                    {"when": {"difficulty": {"within": 3}}, "expect": {"decision": "d"}}
                ]
            }
        )
    with pytest.raises(ContractError, match="when"):
        OraclePolicy.from_dict({"rules": [{"when": {}, "expect": {"decision": "d"}}]})


def test_rules_policy_rejects_bad_models():
    with pytest.raises(ContractError, match="models"):
        OraclePolicy.from_dict(
            {
                "rules": [
                    {
                        "when": {"domain": "code"},
                        "expect": {"decision": "d", "models": "m/code"},
                    }
                ]
            }
        )


def test_rules_policy_rejects_missing_expect_decision():
    with pytest.raises(ContractError, match="decision"):
        OraclePolicy.from_dict({"default": {"models": ["m"]}})


def test_matches_condition_semantics():
    assert matches_condition("code", "code")
    assert not matches_condition("math", "code")
    assert not matches_condition(None, "code")
    assert matches_condition(3, 3)
    assert not matches_condition(3, 4)
    assert matches_condition(3.0, {"gte": 3})
    assert not matches_condition(2.9, {"gte": 3})
    assert matches_condition(1.0, {"gte": 0.5, "lt": 2})
    assert not matches_condition(2.0, {"gt": 2})
    assert not matches_condition(2.0, {"lt": 2, "gte": 3})
    assert not matches_condition("code", {"gte": 1})
    assert not matches_condition(3, "code")


def test_spec_with_rules_policy_roundtrip():
    raw = copy.deepcopy(VALID_SPEC)
    raw["attribution"]["oracle_policy"] = RULES_POLICY.to_dict()
    spec = ExperimentSpec.from_dict(raw)
    assert (
        spec.oracle_policy.evaluate({"domain": "code", "difficulty": 4}).decision
        == "domain_code_complex"
    )
    assert ExperimentSpec.from_dict(spec.to_dict()).to_dict() == spec.to_dict()
