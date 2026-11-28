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
    check_runnable_split,
    dataset_content_hash,
    matches_condition,
)
from tests.fixtures import (
    POLICY,
    RULES_POLICY,
    VALID_SPEC,
    VALID_SPLITS_SPEC,
    VALID_WORKLOADS_SPEC,
)


def test_valid_spec_roundtrip():
    spec = ExperimentSpec.from_dict(copy.deepcopy(VALID_SPEC))
    assert spec.objective.metric_key == "capability.score"
    assert spec.objective.direction == "maximize"
    assert spec.objective.replay_metric == "routed_accuracy"
    assert spec.workload.profile == "quick"
    assert spec.oracle_policy.expected("math") == "local/model-math"
    assert len(spec.gates) == 1
    assert spec.budget.rounds == 5
    assert spec.stack is not None and spec.stack.mode == "external"
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


def test_objective_constraints_parse_and_roundtrip():
    # E: constraints are economic conditions declared on the objective
    raw = copy.deepcopy(VALID_SPEC)
    raw["objective"]["constraints"] = [
        {
            "name": "cost-not-worse",
            "metric": "total_spend_usd",
            "op": "lte",
            "bound": "baseline",
            "tolerance_pct": 5.0,
        }
    ]
    spec = ExperimentSpec.from_dict(raw)
    [constraint] = spec.objective.constraints
    assert (
        constraint.op == "lte"
        and constraint.bound == "baseline"
        and constraint.tolerance_pct == 5.0
    )
    spec2 = ExperimentSpec.from_dict(spec.to_dict())
    assert spec2.to_dict() == spec.to_dict()
    # a numeric bound is legal too
    raw["objective"]["constraints"] = [
        {
            "name": "keep-recall",
            "metric": "guard.pii.recall",
            "op": "gte",
            "bound": 0.95,
        }
    ]
    spec = ExperimentSpec.from_dict(raw)
    assert spec.objective.constraints[0].bound == 0.95


def test_objective_constraints_reject_invalid():
    raw = copy.deepcopy(VALID_SPEC)
    base = {
        "name": "cost-not-worse",
        "metric": "total_spend_usd",
        "op": "lte",
        "bound": "baseline",
    }
    for field, value, match in [
        ("op", "lt", "op"),
        ("bound", "guess", "bound"),
        ("tolerance_pct", -1, "tolerance_pct"),
        ("name", "", "name"),
        ("metric", "", "metric"),
    ]:
        bad = dict(base)
        bad[field] = value
        raw["objective"]["constraints"] = [bad]
        with pytest.raises(ContractError, match=match):
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


def test_check_runnable_split_quarantines_frozen_test():
    with pytest.raises(ContractError, match="frozen_test is quarantined"):
        check_runnable_split("frozen_test")
    check_runnable_split("dev")
    check_runnable_split("calib")


def _write_splits_dataset(tmp_path, overlap_with_dev=False):
    """Three disjoint slice files; optionally leak a dev case into calib."""
    rows = {
        "calib": [{"case_id": "c1", "prompt": "p1"}, {"case_id": "c2", "prompt": "p2"}],
        "dev": [{"case_id": "c3", "prompt": "p3"}],
        "frozen_test": [{"case_id": "c4", "prompt": "p4"}],
    }
    if overlap_with_dev:
        rows["calib"].append({"case_id": "c3", "prompt": "p3"})
    spec = copy.deepcopy(VALID_SPLITS_SPEC)
    for name, case_rows in rows.items():
        target = tmp_path / f"{name}.jsonl"
        target.write_text(
            "".join(json.dumps(r) + "\n" for r in case_rows), encoding="utf-8"
        )
        spec["workload"]["splits"][name]["manifest"] = f"{name}.jsonl"
        spec["workload"]["splits"][name]["content_hash"] = dataset_content_hash(target)
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return spec_path


def test_verify_workload_checks_all_splits_and_disjointness(tmp_path):
    spec_path = _write_splits_dataset(tmp_path)
    spec = ExperimentSpec.from_yaml(spec_path)
    assert spec.verify_workload()["dev"] == spec.workload.split_hash("dev")  # type: ignore[index]


def test_verify_workload_rejects_overlapping_splits(tmp_path):
    spec_path = _write_splits_dataset(tmp_path, overlap_with_dev=True)
    spec = ExperimentSpec.from_yaml(spec_path)
    with pytest.raises(ContractError, match="overlap between 'calib' and 'dev'"):
        spec.verify_workload()


def test_verify_workload_split_selects_one_slice(tmp_path):
    spec_path = _write_splits_dataset(tmp_path)
    spec = ExperimentSpec.from_yaml(spec_path)
    assert spec.verify_workload(split="dev") == spec.workload.split_hash("dev")
    # unknown split names are refused
    with pytest.raises(ContractError, match="unknown workload split"):
        spec.verify_workload(split="train")


# ---- X3: dual-format splits (cases + srbench_dataset) ----


def _write_srbench_splits_dataset(base, mutate_srbench="none"):
    """Splits dataset (written into the existing dir `base`) whose dev split
    also declares an sr-bench rendering of the same cases.

    mutate_srbench: 'hash' drifts the declared srbench hash, 'ids' swaps a
    row id so the rendering no longer pairs with the cases manifest.
    """
    spec_path = _write_splits_dataset(base)
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    srbench_rows = [
        {
            "id": "c3" if mutate_srbench != "ids" else "other-id",
            "benchmark": "recipe-lab",
            "messages": [{"role": "user", "content": "p3"}],
            "answer": "math",
        }
    ]
    dataset = base / "dev.srbench.jsonl"
    dataset.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in srbench_rows),
        encoding="utf-8",
    )
    digest = dataset_content_hash(dataset)
    if mutate_srbench == "hash":
        digest = "sha256:" + "f" * 64
    spec["workload"]["splits"]["dev"]["srbench_dataset"] = "dev.srbench.jsonl"
    spec["workload"]["splits"]["dev"]["srbench_sha256"] = digest
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return spec_path


def test_srbench_split_fields_roundtrip_and_default_absent():
    raw = copy.deepcopy(VALID_SPLITS_SPEC)
    spec = ExperimentSpec.from_dict(raw)
    assert spec.workload.split_srbench_dataset("dev") is None
    assert spec.workload.split_srbench_hash("dev") is None
    # a split without srbench fields round-trips without the new keys
    assert "srbench_dataset" not in spec.to_dict()["workload"]["splits"]["dev"]
    # live binds the cases hash when no rendering is declared (legacy form)
    assert spec.workload.live_dataset_hash("dev") == spec.workload.split_hash("dev")

    declared = copy.deepcopy(VALID_SPLITS_SPEC)
    declared["workload"]["splits"]["dev"]["srbench_dataset"] = "dev.srbench.jsonl"
    declared["workload"]["splits"]["dev"]["srbench_sha256"] = "sha256:" + "b" * 64
    spec2 = ExperimentSpec.from_dict(declared)
    assert spec2.workload.split_srbench_dataset("dev") == "dev.srbench.jsonl"
    assert spec2.workload.split_srbench_hash("dev") == "sha256:" + "b" * 64
    out = spec2.to_dict()["workload"]["splits"]["dev"]
    assert out["srbench_dataset"] == "dev.srbench.jsonl"
    assert out["srbench_sha256"] == "sha256:" + "b" * 64
    # the other splits stay clean and keep the legacy cases binding
    assert "srbench_dataset" not in spec2.to_dict()["workload"]["splits"]["calib"]
    assert spec2.workload.live_dataset_hash("dev") == "sha256:" + "b" * 64
    assert spec2.workload.live_dataset_hash("calib") == (
        spec2.workload.split_hash("calib")
    )


def test_srbench_split_fields_require_both():
    raw = copy.deepcopy(VALID_SPLITS_SPEC)
    raw["workload"]["splits"]["dev"]["srbench_dataset"] = "dev.srbench.jsonl"
    with pytest.raises(ContractError, match="srbench_dataset and srbench_sha256"):
        ExperimentSpec.from_dict(raw)
    raw2 = copy.deepcopy(VALID_SPLITS_SPEC)
    raw2["workload"]["splits"]["dev"]["srbench_sha256"] = "sha256:" + "b" * 64
    with pytest.raises(ContractError, match="srbench_dataset and srbench_sha256"):
        ExperimentSpec.from_dict(raw2)


def test_legacy_workload_binds_srbench_rendering(tmp_path):
    # X3 on the legacy single-workload form: the workload declares the
    # sr-bench rendering of the same cases, verify_workload checks its hash
    # and 1:1 pairing, and live binds the rendering's hash.
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        "".join(
            json.dumps({"case_id": cid, "prompt": f"p{n}", "truth": "math"}) + "\n"
            for n, cid in enumerate(("c1", "c2"), 1)
        ),
        encoding="utf-8",
    )
    rendering = tmp_path / "cases.srbench.jsonl"
    rendering.write_text(
        "".join(
            json.dumps(
                {
                    "id": cid,
                    "benchmark": "recipe-lab",
                    "messages": [{"role": "user", "content": f"p{n}"}],
                    "answer": "math",
                },
                sort_keys=True,
            )
            + "\n"
            for n, cid in enumerate(("c1", "c2"), 1)
        ),
        encoding="utf-8",
    )
    raw = copy.deepcopy(VALID_SPEC)
    raw["workload"]["manifest"] = "cases.jsonl"
    raw["workload"]["content_hash"] = dataset_content_hash(cases)
    raw["workload"]["srbench_dataset"] = "cases.srbench.jsonl"
    raw["workload"]["srbench_sha256"] = dataset_content_hash(rendering)
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    spec = ExperimentSpec.from_yaml(spec_path)
    assert spec.workload.split_srbench_dataset("dev") == "cases.srbench.jsonl"
    assert spec.workload.live_dataset_hash("dev") == dataset_content_hash(rendering)
    # the declared hashes and the 1:1 pairing verify against the real files
    assert spec.verify_workload(split="dev") == dataset_content_hash(cases)
    # the round-trip keeps the legacy single-workload shape with the new keys
    assert spec.to_dict()["workload"] == raw["workload"]


def test_verify_workload_checks_srbench_hash_and_pairing(tmp_path):
    # the happy path: the rendering re-hashes to its declared hash and pairs
    spec_path = _write_srbench_splits_dataset(tmp_path)
    spec = ExperimentSpec.from_yaml(spec_path)
    assert spec.verify_workload(split="dev") == spec.workload.split_hash("dev")

    # a drifted rendering is rejected like any other frozen dataset
    drift = tmp_path / "drift"
    drift.mkdir()
    with pytest.raises(ContractError, match="srbench dataset hash mismatch"):
        ExperimentSpec.from_yaml(
            _write_srbench_splits_dataset(drift, mutate_srbench="hash")
        ).verify_workload(split="dev")

    # a rendering of a different case set is rejected even when it hashes fine
    ids = tmp_path / "ids"
    ids.mkdir()
    with pytest.raises(ContractError, match="does not pair 1:1"):
        ExperimentSpec.from_yaml(
            _write_srbench_splits_dataset(ids, mutate_srbench="ids")
        ).verify_workload(split="dev")


def test_verify_workload_missing_srbench_dataset_is_actionable(tmp_path):
    spec_path = _write_srbench_splits_dataset(tmp_path)
    (tmp_path / "dev.srbench.jsonl").unlink()
    spec = ExperimentSpec.from_yaml(spec_path)
    with pytest.raises(ContractError, match="srbench dataset not found"):
        spec.verify_workload(split="dev")


def test_empty_oracle_policy_rejected():
    raw = copy.deepcopy(VALID_SPEC)
    raw["attribution"]["oracle_policy"] = {}
    with pytest.raises(ContractError, match="oracle_policy"):
        ExperimentSpec.from_dict(raw)


def test_bad_gate_floor_rejected():
    raw = copy.deepcopy(VALID_SPEC)
    raw["gates"][0]["floor"] = "kinda-baseline"
    with pytest.raises(ContractError, match="floor"):
        ExperimentSpec.from_dict(raw)


def test_zero_rounds_rejected():
    raw = copy.deepcopy(VALID_SPEC)
    raw["budget"]["rounds"] = 0
    with pytest.raises(ContractError, match="rounds"):
        ExperimentSpec.from_dict(raw)


def test_bad_stack_mode_rejected():
    raw = copy.deepcopy(VALID_SPEC)
    raw["stack"]["mode"] = "possessed"
    with pytest.raises(ContractError, match=r"stack.mode"):
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


def test_verify_workload_passes_on_exact_bytes(tmp_path):
    spec_path, _ = _write_frozen_spec(tmp_path)
    spec = ExperimentSpec.from_yaml(spec_path)
    assert spec.verify_workload() == {"dev": spec.workload.split_hash("dev")}
    assert spec.verify_workload(split="dev") == spec.workload.split_hash("dev")


def test_verify_workload_rejects_one_byte_change(tmp_path):
    spec_path, dataset = _write_frozen_spec(tmp_path)
    dataset.write_bytes(dataset.read_bytes() + b'{"case_id":"c3"}\n')  # dataset drifted
    spec = ExperimentSpec.from_yaml(spec_path)
    with pytest.raises(ContractError, match=r"dataset hash mismatch.*spec=.*file="):
        spec.verify_workload()


def test_verify_workload_missing_manifest_is_actionable(tmp_path):
    spec_path, _ = _write_frozen_spec(tmp_path)
    spec = ExperimentSpec.from_yaml(spec_path)
    (tmp_path / "datasets" / "dev.jsonl").unlink()
    with pytest.raises(ContractError, match="dataset manifest not found"):
        spec.verify_workload()


def test_require_dataset_file_rejects_drifted_cases(tmp_path):
    spec_path, dataset = _write_frozen_spec(tmp_path)
    spec = ExperimentSpec.from_yaml(spec_path)
    drifted = tmp_path / "other.jsonl"
    drifted.write_bytes(b'{"case_id":"x"}\n')
    with pytest.raises(ContractError, match="cases hash mismatch"):
        spec.require_dataset_file(drifted, "cases")
    with pytest.raises(ContractError, match="cannot read cases"):
        spec.require_dataset_file(tmp_path / "missing.jsonl", "cases")
    # the exact dataset bytes pass
    assert spec.require_dataset_file(dataset, "cases").startswith("sha256:")


# --- F2: multi-signal oracle policy -----------------------------------------


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


# ---- E2: optional top-level cost_model ----


def test_cost_model_roundtrip():
    raw = {
        **copy.deepcopy(VALID_SPEC),
        "cost_model": {"local/model-large": 10, "local/model-small": 1},
    }
    spec = ExperimentSpec.from_dict(raw)
    assert spec.cost_model == {"local/model-large": 10.0, "local/model-small": 1.0}
    assert spec.to_dict()["cost_model"] == {
        "local/model-large": 10.0,
        "local/model-small": 1.0,
    }


def test_cost_model_absent_by_default():
    spec = ExperimentSpec.from_dict(copy.deepcopy(VALID_SPEC))
    assert spec.cost_model is None
    assert "cost_model" not in spec.to_dict()


@pytest.mark.parametrize(
    "bad, message",
    [
        ([("local/model-a", 1)], "cost_model must be a mapping"),
        ({}, "cost_model must not be empty"),
        ({"local/model-a": 0}, "cost_model\\[.{0,40}\\] must be > 0"),
        ({"local/model-a": -1}, "cost_model\\[.{0,40}\\] must be > 0"),
        ({"local/model-a": "cheap"}, "must be a finite number"),
        ({"  ": 1}, "non-empty model names"),
    ],
)
def test_cost_model_validation_rejects(bad, message):
    raw = {**copy.deepcopy(VALID_SPEC), "cost_model": bad}
    with pytest.raises(ContractError, match=message):
        ExperimentSpec.from_dict(raw)


# ---- M4: reference arms and cost basis ----


def _ref_spec() -> dict:
    raw = copy.deepcopy(VALID_SPEC)
    raw["references"] = {"strong": "model-a", "weak": "model-b"}
    raw["cost_basis"] = {
        "source": "provider_pricing",
        "assumed_pricing": {
            "model-a": {"prompt_per_1m": 1.0, "completion_per_1m": 2.0}
        },
    }
    return raw


def test_references_and_cost_basis_roundtrip():
    spec = ExperimentSpec.from_dict(_ref_spec())
    assert spec.references is not None
    assert spec.references.strong == "model-a"
    assert spec.references.weak == "model-b"
    assert spec.cost_basis is not None
    assert spec.cost_basis.source == "provider_pricing"
    assert spec.cost_basis.assumed_pricing == {
        "model-a": {"prompt_per_1m": 1.0, "completion_per_1m": 2.0}
    }
    out = spec.to_dict()
    assert out["references"] == {"strong": "model-a", "weak": "model-b"}
    assert "assumed_pricing" in out["cost_basis"]
    spec2 = ExperimentSpec.from_dict(out)
    assert spec2.to_dict() == out


def test_references_absent_by_default():
    spec = ExperimentSpec.from_dict(copy.deepcopy(VALID_SPEC))
    assert spec.references is None
    assert spec.cost_basis is None
    assert "references" not in spec.to_dict()
    assert "cost_basis" not in spec.to_dict()


def test_references_rejects_empty_or_unknown():
    raw = copy.deepcopy(VALID_SPEC)
    raw["references"] = {}
    with pytest.raises(ContractError, match="references"):
        ExperimentSpec.from_dict(raw)
    raw["references"] = {"strong": "model-a", "weak": "model-b", "middle": "x"}
    with pytest.raises(ContractError, match="middle"):
        ExperimentSpec.from_dict(raw)
    raw["references"] = {"strong": " ", "weak": "model-b"}
    with pytest.raises(ContractError, match="strong"):
        ExperimentSpec.from_dict(raw)


def test_cost_basis_rejects_bad_source_or_rates():
    raw = _ref_spec()
    raw["cost_basis"]["source"] = "magic"
    with pytest.raises(ContractError, match="source"):
        ExperimentSpec.from_dict(raw)
    raw = _ref_spec()
    raw["cost_basis"]["assumed_pricing"] = {
        "model-a": {"prompt_per_1m": -1, "completion_per_1m": 2.0}
    }
    with pytest.raises(ContractError, match="rates must be >= 0"):
        ExperimentSpec.from_dict(raw)
    raw = _ref_spec()
    raw["cost_basis"]["assumed_pricing"] = {"model-a": {"nope": 1.0}}
    with pytest.raises(ContractError, match="nope"):
        ExperimentSpec.from_dict(raw)


@pytest.mark.parametrize("ref_bound", ["ref:strong", "ref:weak"])
def test_ref_bound_requires_references_section(ref_bound):
    # ref-bound constraint without a references section is a contract error
    raw = copy.deepcopy(VALID_SPEC)
    raw["objective"]["constraints"] = [
        {
            "name": "beat-strong",
            "metric": "capability.score",
            "op": "gte",
            "bound": ref_bound,
        }
    ]
    with pytest.raises(ContractError, match=ref_bound):
        ExperimentSpec.from_dict(raw)


def test_cost_ref_bound_requires_cost_basis():
    # a benefit constraint against a reference arm needs a declared basis
    raw = copy.deepcopy(VALID_SPEC)
    raw["references"] = {"strong": "model-a", "weak": "model-b"}
    raw["objective"]["constraints"] = [
        {
            "name": "spend-not-above-strong",
            "metric": "total_spend_usd",
            "op": "lte",
            "bound": "ref:strong",
        }
    ]
    with pytest.raises(ContractError, match="cost_basis"):
        ExperimentSpec.from_dict(raw)
    raw["cost_basis"] = {"source": "provider_pricing"}
    spec = ExperimentSpec.from_dict(raw)
    assert spec.objective.constraints[0].bound == "ref:strong"


def test_non_cost_ref_bound_needs_no_cost_basis():
    raw = copy.deepcopy(VALID_SPEC)
    raw["references"] = {"strong": "model-a", "weak": "model-b"}
    raw["objective"]["constraints"] = [
        {
            "name": "accuracy-vs-strong",
            "metric": "capability.score",
            "op": "gte",
            "bound": "ref:strong",
        }
    ]
    spec = ExperimentSpec.from_dict(raw)
    assert spec.cost_basis is None


def test_constraint_ref_factor_and_tolerance_pp_parse():
    raw = _ref_spec()
    raw["objective"]["constraints"] = [
        {
            "name": "accuracy-vs-weak",
            "metric": "capability.score",
            "op": "gte",
            "bound": "ref:weak",
            "factor": 0.95,
        },
        {
            "name": "cost-not-worse",
            "metric": "total_spend_usd",
            "op": "lte",
            "bound": "baseline",
            "tolerance_pp": 2.0,
        },
    ]
    spec = ExperimentSpec.from_dict(raw)
    first, second = spec.objective.constraints
    assert first.factor == 0.95 and first.tolerance_pp == 0.0
    assert second.tolerance_pp == 2.0 and second.factor is None
    spec2 = ExperimentSpec.from_dict(spec.to_dict())
    assert spec2.to_dict() == spec.to_dict()


def test_constraint_rejects_bad_factor_and_tolerance_pp():
    raw = _ref_spec()
    raw["objective"]["constraints"] = [
        {
            "name": "x",
            "metric": "capability.score",
            "op": "gte",
            "bound": 0.9,
            "factor": 1.5,
        }
    ]
    with pytest.raises(ContractError, match="factor"):
        ExperimentSpec.from_dict(raw)  # factor only with ref bounds
    raw["objective"]["constraints"][0]["bound"] = "ref:strong"
    raw["objective"]["constraints"][0]["factor"] = 0
    with pytest.raises(ContractError, match="factor"):
        ExperimentSpec.from_dict(raw)
    raw["objective"]["constraints"][0]["factor"] = 1.1
    raw["objective"]["constraints"][0]["tolerance_pp"] = -1
    with pytest.raises(ContractError, match="tolerance_pp"):
        ExperimentSpec.from_dict(raw)
    # tolerance_pct and tolerance_pp are mutually exclusive
    raw["objective"]["constraints"][0]["tolerance_pp"] = 2.0
    raw["objective"]["constraints"][0]["tolerance_pct"] = 5.0
    with pytest.raises(ContractError, match="tolerance"):
        ExperimentSpec.from_dict(raw)


# ---- M5: multi-workload contracts ----


def test_legacy_workload_loads_as_single_tune_workload():
    spec = ExperimentSpec.from_dict(copy.deepcopy(VALID_SPEC))
    assert len(spec.workloads) == 1
    workload = spec.workloads[0]
    assert workload.name == "default"
    assert workload.kind == "mcq"
    assert workload.driver == "srbench"
    assert workload.role == "tune"
    assert workload.weight == 1.0
    assert workload.guard_pp == 1.0
    assert workload.legacy is True
    assert spec.workload is workload  # the historical accessor
    assert spec.workload_guards() == {"default": 1.0}


def test_workloads_list_parses_roles_weights_and_guards():
    spec = ExperimentSpec.from_dict(copy.deepcopy(VALID_WORKLOADS_SPEC))
    assert [w.name for w in spec.workloads] == ["mmlu-pro", "office", "gpqa-transfer"]
    first, office, transfer = spec.workloads
    assert first.weight == 2.0 and first.kind == "mcq" and first.role == "tune"
    assert office.driver == "model_eval" and office.split_rule is None
    assert transfer.role == "transfer" and "eval" in transfer.splits
    # the auto "not worse" guard covers every workload, contract-tunable
    assert spec.workload_guards() == {
        "mmlu-pro": 1.0,
        "office": 1.0,
        "gpqa-transfer": 1.0,
    }
    # round-trip: multi-workload specs serialize the workloads list
    spec2 = ExperimentSpec.from_dict(spec.to_dict())
    assert spec2.to_dict() == spec.to_dict()
    assert "workloads" in spec.to_dict()


def test_workloads_reject_both_keys_and_duplicate_names():
    raw = copy.deepcopy(VALID_WORKLOADS_SPEC)
    raw["workload"] = copy.deepcopy(VALID_SPEC["workload"])
    with pytest.raises(ContractError, match="workload or workloads, not both"):
        ExperimentSpec.from_dict(raw)
    raw = copy.deepcopy(VALID_WORKLOADS_SPEC)
    raw["workloads"][1]["name"] = "mmlu-pro"
    with pytest.raises(ContractError, match="unique"):
        ExperimentSpec.from_dict(raw)
    raw = copy.deepcopy(VALID_WORKLOADS_SPEC)
    del raw["workloads"][1]["name"]
    with pytest.raises(ContractError, match=r"workloads\[1\].name is required"):
        ExperimentSpec.from_dict(raw)


def test_workloads_validate_enums_weight_and_tune_presence():
    raw = copy.deepcopy(VALID_WORKLOADS_SPEC)
    raw["workloads"][0]["kind"] = "chat"
    with pytest.raises(ContractError, match="kind"):
        ExperimentSpec.from_dict(raw)
    raw = copy.deepcopy(VALID_WORKLOADS_SPEC)
    raw["workloads"][0]["driver"] = "lm_eval"
    with pytest.raises(ContractError, match="driver"):
        ExperimentSpec.from_dict(raw)
    raw = copy.deepcopy(VALID_WORKLOADS_SPEC)
    raw["workloads"][0]["role"] = "eval"
    with pytest.raises(ContractError, match="role"):
        ExperimentSpec.from_dict(raw)
    raw = copy.deepcopy(VALID_WORKLOADS_SPEC)
    raw["workloads"][0]["weight"] = 0
    with pytest.raises(ContractError, match="weight must be > 0"):
        ExperimentSpec.from_dict(raw)
    raw = copy.deepcopy(VALID_WORKLOADS_SPEC)
    raw["workloads"][0]["guard_pp"] = -0.5
    with pytest.raises(ContractError, match="guard_pp must be >= 0"):
        ExperimentSpec.from_dict(raw)
    # a single transfer workload has nothing to tune on
    raw = copy.deepcopy(VALID_WORKLOADS_SPEC)
    raw["workloads"] = raw["workloads"][2:]
    with pytest.raises(ContractError, match="at least one role=tune"):
        ExperimentSpec.from_dict(raw)
    # an entry without splits is rejected (the legacy manifest form is
    # only accepted at the top-level `workload:` key)
    raw = copy.deepcopy(VALID_WORKLOADS_SPEC)
    del raw["workloads"][1]["splits"]
    with pytest.raises(ContractError, match=r"workloads\[1\] must declare splits"):
        ExperimentSpec.from_dict(raw)


def test_select_and_require_tunable_workload():
    spec = ExperimentSpec.from_dict(copy.deepcopy(VALID_WORKLOADS_SPEC))
    assert spec.select_workload("office").name == "office"
    assert spec.select_workload(None).name == "mmlu-pro"
    with pytest.raises(ContractError, match="unknown workload 'nope'"):
        spec.select_workload("nope")
    # replay/propose/attribute must never touch a transfer workload
    with pytest.raises(ContractError, match="role 'transfer'"):
        spec.require_tunable_workload("gpqa-transfer")
    assert spec.require_tunable_workload("office").name == "office"
    # the final verify may select the transfer workload
    assert spec.select_workload("gpqa-transfer").role == "transfer"


def _write_workloads_spec(
    tmp_path,
    dev_text: str = '{"case_id":"d1","prompt":"d"}\n',
    gpqa_text: str = '{"case_id":"g1","prompt":"g"}\n',
):
    """The multi-workload spec backed by five manifests with real hashes."""
    contents = {
        "calib.jsonl": '{"case_id":"c1","prompt":"c"}\n',
        "dev.jsonl": dev_text,
        "frozen_test.jsonl": '{"case_id":"f1","prompt":"f"}\n',
        "office.jsonl": '{"case_id":"o1","prompt":"o"}\n',
        "gpqa.jsonl": gpqa_text,
    }
    spec = copy.deepcopy(VALID_WORKLOADS_SPEC)
    for name, text in contents.items():
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        digest = dataset_content_hash(path)
        for workload in spec["workloads"]:
            for split in workload["splits"].values():
                if split["manifest"] == name:
                    split["content_hash"] = digest
    spec_path = tmp_path / "spec-workloads.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return spec_path


def test_verify_workload_selects_the_named_workload(tmp_path):
    spec_path = _write_workloads_spec(tmp_path)
    spec = ExperimentSpec.from_yaml(spec_path)
    assert spec.verify_workload(split="dev", workload="office") == (
        spec.select_workload("office").split_hash("dev")
    )
    assert spec.verify_workload(split="eval", workload="gpqa-transfer") == (
        spec.select_workload("gpqa-transfer").split_hash("eval")
    )
    with pytest.raises(ContractError, match="unknown workload 'nope'"):
        spec.verify_workload(split="dev", workload="nope")


def test_verify_workload_rejects_cross_workload_overlap(tmp_path):
    # the transfer workload shares a case with the tune workload's dev split
    shared = json.dumps({"case_id": "leak", "prompt": "same prompt"}) + "\n"
    spec_path = _write_workloads_spec(
        tmp_path,
        dev_text='{"case_id":"d1","prompt":"d"}\n' + shared,
        gpqa_text=shared,
    )
    spec = ExperimentSpec.from_yaml(spec_path)
    with pytest.raises(ContractError, match=r"workloads .* overlap"):
        spec.verify_workload(workload="office")


def test_single_workload_spec_keeps_workload_key_roundtrip():
    spec = ExperimentSpec.from_dict(copy.deepcopy(VALID_SPLITS_SPEC))
    assert "workload" in spec.to_dict() and "workloads" not in spec.to_dict()
    spec2 = ExperimentSpec.from_dict(spec.to_dict())
    assert spec2.to_dict() == spec.to_dict()
