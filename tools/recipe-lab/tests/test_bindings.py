"""Bindings (C4+C5 template path) unit tests - sub-issue-8 §2.7 golden scenarios.

All subprocess boundaries are faked (recipe builtin list/init via run_json,
config validate via monkeypatched validate_config), so the suite is
deterministic and platform-independent.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml
from recipe_lab import bindings as bindings_mod
from recipe_lab.contract import ContractError
from recipe_lab.observe_lanes import (
    capability_satisfied,
    lane_requirements,
)

API_VERSION = "recipe-lab/v1alpha1"

# A small template with the shapes the golden scenarios need: three
# single-candidate measured lanes, one unmeasured lane, one two-candidate
# lane (frontier semantics), and an immediate lane.
TEMPLATE_LANES = {
    "code": {"algorithm": "static", "minimum_candidates": 1, "immediate": False},
    "hard": {"algorithm": "multi_factor", "minimum_candidates": 1, "immediate": False},
    "facts": {"algorithm": "static", "minimum_candidates": 1, "immediate": False},
    "fast": {"algorithm": "static", "minimum_candidates": 1, "immediate": False},
    "frontier": {"algorithm": "decision", "minimum_candidates": 2, "immediate": False},
    "vision": {"algorithm": "static", "minimum_candidates": 1, "immediate": False},
    "agentic": {"algorithm": "static", "minimum_candidates": 1, "immediate": False},
    "standard": {
        "algorithm": "multi_factor",
        "minimum_candidates": 1,
        "immediate": False,
    },
    "long_context": {
        "algorithm": "static",
        "minimum_candidates": 1,
        "immediate": False,
    },
    "recovery": {"algorithm": "static", "minimum_candidates": 1, "immediate": False},
}


def fake_template():
    return {
        "bundle": "mom-v1",
        "recipe": "balance",
        "sha256": "sha256:template",
        "catalog_version": "latest",
        "lanes": dict(TEMPLATE_LANES),
    }


def write_spec(
    tmp_path: Path,
    *,
    capabilities: dict[str, dict] | None = None,
    lambdas: list[float] | None = None,
    min_lane_cases: int = 20,
    with_pricing: bool = True,
) -> Path:
    """A minimal valid spec declaring the template path.

    Without an explicit ``capabilities`` mapping both models declare the
    default set (tools + a 256K window; A also vision), so every template
    lane is satisfiable - the capability-filter tests pass their own
    restricted mapping.
    """
    default_caps = {
        "local/model-a": {"vision": True, "tools": True, "context_window": 262144},
        "local/model-b": {"tools": True, "context_window": 262144},
    }
    models = []
    for alias, endpoint in (
        ("local/model-a", "http://127.0.0.1:9001/v1"),
        ("local/model-b", "http://127.0.0.1:9002/v1"),
    ):
        entry = {"name": alias, "endpoint": endpoint, "served_name": alias}
        if capabilities is None:
            entry["capabilities"] = default_caps[alias]
        elif alias in capabilities:
            entry["capabilities"] = capabilities[alias]
        # else: the capability key is omitted entirely (unsatisfiable lanes)
        models.append(entry)
    spec: dict = {
        "api_version": API_VERSION,
        "objective": {
            "metric_key": "capability.score",
            "direction": "maximize",
            "min_improvement_pct": 1.0,
            "replay_metric": "routed_accuracy",
        },
        "workload": {
            "manifest": "dev.jsonl",
            "content_hash": "sha256:" + "0" * 64,
            "profile": "quick",
        },
        "attribution": {"oracle_policy": {"math": "local/model-a"}},
        "budget": {"rounds": 1, "max_minutes_per_trial": 30},
        "models": models,
        "template": {"bundle": "mom-v1", "recipe": "balance"},
        "providers_config": "providers.yaml",
        "bootstrap": {"min_lane_cases": min_lane_cases},
    }
    if lambdas is not None:
        spec["bootstrap"]["lambdas"] = lambdas
    if with_pricing:
        spec["cost_basis"] = {
            "source": "provider_pricing",
            "assumed_pricing": {
                "local/model-a": {"prompt_per_1m": 0.1, "completion_per_1m": 0.4},
                "local/model-b": {"prompt_per_1m": 0.4, "completion_per_1m": 1.6},
            },
        }
    (tmp_path / "providers.yaml").write_text(
        yaml.safe_dump(
            {
                "version": "v0.5",
                "listeners": [
                    {
                        "name": "gw",
                        "address": "0.0.0.0",
                        "port": 8899,
                        "timeout": "300s",
                    }
                ],
                "providers": {
                    "defaults": {"model": "local/model-a"},
                    "models": [
                        {
                            "name": alias,
                            "provider_model_id": alias,
                            "api_format": "openai",
                            # the provider entry is the model's own card
                            "context_window_size": 131072,
                            "capabilities": ["chat", "text"],
                            "backend_refs": [
                                {
                                    "name": alias + "-backend",
                                    "endpoint": "127.0.0.1:9001",
                                    "protocol": "http",
                                    "provider": "vllm",
                                    "weight": 1,
                                }
                            ],
                        }
                        for alias in ("local/model-a", "local/model-b")
                    ],
                },
                "global": {"stores": {"response_cache": {"backend_type": "memory"}}},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "dev.jsonl").write_text("", encoding="utf-8")
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return spec_path


def write_profile(
    tmp_path: Path,
    *,
    lane_stats: dict[str, dict[str, dict[str, float]]],
    overall: dict[str, dict[str, float]],
) -> Path:
    """A recipe-lab-profile-v1 document (lane mode) the bindings consume.

    ``lane_stats``: {lane: {alias: {"n": int, "accuracy": float,
    "cost_per_question": float}}}; every lane's n is shared by all models
    (all models answer the same tasks).
    """
    stats: dict[str, dict] = {}
    for alias in overall:
        labels = {
            lane: {
                "n": block["n"],
                "accuracy": block[alias]["accuracy"],
                "cost_per_question": block[alias]["cost_per_question"],
                "accuracy_ci95": [0.0, 1.0],
            }
            for lane, block in lane_stats.items()
        }
        stats[alias] = {
            "labels": labels,
            "total": {
                "n": 100,
                "accuracy": overall[alias]["accuracy"],
                "cost_per_question": overall[alias]["cost_per_question"],
            },
        }
    profile = {
        "schema": "recipe-lab-profile-v1",
        "grouping": "lanes",
        "splits": ["calib"],
        "models": list(overall),
        "label_merges": {},
        "uncovered_cases": {},
        "unmeasured_lanes": {},
        "quality": {},
        "model_split_stats": {"calib": stats},
        "inputs": {},
    }
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
    return profile_dir


def fake_run_json_factory(outputs: dict[str, str]):
    """A run_json stand-in: any `init` argv writes the canned output file."""

    def run_json(argv, *, timeout_s, what):
        if "init" in argv:
            out = argv[argv.index("--output") + 1]
            Path(out).write_text(
                outputs.get("init", "version: v0.5\n"), encoding="utf-8"
            )
            return {"valid": True}
        return {"bundles": []}

    return run_json


def plan(tmp_path, *, profile_dir, lam, exclude=(), spec_path=None):
    spec_path = spec_path or write_spec(tmp_path)
    from recipe_lab.contract import ExperimentSpec

    spec = ExperimentSpec.from_yaml(spec_path)
    profile = json.loads((profile_dir / "profile.json").read_text(encoding="utf-8"))
    return bindings_mod.plan_bindings(
        spec=spec,
        profile=profile,
        template=fake_template(),
        lam=lam,
        exclude=exclude,
    )


# --------------------------------------------------------------------------
# Golden scenario (sub-issue-8 §2.7): model A is fine on facts/fast, model B
# is clearly better on code/hard and 4x pricier.
# --------------------------------------------------------------------------


def golden_profile(tmp_path):
    n = {"code": 50, "hard": 50, "facts": 50, "fast": 50, "frontier": 0}
    lane_stats = {}
    for lane, (a_acc, b_acc) in {
        "code": (0.60, 0.90),
        "hard": (0.55, 0.85),
        "facts": (0.80, 0.82),
        "fast": (0.85, 0.84),
    }.items():
        lane_stats[lane] = {
            "n": n[lane],
            "local/model-a": {"accuracy": a_acc, "cost_per_question": 0.001},
            "local/model-b": {"accuracy": b_acc, "cost_per_question": 0.004},
        }
    overall = {
        "local/model-a": {"accuracy": 0.72, "cost_per_question": 0.001},
        "local/model-b": {"accuracy": 0.85, "cost_per_question": 0.004},
    }
    return write_profile(tmp_path, lane_stats=lane_stats, overall=overall)


def test_golden_small_lambda_binds_strong_model_to_hard_lanes(tmp_path):
    profile_dir = golden_profile(tmp_path)
    result = plan(tmp_path, profile_dir=profile_dir, lam=0.0)
    plans = result["plans"]
    assert plans["code"]["selected"] == ["local/model-b"]
    assert plans["hard"]["selected"] == ["local/model-b"]
    assert plans["fast"]["selected"] == ["local/model-a"]
    # unmeasured lane (frontier n=0 < 20): default rule by overall score,
    # marked untested.
    assert plans["frontier"]["measured"] is False
    assert plans["frontier"]["selected"] == ["local/model-b", "local/model-a"]
    assert "unmeasured" in plans["frontier"]["basis"]
    assert result["routing_space"]["verdict"] == "route"


def test_golden_large_lambda_rebinds_hard_to_cheapest(tmp_path):
    profile_dir = golden_profile(tmp_path)
    # lambda = 100: 0.30 accuracy advantage (code) < 100 * (0.004-0.001) = 0.3
    # normalized cost gap 1.0 vs 0.25 -> hard's 0.30 advantage loses to 0.75.
    result = plan(tmp_path, profile_dir=profile_dir, lam=100.0)
    plans = result["plans"]
    # code keeps B (0.30 advantage vs 0.75 cost penalty? 0.90-100*1.0=-10 vs
    # 0.60-100*0.25=-24.4 -> B still wins); hard: A 0.55-25=-24.45 vs B
    # 0.85-100=-99.15 -> A wins.
    assert plans["hard"]["selected"] == ["local/model-a"]
    assert plans["fast"]["selected"] == ["local/model-a"]


def test_frontier_requires_two_distinct_candidates(tmp_path):
    profile_dir = golden_profile(tmp_path)
    result = plan(tmp_path, profile_dir=profile_dir, lam=0.0)
    selected = result["plans"]["frontier"]["selected"]
    assert len(selected) == 2
    assert len(set(selected)) == 2


def test_capability_filter_refuses_unsatisfied_lane_without_explicit_exclude(tmp_path):
    profile_dir = golden_profile(tmp_path)
    spec_path = write_spec(tmp_path, capabilities={})
    from recipe_lab.contract import ExperimentSpec

    spec = ExperimentSpec.from_yaml(spec_path)
    with pytest.raises(ContractError, match="agentic"):
        bindings_mod.plan_bindings(
            spec=spec,
            profile=json.loads(
                (profile_dir / "profile.json").read_text(encoding="utf-8")
            ),
            template=fake_template(),
            lam=0.0,
        )


def test_capability_filter_with_explicit_exclude(tmp_path):
    profile_dir = golden_profile(tmp_path)
    spec_path = write_spec(
        tmp_path,
        capabilities={
            "local/model-a": {
                "vision": True,
                "tools": True,
                "context_window": 262144,
            },
            "local/model-b": {"tools": True, "context_window": 262144},
        },
    )
    result = plan(tmp_path, profile_dir=profile_dir, lam=0.0, spec_path=spec_path)
    # vision now has one eligible model (A); agentic has both.
    assert result["plans"]["vision"]["selected"] == ["local/model-a"]
    assert result["plans"]["agentic"]["measured"] is False
    assert len(result["plans"]["agentic"]["selected"]) == 1


def test_long_context_requires_declared_window(tmp_path):
    requirement = lane_requirements("long_context")
    from recipe_lab.contract import ModelCapabilities

    assert not capability_satisfied(
        requirement, ModelCapabilities(context_window=131072)
    )
    assert capability_satisfied(requirement, ModelCapabilities(context_window=262144))
    # Unlisted lanes carry no requirement.
    assert lane_requirements("fast") == {
        "capabilities": (),
        "min_context_window": None,
    }


def test_tools_lane_never_binds_a_model_without_tool_capability(tmp_path):
    # The Speed and Cost recipes route tool traffic to their `tools` lane.
    # Model A scores higher there but cannot call tools: it must not be
    # picked, measured or not.
    profile_dir = write_profile(
        tmp_path,
        lane_stats={
            "tools": {
                "n": 50,
                "local/model-a": {"accuracy": 1.0, "cost_per_question": 0.001},
                "local/model-b": {"accuracy": 0.8, "cost_per_question": 0.001},
            }
        },
        overall={
            "local/model-a": {"accuracy": 1.0, "cost_per_question": 0.001},
            "local/model-b": {"accuracy": 0.8, "cost_per_question": 0.001},
        },
    )
    spec_path = write_spec(
        tmp_path,
        capabilities={
            "local/model-a": {"context_window": 262144},
            "local/model-b": {"tools": True, "context_window": 262144},
        },
    )
    from recipe_lab.contract import ExperimentSpec

    template = {
        "recipe": "speed",
        "lanes": {"tools": {"immediate": False, "minimum_candidates": 1}},
    }
    result = bindings_mod.plan_bindings(
        spec=ExperimentSpec.from_yaml(spec_path),
        profile=json.loads((profile_dir / "profile.json").read_text(encoding="utf-8")),
        template=template,
        lam=0.0,
    )
    plan = result["plans"]["tools"]
    assert plan["measured"] is True
    assert plan["selected"] == ["local/model-b"]


def test_no_routing_space_report_and_no_candidates(tmp_path, monkeypatch):
    # Degenerate: model B is better everywhere by far and A is never within
    # 5pp -> routing degenerates to one model.
    lane_stats = {
        lane: {
            "n": 50,
            "local/model-a": {"accuracy": 0.60, "cost_per_question": 0.001},
            "local/model-b": {"accuracy": 0.95, "cost_per_question": 0.004},
        }
        for lane in ("code", "hard", "facts", "fast")
    }
    overall = {
        "local/model-a": {"accuracy": 0.60, "cost_per_question": 0.001},
        "local/model-b": {"accuracy": 0.95, "cost_per_question": 0.004},
    }
    profile_dir = write_profile(tmp_path, lane_stats=lane_stats, overall=overall)
    spec_path = write_spec(tmp_path, lambdas=[0.0])
    from recipe_lab.contract import ExperimentSpec

    spec = ExperimentSpec.from_yaml(spec_path)
    monkeypatch.setattr(bindings_mod, "template_info", lambda **kwargs: fake_template())
    result = bindings_mod.run_bindings(
        spec=spec,
        profile_dir=profile_dir,
        out_dir=tmp_path / "cands",
        lambdas_raw=None,
        vllm_sr_bin="unused",
        run_json=fake_run_json_factory({}),
    )
    assert result["candidates"] == []
    assert result["routing_space"]["verdict"] == "no_routing_space"
    assert "No benefit to routing" in result["report"]


def test_run_bindings_generates_and_validates_candidates(tmp_path, monkeypatch):
    profile_dir = golden_profile(tmp_path)
    spec_path = write_spec(
        tmp_path,
        capabilities={
            "local/model-a": {
                "vision": True,
                "tools": True,
                "context_window": 262144,
            },
            "local/model-b": {"tools": True, "context_window": 262144},
        },
        lambdas=[0.0, 100.0],
    )
    from recipe_lab.contract import ExperimentSpec

    spec = ExperimentSpec.from_yaml(spec_path)
    monkeypatch.setattr(bindings_mod, "template_info", lambda **kwargs: fake_template())
    captured: list[list[str]] = []

    def run_json(argv, *, timeout_s, what):
        captured.append(list(argv))
        if "init" in argv:
            out = Path(argv[argv.index("--output") + 1])
            out.write_text("version: v0.5\n", encoding="utf-8")
            return {"valid": True}
        return {"bundles": []}

    monkeypatch.setattr(bindings_mod, "validate_config", lambda path, bin: None)
    result = bindings_mod.run_bindings(
        spec=spec,
        profile_dir=profile_dir,
        out_dir=tmp_path / "cands",
        lambdas_raw=None,
        vllm_sr_bin="fake-vllm-sr",
        run_json=run_json,
    )
    assert [c["lambda"] for c in result["candidates"]] == [0.0, 100.0]
    inits = [argv for argv in captured if "init" in argv]
    assert len(inits) == 2
    for argv in inits:
        assert "--config" in argv and "--bindings" in argv
        assert "--model-name" in argv
        assert (
            argv[argv.index("--catalog-version") + 1]
            == result["template"]["catalog_version"]
        )
    # A canonical provider config rides into the eval copy verbatim: the
    # response cache store has no switch (a decision mounts the plugin or it
    # does not), so nothing is written to turn it off.
    eval_copy = yaml.safe_load(
        (tmp_path / "cands" / "providers-eval.yaml").read_text(encoding="utf-8")
    )
    original = yaml.safe_load((tmp_path / "providers.yaml").read_text(encoding="utf-8"))
    assert eval_copy == original
    # Bindings are the raw {lane: refs} mapping init consumes.
    bindings_doc = yaml.safe_load(
        (tmp_path / "cands" / "bindings-lambda-0.yaml").read_text(encoding="utf-8")
    )
    assert bindings_doc["frontier"][0]["model"]
    assert bindings_doc["code"] == [{"model": "local/model-b"}]
    # Sidecar records the template hash and the per-lane basis.
    sidecar = json.loads(
        (tmp_path / "cands" / "bindings-lambda-0.provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert sidecar["template"]["sha256"] == "sha256:template"
    assert sidecar["lanes"]["frontier"]["measured"] is False
    assert sidecar["unmeasured_lanes"] == [
        "agentic",
        "frontier",
        "long_context",
        "recovery",
        "standard",
        "vision",
    ]


def test_lambda_above_zero_without_costs_is_rejected(tmp_path):
    # Measured lane without cost data + lambda > 0 -> explicit error, never
    # a silent degrade to pure accuracy (the §3.2-5 lesson).
    lane_stats = {
        "code": {
            "n": 50,
            "local/model-a": {"accuracy": 0.60, "cost_per_question": None},
            "local/model-b": {"accuracy": 0.90, "cost_per_question": 0.004},
        },
        "hard": {
            "n": 50,
            "local/model-a": {"accuracy": 0.55, "cost_per_question": None},
            "local/model-b": {"accuracy": 0.85, "cost_per_question": 0.004},
        },
        "facts": {
            "n": 50,
            "local/model-a": {"accuracy": 0.80, "cost_per_question": None},
            "local/model-b": {"accuracy": 0.82, "cost_per_question": 0.004},
        },
        "fast": {
            "n": 50,
            "local/model-a": {"accuracy": 0.85, "cost_per_question": None},
            "local/model-b": {"accuracy": 0.84, "cost_per_question": 0.004},
        },
    }
    overall = {
        "local/model-a": {"accuracy": 0.72, "cost_per_question": None},
        "local/model-b": {"accuracy": 0.85, "cost_per_question": 0.004},
    }
    profile_dir = write_profile(tmp_path, lane_stats=lane_stats, overall=overall)
    spec_path = write_spec(tmp_path, lambdas=[0.5])
    from recipe_lab.contract import ExperimentSpec

    spec = ExperimentSpec.from_yaml(spec_path)
    with pytest.raises(ContractError, match="cost"):
        bindings_mod.plan_bindings(
            spec=spec,
            profile=json.loads(
                (profile_dir / "profile.json").read_text(encoding="utf-8")
            ),
            template=fake_template(),
            lam=0.5,
        )


def test_unknown_exclude_lane_is_rejected(tmp_path):
    profile_dir = golden_profile(tmp_path)
    spec_path = write_spec(tmp_path)
    from recipe_lab.contract import ExperimentSpec

    spec = ExperimentSpec.from_yaml(spec_path)
    with pytest.raises(ContractError, match="does not declare"):
        bindings_mod.plan_bindings(
            spec=spec,
            profile=json.loads(
                (profile_dir / "profile.json").read_text(encoding="utf-8")
            ),
            template=fake_template(),
            lam=0.0,
            exclude=("nonexistent-lane",),
        )


def test_routing_space_no_advantage(tmp_path):
    # Condition A: the best model's advantage over the cheapest is < 5pp on
    # every measured lane.
    lane_stats = {
        lane: {
            "n": 50,
            "local/model-a": {"accuracy": 0.80, "cost_per_question": 0.001},
            "local/model-b": {"accuracy": 0.83, "cost_per_question": 0.004},
        }
        for lane in ("code", "hard")
    }
    overall = {
        "local/model-a": {"accuracy": 0.80, "cost_per_question": 0.001},
        "local/model-b": {"accuracy": 0.83, "cost_per_question": 0.004},
    }
    profile_dir = write_profile(tmp_path, lane_stats=lane_stats, overall=overall)
    result = plan(tmp_path, profile_dir=profile_dir, lam=0.0)
    assert result["routing_space"]["verdict"] == "no_routing_space"


def test_routing_space_cost_complement_with_equal_accuracy(tmp_path):
    a, b = "local/model-a", "local/model-b"
    lane_stats = {
        lane: {
            "n": 50,
            a: {"accuracy": 1.0, "cost_per_question": cost_a},
            b: {"accuracy": 1.0, "cost_per_question": cost_b},
        }
        for lane, cost_a, cost_b in [("fast", 1, 100), ("hard", 100, 2)]
    }
    overall = {
        a: {"accuracy": 1.0, "cost_per_question": 50.5},
        b: {"accuracy": 1.0, "cost_per_question": 51.0},
    }
    profile = write_profile(tmp_path, lane_stats=lane_stats, overall=overall)
    result = plan(tmp_path, profile_dir=profile, lam=0.1)
    assert result["routing_space"]["verdict"] == "route"
    assert result["plans"]["fast"]["selected"] == [a]
    assert result["plans"]["hard"]["selected"] == [b]


def test_routing_space_preserves_intermediate_model_on_one_lane(tmp_path):
    a, b, c = "local/model-a", "local/model-b", "local/model-c"
    spec_path = write_spec(tmp_path, with_pricing=False)
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    spec["models"].append(
        {"name": c, "served_name": c, "endpoint": "http://127.0.0.1:9003/v1"}
    )
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    # C is most accurate everywhere. A is cheapest everywhere and always
    # trails C by more than the threshold. B can replace C on fast only.
    stats = {}
    for lane, accuracies in [("fast", [0.70, 0.99, 1.0]), ("hard", [0.45, 0.55, 1.0])]:
        stats[lane] = {
            "n": 100,
            **{
                alias: {"accuracy": acc, "cost_per_question": cost}
                for alias, acc, cost in zip(
                    (a, b, c), accuracies, (1, 3, 100), strict=False
                )
            },
        }
    overall = {alias: stats["fast"][alias] for alias in (a, b, c)}
    profile = write_profile(tmp_path, lane_stats=stats, overall=overall)
    result = plan(tmp_path, profile_dir=profile, spec_path=spec_path, lam=0.2)
    assert result["routing_space"]["verdict"] == "route"
    assert result["plans"]["fast"]["selected"] == [b]
    assert result["plans"]["hard"]["selected"] == [c]


def test_routing_space_preserves_different_break_even_costs_for_two_models():
    result = bindings_mod.check_routing_space(
        measured_lanes=["fast", "hard"],
        accuracies={"fast": {"a": 0.9, "b": 1.0}, "hard": {"a": 0.5, "b": 1.0}},
        costs={"fast": {"a": 1, "b": 100}, "hard": {"a": 1, "b": 100}},
        min_advantage_pp=5,
    )
    assert result["verdict"] == "route"


def test_routing_space_cannot_rule_out_unmeasured_eligible_model():
    result = bindings_mod.check_routing_space(
        measured_lanes=["fast", "hard"],
        accuracies={"fast": {"a": 0.9, "b": None}, "hard": {"a": 0.9, "b": 0.9}},
        costs={"fast": {"a": 1, "b": 3}, "hard": {"a": 1, "b": 3}},
        min_advantage_pp=5,
    )
    assert result["verdict"] == "route"


def test_routing_space_classifies_unmeasured_and_unpriced_lanes():
    kwargs = {
        "accuracies": {"fast": {"a": 0.7, "b": 0.9}},
        "costs": {},
        "min_advantage_pp": 5,
    }
    no_lanes = bindings_mod.check_routing_space(measured_lanes=[], **kwargs)
    assert no_lanes["verdict"] == "no_routing_space"
    assert "min_lane_cases" in no_lanes["reason"]
    # A missing cost cannot establish that a fixed model is cheapest.
    no_prices = bindings_mod.check_routing_space(measured_lanes=["fast"], **kwargs)
    assert no_prices["verdict"] == "route"
    assert no_prices["cheapest_model"] is None


def test_bindings_output_is_byte_stable(tmp_path, monkeypatch):
    profile_dir = golden_profile(tmp_path)
    spec_path = write_spec(tmp_path, lambdas=[0.0])
    from recipe_lab.contract import ExperimentSpec

    spec = ExperimentSpec.from_yaml(spec_path)
    monkeypatch.setattr(bindings_mod, "template_info", lambda **kwargs: fake_template())
    monkeypatch.setattr(bindings_mod, "validate_config", lambda path, bin: None)

    def run_json(argv, *, timeout_s, what):
        if "init" in argv:
            Path(argv[argv.index("--output") + 1]).write_text(
                "version: v0.5\n", encoding="utf-8"
            )
            return {"valid": True}
        return {"bundles": []}

    digests = []
    for run in ("run1", "run2"):
        out = tmp_path / run
        bindings_mod.run_bindings(
            spec=spec,
            profile_dir=profile_dir,
            out_dir=out,
            lambdas_raw=None,
            vllm_sr_bin="fake-vllm-sr",
            run_json=run_json,
        )
        # Volatile fields (absolute output paths, hash-of-paths) are excluded;
        # everything selection-relevant must be identical across runs.
        stable_keys = (
            "template",
            "routing_space",
            "measured_lanes",
            "unmeasured_lanes",
            "excluded_lanes",
            "lanes",
        )
        digests.append(
            (out / "bindings-lambda-0.yaml").read_bytes()
            + json.dumps(
                {
                    k: v
                    for k, v in json.loads(
                        (out / "bindings-lambda-0.provenance.json").read_text(
                            encoding="utf-8"
                        )
                    ).items()
                    if k in stable_keys
                },
                sort_keys=True,
            ).encode("utf-8")
        )
    assert digests[0] == digests[1]


@pytest.mark.parametrize(
    "bad_card", [None, "max_output_tokens", "context_window_size", "capabilities"]
)
def test_bindings_checks_rendered_strict_model_cards(tmp_path, monkeypatch, bad_card):
    from recipe_lab.contract import ExperimentSpec

    spec = ExperimentSpec.from_yaml(write_spec(tmp_path))
    profile = golden_profile(tmp_path)
    providers_path = tmp_path / "providers.yaml"
    providers = yaml.safe_load(providers_path.read_text())
    models = providers["providers"]["models"]
    for model in models:
        model["max_output_tokens"] = 8192
        model["context_window_size"] = 262144
        model["capabilities"] = ["chat", "tool_use", "image_input"]
    if bad_card:
        del models[1][bad_card]
    providers_path.write_text(yaml.safe_dump(providers))
    template = fake_template()
    monkeypatch.setattr(bindings_mod, "template_info", lambda **kw: template)
    monkeypatch.setattr(bindings_mod, "validate_config", lambda path, binary: None)

    def render(argv, **kw):
        doc = yaml.safe_load(Path(argv[argv.index("--config") + 1]).read_text())
        assigned = yaml.safe_load(Path(argv[argv.index("--bindings") + 1]).read_text())
        doc["recipes"] = [
            {
                "name": template["recipe"],
                "candidate_requirements": {
                    "context": "known_limits",
                    "capabilities": "declared",
                },
                "decisions": [
                    {"name": name, "modelRefs": refs} for name, refs in assigned.items()
                ],
            }
        ]
        Path(argv[argv.index("--output") + 1]).write_text(yaml.safe_dump(doc))
        return {}

    kwargs = {
        "spec": spec,
        "profile_dir": profile,
        "out_dir": tmp_path / "out",
        "lambdas_raw": "0",
        "vllm_sr_bin": "fixture",
        "run_json": render,
    }
    if bad_card:
        with pytest.raises(ContractError) as exc:
            bindings_mod.run_bindings(**kwargs)
        assert "frontier: 1/2 eligible" in str(exc.value)
        assert "local/model-b" in str(exc.value)
        assert bad_card in str(exc.value)
    else:
        assert len(bindings_mod.run_bindings(**kwargs)["candidates"]) == 1


def test_card_preflight_allows_inherited_catalog_metadata_and_external_id(tmp_path):
    candidate = tmp_path / "candidate.yaml"
    candidate.write_text(
        yaml.safe_dump(
            {
                "providers": {
                    "models": [
                        {
                            "name": "alias",
                            "catalog": "vendor/card",
                            "provider_model_id": "served",
                        },
                    ]
                },
                "recipes": [
                    {
                        "name": "balance",
                        "candidate_requirements": {
                            "context": "known_limits",
                            "capabilities": "declared",
                        },
                        "decisions": [
                            {"name": "vision", "modelRefs": [{"model": "served"}]}
                        ],
                    }
                ],
            }
        )
    )
    bindings_mod._check_bound_model_cards(candidate, fake_template())


def test_card_preflight_reads_recipe_level_requirements(tmp_path):
    """Candidates (the current recipe builtin init output) carry
    candidate_requirements and decisions at the recipe level; the preflight
    reads them there and treats the provider entry as the model's own card."""
    candidate = tmp_path / "candidate.yaml"
    candidate.write_text(
        yaml.safe_dump(
            {
                "providers": {
                    "models": [
                        {
                            "name": "alias",
                            "provider_model_id": "served",
                        },
                    ]
                },
                "recipes": [
                    {
                        "name": "balance",
                        "candidate_requirements": {
                            "context": "known_limits",
                            "capabilities": "declared",
                        },
                        "decisions": [
                            {"name": "vision", "modelRefs": [{"model": "served"}]}
                        ],
                    }
                ],
            }
        )
    )
    with pytest.raises(ContractError) as exc:
        bindings_mod._check_bound_model_cards(candidate, fake_template())
    assert "vision" in str(exc.value)
    assert "context_window_size" in str(exc.value)


def test_card_preflight_reads_facts_from_the_provider_entries(tmp_path):
    # the provider entry is the model's own card: declared limits and
    # capabilities on it satisfy a strict recipe without any card table
    candidate = tmp_path / "candidate.yaml"
    candidate.write_text(
        yaml.safe_dump(
            {
                "providers": {
                    "models": [
                        {
                            "name": "local/model-a",
                            "capabilities": ["chat", "tools", "vision"],
                            "context_window_size": 262144,
                            "max_output_tokens": 8192,
                        },
                        {
                            "name": "local/model-b",
                            "capabilities": ["chat", "tools", "vision"],
                            "context_window_size": 262144,
                            "max_output_tokens": 8192,
                        },
                    ]
                },
                "recipes": [
                    {
                        "name": "balance",
                        "candidate_requirements": {
                            "context": "known_limits",
                            "capabilities": "declared",
                        },
                        "decisions": [
                            {
                                "name": "frontier",
                                "modelRefs": [
                                    {"model": "local/model-a"},
                                    {"model": "local/model-b"},
                                ],
                            }
                        ],
                    }
                ],
            }
        )
    )
    bindings_mod._check_bound_model_cards(candidate, fake_template())


def test_bindings_validates_and_hashes_effective_replay_store(tmp_path, monkeypatch):
    from recipe_lab.contract import ExperimentSpec, dataset_content_hash

    spec = ExperimentSpec.from_yaml(write_spec(tmp_path))
    template = fake_template()
    monkeypatch.setattr(bindings_mod, "template_info", lambda **kw: template)
    validated = []

    def render(argv, **kw):
        doc = yaml.safe_load(Path(argv[argv.index("--config") + 1]).read_text())
        doc["global"]["services"] = {
            "router_replay": {
                "enabled": True,
                "store_backend": "postgres",
                "ttl_seconds": 600,
            }
        }
        Path(argv[argv.index("--output") + 1]).write_text(yaml.safe_dump(doc))
        return {}

    def validate(path, binary):
        settings = yaml.safe_load(path.read_text())["global"]["services"][
            "router_replay"
        ]
        assert settings == {
            "enabled": True,
            "store_backend": "memory",
            "ttl_seconds": 600,
        }
        validated.append(dataset_content_hash(path))

    monkeypatch.setattr(bindings_mod, "validate_config", validate)
    provider_hash = dataset_content_hash(tmp_path / "providers.yaml")
    result = bindings_mod.run_bindings(
        spec=spec,
        profile_dir=golden_profile(tmp_path),
        out_dir=tmp_path / "out",
        lambdas_raw="0",
        vllm_sr_bin="fixture",
        run_json=render,
    )
    sidecar = json.loads(Path(result["candidates"][0]["provenance"]).read_text())
    assert validated == [sidecar["candidate"]["sha256"]]
    (change,) = sidecar["evaluation_overrides"]
    assert (change["field"], change["old"], change["new"]) == (
        "global.services.router_replay.store_backend",
        "postgres",
        "memory",
    )
    assert "template" in change["reason"]
    assert dataset_content_hash(tmp_path / "providers.yaml") == provider_hash


@pytest.mark.parametrize(
    "source,rendered,invalid",
    [
        ({"store_backend": "postgres"}, {"store_backend": "postgres"}, True),
        (
            {"postgres": {"host": "db"}},
            {"store_backend": "postgres", "postgres": {"host": "db"}},
            True,
        ),
        (
            {"redis": {"address": "db:6379"}},
            {"store_backend": "postgres", "redis": {"address": "db:6379"}},
            True,
        ),
        (
            {
                "store_backend": "postgres",
                "postgres": {"database": "archive", "user": "owner"},
            },
            None,
            False,
        ),
        ({"store_backend": "redis", "redis": {"address": "db:6379"}}, None, False),
        ({"store_backend": "memory"}, None, False),
    ],
)
def test_explicit_replay_storage_is_preserved_or_diagnosed(
    tmp_path, source, rendered, invalid
):
    def config(replay):
        return {"global": {"services": {"router_replay": replay}}}

    candidate = tmp_path / "candidate.yaml"
    initial = yaml.safe_dump(config(rendered or source))
    candidate.write_text(initial)
    if invalid:
        with pytest.raises(ContractError, match=r"postgres\.database"):
            bindings_mod._derive_eval_replay_store(candidate, config(source))
    else:
        assert bindings_mod._derive_eval_replay_store(candidate, config(source)) == []
    assert candidate.read_text() == initial


def test_unbound_template_postgres_selects_memory(tmp_path):
    # The Router validates the replay store at startup whenever the services
    # block selects a backend (every request carries a default replay
    # policy), so a template postgres selection without a connection must be
    # rewritten even though no decision mounts a router_replay plugin.
    for rendered in (
        {"store_backend": "postgres"},
        {"postgres": {}},  # no store_backend: the postgres block selects it
    ):
        document = {"global": {"services": {"router_replay": rendered}}}
        candidate = tmp_path / "candidate.yaml"
        candidate.write_text(yaml.safe_dump(document))
        overrides = bindings_mod._derive_eval_replay_store(candidate, {})
        assert [change["new"] for change in overrides] == ["memory"]
        service = yaml.safe_load(candidate.read_text())["global"]["services"][
            "router_replay"
        ]
        assert service["store_backend"] == "memory"


@pytest.mark.parametrize("scope", ["only", "later"])
@pytest.mark.parametrize("enabled", [None, False, True])
def test_decision_replay_plugins_do_not_gate_store_rewrite(tmp_path, scope, enabled):
    # A decision-level router_replay plugin never gates store
    # initialization: the store is built and validated from the services
    # block alone, whatever the decisions record.
    plugin = {"type": "router_replay"}
    if enabled is not None:
        plugin["configuration"] = {"enabled": enabled}
    routing = {"decisions": [{"name": "observed", "plugins": [plugin]}]}
    document = {
        "global": {"services": {"router_replay": {"store_backend": "postgres"}}}
    }
    if scope == "only":
        document["recipes"] = [routing]
    else:
        document["recipes"] = [{"name": "other", "decisions": []}, routing]
    candidate = tmp_path / "candidate.yaml"
    candidate.write_text(yaml.safe_dump(document))
    overrides = bindings_mod._derive_eval_replay_store(candidate, {})
    assert [change["new"] for change in overrides] == ["memory"]
    assert (
        yaml.safe_load(candidate.read_text())["global"]["services"]["router_replay"][
            "store_backend"
        ]
        == "memory"
    )


@pytest.mark.parametrize(
    "store, expected, disabled",
    [
        (None, {"backend_type": "memory"}, True),
        ({"backend_type": "redis"}, {"backend_type": "redis"}, False),
        # the store has no switch: a stale key is dropped, never written back
        ({"enabled": True, "ttl_seconds": 3600}, {"ttl_seconds": 3600}, True),
    ],
)
def test_eval_provider_copy_writes_no_store_switch(store, expected, disabled):
    providers = {"global": {"stores": {}}}
    if store is not None:
        providers["global"]["stores"]["response_cache"] = store
    before = copy.deepcopy(providers)
    eval_copy, cache_disabled = bindings_mod._derived_eval_providers(providers)
    assert eval_copy["global"]["stores"]["response_cache"] == expected
    assert cache_disabled is disabled
    assert providers == before  # the input is never edited in place
