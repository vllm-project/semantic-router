"""Threshold-search proposer tests: determinism, clamping, one-variable rule,
and recipe selection in the canonical v0.5 layout."""

from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest
import yaml
from recipe_lab.contract import ContractError
from recipe_lab.layout import find_signal, recipe_signals, select_recipe
from recipe_lab.proposers import rule_threshold
from recipe_lab.proposers.threshold_search import propose_thresholds, write_candidates
from tests.fixtures import VALID_SPEC  # noqa: F401  (keeps import sanity)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
MAINTAINED_RECIPES = Path(__file__).resolve().parents[3] / "config" / "recipes"

BASE_CONFIG = {
    "version": "v0.5",
    "recipes": [
        {
            "name": "default",
            "signals": {"systemone": [{"name": "depth", "predicate": {"gte": 0.6}}]},
            "decisions": [{"name": "cloud_frontier_reasoning"}],
        }
    ],
}


def _two_recipes() -> dict:
    return yaml.safe_load((FIXTURES / "two_recipes.yaml").read_text(encoding="utf-8"))


def _depth(config: dict) -> float:
    return config["recipes"][0]["signals"]["systemone"][0]["predicate"]["gte"]


def test_propose_generates_one_variable_candidates():
    candidates = propose_thresholds(BASE_CONFIG, "depth", [-0.1, 0.05, 0.1])
    assert [c.new_value for c in candidates] == [0.5, 0.65, 0.7]
    for candidate in candidates:
        assert candidate.old_value == 0.6
        # exactly one variable changed per candidate
        assert _depth(candidate.config) == candidate.new_value
        assert (
            candidate.config["recipes"][0]["decisions"]
            == BASE_CONFIG["recipes"][0]["decisions"]
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
    with pytest.raises(ContractError, match="not found in recipe 'default'"):
        propose_thresholds(BASE_CONFIG, "no-such-signal", [0.05])


def test_propose_signal_without_gte_rejected():
    config = copy.deepcopy(BASE_CONFIG)
    config["recipes"][0]["signals"]["systemone"][0] = {
        "name": "depth",
        "predicate": {"lt": 0.6},
    }
    with pytest.raises(ContractError, match="gte"):
        propose_thresholds(config, "depth", [0.05])


def test_propose_finds_the_signal_whatever_its_family():
    config = copy.deepcopy(BASE_CONFIG)
    config["recipes"][0]["signals"] = {
        "input_modality": [{"name": "has_images", "modality": "image"}],
        "privacy": [{"name": "depth", "predicate": {"gte": 0.6}}],
    }
    candidates = propose_thresholds(config, "depth", [0.1])
    assert [c.new_value for c in candidates] == [0.7]


def test_propose_signal_name_declared_twice_rejected():
    config = copy.deepcopy(BASE_CONFIG)
    config["recipes"][0]["signals"]["privacy"] = [
        {"name": "depth", "predicate": {"gte": 0.2}}
    ]
    with pytest.raises(ContractError, match="declared 2 times"):
        propose_thresholds(config, "depth", [0.1])


def test_legacy_routing_block_is_not_read():
    # Before v0.5 the signals and decisions lived under a top-level routing
    # block; recipe-lab reads the canonical per-recipe layout only.
    legacy = {
        "version": "v0.3",
        "routing": {
            "signals": {"decision": [{"name": "depth", "predicate": {"gte": 0.6}}]},
            "decisions": [{"name": "cloud_frontier_reasoning"}],
        },
    }
    rejected = re.escape("no v0.5 'recipes' list")
    with pytest.raises(ContractError, match=rejected):
        propose_thresholds(legacy, "depth", [0.1])
    with pytest.raises(ContractError, match=rejected):
        rule_threshold.find_threshold_sites(legacy)


def test_several_recipes_need_a_selection():
    with pytest.raises(ContractError, match=r"--recipe <name> or --public-model"):
        propose_thresholds(_two_recipes(), "depth", [0.1])
    with pytest.raises(ContractError, match=r"--recipe <name> or --public-model"):
        rule_threshold.find_threshold_sites(_two_recipes())


def test_named_recipe_is_edited_and_the_others_left_alone():
    base = _two_recipes()
    untouched = copy.deepcopy(base["recipes"][1])
    candidates = propose_thresholds(
        base, "depth", [0.1], recipe=select_recipe(base, recipe="balanced")
    )
    assert [c.new_value for c in candidates] == [0.7]
    assert _depth(candidates[0].config) == 0.7
    assert candidates[0].config["recipes"][1] == untouched
    assert _depth(base) == 0.6


def test_public_model_selects_the_recipe_its_entrypoint_serves():
    base = _two_recipes()
    recipe = select_recipe(base, public_model="vllm-sr/mom-flash")
    assert recipe.name == "speed-first"
    candidates = rule_threshold.propose_rule_thresholds(
        base, "depth", values=[0.5], recipe=recipe
    )
    applied = rule_threshold.find_threshold_sites(candidates[0].config, recipe)
    assert [(s.decision, s.value, s.negated) for s in applied] == [
        ("quick_route", 0.5, True)
    ]
    balanced = select_recipe(candidates[0].config, recipe="balanced")
    assert [
        s.value
        for s in rule_threshold.find_threshold_sites(candidates[0].config, balanced)
    ] == [0.6]


@pytest.mark.parametrize(
    ("selector", "message", "alternatives"),
    [
        (
            {"recipe": "no-such"},
            "has no recipe 'no-such'",
            "'balanced' (vllm-sr/mom-blend)",
        ),
        (
            {"public_model": "vllm-sr/unknown"},
            "no entrypoint serves",
            "served: vllm-sr/mom-blend",
        ),
    ],
)
def test_unknown_selection_names_the_alternatives(selector, message, alternatives):
    with pytest.raises(ContractError, match=re.escape(message)) as caught:
        select_recipe(_two_recipes(), **selector)
    assert alternatives in str(caught.value)


@pytest.mark.parametrize(
    "document", ["agent/config.yaml", "multi-objective/config.yaml"]
)
def test_maintained_documents_resolve_their_recipes(document):
    # The real recipes ship one recipe per public model, or a single one.
    config = yaml.safe_load((MAINTAINED_RECIPES / document).read_text(encoding="utf-8"))
    with_decisions = {r["name"] for r in config["recipes"] if r.get("decisions")}
    entry = next(e for e in config["entrypoints"] if e["recipe"] in with_decisions)
    recipe = select_recipe(config, public_model=entry["model_names"][0])
    assert recipe.name == entry["recipe"]
    if len(with_decisions) == 1:  # a single-recipe document needs no selector
        assert select_recipe(config).name == entry["recipe"]
    # the recipe's threshold-bearing signals are the proposer's material
    thresholds = {
        signal["name"]: signal["predicate"]["gte"]
        for signal in recipe_signals(recipe.get(config))
        if isinstance(signal.get("predicate"), dict)
        and isinstance(signal["predicate"].get("gte"), (int, float))
    }
    assert thresholds
    signal_name = next(iter(thresholds))
    candidate = propose_thresholds(config, signal_name, [0.05], recipe=recipe)[0]
    moved = find_signal(candidate.config, recipe, signal_name)["predicate"]["gte"]
    assert moved == round(min(1.0, max(0.0, thresholds[signal_name] + 0.05)), 6)
    for site in rule_threshold.find_threshold_sites(config, recipe):
        node = config
        for key in site.path:
            node = node[key]
        assert node[site.op] == site.value


def test_write_candidates_writes_yaml(tmp_path):
    candidates = propose_thresholds(BASE_CONFIG, "depth", [-0.1])
    paths = write_candidates(candidates, tmp_path)
    assert len(paths) == 1
    loaded = yaml.safe_load(paths[0].read_text(encoding="utf-8"))
    assert _depth(loaded) == 0.5
