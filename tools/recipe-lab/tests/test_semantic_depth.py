"""Standalone cutoff analysis for the Speed depth threshold (sub-issue 7).

The planner-side enrollment and per-prefix probes belong to the schedule
owner (sub-issue 9); these cases exercise only the recipes-free cutoff
surface: threshold site inventory and the protected-score proposal.
"""

from __future__ import annotations

import copy

import pytest
import yaml
from recipe_lab.semantic_depth import propose_threshold, score_value, threshold_sites
from tests.fixtures import DEPTH_FAST, DEPTH_PUBLIC, DEPTH_STRONG, depth_doc


@pytest.fixture
def doc():
    document = depth_doc()
    document["recipes"].append({"name": "unrelated", "strategy": "priority"})
    return document


@pytest.fixture
def site(doc):
    (found,) = threshold_sites(doc["recipes"][0], DEPTH_STRONG, DEPTH_FAST)
    return found


@pytest.mark.parametrize(
    "value,expected",
    [(0, True), (1, True), (0.5, True), (-0.1, False), (1.0001, False), (True, False)],
)
def test_score_value_bounds(value, expected):
    assert bool(score_value(value)) is expected


def test_inventory_reports_the_supported_edit(site):
    assert site["feature"] == "semantic_depth"
    assert site["signal"] == "depth" and site["before"] == 0.6
    assert site["primary"] == "reasoning"
    assert site["overlap"] == "tools" and site["fallback"] == "fast"
    assert site["bindings"] == {
        "reasoning": [DEPTH_STRONG],
        "tools": [DEPTH_STRONG],
        "fast": [DEPTH_FAST],
    }


def test_unsupported_shapes_return_no_sites():
    recipe = depth_doc()["recipes"][0]
    assert threshold_sites(recipe, DEPTH_STRONG, DEPTH_STRONG) == []
    changed = copy.deepcopy(recipe)
    changed["strategy"] = "RoundRobin"
    assert threshold_sites(changed, DEPTH_STRONG, DEPTH_FAST) == []
    changed = copy.deepcopy(recipe)
    changed["candidate_requirements"]["capabilities"] = "any"
    assert threshold_sites(changed, DEPTH_STRONG, DEPTH_FAST) == []
    changed = copy.deepcopy(recipe)
    changed["decisions"][0]["priority"] = 2
    assert threshold_sites(changed, DEPTH_STRONG, DEPTH_FAST) == []
    changed = copy.deepcopy(recipe)
    changed["decisions"][1]["algorithm"] = {"type": "loop"}
    assert threshold_sites(changed, DEPTH_STRONG, DEPTH_FAST) == []
    changed = copy.deepcopy(recipe)
    changed["decisions"][2]["rules"] = changed["decisions"][1]["rules"]
    assert threshold_sites(changed, DEPTH_STRONG, DEPTH_FAST) == []
    changed = copy.deepcopy(recipe)
    changed["decisions"][1]["modelRefs"] = [{"model": "C"}]
    assert threshold_sites(changed, DEPTH_STRONG, DEPTH_FAST) == []
    changed = copy.deepcopy(recipe)
    changed["signals"]["systemone"][0]["predicate"] = {"gte": 1.5}
    assert threshold_sites(changed, DEPTH_STRONG, DEPTH_FAST) == []
    changed = copy.deepcopy(recipe)
    changed["projections"] = [{"name": "x"}]
    assert threshold_sites(changed, DEPTH_STRONG, DEPTH_FAST) == []


def test_graded_scores_protect_every_retained_reasoning_observation(doc, site):
    draft, predicate = propose_threshold(
        doc, DEPTH_PUBLIC, site, [0.62, 0.71], [0.83, 0.94]
    )
    assert predicate["after"] == (0.71 + 0.83) / 2
    # the unrelated recipe is preserved verbatim; only the signal moves
    assert draft["recipes"][1] == doc["recipes"][1]
    assert draft["recipes"][0] is not doc["recipes"][0]
    rendered = yaml.safe_load(
        yaml.safe_dump(draft["recipes"][0], sort_keys=False, allow_unicode=True)
    )
    moved = rendered["signals"]["systemone"][0]["predicate"]["gte"]
    assert moved == (0.71 + 0.83) / 2
    assert rendered["decisions"] == doc["recipes"][0]["decisions"]
    assert propose_threshold(doc, DEPTH_PUBLIC, site, [0.83], [0.83]) is None
    assert propose_threshold(doc, DEPTH_PUBLIC, site, [0.7], []) is None
    assert propose_threshold(doc, DEPTH_PUBLIC, site, [0.65], [0.5]) is None


def test_site_mismatch_is_refused(doc, site):
    other_site = {**site, "before": 0.2}
    with pytest.raises(ValueError, match="site differs"):
        propose_threshold(doc, DEPTH_PUBLIC, other_site, [0.4], [0.8])
