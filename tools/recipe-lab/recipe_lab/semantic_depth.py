"""Propose one observable SystemOne cutoff change on a bound priority recipe.

The supported edit (sub-issue 7, Speed depth): one private binary signal
with an ordered static fallback, so moving its observed cutoff changes which
of the two bound backends serves the requests in between. The planner-side
enrollment (`add_candidate`, per-prefix evidence, winner/read_observation
and the paired probes) arrives with the schedule owner; this module keeps
the standalone cutoff analysis only.
"""

import copy

from .loop_evidence import check, is_number, require_isolated

# The supported edit is exactly one primary lane, one overlap lane and one
# rule-less fallback lane.
SUPPORTED_LANES = 3


def score_value(value):
    return is_number(value) and 0 <= value <= 1


def references(rule, name):
    if not isinstance(rule, dict):
        return False
    return (rule.get("type"), rule.get("name")) == ("systemone", name) or any(
        references(child, name) for child in rule.get("conditions", [])
    )


def threshold_sites(recipe, strong, fast):
    """The supported edit has one private binary signal and an ordered static fallback."""
    if strong == fast:
        return []
    if recipe.get("strategy") != "priority" or recipe.get("candidate_requirements") != {
        "context": "known_limits",
        "capabilities": "declared",
    }:
        return []
    lanes = recipe.get("decisions", [])
    priorities = [lane.get("priority") for lane in lanes]
    names = [lane.get("name") for lane in lanes]
    if (
        len(lanes) != SUPPORTED_LANES
        or len(set(names)) != len(lanes)
        or any(type(p) is not int for p in priorities)
        or len(set(priorities)) != len(lanes)
        or any(lane.get("algorithm") != {"type": "static"} for lane in lanes)
    ):
        return []
    primary, overlap, fallback = sorted(lanes, key=lambda d: -d["priority"])
    if not overlap.get("rules") or fallback.get("rules"):
        return []
    bindings = {}
    for lane in lanes:
        refs = lane.get("modelRefs", [])
        if not refs or any(r not in ({"model": strong}, {"model": fast}) for r in refs):
            return []
        bindings[lane["name"]] = [r["model"] for r in refs]
    if bindings[primary["name"]][0] != strong or bindings[fallback["name"]][0] != fast:
        return []
    result = []
    signals = recipe.get("signals", {}).get("systemone", [])
    for signal in signals:
        name = signal["name"]
        rules = primary.get("rules", {})
        if (
            sum(s["name"] == name for s in signals) != 1
            or signal.get("question", {}).get("type") != "noul"
            or set(signal.get("predicate", {})) != {"gte"}
            or not score_value(signal["predicate"]["gte"])
            or rules
            != {
                "operator": rules.get("operator"),
                "conditions": [{"type": "systemone", "name": name}],
                "on_unknown": "match",
            }
            or rules.get("operator") not in ("OR", "AND")
            or references(overlap["rules"], name)
            or recipe.get("projections")
        ):
            continue
        result.append(
            {
                "feature": "semantic_depth",
                "signal": name,
                "before": signal["predicate"]["gte"],
                "primary": primary["name"],
                "overlap": overlap["name"],
                "fallback": fallback["name"],
                "bindings": bindings,
            }
        )
    return result


def propose_threshold(doc, public_model, site, eligible_scores, protected_scores):
    """Move observed eligible scores below the lowest protected reasoning score."""
    original = require_isolated(doc, public_model)
    strong = site["bindings"][site["primary"]][0]
    fast = site["bindings"][site["fallback"]][0]
    check(
        site in threshold_sites(original, strong, fast),
        "threshold site differs from the target recipe",
    )
    eligible, protected = list(eligible_scores), list(protected_scores)
    check(
        all(score_value(s) for s in [*eligible, *protected]), "invalid normalized score"
    )
    if not protected or min(protected) < site["before"]:
        return None
    higher = min(protected)
    lower = max((s for s in eligible if site["before"] <= s < higher), default=None)
    if lower is None:
        return None
    cutoff = lower + (higher - lower) / 2
    if cutoff <= lower:
        cutoff = higher
    recipe = copy.deepcopy(original)
    signals = recipe["signals"]["systemone"]
    recipe["signals"] = {
        **recipe["signals"],
        "systemone": [
            {**s, "predicate": {"gte": cutoff}} if s["name"] == site["signal"] else s
            for s in signals
        ],
    }
    draft = {**doc, "recipes": [recipe if r is original else r for r in doc["recipes"]]}
    return draft, {**site, "after": cutoff}
