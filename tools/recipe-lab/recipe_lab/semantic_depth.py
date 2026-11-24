"""Propose one observable SystemOne cutoff change on a bound priority recipe.

The supported edit (sub-issue 7, Speed depth): one private binary signal
with an ordered static fallback, so moving its observed cutoff changes which
of the two bound backends serves the requests in between. The standalone
cutoff analysis (`threshold_sites`, `propose_threshold`) backs the
`suggest-depth` CLI; the planner-side enrollment (`add_candidate`, the
per-prefix observations and the paired public-gateway probes) feeds the
development planner.
"""

import copy
import json

import yaml

if __package__:
    from .headroom import unique_fields
    from .loop_evidence import check, digest, is_number, require_isolated
    from .planning_replay import paired_requests, source_binding, token_fit
else:
    from headroom import unique_fields
    from loop_evidence import check, digest, is_number, require_isolated
    from planning_replay import paired_requests, source_binding, token_fit

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


def winner(site, score, others, cutoff):
    """Use the verified priority order, including the parent's on_unknown: match."""
    if score is None or score >= cutoff:
        return site["primary"]
    return site["overlap"] if others else site["fallback"]


def read_observation(row, site):
    score = row["score"]
    check(
        score is None or score_value(score),
        "expected a normalized SystemOne score or null",
    )
    others = row["matched_other_decisions"]
    check(
        others in ([], [site["overlap"]]), "unexpected competing decision observations"
    )
    check(type(row["fast_eligible"]) is bool, "missing fast model eligibility")
    return score, others


def add_candidate(path, doc, space, configs, tasks, cards):
    raw = path.read_bytes()
    packet = json.loads(raw, object_pairs_hook=unique_fields)
    check(
        packet["schema"] == "recipe-lab-depth-evidence-v1"
        and packet["public_model"] == space["public_model"]
        and packet["base_config_sha256"] == space["base_config_sha256"],
        "depth evidence must identify the exact parent configuration and public model",
    )
    sites = space["inventory"].get("semantic_thresholds", [])
    site = next((s for s in sites if s["signal"] == packet["signal"]), None)
    check(site, "signal has no supported single-site priority edit")
    fast, strong = space["fast"], space["strong"]
    aliases = {m: cards[m].get("provider_model_id") or m for m in (fast, strong)}
    observed = {}
    for info, trace in tasks:
        for call, req, reply in paired_requests(trace, space["public_model"]):
            key = (info["measurement"], info["task"], call)
            observed[key] = (info, req, reply)
    scores = set()
    eligible_scores = set()
    by_trace = {}
    seen = set()
    check(
        isinstance(packet["observations"], list),
        "expected development observation rows",
    )
    for row in packet["observations"]:
        call = row["call"]
        key = (row["measurement"], row["task"], call)
        check(
            type(call) is int and key in observed and key not in seen,
            "depth observation must identify one distinct supplied development call",
        )
        seen.add(key)
        info, request, reply = observed[key]
        check(
            row["trace_sha256"] == info["request_sha256"],
            "depth trace fingerprint mismatch",
        )
        score, others = read_observation(row, site)
        actual = source_binding(reply, fast, strong, aliases)
        lane = winner(site, score, others, site["before"])
        check(
            actual in site["bindings"][lane] and reply.get("selected_decision") == lane,
            "depth observation conflicts with the recorded parent decision or backend",
        )
        by_trace.setdefault(key[:2], {})[call] = row
        if (
            score is None
            or score < site["before"]
            or not info["parent_strict"]
            or info["parent_termination"] != "natural_completion"
            or actual != strong
        ):
            continue
        scores.add(score)
        fit = token_fit(
            request,
            reply,
            cards[fast].get("context_window_size"),
            cards[fast].get("max_output_tokens"),
        )
        if (
            not others
            and row["fast_eligible"]
            and fit["status"] in ("unknown", "source_token_estimate_within_limits")
        ):
            eligible_scores.add(score)
    result = {
        "evidence_sha256": digest(raw),
        "observations": len(seen),
        "signal": site["signal"],
        "eligible_scores": sorted(eligible_scores),
        "parent_reasoning_scores": sorted(scores),
    }
    space["depth_evidence"] = result
    for lower in sorted(eligible_scores):
        higher = min((s for s in scores if s > lower), default=None)
        if higher is None:
            continue
        # With only parent grades, probe the smallest observed change first.
        draft, predicate = propose_threshold(
            doc, space["public_model"], site, [lower], [higher]
        )
        data = yaml.safe_dump(draft, sort_keys=False, allow_unicode=True).encode()
        ident = "semantic-threshold-" + site["signal"]
        configs[ident] = data
        space["candidates"].append(
            {
                "id": ident,
                "kind": "semantic_depth",
                "reason": "a passing parent used the strong model on a fast-eligible call below a protected observed score",
                "candidate_sha256": digest(data),
                "predicate": predicate,
                "public_probe_predicate": {
                    **predicate,
                    "parent_sha": space["base_config_sha256"],
                },
            }
        )
        result["protected_score"] = higher
        return by_trace
    result["reason"] = (
        "need a fast-eligible passing parent call and a higher observed reasoning score to protect"
    )
    return by_trace


def score_region(score, site):
    if score is None:
        return "missing"
    if score < site["before"]:
        return "below_parent"
    return "changed" if score < site["after"] else "retained"


def check_probe(case, row, site, fast, aliases, issue, expect):
    """Compare both deployed recipes on one identical public request."""
    parent = row.get("parent")
    if not isinstance(parent, dict):
        issue(case, "parent", "missing paired parent gateway observation", "incomplete")
        return
    expect(case, parent, "request_sha256", row.get("request_sha256"))
    for name in ("score", "matched_other_decisions", "fast_eligible"):
        if name not in row or name not in parent:
            issue(case, name, "missing paired routing input", "incomplete")
            return
        if name != "score" and row[name] != parent[name]:
            issue(case, name, "paired observations have different routing inputs")
            return
    try:
        score, others = read_observation(row, site)
        parent_score, _ = read_observation(parent, site)
    except ValueError as error:
        issue(case, "observation", str(error))
        return
    region = score_region(score, site)
    if score_region(parent_score, site) != region:
        issue(case, "score", "paired scores fall in different threshold regions")
        return
    band = region == "changed"
    if case == "depth-changed" and not (band and not others and row["fast_eligible"]):
        issue(
            case, "score", "expected an eligible request inside the changed score band"
        )
    if case == "depth-protected" and (score is None or score < site["after"]):
        issue(case, "score", "expected a protected reasoning score")
    if case == "depth-overlap" and not (band and others):
        issue(
            case, "score", "expected the competing branch inside the changed score band"
        )
    if case == "depth-missing" and score is not None:
        issue(case, "score", "expected an explicitly unavailable score")
    if case == "long-input" and not (band and not others and not row["fast_eligible"]):
        issue(
            case,
            "fast_eligible",
            "expected a fast-lane context fallback inside the changed band",
        )
    for reading, cutoff in ((parent, site["before"]), (row, site["after"])):
        decision = winner(site, reading["score"], others, cutoff)
        expect(case, reading, "http", 200)
        expect(case, reading, "done", True)
        expect(case, reading, "decision", decision)
        preferred = site["bindings"][decision]
        eligible = [m for m in preferred if m != fast or row["fast_eligible"]]
        if not eligible:
            issue(case, "selected", "no eligible model in the predicted lane")
            continue
        model = eligible[0]
        expect(case, reading, "selected", model)
        if reading.get("returned") not in {model, aliases.get(model, model)}:
            issue(
                case,
                "returned",
                "expected a completed reply from the predicted backend",
            )
