"""Fix selection, conflict analysis, and config mutation for the DSL tuning engine.

Split from engine.py to satisfy the structural line-count limit.  This module
contains the higher-level orchestration that sits on top of the core analytical
primitives (trace walking, decomposition, score computation, regression checking)
which remain in engine.py.
"""

from __future__ import annotations

import copy
from typing import Any

from .engine import (
    EPSILON,
    DSLConfig,
    FailureClassification,
    Fix,
    StructuralFix,
    TraceLeaf,
    check_regression_threshold,
    check_regression_weight,
    compute_score,
    compute_threshold_fix,
    compute_weight_fix,
    decompose_projection,
    find_failing_leaves,
    find_false_positive_leaves,
    probe_severity,
)

# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _leaf_to_dict(leaf: TraceLeaf) -> dict:
    return {
        "signal_type": leaf.signal_type,
        "signal_name": leaf.signal_name,
        "matched": leaf.matched,
        "confidence": leaf.confidence,
        "path": leaf.path,
    }


def _fc_to_dict(fc: FailureClassification) -> dict:
    d: dict[str, Any] = {
        "kind": fc.kind,
        "signal_type": fc.leaf.signal_type,
        "signal_name": fc.leaf.signal_name,
    }
    if fc.score_name:
        d["score_name"] = fc.score_name
    if fc.mapping_name:
        d["mapping_name"] = fc.mapping_name
    if fc.threshold is not None:
        d["threshold"] = fc.threshold
    if fc.threshold_dir:
        d["threshold_dir"] = fc.threshold_dir
    if fc.current_score is not None:
        d["current_score"] = fc.current_score
    if fc.gap is not None:
        d["gap"] = fc.gap
    if fc.active_terms:
        d["active_terms"] = fc.active_terms
    if fc.silent_terms:
        d["silent_terms"] = fc.silent_terms
    return d


# ---------------------------------------------------------------------------
# Priority conflict analysis
# ---------------------------------------------------------------------------


def _classify_fp_leaf(
    leaf: TraceLeaf,
    dsl: DSLConfig,
    sc: dict[str, float],
    ms: dict[str, list[str]],
) -> dict | None:
    """Classify a false-positive leaf; returns parametric dict or None (structural)."""
    if leaf.signal_type != "projection":
        return None
    mapping, output_band = dsl.find_mapping_for_projection(leaf.signal_name)
    if not mapping or not output_band:
        return None
    score_formula = dsl.find_score(mapping.source_score)
    if not score_formula:
        return None
    current_score = compute_score(score_formula, sc, ms)
    base = {
        "leaf": leaf,
        "score_name": mapping.source_score,
        "mapping_name": mapping.name,
        "current_score": current_score,
    }
    if "gte" in output_band:
        return {
            **base,
            "threshold": output_band["gte"],
            "threshold_dir": "gte",
            "gap": current_score - output_band["gte"],
            "fix_direction": "raise_threshold",
        }
    if "lt" in output_band:
        return {
            **base,
            "threshold": output_band["lt"],
            "threshold_dir": "lt",
            "gap": output_band["lt"] - current_score,
            "fix_direction": "lower_threshold",
        }
    return None


def analyze_priority_conflict(probe_result: dict, dsl: DSLConfig) -> dict:
    """When expected decision matches but a higher-priority one also matches."""
    expected = probe_result["expected"]
    actual = probe_result["actual"]
    traces = probe_result.get("eval_trace", [])
    sc = probe_result.get("signal_confidences", {})
    ms = probe_result.get("matched_signals", {})

    expected_trace = actual_trace = None
    for t in traces:
        if t.get("decision_name") == expected:
            expected_trace = t
        if t.get("decision_name") == actual:
            actual_trace = t

    if not actual_trace or not actual_trace.get("matched"):
        return {
            "failure_kind": "unknown",
            "error": "competing decision trace not found",
        }

    expected_matched = expected_trace.get("matched", False) if expected_trace else False
    actual_root = actual_trace.get("root_trace", {})
    fp_leaves = find_false_positive_leaves(actual_root)

    structural_leaves, parametric_leaves = [], []
    for leaf in fp_leaves:
        result = _classify_fp_leaf(leaf, dsl, sc, ms)
        if result:
            parametric_leaves.append(result)
        else:
            structural_leaves.append(leaf)

    kind = "priority_conflict"
    if structural_leaves and not parametric_leaves:
        kind = "priority_conflict_structural"
    elif parametric_leaves:
        kind = "priority_conflict_parametric"

    rule_fix = None
    if structural_leaves:
        rule_fix = {
            "fix_type": "structural_rule_change",
            "decision": actual,
            "action": "remove_false_positive_branches",
            "description": (
                f"Decision '{actual}' has {len(structural_leaves)} "
                f"non-projection signals that false-positive match."
            ),
            "remove_signals": [
                {"type": leaf.signal_type, "name": leaf.signal_name}
                for leaf in structural_leaves
            ],
        }

    return {
        "failure_kind": kind,
        "expected_decision": expected,
        "expected_matched": expected_matched,
        "competing_decision": actual,
        "false_positive_leaves": [_leaf_to_dict(leaf) for leaf in fp_leaves],
        "structural_fp": [_leaf_to_dict(leaf) for leaf in structural_leaves],
        "parametric_fp": [
            {
                "signal_type": p["leaf"].signal_type,
                "signal_name": p["leaf"].signal_name,
                "score_name": p["score_name"],
                "mapping_name": p["mapping_name"],
                "threshold": p["threshold"],
                "threshold_dir": p["threshold_dir"],
                "current_score": round(p["current_score"], 6),
                "gap": round(p["gap"], 6),
                "fix_direction": p["fix_direction"],
            }
            for p in parametric_leaves
        ],
        "structural_rule_fix": rule_fix,
    }


# ---------------------------------------------------------------------------
# Full analytical diagnosis for one probe
# ---------------------------------------------------------------------------


def diagnose_probe(probe_result: dict, dsl: DSLConfig) -> dict:
    """Run the full trace-based diagnostic pipeline for a single misrouted probe.

    Handles:
      A. Expected decision doesn't match -> FindFailingLeaves
      B. Expected decision matches but is outprioritized -> Priority conflict
    """
    expected = probe_result["expected"]
    traces = probe_result.get("eval_trace", [])
    sc = probe_result.get("signal_confidences", {})

    expected_trace = None
    for t in traces:
        if t.get("decision_name") == expected:
            expected_trace = t
            break

    if expected_trace is None:
        return {
            "error": f"No trace found for expected decision {expected}",
            "failing_leaves": [],
            "decompositions": [],
            "failure_kind": "missing_trace",
        }

    if expected_trace.get("matched", False):
        return analyze_priority_conflict(probe_result, dsl)

    ms = probe_result.get("matched_signals", {})
    root = expected_trace.get("root_trace", {})
    failing_leaves = find_failing_leaves(root)

    decompositions = []
    for leaf in failing_leaves:
        if leaf.signal_type == "projection":
            dec = decompose_projection(leaf, sc, dsl, ms)
        else:
            dec = FailureClassification(kind="structural", leaf=leaf)
        decompositions.append(dec)

    kinds = [d.kind for d in decompositions]
    overall = "parametric" if any(k == "parametric" for k in kinds) else "structural"

    return {
        "expected_decision": expected,
        "trace_matched": False,
        "trace_confidence": expected_trace.get("confidence", 0.0),
        "failing_leaves": [_leaf_to_dict(leaf) for leaf in failing_leaves],
        "decompositions": [_fc_to_dict(d) for d in decompositions],
        "failure_kind": overall,
    }


# ---------------------------------------------------------------------------
# Fix selection — severity-weighted constraint-satisfying strategy
# ---------------------------------------------------------------------------


def _find_probe_best_category(probe: dict, decision_name: str) -> str | None:
    """Find the best-matching KB category leaf for a probe in a given decision."""
    for t in probe.get("eval_trace", []):
        if t.get("decision_name") != decision_name or not t.get("matched"):
            continue
        for child in t.get("root_trace", {}).get("children", []):
            if not child.get("matched") or child.get("node_type") != "leaf":
                continue
            sname = child.get("signal_name", "")
            stype = child.get("signal_type", "")
            if stype in ("category_kb", "kb") or sname.startswith("__best__:"):
                return sname
    return None


def _batch_structural_fix(
    diagnoses: list[dict],
    probe_cache: list[dict],
    severity_fn=None,
) -> StructuralFix | None:
    """Regression-aware batch structural fix.  Collects all proposed category
    removals, then keeps only those with positive severity-weighted net."""
    if severity_fn is None:
        severity_fn = probe_severity

    removal_candidates: dict[tuple[str, str, str], list[str]] = {}
    decision_name = None

    for diag in diagnoses:
        rule_fix = diag.get("structural_rule_fix")
        if not rule_fix:
            continue
        decision_name = rule_fix["decision"]
        for sig in rule_fix.get("remove_signals", []):
            key = (rule_fix["decision"], sig["type"], sig["name"])
            removal_candidates.setdefault(key, [])

    if not removal_candidates or not decision_name:
        return None

    probe_best_cats: dict[str, str] = {}
    for probe in probe_cache:
        cat = _find_probe_best_category(probe, decision_name)
        if cat:
            probe_best_cats[probe.get("id", "?")] = cat

    net_positive = []
    for (_dec, sig_type, sig_name), _ in removal_candidates.items():
        gain, loss, fixes, regressions = 0, 0, [], []
        for probe in probe_cache:
            pid = probe.get("id", "?")
            if probe_best_cats.get(pid, "") != sig_name:
                continue
            w = severity_fn(probe)
            if not probe.get("correct", False):
                gain += w
                fixes.append(pid)
            else:
                loss += w
                regressions.append(pid)
        net = gain - loss
        if net > 0:
            net_positive.append(
                {
                    "type": sig_type,
                    "name": sig_name,
                    "net": net,
                    "fixes": len(fixes),
                    "regressions": len(regressions),
                }
            )

    if not net_positive:
        return None

    net_positive.sort(key=lambda x: x["net"], reverse=True)
    return StructuralFix(
        decision_name=decision_name,
        action="remove_false_positive_branches",
        description=(
            f"Batch-remove {len(net_positive)} false-positive categories from "
            f"'{decision_name}' OR condition."
        ),
        remove_signals=[{"type": r["type"], "name": r["name"]} for r in net_positive],
    )


def _reconstruct_fc(dec_dict: dict, matched: bool) -> FailureClassification:
    return FailureClassification(
        kind="parametric",
        leaf=TraceLeaf(
            signal_type=dec_dict["signal_type"],
            signal_name=dec_dict["signal_name"],
            matched=matched,
            confidence=0.0,
            path="",
        ),
        score_name=dec_dict.get("score_name"),
        mapping_name=dec_dict.get("mapping_name"),
        threshold=dec_dict.get("threshold"),
        threshold_dir=dec_dict.get("threshold_dir"),
        current_score=dec_dict.get("current_score"),
        gap=dec_dict.get("gap"),
        active_terms=dec_dict.get("active_terms", []),
        silent_terms=dec_dict.get("silent_terms", []),
    )


def _add_parametric_candidates(
    fc: FailureClassification,
    candidates: list,
    probe_cache: list[dict],
    dsl: DSLConfig,
    severity_fn=None,
):
    threshold_fix = compute_threshold_fix(fc)
    if threshold_fix:
        mapping, _ = dsl.find_mapping_for_projection(fc.leaf.signal_name)
        if mapping:
            threshold_fix = check_regression_threshold(
                threshold_fix, mapping, probe_cache, dsl, severity_fn
            )
            candidates.append((threshold_fix, fc))

    weight_fix = compute_weight_fix(fc)
    if weight_fix and fc.active_terms:
        score_formula = dsl.find_score(fc.score_name or "")
        mapping, _ = dsl.find_mapping_for_projection(fc.leaf.signal_name)
        if score_formula and mapping:
            best_term = max(fc.active_terms, key=lambda t: t["confidence"])
            weight_fix = check_regression_weight(
                weight_fix,
                score_formula,
                mapping,
                probe_cache,
                best_term["signal"],
                severity_fn,
            )
            candidates.append((weight_fix, fc))


def _fp_threshold_candidate(
    pf: dict,
    dsl: DSLConfig,
    probe_cache: list[dict],
    severity_fn,
) -> tuple[Fix, FailureClassification] | None:
    """Build a threshold fix candidate from a false-positive parametric entry."""
    fix_dir = pf.get("fix_direction", "")
    if fix_dir != "raise_threshold":
        return None
    threshold = pf.get("threshold")
    current_score = pf.get("current_score")
    if threshold is None or current_score is None:
        return None
    fc = FailureClassification(
        kind="parametric",
        leaf=TraceLeaf(
            signal_type=pf["signal_type"],
            signal_name=pf["signal_name"],
            matched=True,
            confidence=0.0,
            path="",
        ),
        score_name=pf.get("score_name"),
        mapping_name=pf.get("mapping_name"),
        threshold=threshold,
        threshold_dir=pf.get("threshold_dir"),
        current_score=current_score,
        gap=pf.get("gap"),
    )
    new_t = current_score + EPSILON
    fix = Fix(
        fix_type="threshold",
        target=fc.mapping_name or "",
        param_path=(
            f"projections.mappings[{fc.mapping_name}]"
            f".outputs[{fc.leaf.signal_name}].gte"
        ),
        old_value=threshold,
        new_value=round(new_t, 4),
        explanation=(
            f"Raise threshold for {fc.leaf.signal_name} to prevent false-positive"
        ),
    )
    if abs(fix.new_value - fix.old_value) <= 1e-6:
        return None
    mapping, _ = dsl.find_mapping_for_projection(fc.leaf.signal_name)
    if not mapping:
        return None
    fix = check_regression_threshold(fix, mapping, probe_cache, dsl, severity_fn)
    return (fix, fc)


def select_fix(
    diagnoses: list[dict],
    probe_cache: list[dict],
    dsl: DSLConfig,
    severity_fn=None,
) -> Fix | StructuralFix | None:
    """Choose the best analytical fix from all failures.

    Strategy:
      1. Regression-aware batch structural fix (priority conflicts).
      2. Parametric threshold/weight fixes with regression checking.
      3. Accept the fix with the best severity-weighted net improvement.
    """
    has_structural = any(d.get("structural_rule_fix") for d in diagnoses)
    if has_structural:
        batch_fix = _batch_structural_fix(diagnoses, probe_cache, severity_fn)
        if batch_fix:
            return batch_fix

    candidates: list[tuple[Fix, FailureClassification]] = []

    for diag in diagnoses:
        for dec_dict in diag.get("decompositions", []):
            if dec_dict["kind"] != "parametric":
                continue
            fc = _reconstruct_fc(dec_dict, matched=False)
            _add_parametric_candidates(fc, candidates, probe_cache, dsl, severity_fn)

        for pf in diag.get("parametric_fp", []):
            candidate = _fp_threshold_candidate(pf, dsl, probe_cache, severity_fn)
            if candidate:
                candidates.append(candidate)

    if not candidates:
        return None

    best_fix, best_score = None, -float("inf")
    for fix, _fc in candidates:
        score = fix.net_improvement
        if len(fix.regressions) == 0:
            score += 10
        if score > best_score:
            best_score = score
            best_fix = fix

    if best_fix and best_fix.net_improvement <= 0 and best_fix.regressions:
        return None
    return best_fix


# ---------------------------------------------------------------------------
# Apply fixes to YAML config dict
# ---------------------------------------------------------------------------


def _update_mapping_outputs(m: dict, fix: Fix) -> None:
    """Update threshold values in mapping outputs and maintain partition consistency."""
    for out in m.get("outputs", []):
        if "gte" in out and abs(out["gte"] - fix.old_value) < 1e-9:
            out["gte"] = fix.new_value
        elif "lt" in out and abs(out["lt"] - fix.old_value) < 1e-9:
            out["lt"] = fix.new_value
    gte_outputs = [o for o in m["outputs"] if "gte" in o]
    lt_outputs = [o for o in m["outputs"] if "lt" in o]
    if gte_outputs and lt_outputs:
        boundary = gte_outputs[0]["gte"]
        for lt_out in lt_outputs:
            lt_out["lt"] = boundary


def _update_score_weight(score: dict, signal_name: str, new_weight: float) -> None:
    """Update a specific input weight in a score formula."""
    for inp in score.get("inputs", []):
        if inp["name"] == signal_name:
            inp["weight"] = new_weight
            break


def apply_structural_fix(cfg: dict, sfix: StructuralFix) -> dict:
    """Remove specified signal leaves from a decision's OR condition."""
    cfg = copy.deepcopy(cfg)
    for dec in cfg["routing"]["decisions"]:
        if dec["name"] != sfix.decision_name:
            continue
        if sfix.action == "remove_false_positive_branches":
            remove_set = {(s["type"], s["name"]) for s in sfix.remove_signals}
            rules = dec.get("rules", {})
            conditions = rules.get("conditions", [])
            new_conditions = [
                c
                for c in conditions
                if (c.get("type", ""), c.get("name", "")) not in remove_set
            ]
            if new_conditions:
                rules["conditions"] = new_conditions
                if len(new_conditions) == 1:
                    rules["operator"] = "AND"
            break
    return cfg


def apply_fix_to_config(cfg: dict, fix: Fix, dsl: DSLConfig) -> dict:
    """Mutate config dict for a parametric fix, maintaining partition consistency."""
    cfg = copy.deepcopy(cfg)
    proj = cfg["routing"]["projections"]
    if fix.fix_type == "threshold":
        for m in proj.get("mappings", []):
            if m["name"] != fix.target:
                continue
            _update_mapping_outputs(m, fix)
            break
    elif fix.fix_type == "weight":
        signal_name = fix.param_path.split("inputs[")[1].split("]")[0]
        for s in proj.get("scores", []):
            if s["name"] != fix.target:
                continue
            _update_score_weight(s, signal_name, fix.new_value)
            break
    return cfg
