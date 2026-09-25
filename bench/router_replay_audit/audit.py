#!/usr/bin/env python3
"""Semantic Router counterfactual/state-residual PoC v0.1.

Zero third-party dependencies.
Synthetic order/grouping fixtures are stress-test counterexamples, not upstream bug claims.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

from replay_adapter import adapt_replay

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "fixtures"


def has_signal(bundle: dict[str, Any], family: str, name: str) -> bool:
    values = bundle.get("signals", {}).get(family, [])
    return isinstance(values, list) and name in values


def condition_matches(bundle: dict[str, Any], cond: dict[str, Any]) -> bool:
    kind = cond["type"]
    if kind == "signal_contains":
        return has_signal(bundle, cond["family"], cond["name"])
    if kind == "signal_value_gte":
        value = bundle.get("signal_values", {}).get(cond["key"])
        return value is not None and float(value) >= float(cond["threshold"])
    if kind == "signal_value_lt":
        value = bundle.get("signal_values", {}).get(cond["key"])
        return value is not None and float(value) < float(cond["threshold"])
    if kind == "always":
        return True
    raise ValueError(f"unsupported condition type: {kind}")


def evaluate_policy(bundle: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    decisions = sorted(
        policy.get("decisions", []),
        key=lambda d: int(d.get("priority", 0)),
        reverse=True,
    )
    for decision in decisions:
        op = decision.get("operator", "AND").upper()
        matches = [condition_matches(bundle, c) for c in decision.get("conditions", [])]
        ok = all(matches) if op == "AND" else any(matches)
        if ok:
            return {
                "decision": decision["name"],
                "selected_model": decision.get("selected_model"),
                "priority": decision.get("priority"),
            }
    return {
        "decision": policy.get("default_decision"),
        "selected_model": policy.get("default_model"),
        "priority": None,
    }


def baseline_reproduction(fixture: dict[str, Any]) -> dict[str, Any]:
    bundle = adapt_replay(fixture["replay"])
    got = evaluate_policy(bundle, fixture["policy"])
    expected_decision = bundle.get("decision")
    expected_model = bundle.get("selected_model")
    decision_ok = got["decision"] == expected_decision
    model_ok = expected_model in (None, "") or got["selected_model"] == expected_model
    return {
        "status": "PASS" if decision_ok and model_ok else "FAIL",
        "expected_decision": expected_decision,
        "actual_decision": got["decision"],
        "expected_model": expected_model,
        "actual_model": got["selected_model"],
    }


def enumerate_single_signal_removals(bundle: dict[str, Any]):
    signals = bundle.get("signals", {})
    for family, names in signals.items():
        if not isinstance(names, list):
            continue
        for name in names:
            mutated = copy.deepcopy(bundle)
            mutated["signals"][family] = [x for x in names if x != name]
            yield {
                "cost": 1,
                "mutation": f"remove {family}:{name}",
                "bundle": mutated,
            }


def discrete_counterfactual_margin(fixture: dict[str, Any]) -> dict[str, Any]:
    bundle = adapt_replay(fixture["replay"])
    baseline = evaluate_policy(bundle, fixture["policy"])
    scope = fixture.get("counterfactual_scope")
    allowed = None
    if isinstance(scope, list):
        allowed = {(str(x["family"]), str(x["name"])) for x in scope}

    candidates = []
    for item in enumerate_single_signal_removals(bundle):
        # Optional fixture scope keeps a counterfactual semantically focused.
        # Without it, all single-signal removals are considered.
        mutation_tokens = item["mutation"].removeprefix("remove ").split(":", 1)
        if allowed is not None and tuple(mutation_tokens) not in allowed:
            continue
        got = evaluate_policy(item["bundle"], fixture["policy"])
        if (
            got["decision"] != baseline["decision"]
            or got["selected_model"] != baseline["selected_model"]
        ):
            candidates.append(
                {
                    "cost": item["cost"],
                    "mutation": item["mutation"],
                    "decision": got["decision"],
                    "selected_model": got["selected_model"],
                }
            )
    if not candidates:
        return {
            "status": "NO_FLIP_FOUND",
            "margin": None,
            "baseline_decision": baseline["decision"],
        }
    best = sorted(candidates, key=lambda x: (x["cost"], x["mutation"]))[0]
    return {
        "status": "FLIP",
        "margin": best["cost"],
        "unit": "signal_edit",
        "baseline_decision": baseline["decision"],
        "counterfactual_decision": best["decision"],
        "mutation": best["mutation"],
    }


def numeric_boundary_margin(fixture: dict[str, Any]) -> dict[str, Any]:
    probe = fixture["boundary_probe"]
    replay = adapt_replay(fixture["replay"])
    value = float(replay["signal_values"][probe["key"]])
    threshold = float(probe["threshold"])
    margin = abs(value - threshold)
    status = "CRITICAL" if margin <= float(probe.get("critical_margin", 0.05)) + 1e-12 else "OK"
    return {
        "status": status,
        "margin": margin,
        "key": probe["key"],
        "value": value,
        "threshold": threshold,
        "baseline_route": probe["route_if_gte"] if value >= threshold else probe["route_if_lt"],
        "counterfactual_route": probe["route_if_lt"] if value >= threshold else probe["route_if_gte"],
    }


def apply_state_event(state: dict[str, float], event: dict[str, Any]) -> dict[str, float]:
    out = dict(state)
    field = event["field"]
    cur = float(out.get(field, 0.0))
    if event["op"] == "add":
        cur += float(event["value"])
    elif event["op"] == "scale":
        cur *= float(event["value"])
    else:
        raise ValueError(f"unsupported event op: {event['op']}")
    out[field] = cur
    return out


def route_from_state(state: dict[str, float], route: dict[str, Any]) -> str:
    value = float(state[route["field"]])
    return route["if_gte"] if value >= float(route["threshold"]) else route["if_lt"]


def order_commutator(fixture: dict[str, Any]) -> dict[str, Any]:
    state0 = {k: float(v) for k, v in fixture["initial_state"].items()}
    a = fixture["events"]["A"]
    b = fixture["events"]["B"]
    ab = apply_state_event(apply_state_event(state0, a), b)
    ba = apply_state_event(apply_state_event(state0, b), a)
    route_ab = route_from_state(ab, fixture["route"])
    route_ba = route_from_state(ba, fixture["route"])
    divergent = ab != ba
    route_flip = route_ab != route_ba
    return {
        "status": "FAIL" if route_flip else ("STATE_DIVERGENT" if divergent else "PASS"),
        "route_flip": route_flip,
        "state_divergent": divergent,
        "A_then_B_state": ab,
        "B_then_A_state": ba,
        "A_then_B_route": route_ab,
        "B_then_A_route": route_ba,
    }


def project_scalar(value: float, spec: dict[str, Any]) -> float:
    lo = float(spec.get("clamp_min", 0.0))
    hi = float(spec.get("clamp_max", 1.0))
    value = min(max(value, lo), hi)
    cutoff = float(spec["compression_cutoff"])
    retention = float(spec["low_band_retention"])
    if value < cutoff:
        value *= retention
    return round(value, int(spec.get("round_digits", 12)))


def star(left: float, right: float, spec: dict[str, Any]) -> float:
    # ★ := merge/add followed by lossy projection/reduction.
    return project_scalar(float(left) + float(right), spec)


def grouping_associator(fixture: dict[str, Any]) -> dict[str, Any]:
    a = float(fixture["values"]["A"])
    b = float(fixture["values"]["B"])
    c = float(fixture["values"]["C"])
    spec = fixture["projection"]
    left_inner = star(a, b, spec)
    left = star(left_inner, c, spec)
    right_inner = star(b, c, spec)
    right = star(a, right_inner, spec)
    route = fixture["route"]
    left_route = route["if_gte"] if left >= float(route["threshold"]) else route["if_lt"]
    right_route = route["if_gte"] if right >= float(route["threshold"]) else route["if_lt"]
    residual = abs(left - right)
    return {
        "status": "FAIL" if left_route != right_route else ("STATE_DIVERGENT" if residual else "PASS"),
        "route_flip": left_route != right_route,
        "associator_residual": residual,
        "left": {
            "expr": "(A★B)★C",
            "A_star_B": left_inner,
            "terminal_state": left,
            "route": left_route,
        },
        "right": {
            "expr": "A★(B★C)",
            "B_star_C": right_inner,
            "terminal_state": right,
            "route": right_route,
        },
    }


def load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def result_rows(results: dict[str, Any]) -> list[tuple[str, str, str]]:
    base = results["baseline"]
    official = results["official_counterfactual"]
    gray = results["gray_boundary"]
    order = results["order"]
    grouping = results["grouping"]
    return [
        (
            "Baseline reproduction",
            base["status"],
            f"{base['actual_decision']} / {base['actual_model']}",
        ),
        (
            "Official counterfactual",
            official["status"],
            (
                f"{official['margin']} {official['unit']}: "
                f"{official['baseline_decision']} → {official['counterfactual_decision']}"
                if official["status"] == "FLIP"
                else "no route flip found"
            ),
        ),
        (
            "Counterfactual safety margin",
            gray["status"],
            f"{gray['margin']:.3f}: {gray['baseline_route']} → {gray['counterfactual_route']}",
        ),
        (
            "Order commutator",
            order["status"],
            f"{order['A_then_B_route']} != {order['B_then_A_route']}",
        ),
        (
            "Grouping associator",
            grouping["status"],
            (
                f"residual={grouping['associator_residual']:.3f}; "
                f"{grouping['left']['route']} != {grouping['right']['route']}"
            ),
        ),
    ]


def render_table(rows: list[tuple[str, str, str]]) -> str:
    headers = ("Metric", "Status", "Evidence")
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows))
        for i in range(3)
    ]
    def line(parts):
        return " | ".join(parts[i].ljust(widths[i]) for i in range(3))
    return "\n".join([
        line(headers),
        "-+-".join("-" * w for w in widths),
        *(line(r) for r in rows),
    ])


def render_markdown(results: dict[str, Any], rows: list[tuple[str, str, str]]) -> str:
    grouping = results["grouping"]
    order = results["order"]
    return "\n".join([
        "# Counterfactual / Stateful Routing Audit — PoC v0.1",
        "",
        "| Metric | Status | Evidence |",
        "|---|---|---|",
        *[f"| {m} | **{s}** | {e} |" for m, s, e in rows],
        "",
        "## Grouping fixture detail",
        "",
        f"- `(A★B)★C`: terminal state `{grouping['left']['terminal_state']:.3f}` → `{grouping['left']['route']}`",
        f"- `A★(B★C)`: terminal state `{grouping['right']['terminal_state']:.3f}` → `{grouping['right']['route']}`",
        f"- Associator residual: `{grouping['associator_residual']:.3f}`",
        "",
        "`★` is explicitly `merge/update → lossy projection/reduction`; it is not ordinary function composition.",
        "",
        "## Order fixture detail",
        "",
        f"- `A→B`: `{order['A_then_B_state']}` → `{order['A_then_B_route']}`",
        f"- `B→A`: `{order['B_then_A_state']}` → `{order['B_then_A_route']}`",
        "",
        "## Evidence classification",
        "",
        "- Baseline + official counterfactual: derived from the public AuthZ-RBAC E2E policy/testcase.",
        "- Gray-boundary, order, grouping: explicitly synthetic stress fixtures.",
        "- A synthetic `FAIL` demonstrates the audit primitive, **not an upstream vSR bug**.",
        "",
    ])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT)
    args = ap.parse_args()

    official = load("official_authz_premium_complex.json")
    gray = load("synthetic_gray_boundary.json")
    order_fx = load("synthetic_order.json")
    grouping_fx = load("synthetic_grouping.json")

    results = {
        "schema": "semantic-router-counterfactual-audit/0.1",
        "baseline": baseline_reproduction(official),
        "official_counterfactual": discrete_counterfactual_margin(official),
        "gray_boundary": numeric_boundary_margin(gray),
        "order": order_commutator(order_fx),
        "grouping": grouping_associator(grouping_fx),
        "evidence_note": (
            "Synthetic order/grouping failures are stress-test counterexamples, "
            "not evidence of an upstream Semantic Router defect."
        ),
    }

    rows = result_rows(results)
    print(render_table(rows))

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "REPORT.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out / "REPORT.md").write_text(
        render_markdown(results, rows),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
