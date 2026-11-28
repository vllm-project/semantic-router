"""The lane-binding selection core (minimal prerequisite, sub-issue-8 §2.2).

PR-6's session-policy bootstrap routes lane assignment through the same
selection core the measured-binding contract uses (`plan_lane`,
unmeasured-lane semantics included). The full measured-binding contract -
profile observation, binding plans, run wiring - arrives with its own PR;
this module carries only the pure selection functions its consumers need,
verbatim from the shared implementation.
"""

from __future__ import annotations

from typing import Any

from .contract import (
    LONG_CONTEXT_MIN_TOKENS,
    ContractError,
)

LANE_CAPABILITY_REQUIREMENTS: dict[str, dict[str, Any]] = {
    "vision": {"capabilities": ("vision",), "min_context_window": None},
    "agentic": {"capabilities": ("tools",), "min_context_window": None},
    "long_context": {
        "capabilities": (),
        "min_context_window": LONG_CONTEXT_MIN_TOKENS,
    },
}


def lane_requirements(lane: str) -> dict[str, Any]:
    """The capability requirement of one lane (empty for unlisted lanes)."""
    return LANE_CAPABILITY_REQUIREMENTS.get(
        lane, {"capabilities": (), "min_context_window": None}
    )


def capability_satisfied(requirement: dict[str, Any], capabilities: Any) -> bool:
    """True when one model's declared ModelCapabilities meet a requirement."""
    for name in requirement.get("capabilities") or ():
        if not getattr(capabilities, name, False):
            return False
    window = requirement.get("min_context_window")
    if window is None:
        return True
    return not (
        capabilities.context_window is None or capabilities.context_window < window
    )


def _calib_stats(profile: dict[str, Any]) -> dict[str, Any]:
    stats = (profile.get("model_split_stats") or {}).get("calib") or {}
    if not stats:
        raise ContractError(
            "profile has no calib stats: bindings consumes the C3 profile's "
            "calib split only (rerun profile with --splits calib)"
        )
    return stats


def _lane_stat(
    profile: dict[str, Any], alias: str, lane: str, key: str
) -> float | None:
    block = ((_calib_stats(profile).get(alias) or {}).get("labels") or {}).get(lane)
    if not isinstance(block, dict):
        return None
    value = block.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _overall_stat(profile: dict[str, Any], alias: str, key: str) -> float | None:
    block = (_calib_stats(profile).get(alias) or {}).get("total")
    if not isinstance(block, dict):
        return None
    value = block.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _normalize_costs(
    costs: dict[str, float | None],
) -> dict[str, float] | None:
    """Per-lane cost normalization (``cost_aware_calib`` convention: divide
    by the max across models). None when no model has a cost."""
    values = {alias: cost for alias, cost in costs.items() if cost is not None}
    if not values:
        return None
    top = max(values.values())
    if top <= 0:
        return dict.fromkeys(costs, 0.0)
    return {
        alias: (cost / top if cost is not None else None)
        for alias, cost in costs.items()
    }


def plan_lane(
    *,
    lane: str,
    template_lane: dict[str, Any],
    models: Any,
    profile: dict[str, Any],
    lam: float,
    min_lane_cases: int,
    lane_counts: dict[str, int],
) -> dict[str, Any]:
    """One lane's binding plan (sub-issue-8 §2.2 step 6).

    Returns the plan dict with ``selected`` refs, the ``measured`` flag, the
    selection basis and the per-model scores the sidecar records.
    """
    requirement = lane_requirements(lane)
    eligible = [
        model
        for model in models
        if capability_satisfied(requirement, model.capabilities)
    ]
    basis = "accuracy - lambda * normalized_cost"
    if template_lane["immediate"]:
        return {
            "lane": lane,
            "immediate": True,
            "measured": False,
            "n": lane_counts.get(lane, 0),
            "selected": [],
            "basis": "immediate response (no backend candidates)",
            "scores": {},
        }
    minimum = template_lane["minimum_candidates"]
    if not eligible:
        raise ContractError(
            f"lane {lane!r} requires capabilities "
            f"{requirement.get('capabilities') or ()} / context window "
            f"{requirement.get('min_context_window')} that none of the declared "
            "models satisfy; pass --exclude-decision "
            f"{lane} explicitly or declare the capability"
        )
    n = lane_counts.get(lane, 0)
    measured = n >= min_lane_cases
    if measured:
        costs = {
            model.name: _lane_stat(profile, model.name, lane, "cost_per_question")
            for model in eligible
        }
        normalized = _normalize_costs(costs)
        if lam > 0 and (normalized is None or any(c is None for c in costs.values())):
            raise ContractError(
                f"lane {lane!r} has incomplete cost data: a lambda > 0 would "
                "silently rank cost-less models as free; declare cost_basis "
                "assumed_pricing (or rerun profile) or drop the lambda"
            )

        def score(model) -> tuple[float, str]:
            acc = _lane_stat(profile, model.name, lane, "accuracy")
            acc = acc if acc is not None else 0.0
            norm = (normalized or {}).get(model.name)
            return acc - lam * (norm if norm is not None else 0.0), model.name

        ranked = sorted(eligible, key=lambda model: (-score(model)[0], score(model)[1]))
        basis_scores = {
            model.name: {
                "accuracy": _lane_stat(profile, model.name, lane, "accuracy"),
                "normalized_cost": (normalized or {}).get(model.name),
                "score": score(model)[0],
            }
            for model in eligible
        }
    else:
        # Unmeasured lane: default rule on the lambda-adjusted overall score,
        # marked untested (sub-issue-8 §2.2 step 6).
        basis = "default rule (unmeasured lane): overall accuracy - lambda * normalized overall cost"
        costs = {
            model.name: _overall_stat(profile, model.name, "cost_per_question")
            for model in eligible
        }
        normalized = _normalize_costs(costs)
        if lam > 0 and (normalized is None or any(c is None for c in costs.values())):
            raise ContractError(
                f"lane {lane!r} (unmeasured) has incomplete overall cost data: "
                "a lambda > 0 would silently rank cost-less models as free; "
                "declare cost_basis assumed_pricing or drop the lambda"
            )

        def oscore(model) -> tuple[float, str]:
            acc = _overall_stat(profile, model.name, "accuracy")
            acc = acc if acc is not None else 0.0
            norm = (normalized or {}).get(model.name)
            return acc - lam * (norm if norm is not None else 0.0), model.name

        ranked = sorted(
            eligible, key=lambda model: (-oscore(model)[0], oscore(model)[1])
        )
        basis_scores = {
            model.name: {
                "overall_accuracy": _overall_stat(profile, model.name, "accuracy"),
                "normalized_overall_cost": (normalized or {}).get(model.name),
                "score": oscore(model)[0],
            }
            for model in eligible
        }
    if len(ranked) < minimum:
        raise ContractError(
            f"lane {lane!r} requires {minimum} distinct candidates but only "
            f"{len(ranked)} model(s) satisfy its capability requirement; pass "
            f"--exclude-decision {lane} explicitly or declare more capable models"
        )
    selected = [model.name for model in ranked[:minimum]]
    return {
        "lane": lane,
        "immediate": False,
        "measured": measured,
        "n": n,
        "minimum_candidates": minimum,
        "selected": selected,
        "basis": basis,
        "scores": basis_scores,
    }
