"""Decisions and labels the calibrate tests build."""

import fitting
from decisions import TIERS, Decision


def guard(i, score, verdict=None, learnable=True, **detail):
    return Decision(
        f"g{i}",
        None,
        "s",
        "guard",
        verdict or ("held" if score >= 0.9 else "allowed"),
        score,
        0.9,
        f"cmd {i}",
        detail,
        learnable,
    )


def labeled(scores_yes, scores_no):
    return [((s,), True) for s in scores_yes] + [((s,), False) for s in scores_no]


def route(i, level, model, confidence=0.8, verdict="routed", **extra):
    score = confidence if verdict == "kept" else level
    return Decision(
        f"r{i}",
        None,
        "s",
        "router",
        verdict,
        score,
        None,
        f"task {i}",
        {
            "level": level,
            "confidence": confidence,
            "model": model,
            "escalated": None,
            **extra,
        },
        not extra.get("escalated"),
    )


def tier_model(level, easy=0.5, hard=1.5):
    return TIERS[fitting.route_tier(level, easy, hard)]
