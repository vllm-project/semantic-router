"""Predeclared random controls and calibration-only comparator selection."""

from __future__ import annotations

import random
from collections import defaultdict
from statistics import mean

from .metrics import summarize

RANDOM_SEEDS = tuple(range(20))
BOOTSTRAP_REPLICATES = 1000


def group_interval(values: dict[str, list[float]]) -> list[float]:
    groups = list(values.values())
    generator = random.Random(42)
    samples = sorted(
        mean(value for _ in groups for value in generator.choice(groups))
        for _ in range(BOOTSTRAP_REPLICATES)
    )
    return [samples[25], samples[974]]


def random_assignment_control(
    rows: list[dict], selected: list[dict], kind: str, run, aggregate
) -> dict:
    """Preserve action counts; randomize which source receives each action."""
    actions = [row["action"] for row in selected]
    per_seed, source_correct = [], defaultdict(list)
    for seed in RANDOM_SEEDS:
        shuffled = actions.copy()
        random.Random(seed).shuffle(shuffled)
        assigned = {
            row["id"]: action for row, action in zip(rows, shuffled, strict=True)
        }
        samples = run(
            rows, {"kind": "assignment", "delivery_kind": kind, "actions": assigned}
        )
        per_seed.append({"seed": seed, **aggregate(samples)})
        for sample in samples:
            source_correct[sample["group_id"]].append(
                float(sample["result"]["correct"])
            )
    # Randomization repetitions are not additional independent labelled sources.
    per_group = {key: [mean(values)] for key, values in source_correct.items()}
    metrics = (
        "bundle_accuracy",
        "bundle_error",
        "mean_typed_loss",
        "mean_policy_cost_ms",
        "mean_accumulated_call_ms",
        "mean_calls",
        "escalation_fraction",
    )
    return {
        "kind": "same_action_mix_random_assignment",
        "seeds": list(RANDOM_SEEDS),
        "source_groups": len(per_group),
        "mean": {key: mean(run[key] for run in per_seed) for key in metrics},
        "bundle_accuracy_group_bootstrap_95pct": group_interval(per_group),
        "per_seed": per_seed,
        "interpretation": "Same per-model action counts as the selected policy; realized input-dependent cost is reported, not assumed identical. Group interval conditions on these predeclared randomizations; twenty repetitions do not multiply sample size.",
    }


def budget_controls(
    calibration_rows: list[dict],
    held_rows: list[dict],
    matrix: dict,
    native: list[str],
    base: str,
    budget: float,
    run,
    aggregate,
) -> dict:
    direct = []
    for name in native:
        observations = [matrix[row["id"]][name] for row in calibration_rows]
        cost = mean(row["policy_cost_ms"] for row in observations)
        score = summarize([row["result"] for row in observations])
        if cost <= budget:
            direct.append((score["bundle_error"], score["mean_typed_loss"], cost, name))
    direct_name = min(direct)[-1] if direct else None
    direct_report = None
    if direct_name is not None:
        observations = [matrix[row["id"]][direct_name] for row in held_rows]
        direct_report = {
            "model": direct_name,
            "calls_per_request": 1,
            **summarize([row["result"] for row in observations]),
            "mean_policy_cost_ms": mean(row["policy_cost_ms"] for row in observations),
            "mean_client_elapsed_ms": mean(
                row["client_elapsed_ms"] for row in observations
            ),
        }
    upgrades = []
    for name in native:
        if name == base:
            continue
        setting = {"kind": "cascade", "action": name, "risk_threshold": -1.0}
        score = aggregate(run(calibration_rows, setting))
        if score["mean_policy_cost_ms"] <= budget:
            upgrades.append(
                (
                    score["bundle_error"],
                    score["mean_typed_loss"],
                    score["mean_policy_cost_ms"],
                    name,
                )
            )
    upgraded = None
    if upgrades:
        name = min(upgrades)[-1]
        upgraded = {
            "model": name,
            **aggregate(
                run(
                    held_rows,
                    {"kind": "cascade", "action": name, "risk_threshold": -1.0},
                )
            ),
        }
    return {
        "selection": "best calibration quality within the same cost ceiling; held-out cost is not forced to match",
        "direct": direct_report,
        "always_escalate": upgraded,
    }


def error_complementarity(rows: list[dict], matrix: dict, native: list[str]) -> dict:
    """Explore only training/calibration labels, never choose from held-out errors."""
    output = {}
    for source in native:
        output[source] = {}
        for target in native:
            if target == source:
                continue
            pairs = [
                (
                    matrix[row["id"]][source]["result"]["correct"],
                    matrix[row["id"]][target]["result"]["correct"],
                )
                for row in rows
            ]
            rescue = sum(not first and second for first, second in pairs)
            harm = sum(first and not second for first, second in pairs)
            output[source][target] = {
                "source_groups": len(rows),
                "rescue": rescue,
                "harm": harm,
                "both_correct": sum(first and second for first, second in pairs),
                "both_wrong": sum(not first and not second for first, second in pairs),
                "net_rescue_rate": (rescue - harm) / len(rows),
            }
    return output
