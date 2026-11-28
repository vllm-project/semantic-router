"""Descriptive task-level headroom from a pinned, balanced strict-grade matrix."""

import json
import math
from itertools import product
from pathlib import Path
from statistics import mean, stdev

if __package__:
    from .loop_evidence import check, digest, sha
else:
    from loop_evidence import check, digest, sha


def unique_fields(pairs):
    result = {}
    for key, value in pairs:
        check(key not in result, f"duplicate JSON field: {key}")
        result[key] = value
    return result


def named(value):
    return isinstance(value, str) and value and value.strip() == value


def axis(manifest, key, valid, minimum):
    values = manifest[key]
    check(
        isinstance(values, list) and len(values) >= minimum and all(map(valid, values)),
        f"invalid {key} in manifest",
    )
    check(len(values) == len(set(values)), f"duplicate {key} in manifest")
    return values


def read_matrix(path, scope, *, minimum_arms=2, sealed_search=False):
    manifest_raw = path.read_bytes()
    manifest = json.loads(manifest_raw, object_pairs_hook=unique_fields)
    check(isinstance(manifest, dict) and named(manifest["study"]), "invalid study")
    check(
        (scope in ("dev", "retired-holdout") or (sealed_search and scope == "holdout"))
        and scope == manifest["split"],
        "scope must match dev or explicitly retired holdout evidence",
    )
    tasks = axis(manifest, "tasks", named, 1)
    arms = axis(manifest, "arms", named, minimum_arms)
    repeats = axis(manifest, "repeats", lambda r: type(r) is int and r > 0, 1)
    scores = Path(manifest["scores"])
    check(
        scores.name == str(scores) and scores.name not in ("", ".", ".."),
        "scores must name a sibling JSON file",
    )
    raw = (path.parent / scores).read_bytes()
    check(digest(raw) == sha(manifest["scores_sha256"]), "scores fingerprint mismatch")
    rows = json.loads(raw, object_pairs_hook=unique_fields)
    check(isinstance(rows, list), "scores must be an array of episodes")

    cells, controls, policies, episodes = {}, {}, {}, set()
    for row in rows:
        check(
            row["study"] == manifest["study"] and row["split"] == scope,
            "mixed study or split",
        )
        check(
            row.get("assessment") == "valid",
            "episode requires a valid evidence assessment",
        )
        task, arm, repeat = row["task"], row["arm"], row["repeat"]
        check(
            task in tasks and arm in arms and type(repeat) is int and repeat in repeats,
            f"undeclared task, arm or repeat: {(task, arm, repeat)}",
        )
        key = task, arm, repeat
        check(key not in cells, f"duplicate cell: {key}")
        episode = sha(row["episode_sha256"])
        check(episode not in episodes, "physical episode reused")
        episodes.add(episode)
        control, policy = sha(row["controls_sha256"]), sha(row["policy_sha256"])
        check(
            controls.setdefault(task, control) == control, f"controls differ for {task}"
        )
        check(
            policies.setdefault(arm, policy) == policy, f"policy changed within {arm}"
        )
        check(type(row["strict"]) is bool, "strict must be a boolean")
        seconds = row["seconds"]
        check(
            type(seconds) in (float, int) and math.isfinite(seconds) and seconds > 0,
            "seconds must be finite and positive",
        )
        cells[key] = row
    for key in product(tasks, arms, repeats):
        check(key in cells, f"missing cell: {key}")
    # Index each fixed arm by the same (task, repeat) for all paired comparisons.
    fixed = {
        arm: {(t, r): cells[t, arm, r] for t, r in product(tasks, repeats)}
        for arm in arms
    }
    return manifest, digest(manifest_raw), fixed


def summarize(rows):
    seconds = [row["seconds"] for row in rows if row["strict"]]
    durations = [row["seconds"] for row in rows]
    return {
        "strict_passes": len(seconds),
        "episodes": len(durations),
        "successful_seconds_sum": math.fsum(seconds),
        "wall_seconds_sum": math.fsum(durations),
    }


def task_summary(rows):
    times = [row["seconds"] for row in rows]
    passed = [row["seconds"] for row in rows if row["strict"]]
    return {
        **summarize(rows),
        "mixed_strict_over_repeats": 0 < len(passed) < len(times),
        "all_wall_seconds_range": [min(times), max(times)],
        "all_wall_cv": stdev(times) / mean(times) if len(times) >= 2 else None,
        "successful_wall_cv": (
            stdev(passed) / mean(passed) if len(passed) >= 2 else None
        ),
    }


def choose(fixed, keys, arms):
    # Tie-break on exactly the same successful cells for all quality-tied arms.
    # Using their separate successful totals would reward a different task mix.
    counts = {arm: sum(fixed[arm][k]["strict"] for k in keys) for arm in arms}
    best_count = max(counts.values())
    tied = [arm for arm in arms if counts[arm] == best_count]
    common = [k for k in keys if all(fixed[arm][k]["strict"] for arm in tied)]
    winner = min(
        tied, key=lambda arm: math.fsum(fixed[arm][k]["seconds"] for k in common)
    )
    return winner


def paired(base, other):
    gains, losses, common = [], [], []
    for (task, repeat), left in base.items():
        right = other[task, repeat]
        if left["strict"] and right["strict"]:
            common.append((left["seconds"], right["seconds"]))
        elif left["strict"] != right["strict"]:
            (losses if left["strict"] else gains).append([task, repeat])
    before = math.fsum(a for a, _ in common)
    after = math.fsum(b for _, b in common)
    return {
        "gained_strict_cells": gains,
        "lost_strict_cells": losses,
        "joint_successes": len(common),
        "joint_base_seconds": before,
        "joint_other_seconds": after,
        "joint_seconds_saved": before - after,
        "joint_percent_saved": (before - after) / before * 100 if common else None,
    }


def oracle(fixed, tasks, arms, repeats, baseline, preserve):
    choices, selected = {}, {}
    for task in tasks:
        keys = [(task, r) for r in repeats]
        eligible = arms
        if preserve:
            eligible = [
                arm
                for arm in arms
                if all(
                    not fixed[baseline][k]["strict"] or fixed[arm][k]["strict"]
                    for k in keys
                )
            ]
        winner = choose(fixed, keys, eligible)
        choices[task] = winner
        selected.update({k: fixed[winner][k] for k in keys})
    return {
        "chosen_arm_by_task": choices,
        **summarize(selected.values()),
        "versus_best_fixed": paired(fixed[baseline], selected),
    }


def mde(fixed, tasks, repeats, pair):
    result = {
        "unit": "mean seconds saved per jointly successful task",
        "estimate_seconds": None,
        "fixed_pair": pair,
    }
    if not pair:
        return {**result, "reason": "a predeclared fixed pair was not supplied"}
    a, b = pair
    check(a in fixed and b in fixed and a != b, "MDE needs two distinct declared arms")
    differences = {}
    pair_counts = {}
    for task in tasks:
        deltas = [
            fixed[a][task, r]["seconds"] - fixed[b][task, r]["seconds"]
            for r in repeats
            if fixed[a][task, r]["strict"] and fixed[b][task, r]["strict"]
        ]
        pair_counts[task] = len(deltas)
        if deltas:
            differences[task] = mean(deltas)
    n = len(differences)
    result.update(
        independent_tasks=n,
        joint_repeats_by_task=pair_counts,
        mean_paired_delta_seconds_by_task=differences,
    )
    # A conservative pilot reporting threshold, not a power guarantee.
    if n < 8:
        return {
            **result,
            "reason": "fewer than eight independent jointly successful tasks",
        }
    if any(pair_counts[t] < 2 for t in differences):
        return {**result, "reason": "fewer than two joint repeats on an included task"}
    sigma = stdev(differences.values())
    if sigma == 0:
        return {
            **result,
            "reason": "no observed between-task dispersion for the fixed pair",
        }
    return {
        **result,
        "estimate_seconds": 2.8 * sigma / math.sqrt(n),
        "method": "pilot normal approximation, two-sided alpha 0.05, power 0.8",
    }


def headroom(args):
    if __package__:
        from .objective import evaluate_objective
    else:
        from objective import evaluate_objective

    manifest, manifest_hash, fixed = read_matrix(args.manifest, args.scope)
    tasks, arms, repeats = manifest["tasks"], manifest["arms"], manifest["repeats"]
    # Picking between two samples of one policy after seeing the task outcome
    # would turn run-to-run noise into apparent routing headroom.
    policies = {fixed[arm][tasks[0], repeats[0]]["policy_sha256"] for arm in arms}
    check(
        len(policies) == len(arms),
        "headroom requires distinct policies; pool policy aliases as repeats",
    )
    keys = list(product(tasks, repeats))
    best = choose(fixed, keys, arms)
    objective = evaluate_objective(args.manifest, manifest, fixed)
    declared_pair = objective.get("mde_pair")
    check(
        not (declared_pair and args.mde_pair) or declared_pair == list(args.mde_pair),
        "requested MDE pair differs from the frozen objective",
    )
    return {
        "study": manifest["study"],
        "scope": args.scope,
        "manifest_sha256": manifest_hash,
        "scores_sha256": manifest["scores_sha256"],
        "objective": objective,
        "tasks": len(tasks),
        "repeats_per_task": len(repeats),
        "independent_unit": "task",
        "fixed": {arm: summarize(fixed[arm].values()) for arm in arms},
        "by_task": {
            t: {arm: task_summary([fixed[arm][t, r] for r in repeats]) for arm in arms}
            for t in tasks
        },
        "best_fixed": best,
        "fixed_versus_best": {
            arm: paired(fixed[best], fixed[arm]) for arm in arms if arm != best
        },
        "l0": oracle(fixed, tasks, arms, repeats, best, preserve=False),
        "l0_preserving_baseline_passes": oracle(
            fixed, tasks, arms, repeats, best, preserve=True
        ),
        "mde": mde(fixed, tasks, repeats, declared_pair or args.mde_pair),
        "l1": {
            "status": "unidentified",
            "requires": "matched handoff snapshot continuations",
        },
        "l2": {
            "status": "unidentified",
            "requires": "matched request alternatives and downstream effects",
        },
        "time_breakdown": {
            "status": "unavailable",
            "requires": "non-overlapping attributable spans",
        },
    }
