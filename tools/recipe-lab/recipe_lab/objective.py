"""Check a development matrix against its predeclared experimental objective."""

import json
import math
from pathlib import Path

if __package__:
    from .headroom import axis, named, paired, unique_fields
    from .loop_evidence import check, digest, sha
else:
    from headroom import axis, named, paired, unique_fields
    from loop_evidence import check, digest, sha


PRIORITIES = ["strict_success", "joint_success_latency", "total_attempt_cost"]


def evaluate_objective(path, manifest, fixed):
    ref = manifest.get("objective")
    if "objective" not in manifest:
        return {"status": "not_declared"}
    check(
        manifest["split"] == "dev",
        "a development objective requires a development matrix",
    )
    check(
        isinstance(ref, dict) and set(ref) == {"file", "sha256"},
        "invalid objective reference",
    )
    name = ref["file"]
    check(
        named(name)
        and name not in (".", "..")
        and "/" not in name
        and "\\" not in name
        and Path(name).name == name,
        "objective must be a sibling JSON file",
    )
    raw = (path.parent / name).read_bytes()
    check(digest(raw) == sha(ref["sha256"]), "objective fingerprint mismatch")
    spec = json.loads(raw, object_pairs_hook=unique_fields)
    check(
        isinstance(spec, dict)
        and set(spec) - {"mde_pair"}
        == {
            "schema",
            "study",
            "priorities",
            "baseline",
            "arms",
            "repeats",
            "dev_targets",
            "dev_guards",
            "holdout_task_id_sha256",
            "thresholds",
            "external_checks",
            "policy_sha256_by_arm",
            "controls_sha256_by_dev_task",
        }
        and spec["schema"] == "recipe-lab-objective-v1",
        "invalid objective declaration",
    )
    check(
        spec["study"] == manifest["study"] and spec["priorities"] == PRIORITIES,
        "objective study or priority order differs from the protocol",
    )
    arms = axis(spec, "arms", named, 2)
    repeats = axis(spec, "repeats", lambda r: type(r) is int and r > 0, 1)
    check(
        arms == manifest["arms"]
        and repeats == manifest["repeats"]
        and spec["baseline"] in arms,
        "matrix arms, repeat allocation or baseline differ from objective",
    )
    mde_pair = spec.get("mde_pair")
    if mde_pair is not None:
        check(
            isinstance(mde_pair, list)
            and len(mde_pair) == 2
            and mde_pair[0] != mde_pair[1]
            and all(arm in arms for arm in mde_pair),
            "objective MDE pair must name two distinct arms",
        )
    targets, guards = (
        set(axis(spec, key, named, 1)) for key in ("dev_targets", "dev_guards")
    )
    check(
        not targets.intersection(guards) and targets | guards == set(manifest["tasks"]),
        "development targets and guards must partition the measured tasks",
    )
    first_cell = (manifest["tasks"][0], repeats[0])
    actual_policies = {arm: fixed[arm][first_cell]["policy_sha256"] for arm in arms}
    actual_controls = {
        task: fixed[arms[0]][task, repeats[0]]["controls_sha256"]
        for task in manifest["tasks"]
    }
    check(
        spec["policy_sha256_by_arm"] == actual_policies,
        "measured policies differ from the objective",
    )
    check(
        spec["controls_sha256_by_dev_task"] == actual_controls,
        "measured controls differ from the objective",
    )
    sealed = axis(spec, "holdout_task_id_sha256", sha, 1)
    check(
        not {digest(t.encode("utf-8")) for t in targets | guards}.intersection(sealed),
        "a development task also appears in the sealed holdout allocation",
    )
    external = axis(spec, "external_checks", named, 0)
    limits = spec["thresholds"]
    check(
        isinstance(limits, dict)
        and set(limits)
        == {
            "max_lost_strict_cells",
            "min_joint_tasks",
            "min_joint_seconds_saved",
            "max_solver_wall_ratio",
        },
        "unexpected objective thresholds",
    )
    check(
        type(limits["max_lost_strict_cells"]) is int
        and limits["max_lost_strict_cells"] == 0,
        "the objective must preserve every baseline strict pass",
    )
    check(
        type(limits["min_joint_tasks"]) is int and limits["min_joint_tasks"] > 0,
        "min_joint_tasks must be a positive integer",
    )
    saved, ratio = limits["min_joint_seconds_saved"], limits["max_solver_wall_ratio"]
    check(
        type(saved) in (int, float) and math.isfinite(saved) and saved >= 0,
        "min_joint_seconds_saved must be finite and nonnegative",
    )
    check(
        ratio is None
        or (type(ratio) in (int, float) and math.isfinite(ratio) and ratio > 0),
        "max_solver_wall_ratio must be null or finite and positive",
    )

    baseline = fixed[spec["baseline"]]
    missing_guards = sorted(
        t for t in guards if not any(baseline[t, r]["strict"] for r in repeats)
    )
    before = math.fsum(row["seconds"] for row in baseline.values())
    by_arm = {}
    for arm in arms:
        if arm == spec["baseline"]:
            continue
        challenger = fixed[arm]
        pair = paired(baseline, challenger)
        count = len(
            {
                t
                for (t, r), row in baseline.items()
                if row["strict"] and challenger[t, r]["strict"]
            }
        )
        wall_ratio = math.fsum(row["seconds"] for row in challenger.values()) / before
        has_gain = bool(pair["gained_strict_cells"]) or pair["joint_seconds_saved"] > 0
        checks = {
            "guards_observed_passing_on_baseline": not missing_guards,
            "preserves_strict_passes": not pair["lost_strict_cells"],
            "minimum_joint_tasks": count >= limits["min_joint_tasks"],
            "minimum_joint_seconds_saved": pair["joint_seconds_saved"] >= saved,
            "improves_strict_success_or_joint_latency": has_gain,
        }
        if ratio is not None:
            checks["maximum_solver_wall_ratio"] = wall_ratio <= ratio
        by_arm[arm] = {
            "observed_dev_thresholds_met": all(checks.values()),
            "checks": checks,
            "joint_tasks": count,
            "joint_seconds_saved": pair["joint_seconds_saved"],
            "gained_strict_cells": pair["gained_strict_cells"],
            "lost_strict_cells": pair["lost_strict_cells"],
            "solver_wall_ratio": wall_ratio,
        }
    return {
        "status": "verified_declaration",
        "objective_sha256": ref["sha256"],
        "priorities": PRIORITIES,
        "baseline": spec["baseline"],
        "thresholds": limits,
        "mde_pair": mde_pair,
        "dev_targets": sorted(targets),
        "dev_guards": sorted(guards),
        "guards_without_baseline_pass": missing_guards,
        "holdout_task_count": len(sealed),
        "by_arm": by_arm,
        "declared_external_checks": external,
        "independent_holdout_verified": False,
    }
