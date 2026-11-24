"""Screen development tasks for routing opportunities before budgeting solver runs."""

import json
import math
from itertools import combinations
from pathlib import Path

if __package__:
    from .headroom import named, read_matrix, unique_fields
    from .loop_evidence import check, digest, sha
else:
    from headroom import named, read_matrix, unique_fields
    from loop_evidence import check, digest, sha


def required_costs(path, manifest, controls):
    """Read an adapter's audited, recipe-independent work receipts."""
    result = {}
    references = manifest.get("required_cost_receipts", {})
    check(
        isinstance(references, dict),
        "required_cost_receipts must map tasks to receipts",
    )
    for task, ref in references.items():
        check(task in controls, "required cost for undeclared task")
        name = ref["file"]
        check(
            Path(name).name == name and name not in ("", ".", ".."),
            "expected sibling receipt",
        )
        raw = (path.parent / name).read_bytes()
        check(digest(raw) == sha(ref["sha256"]), "required-cost receipt changed")
        evidence = json.loads(raw, object_pairs_hook=unique_fields)
        check(
            evidence["study"] == manifest["study"]
            and evidence["split"] == "dev"
            and evidence["task"] == task
            and evidence["controls_sha256"] == controls[task]
            and evidence["required_for_every_attempt"] is True
            and named(evidence["requirement"])
            and evidence["stage"] in ("required_solver_work", "official_grading"),
            "required cost must be bound to this task and its mandatory protocol",
        )
        sha(evidence["protocol_sha256"])
        measurements = evidence["observations"]
        check(
            isinstance(measurements, list) and measurements,
            "missing mandatory-work measurements",
        )
        runs = {sha(m["run_sha256"]) for m in measurements}
        times = [m["seconds"] for m in measurements]
        check(
            len(runs) == len(times)
            and all(
                type(t) in (float, int) and math.isfinite(t) and t > 0 for t in times
            ),
            "invalid mandatory-work measurements",
        )
        result[task] = {
            "stage": evidence["stage"],
            "requirement": evidence["requirement"],
            "min_observed_seconds": min(times),
            "observations": len(times),
            "receipt_sha256": ref["sha256"],
        }
    return result


def screen_task(fixed, task, repeats, limit, cost):
    rows = [arm[task, r] for arm in fixed.values() for r in repeats]
    natural = [row for row in rows if row.get("termination") == "natural_completion"]
    successes = [row["seconds"] for row in natural if row["strict"]]
    longest = max((row["seconds"] for row in rows), default=None)
    quality, latency, request_profiles = [], [], []
    for row in natural:
        union = row.get("model_request_union_seconds")
        count = row.get("model_requests")
        if union is None and count is None:
            continue
        check(
            type(count) is int
            and count >= 0
            and type(union) in (int, float)
            and math.isfinite(union)
            and 0 <= union <= row["seconds"],
            "expected matched model request count and in-episode interval union",
        )
        request_profiles.append(
            {
                "arm": row["arm"],
                "repeat": row["repeat"],
                "requests": count,
                "model_in_flight_seconds": union,
                "outside_model_seconds": row["seconds"] - union,
            }
        )
    for left, right in combinations(fixed, 2):
        for repeat in repeats:
            pair = [fixed[left][task, repeat], fixed[right][task, repeat]]
            if pair[0]["policy_sha256"] == pair[1]["policy_sha256"] or any(
                row.get("termination") != "natural_completion" for row in pair
            ):
                continue
            if pair[0]["strict"] != pair[1]["strict"]:
                good, bad = sorted(pair, key=lambda r: not r["strict"])
                quality.append(
                    {
                        "repeat": repeat,
                        "passed": good["arm"],
                        "failed": bad["arm"],
                        "passed_seconds": good["seconds"],
                        "failed_seconds": bad["seconds"],
                    }
                )
            elif pair[0]["strict"] and pair[0]["seconds"] != pair[1]["seconds"]:
                fast, slow = sorted(pair, key=lambda r: r["seconds"])
                latency.append(
                    {
                        "repeat": repeat,
                        "faster": fast["arm"],
                        "slower": slow["arm"],
                        "faster_seconds": fast["seconds"],
                        "slower_seconds": slow["seconds"],
                        "gap_seconds": slow["seconds"] - fast["seconds"],
                    }
                )

    # A successful solution below the claimed mandatory solver cost disproves
    # its use as a floor for this task. Grading is timed outside the solver.
    contradicted = bool(
        cost
        and cost["stage"] == "required_solver_work"
        and successes
        and min(successes) < cost["min_observed_seconds"]
    )
    if contradicted:
        pool = "investigate_cost_conflict"
    elif cost and cost["min_observed_seconds"] > limit:
        pool = "exclude_required_cost"
    elif quality or latency:
        pool = "routing_opportunity"
    elif not rows or len(natural) != len(rows):
        pool = "needs_evidence"
    elif longest > limit:
        pool = "investigate_slow"
    elif successes:
        pool = "quick_control"
    else:
        pool = "quick_quality_probe"
    return {
        "pool": pool,
        "observations": len(rows),
        "strict_passes": len(successes),
        "all_natural": len(natural) == len(rows),
        "request_profiles": request_profiles,
        "max_agent_seconds": longest,
        "fastest_success_seconds": min(successes) if successes else None,
        "quality_contrasts": quality,
        "paired_latency_contrasts": sorted(
            latency, key=lambda p: p["gap_seconds"], reverse=True
        ),
        "required_cost": cost,
        "required_solver_cost_contradicted": contradicted,
    }


def screen_dev(args):
    limit = args.max_seconds
    check(math.isfinite(limit) and limit > 0, "max-seconds must be finite and positive")
    raw = args.manifest.read_bytes()
    manifest = json.loads(raw, object_pairs_hook=unique_fields)
    check(isinstance(manifest, dict), "expected a development manifest")
    if manifest.get("tasks") == []:
        check(
            manifest["split"] == "dev"
            and named(manifest["study"])
            and manifest.get("arms") == []
            and manifest.get("repeats") == []
            and "scores" not in manifest
            and "scores_sha256" not in manifest
            and manifest.get("unmeasured_controls"),
            "invalid unmeasured development manifest",
        )
        manifest_hash, fixed = digest(raw), {}
    else:
        manifest, manifest_hash, fixed = read_matrix(
            args.manifest, "dev", minimum_arms=1
        )
    controls = {
        t: fixed[manifest["arms"][0]][t, manifest["repeats"][0]]["controls_sha256"]
        for t in manifest["tasks"]
    }
    unmeasured = manifest.get("unmeasured_controls", {})
    check(
        isinstance(unmeasured, dict)
        and all(named(t) and t not in controls for t in unmeasured),
        "unmeasured tasks must be distinct from the measured cohort",
    )
    controls.update({t: sha(control) for t, control in unmeasured.items()})
    costs = required_costs(args.manifest, manifest, controls)
    tasks = {
        task: screen_task(fixed, task, manifest["repeats"], limit, costs.get(task))
        for task in manifest["tasks"]
    }
    tasks.update(
        {task: screen_task({}, task, [], limit, costs.get(task)) for task in unmeasured}
    )

    def priority(task):
        row = tasks[task]
        gap = max(
            (p["gap_seconds"] for p in row["paired_latency_contrasts"]), default=0.0
        )
        return (-bool(row["quality_contrasts"]), -gap, task)

    return {
        "study": manifest["study"],
        "split": "dev",
        "source_manifest_sha256": manifest_hash,
        "baseline_arms": manifest["arms"],
        "reference_seconds": limit,
        "by_task": tasks,
        "routing_candidates": sorted(
            (t for t, r in tasks.items() if r["pool"] == "routing_opportunity"),
            key=priority,
        ),
        "quick_controls": [
            t
            for t, r in tasks.items()
            if r["all_natural"]
            and r["strict_passes"]
            and r["max_agent_seconds"] <= limit
            and r["pool"] in ("quick_control", "routing_opportunity")
        ],
        "excluded_tasks": [
            t for t, r in tasks.items() if r["pool"] == "exclude_required_cost"
        ],
    }
