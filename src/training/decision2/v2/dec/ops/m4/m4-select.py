"""Decoder M4 finalist selection from development readouts (prereg dec-m4-prereg-2026-09-29.md, rules 1-5).

Reads only the soup watchers' development readouts (typed DEV + CSS pilot) and dev predictions on node B;
never a v3 / public-231 / mlx-diag result. R is ``proxy_mean_H`` = 100·sqrt(T_dev·H_mean3) unless another
readout statistic is named (proxy v2, if the eval track publishes it first).

usage: python3 m4-select.py [--stat proxy_mean_H] [--band 4] [--cross N4LX] <group> ...
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

M = Path("/data/dev2/runs/dec/m4")
NOX = {"choice": 460, "noul": 228, "score": 378}
FLOORS = {"choice": 345, "noul": 171, "score": 284}


def categories(path: Path) -> dict[str, Counter]:
    """Answer categories per type: Choice none/option, Noul true/false/abstain, Score argmax level."""
    by_type: dict[str, Counter] = {}
    for line in path.open(encoding="utf-8"):
        answer = json.loads(line)["answers"]["decision"]
        kind = answer["type"]
        if kind == "choice":
            category = "none" if answer.get("choice") == "none" else "option"
        elif kind == "noul":
            value = answer.get("noul")
            category = None if value is None or value == 0.5 else value > 0.5
        else:
            probs = answer.get("probabilities")
            category = max(probs, key=probs.get) if probs else answer.get("score")
        by_type.setdefault(kind, Counter())[category] += 1
    return by_type


def collapsed(by_type: dict[str, Counter]) -> list[str]:
    out = []
    for kind, counts in by_type.items():
        share = max(counts.values()) / sum(counts.values())
        if kind == "choice" and counts.get("none", 0) / sum(counts.values()) >= 0.95:
            out.append(kind)
        elif kind != "choice" and share >= 0.95:
            out.append(kind)
    return out


def artifact(group: str, stat: str) -> dict:
    readout = json.loads((M / "soup" / group / "readout.json").read_text())
    arms = readout["arms"]
    seeds = sorted(k for k in arms if k not in ("nox1", "n4lkr", "soup"))
    values = {k: arms[k][stat] for k in seeds}
    mean = statistics.mean(values.values())
    if arms["soup"][stat] >= mean:
        chosen, pred = "soup", M / "soup" / group / "dev" / "dev.predictions.jsonl"
    else:
        chosen = sorted(seeds, key=values.get)[len(seeds) // 2]
        pred = (
            M
            / "arms"
            / "full"
            / f"m4-{group}-{chosen}-post"
            / "dev"
            / "dev.predictions.jsonl"
        )
    data = arms[chosen]
    counts = {k: data["by_type"][k]["correct"] for k in NOX}
    invalid = sum(v["invalid"] for v in data["by_type"].values())
    flat = collapsed(categories(pred))
    reasons = [f"{k} {counts[k]} < {FLOORS[k]}" for k in NOX if counts[k] < FLOORS[k]]
    reasons += [f"{invalid} invalid"] if invalid else []
    reasons += [f"{k} constant" for k in flat]
    return {
        "group": group,
        "artifact": chosen,
        "R": data[stat],
        "seed_R": values,
        "seed_mean_R": mean,
        "soup_R": arms["soup"][stat],
        "T": data["T"],
        "H_mean3": data["H_mean"],
        "H_median": data["H"],
        "P_v1": data["proxy"],
        "tasks": data["css_task_macro_f1"],
        "typed": counts,
        "min_ratio": min(counts[k] / NOX[k] for k in NOX),
        "score_levels": data.get("score_predicted_levels"),
        "eligible": not reasons,
        "reasons": reasons,
        "n4lkr_R": arms["n4lkr"][stat],
        "nox1_R": arms["nox1"][stat],
        "soup_vs_n4lkr": readout["comparisons"]
        .get("soup-minus-n4lkr", {})
        .get("delta_b_minus_a", {})
        .get(stat),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("groups", nargs="+")
    parser.add_argument("--stat", default="proxy_mean_H")
    parser.add_argument("--band", type=float, default=4.0)
    parser.add_argument("--cross", help="cross-arm soup group, if built (rule 5)")
    args = parser.parse_args()
    rows = [artifact(g, args.stat) for g in args.groups]
    eligible = sorted((r for r in rows if r["eligible"]), key=lambda r: -r["R"])
    cross_members = [r["group"] for r in eligible[:2]]
    pool = list(eligible)
    if args.cross and (M / "soup" / args.cross / "readout.json").is_file():
        cross = artifact(args.cross, args.stat)
        if cross["artifact"] != "soup":
            cross["eligible"] = False
            cross["reasons"].append("cross soup below the mean of its seeds")
        rows.append(cross)
        if cross["eligible"]:
            pool = sorted(pool + [cross], key=lambda r: -r["R"])
    reference = rows[0]["n4lkr_R"]
    pool = [r for r in pool if r["R"] >= reference - args.band]
    finalists = pool[:1]
    if len(pool) >= 2:
        second = pool[1]
        if len(pool) >= 3 and pool[1]["R"] - pool[2]["R"] < args.band:
            second = max(pool[1:3], key=lambda r: r["min_ratio"])
        finalists.append(second)
    print(
        json.dumps(
            {
                "stat": args.stat,
                "band": args.band,
                "n4lkr_R": reference,
                "nox1_R": rows[0]["nox1_R"],
                "rows": rows,
                "cross_members": cross_members,
                "finalists": [(r["group"], r["artifact"]) for r in finalists],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
