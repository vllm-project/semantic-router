"""Decoder M5 finalist selection from development readouts (prereg dec-m5-prereg-2026-09-29.md, Selection 1-4;
amendment 1 e574bbf56).

Reads only the soup pipeline's development readouts on node B (typed DEV + CSS pilot ``soup/<arm>/readout.json``,
dev predictions) and the MLX-DEV readouts (``mlxdev/readouts/<name>/{score,vs-n4xf-soup}.json``); never a v3 /
public-231 / mlx-diag result.

1. Artifact per arm: the uniform soup if its R >= the seed mean of R, else the median seed by R.
2. Eligible: typed-DEV correct per type >= 345 / 171 / 284 (Choice / Noul / Score), no invalid answer, no type
   answered with one constant value (the share of the most common answer category is reported as well).
3. R = proxy v2 P = 100*sqrt(T_dev*H_pilot) with the pilot median (``proxy``); P_mean3 (``proxy_mean_H``) is
   reported. Dropped: R <= R(N4XF soup, same readout) - 8, or Noul-ML paired B - A (A = N4XF soup) with a 95%
   upper bound < 0.
4. Finalists: every eligible, non-dropped artifact (at most one per arm). Verdict "multilingual-Noul fix (dev)":
   Noul-ML paired lower bound > 0 vs the N4XF soup and typed-DEV correct per type >= 0.95 x the N4XF soup's.

An artifact whose MLX-DEV comparison is missing (a median seed is not read out automatically) is reported as
pending with the command that produces it, and is neither dropped nor a finalist until it exists.

usage: python3 m5-select.py [--root /data/dev2/runs/dec/m5] --output <prefix> [N5N N5B N5BN]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

TYPES = ("choice", "noul", "score")
FLOORS = {"choice": 345, "noul": 171, "score": 284}
BAND = 8.0
TYPED_RATIO = 0.95
STAT, STAT_MEAN3 = "proxy", "proxy_mean_H"
SEEDS = ("s1", "s2", "s3")


def categories(path: Path) -> dict[str, Counter]:
    """Answer categories per type: Choice key, Noul true/false/abstain, Score argmax level."""
    by_type: dict[str, Counter] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            answer = json.loads(line)["answers"]["decision"]
            kind = answer["type"]
            if kind == "choice":
                category = answer.get("choice")
            elif kind == "noul":
                value = answer.get("noul")
                category = None if value is None or value == 0.5 else value > 0.5
            else:
                probs = answer.get("probabilities")
                category = max(probs, key=probs.get) if probs else answer.get("score")
            by_type.setdefault(kind, Counter())[category] += 1
    return by_type


def top_share(by_type: dict[str, Counter]) -> dict[str, float]:
    return {k: max(c.values()) / sum(c.values()) for k, c in sorted(by_type.items())}


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def artifact(root: Path, arm: str) -> dict:
    readout = json.loads((root / "soup" / arm / "readout.json").read_text())
    arms = readout["arms"]
    seeds = [s for s in SEEDS if s in arms]
    values = {s: arms[s][STAT] for s in seeds}
    mean = statistics.mean(values.values())
    if arms["soup"][STAT] >= mean:
        chosen, name = "soup", f"m5-{arm}-soup"
        pred = root / "soup" / arm / "dev" / "dev.predictions.jsonl"
    else:
        chosen = sorted(seeds, key=lambda s: (values[s], s))[len(seeds) // 2]
        name = f"m5-{arm}-{chosen}"
        pred = (
            root
            / "arms"
            / "full"
            / f"m5-{arm}-{chosen}-post"
            / "dev"
            / "dev.predictions.jsonl"
        )
    data, ref = arms[chosen], arms["n4xf"]
    typed = {k: data["by_type"][k]["correct"] for k in TYPES}
    ref_typed = {k: ref["by_type"][k]["correct"] for k in TYPES}
    invalid = sum(v.get("invalid", 0) for v in data["by_type"].values())
    shares = top_share(categories(pred))
    reasons = [f"{k} {typed[k]} < {FLOORS[k]}" for k in TYPES if typed[k] < FLOORS[k]]
    reasons += [f"{invalid} invalid"] if invalid else []
    reasons += [f"{k} constant" for k, v in shares.items() if v >= 1.0]
    mlx = root / "mlxdev" / "readouts" / name
    vs = load(mlx / "vs-n4xf-soup.json")
    noul = vs["metrics"]["noul_ml"] if vs else None
    drops = []
    if data[STAT] <= ref[STAT] - BAND:
        drops.append(f"R {data[STAT]:.2f} <= N4XF soup R {ref[STAT]:.2f} - {BAND:g}")
    if noul and noul["ci95"] and noul["ci95"][1] < 0:
        drops.append(f"Noul-ML vs N4XF soup upper bound {noul['ci95'][1]:.4f} < 0")
    pending = [] if vs else [f"MLX-DEV comparison missing: {mlx}/vs-n4xf-soup.json"]
    if not vs and chosen != "soup":
        pending.append(
            f"run: m5-mlx.sh <mirror> <gpu> {name} checkpoint "
            f"/runs/m5/arms/full/m5-{arm}-{chosen}/<BEST.json checkpoint>"
        )
    typed_ok = all(typed[k] >= TYPED_RATIO * ref_typed[k] for k in TYPES)
    fix = bool(noul and noul["ci95"] and noul["ci95"][0] > 0 and typed_ok)
    return {
        "arm": arm,
        "artifact": chosen,
        "mlxdev_name": name,
        "R": data[STAT],
        "P_mean3": data[STAT_MEAN3],
        "T": data["T"],
        "H_median": data["H"],
        "H_mean3": data["H_mean"],
        "seed_R": values,
        "seed_mean_R": mean,
        "soup_R": arms["soup"][STAT],
        "n4xf_soup_R": ref[STAT],
        "nox1_R": arms["nox1"][STAT],
        "typed": typed,
        "n4xf_soup_typed": ref_typed,
        "typed_within_5pct_of_n4xf_soup": typed_ok,
        "invalid": invalid,
        "top_answer_share": shares,
        "eligible": not reasons,
        "reasons": reasons,
        "mlxdev_score": load(mlx / "score.json"),
        "noul_ml_vs_n4xf_soup": noul,
        "dropped": bool(drops),
        "drop_reasons": drops,
        "pending": pending,
        "finalist": not reasons and not drops and not pending,
        "multilingual_noul_fix_dev": fix,
    }


def fmt(value: float | None, digits: int = 2) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def table(rows: list[dict]) -> str:
    head = (
        "| arm | artifact | R | P_mean3 | seed-mean R | N4XF soup R | typed C/N/S | eligible | "
        "Noul-ML Δ vs N4XF soup [95% CI] | M_dev | dropped | finalist | Noul fix (dev) |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for r in rows:
        noul = r["noul_ml_vs_n4xf_soup"]
        ci = (
            f"{noul['diff']:+.4f} [{noul['ci95'][0]:+.4f}, {noul['ci95'][1]:+.4f}]"
            if noul and noul["ci95"]
            else "pending"
        )
        score = r["mlxdev_score"] or {}
        lines.append(
            f"| {r['arm']} | {r['artifact']} | {fmt(r['R'])} | {fmt(r['P_mean3'])} | {fmt(r['seed_mean_R'])} | "
            f"{fmt(r['n4xf_soup_R'])} | {'/'.join(str(r['typed'][k]) for k in TYPES)} | "
            f"{'yes' if r['eligible'] else 'no: ' + '; '.join(r['reasons'])} | {ci} | "
            f"{fmt(score.get('m_dev'), 4)} | {'; '.join(r['drop_reasons']) or 'no'} | "
            f"{'yes' if r['finalist'] else ('pending' if r['pending'] else 'no')} | "
            f"{'yes' if r['multilingual_noul_fix_dev'] else 'no'} |"
        )
    return head + "\n".join(lines) + "\n"


def select(root: Path, arms: list[str]) -> dict:
    rows = [artifact(root, a) for a in arms]
    return {
        "rule": "dec-m5-prereg-2026-09-29.md Selection 1-4; amendment 1 e574bbf56",
        "stat": STAT,
        "band": BAND,
        "floors": FLOORS,
        "rows": rows,
        "finalists": [(r["arm"], r["artifact"]) for r in rows if r["finalist"]],
        "pending": [r["arm"] for r in rows if r["pending"]],
        "multilingual_noul_fix_dev": [
            r["arm"] for r in rows if r["finalist"] and r["multilingual_noul_fix_dev"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arms", nargs="*", default=["N5N", "N5B", "N5BN"])
    parser.add_argument("--root", type=Path, default=Path("/data/dev2/runs/dec/m5"))
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="prefix; writes <prefix>.json and <prefix>.md",
    )
    args = parser.parse_args()
    result = select(args.root, args.arms)
    Path(f"{args.output}.json").write_text(json.dumps(result, indent=1) + "\n")
    md = (
        "Decoder M5 selection (development readouts; MLX-DEV Noul is answerability / relevance, not paraphrase, "
        "and in-distribution for arms that train more on these sub-arms).\n\n"
        + table(result["rows"])
        + f"\nFinalists: {result['finalists'] or 'none'}; pending: {result['pending'] or 'none'}; "
        f"multilingual-Noul fix (dev): {result['multilingual_noul_fix_dev'] or 'none'}.\n"
    )
    Path(f"{args.output}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
