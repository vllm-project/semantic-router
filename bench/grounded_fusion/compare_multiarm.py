"""Paired, multi-arm comparison of cached-panel fusion arms + a KEEP/KILL verdict.

The cached-panel evaluator runs every arm on the byte-identical panel, so deltas
isolate the intervention. This joins ``samples_{arm}.jsonl`` across arms on a STRICT
paired set (ids present and error-free in EVERY arm) and applies the pre-registered
decision rule:

    A judge-solo   B plain-fusion   C weight (default)   D random-weight placebo

    KEEP_GROUNDING       if C > B and C > D and B >= A   (all paired, CI excludes 0)
    KILL_GROUNDING_ADDON if C ~= B or C ~= D             (grounding adds nothing)
    KILL_FUSION          if A significantly beats B      (fusion loses to one model)
    INCONCLUSIVE         otherwise (e.g. fusion vs solo unproven)

Gating is on the normalized DRACO score (higher = better); negative_penalty (toward
0 = better) is reported as corroboration. "favorable" = bootstrap CI lower bound > 0.

    python -m bench.grounded_fusion.compare_multiarm \
        --results-dir results --arms A,B,C,D --json-out results/verdict.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import paired_bootstrap_ci

METRICS = ("normalized", "negative_penalty")


def _load_arm(results_dir: Path, arm: str) -> dict[str, dict]:
    path = results_dir / f"samples_{arm}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path} -- grade arm {arm} first (grade_only.py)")
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r
    return out


def _paired_ids(byarm: dict[str, dict[str, dict]], arms: list[str]) -> list[str]:
    """Ids present and error-free in EVERY arm (strict paired set)."""
    common: set[str] | None = None
    for arm in arms:
        ok = {i for i, r in byarm[arm].items() if not r.get("error")}
        common = ok if common is None else (common & ok)
    return sorted(common or set())


def _delta_block(
    byarm: dict[str, dict[str, dict]], x: str, y: str, key: str, ids: list[str]
) -> dict:
    deltas = [byarm[x][i]["final"][key] - byarm[y][i]["final"][key] for i in ids]
    mean, lo, hi = paired_bootstrap_ci(deltas)
    return {
        "n": len(ids),
        "mean_delta": mean,
        "ci95": [lo, hi],
        "significant": (lo > 0) or (hi < 0),
        # Higher is better for both metrics (normalized up; penalty toward 0), so a
        # favorable X-vs-Y is a strictly positive lower CI bound.
        "favorable": lo > 0,
    }


def _pair(byarm, x, y, ids) -> dict:
    return {m: _delta_block(byarm, x, y, m, ids) for m in METRICS}


def _contested(byarm: dict[str, dict[str, dict]], ids: list[str]) -> int:
    """Items where arm C flagged or dropped a panel response (where grounding acts)."""
    if "C" not in byarm:
        return 0
    n = 0
    for i in ids:
        panel = byarm["C"].get(i, {}).get("panel", [])
        if any(p.get("dropped") or p.get("flagged") for p in panel):
            n += 1
    return n


def _decide(pairs: dict[str, dict], have: set[str]) -> tuple[str, str]:
    """Apply the pre-registered rule on the normalized-score pair blocks."""
    norm = {k: v["normalized"] for k, v in pairs.items()}

    # KILL_FUSION: arm A (one model) significantly beats arm B (plain fusion).
    if {"A", "B"} <= have:
        ba = norm["B_vs_A"]
        if ba["ci95"][1] < 0:
            return "KILL_FUSION", (
                f"plain fusion loses to the judge model alone "
                f"(B-A normalized {ba['mean_delta']:+.4f}, CI {ba['ci95']}); "
                "fusion itself is not worth its cost."
            )
        if not ba["favorable"]:
            return "INCONCLUSIVE", (
                f"fusion is not shown to beat one model (B-A normalized {ba['mean_delta']:+.4f}, "
                f"CI {ba['ci95']} includes 0); need more N before judging grounding."
            )

    if not ({"B", "C"} <= have):
        return "INCONCLUSIVE", "arms B and C are both required to judge grounding."

    cb = norm["C_vs_B"]
    cd = norm.get("C_vs_D")
    if not cb["favorable"]:
        return "KILL_GROUNDING_ADDON", (
            f"weight does not beat plain fusion (C-B normalized {cb['mean_delta']:+.4f}, "
            f"CI {cb['ci95']}); the grounding stage adds nothing over plain synthesis."
        )
    if cd is not None and not cd["favorable"]:
        return "KILL_GROUNDING_ADDON", (
            f"weight does not beat the random-weight placebo (C-D normalized "
            f"{cd['mean_delta']:+.4f}, CI {cd['ci95']}); the gain is from weighting, "
            "not the groundedness score."
        )
    if cd is None:
        return "INCONCLUSIVE", (
            f"weight beats plain fusion (C-B {cb['mean_delta']:+.4f}, CI {cb['ci95']}) but the "
            "placebo arm D is missing — cannot rule out that any weighting would do."
        )
    return "KEEP_GROUNDING", (
        f"weight beats both plain fusion (C-B {cb['mean_delta']:+.4f}, CI {cb['ci95']}) and the "
        f"random-weight placebo (C-D {cd['mean_delta']:+.4f}, CI {cd['ci95']}); the groundedness "
        "score earns its place."
    )


def compare(results_dir: str, arms: list[str]) -> dict:
    d = Path(results_dir)
    byarm = {arm: _load_arm(d, arm) for arm in arms}
    have = set(arms)
    ids = _paired_ids(byarm, arms)
    if not ids:
        raise SystemExit("no overlapping error-free samples across all arms")

    # Pairs needed for the decision rule (only those whose both arms are present).
    wanted = [
        ("C_vs_B", "C", "B"),
        ("C_vs_D", "C", "D"),
        ("B_vs_A", "B", "A"),
        ("C_vs_A", "C", "A"),
    ]
    pairs = {name: _pair(byarm, x, y, ids) for name, x, y in wanted if {x, y} <= have}

    decision, rationale = _decide(pairs, have)
    per_arm_mean = {
        arm: (sum(byarm[arm][i]["final"]["normalized"] for i in ids) / len(ids))
        for arm in arms
    }
    return {
        "arms": arms,
        "n_paired": len(ids),
        "n_contested": _contested(byarm, ids),
        "per_arm_mean_normalized": per_arm_mean,
        "pairs": pairs,
        "decision": decision,
        "rationale": rationale,
    }


def _print(report: dict) -> None:
    print("\n" + "=" * 64)
    print("CACHED-PANEL FUSION: multi-arm paired comparison")
    print("=" * 64)
    print(
        f"paired samples (all arms, error-free): {report['n_paired']} | contested: {report['n_contested']}"
    )
    print("\n[per-arm mean normalized]")
    for arm, m in report["per_arm_mean_normalized"].items():
        print(f"  {arm:10} {m:.4f}")
    for name, block in report["pairs"].items():
        print(f"\n[{name}]")
        for metric in METRICS:
            b = block[metric]
            sig = "SIGNIFICANT" if b["significant"] else "ns"
            print(
                f"  {metric:18} delta={b['mean_delta']:+.4f}  ci95=[{b['ci95'][0]:+.4f},{b['ci95'][1]:+.4f}]  {sig}"
            )
    print("\n" + "-" * 64)
    print(f"DECISION: {report['decision']}")
    print(f"  {report['rationale']}")
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser(
        description="Multi-arm paired compare + KEEP/KILL verdict"
    )
    ap.add_argument("--results-dir", default="bench/grounded_fusion/results")
    ap.add_argument("--arms", default="A,B,C,D", help="comma-separated arms to compare")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    report = compare(args.results_dir, arms)
    _print(report)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
