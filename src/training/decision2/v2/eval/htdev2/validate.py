"""HT-DEV v2 validation: does ΔH_dev2 track formal within-tier ΔH better than the pilot mean?

Preregistration `v2/eval/records/htdev2-prereg-2026-09-30.md` §5 (amendments 1-2).

    python3 -m v2.eval.htdev2.validate extract --spec <spec.json> --collect-root <dir> \
        [--htdev1-features <features.json>] --output <features.json>
    python3 -m v2.eval.htdev2.validate analyze --features <features.json> [--c1 <c1.json>] --output <analysis.json>

`extract` (node A) reads, per model, the HT-DEV v2 report of its collection, its CSS-pilot
predictions (the spec's stored readout or the same job's collection) and the stored formal
`REPORT.json`; it writes per-model aggregates only. `analyze` runs anywhere.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from v2.eval.htdev.validate import (
    agreement,
    binned,
    correlations,
    quantile,
    tier_r,
    within_pairs,
)
from v2.eval.proxy_calibration import ols, pearson
from v2.eval.same_panel import sha_file, utc_now, write_json

FEATURES_SCHEMA = "dev2-htdev2-validation-features/1"
ANALYSIS_SCHEMA = "dev2-htdev2-validation/1"
THRESHOLDS = {"all": 0.0, "ge_0.01": 0.01, "ge_0.02": 0.02, "ge_0.03": 0.03}
PRIMARY = "ge_0.02"
DRAWS = 5000
SEED = 20260930
P_BETTER_MIN = 0.90
Z_90 = 1.2816
CANDIDATE = "H_dev2"
BAR = "H_mean3"
SECONDARY = ("H_dev2_median", "H_pilot", "H_comb16")
GAP_BINS = (0.0, 0.01, 0.02, 0.04, math.inf)


# ---------------------------------------------------------------- extract (node A)


def pilot_features(path: Path, panel_root: Path) -> dict[str, Any]:
    from transfer.score import score
    from v2.eval import panels

    report = score(panels.path(panel_root, "css-pilot", "gold"), path)
    f1 = {t: v["macro_f1_all"] for t, v in report["tasks"].items()}
    return {
        "predictions_sha256": report["predictions_sha256"],
        "tasks": f1,
        "H_mean3": statistics.fmean(f1.values()),
        "H_pilot": statistics.median(f1.values()),
        "invalid": sum(v["invalid_or_missing_n"] for v in report["tasks"].values()),
    }


def extract(args: argparse.Namespace) -> dict[str, Any]:
    from v2.eval import panels

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    v1 = {}
    if args.htdev1_features:
        for m in json.loads(args.htdev1_features.read_text(encoding="utf-8"))["models"]:
            v1[m["key"]] = {
                "H_dev_v1": m["htdev"]["H_dev"],
                "H_dev_v1_mean": m["htdev"]["H_dev_mean"],
            }
    hashes = panels.verify(args.panel_root, ["css15", "css-pilot", "ht-dev2"])
    models = []
    for entry in spec["models"]:
        run = args.collect_root / entry["key"]
        dev = json.loads((run / "REPORT-HTDEV2.json").read_text(encoding="utf-8"))
        pilot_path = (
            Path(entry["pilot_predictions"])
            if entry.get("pilot_predictions")
            else run / "output/css-pilot.predictions.jsonl"
        )
        formal = json.loads(
            (Path(entry["formal_run"]) / "REPORT.json").read_text(encoding="utf-8")
        )
        css = formal["panels"]["css15"]
        models.append(
            {
                "key": entry["key"],
                "tier": entry["tier"],
                "group": entry["group"],
                "lineage": entry["lineage"],
                "formal": {
                    "report_sha256": sha_file(
                        Path(entry["formal_run"]) / "REPORT.json"
                    ),
                    "H_formal": css["H"],
                    "css15_task_mean": statistics.fmean(
                        t["macro_f1"] for t in css["tasks"].values()
                    ),
                    "v3": formal["v3"]["score"],
                },
                "htdev2": {
                    "report_sha256": sha_file(run / "REPORT-HTDEV2.json"),
                    "H_dev2": dev["H_dev2"],
                    "H_dev2_median": dev["H_dev2_median"],
                    "sd": dev["bootstrap"]["H_dev2"]["sd"],
                    "tasks": {t: v["macro_f1_all"] for t, v in dev["tasks"].items()},
                    "items": dev["items"],
                    "invalid": dev["items"] - dev["valid"],
                },
                "pilot": {
                    "source": (
                        entry.get("pilot_source")
                        if entry.get("pilot_predictions")
                        else "same-job collection"
                    ),
                    **pilot_features(pilot_path, args.panel_root),
                },
                "htdev1": v1.get(entry["key"]),
            }
        )
    return {
        "schema": FEATURES_SCHEMA,
        "created_utc": utc_now(),
        "scope": "per-model aggregates only (HT-DEV v2 and CSS-pilot readouts, stored formal REPORT.json fields)",
        "spec_sha256": sha_file(args.spec),
        "panels": hashes,
        "models": models,
    }


# ---------------------------------------------------------------- analyze (anywhere)


def flatten(model: dict[str, Any], c1: dict[str, float]) -> dict[str, Any]:
    dev_tasks = list(model["htdev2"]["tasks"].values())
    pilot_tasks = list(model["pilot"]["tasks"].values())
    v1 = model.get("htdev1") or {}
    return {
        "key": model["key"],
        "tier": model["tier"],
        "group": model["group"],
        "lineage": model["lineage"],
        "H_formal": model["formal"]["H_formal"],
        "css15_task_mean": model["formal"]["css15_task_mean"],
        "v3": model["formal"]["v3"],
        "H_dev2": model["htdev2"]["H_dev2"],
        "H_dev2_median": model["htdev2"]["H_dev2_median"],
        "H_mean3": model["pilot"]["H_mean3"],
        "H_pilot": model["pilot"]["H_pilot"],
        "H_comb16": statistics.fmean(dev_tasks + pilot_tasks),
        "H_dev_v1": v1.get("H_dev_v1"),
        "c1": c1.get(model["key"]),
    }


def bootstrap(
    rows, x: str, y: str, target: str, draws: int, seed: int
) -> dict[str, Any]:
    """Paired model bootstrap within tiers of agreement(x) - agreement(y) and r(x) - r(y)."""
    rng = random.Random(seed)
    by_tier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_tier[row["tier"]].append(row)
    deltas, r_deltas, empty = [], [], 0
    for _ in range(draws):
        sample = []
        for tier in sorted(by_tier):
            members = by_tier[tier]
            sample.extend(members[rng.randrange(len(members))] for _ in members)
        pairs = within_pairs(sample)
        a = agreement(sample, pairs, x, target, THRESHOLDS[PRIMARY])
        b = agreement(sample, pairs, y, target, THRESHOLDS[PRIMARY])
        if a["rate"] is None:
            empty += 1
            continue
        deltas.append(a["rate"] - b["rate"])
        r_deltas.append(tier_r(sample, x, target) - tier_r(sample, y, target))

    def summarize(values):
        return {
            "draws": len(values),
            "p_better": (sum(v > 0 for v in values) + 0.5 * sum(v == 0 for v in values))
            / len(values),
            "delta_mean": statistics.fmean(values),
            "delta_ci95": [quantile(values, 0.025), quantile(values, 0.975)],
        }

    return {
        "draws": draws,
        "seed": seed,
        "unit": "models within tiers (with replacement); two draws of one model form no pair",
        "draws_without_decidable_pairs": empty,
        "agreement": summarize(deltas),
        "within_tier_r": summarize(r_deltas),
    }


def pair_delta_r(rows, pairs, x: str, y: str) -> float:
    return pearson(
        [rows[i][x] - rows[j][x] for i, j in pairs],
        [rows[i][y] - rows[j][y] for i, j in pairs],
    )


def screen_band(rows, x: str) -> dict[str, Any]:
    tiers = sorted({r["tier"] for r in rows})
    dummies = [[float(r["tier"] == t) for r in rows] for t in tiers[1:]]
    beta = ols([[r[x] for r in rows], *dummies], [r["H_formal"] for r in rows])
    fitted = [
        beta[0] + beta[1] * r[x] + sum(b * d[k] for b, d in zip(beta[2:], dummies))
        for k, r in enumerate(rows)
    ]
    dof = len(rows) - len(beta)
    sigma = math.sqrt(sum((r["H_formal"] - f) ** 2 for r, f in zip(rows, fitted)) / dof)
    gap = Z_90 * math.sqrt(2) * sigma / beta[1] if beta[1] > 0 else None
    return {
        "slope": beta[1],
        "residual_sd": sigma,
        "dof": dof,
        "gap": gap,
        "band": max(1, math.ceil(round(gap / 0.005, 9))) * 0.005 if gap else None,
        "rule": "formal H = tier effect + b * x; band = 1.2816*sqrt(2)*sigma/b rounded up to 0.005",
    }


def block(rows, pairs, x: str, target: str = "H_formal") -> dict[str, Any]:
    return {
        "agreement": {
            n: agreement(rows, pairs, x, target, t) for n, t in THRESHOLDS.items()
        },
        **correlations(rows, x, target),
        "pair_delta_pearson": pair_delta_r(rows, pairs, x, target),
    }


def analyze(
    data: dict[str, Any], c1: dict[str, float], draws: int = DRAWS
) -> dict[str, Any]:
    rows = [flatten(m, c1) for m in data["models"]]
    pairs = within_pairs(rows)
    names = (CANDIDATE, BAR, *SECONDARY)
    main = {x: block(rows, pairs, x) for x in names}
    boot = bootstrap(rows, CANDIDATE, BAR, "H_formal", draws, SEED)
    rate = main[CANDIDATE]["agreement"][PRIMARY]["rate"]
    bar_rate = main[BAR]["agreement"][PRIMARY]["rate"]
    r_dev, r_bar = (
        main[CANDIDATE]["within_tier_pearson"],
        main[BAR]["within_tier_pearson"],
    )
    cond_i = rate > bar_rate and boot["agreement"]["p_better"] >= P_BETTER_MIN
    cond_ii = r_dev > r_bar
    siblings = [
        (i, j)
        for i, j in pairs
        if rows[i]["lineage"] == rows[j]["lineage"]
        and rows[i]["group"] == rows[j]["group"] == "2.0 candidate"
    ]
    per_tier = {}
    for tier in sorted({r["tier"] for r in rows}):
        tier_pairs = [(i, j) for i, j in pairs if rows[i]["tier"] == tier]
        per_tier[tier] = {
            "models": sum(r["tier"] == tier for r in rows),
            **{
                x: agreement(rows, tier_pairs, x, "H_formal", THRESHOLDS[PRIMARY])
                for x in (CANDIDATE, BAR, "H_dev2_median")
            },
            "H_formal_range": [
                min(r["H_formal"] for r in rows if r["tier"] == tier),
                max(r["H_formal"] for r in rows if r["tier"] == tier),
            ],
        }
    v1_rows = [r for r in rows if r["H_dev_v1"] is not None]
    v1_pairs = within_pairs(v1_rows)
    c1_rows = [r for r in rows if r["c1"] is not None]
    c1_pairs = within_pairs(c1_rows)
    band = screen_band(rows, CANDIDATE)
    return {
        "schema": ANALYSIS_SCHEMA,
        "created_utc": utc_now(),
        "label": "development readout (HT-DEV v2 validation); never a release score",
        "features_sha256": data.get("_sha256"),
        "models": len(rows),
        "tiers": {
            t: sum(r["tier"] == t for r in rows)
            for t in sorted({r["tier"] for r in rows})
        },
        "within_tier_pairs": len(pairs),
        "primary": {
            "target": "formal CSS15 H (median task macro-F1)",
            "blocks": main,
            "paired_bootstrap": boot,
            "decision": {
                "H_dev2_agreement": rate,
                "H_mean3_agreement": bar_rate,
                "p_better": boot["agreement"]["p_better"],
                "H_dev2_within_tier_r": r_dev,
                "H_mean3_within_tier_r": r_bar,
                "condition_i": cond_i,
                "condition_ii": cond_ii,
                "passes": cond_i and cond_ii,
                "rule": "passes iff (i) agreement on |dH_formal| >= 0.02 higher with P(better) >= 0.90 and (ii) within-tier Pearson r higher",
            },
        },
        "secondary": {
            "per_tier": per_tier,
            "siblings": {
                "pairs": len(siblings),
                **{
                    x: {
                        n: agreement(rows, siblings, x, "H_formal", t)
                        for n, t in THRESHOLDS.items()
                    }
                    for x in (CANDIDATE, BAR, "H_dev2_median")
                },
            },
            "css15_task_mean_target": {
                x: block(rows, pairs, x, "css15_task_mean") for x in (CANDIDATE, BAR)
            },
            "agreement_by_H_dev2_gap": binned(
                rows, pairs, CANDIDATE, "H_formal", GAP_BINS, None
            ),
            "htdev_v1_subset": (
                {
                    "models": len(v1_rows),
                    **{
                        x: block(v1_rows, v1_pairs, x)
                        for x in (CANDIDATE, BAR, "H_dev_v1")
                    },
                }
                if v1_pairs
                else None
            ),
            "c1_postkey_informational": (
                None
                if not c1_pairs
                else {
                    "note": "stored C1 aggregates of the event models (events 1-2 on v1.1, event 3 on v1.2); reported only, never tuned to",
                    "models": len(c1_rows),
                    "pairs": len(c1_pairs),
                    **{
                        x: {
                            "agreement_all": agreement(c1_rows, c1_pairs, x, "c1", 0.0),
                            "agreement_ge_1pt": agreement(
                                c1_rows, c1_pairs, x, "c1", 1.0
                            ),
                        }
                        for x in (CANDIDATE, BAR, "H_formal")
                    },
                }
            ),
        },
        "screen_band": band,
        "table": [
            {
                k: r[k]
                for k in (
                    "key",
                    "tier",
                    "group",
                    "H_formal",
                    "H_dev2",
                    "H_dev2_median",
                    "H_mean3",
                    "H_pilot",
                    "H_comb16",
                    "c1",
                )
            }
            for r in rows
        ],
    }


def main(argv: list[str] | None = None) -> int:
    from v2.eval import panels

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("extract")
    one.add_argument("--spec", type=Path, required=True)
    one.add_argument("--collect-root", type=Path, required=True)
    one.add_argument("--htdev1-features", type=Path)
    one.add_argument("--panel-root", type=Path, default=panels.DEFAULT_ROOT)
    one.add_argument("--output", type=Path, required=True)
    two = commands.add_parser("analyze")
    two.add_argument("--features", type=Path, required=True)
    two.add_argument("--c1", type=Path)
    two.add_argument("--draws", type=int, default=DRAWS)
    two.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "extract":
        write_json(args.output, extract(args))
        return 0
    data = json.loads(args.features.read_text(encoding="utf-8"))
    data["_sha256"] = sha_file(args.features)
    c1 = json.loads(args.c1.read_text(encoding="utf-8"))["c1"] if args.c1 else {}
    result = analyze(data, c1, args.draws)
    write_json(args.output, result)
    print(json.dumps(result["primary"]["decision"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
