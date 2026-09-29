"""HT-DEV v1 validation: does H_dev track formal human transfer better than the CSS pilot?

Preregistration `v2/eval/records/htdev-prereg-2026-09-29.md` §5-§6 (amendments 1-2).

    python3 -m v2.eval.htdev.validate extract --spec <spec.json> --output <features.json>
    python3 -m v2.eval.htdev.validate analyze --features <features.json> --output <analysis.json>

`extract` (node A) reads, per model, the same-job HT-DEV report (`REPORT-HTDEV.json` from
`v2.eval.htdev.score`), the typed-DEV and CSS-pilot predictions of the same job (T_dev, H_pilot,
pilot three-task mean and their bootstrap noise via `v2.eval.proxy_calibration`, cross-checked
with `v2.eval.dev_readout`) and the stored formal `REPORT.json` (css15 H, css15 task-mean
macro-F1, v3). It writes per-model aggregates only: no items, labels or answers, and no v3 item
is re-read. `analyze` runs anywhere on the features file.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Any

from v2.eval import panels
from v2.eval.proxy_calibration import (
    error_summary,
    linear_1d,
    loo_1d,
    ols,
    pearson,
    ranks,
    within_tier_r,
)
from v2.eval.same_panel import sha_file, utc_now, write_json

FEATURES_SCHEMA = "dev2-htdev-validation-features/1"
ANALYSIS_SCHEMA = "dev2-htdev-validation/1"
TIERS = ("0.6B", "0.8B", "2B", "4B", "9B")
THRESHOLDS = {"all": 0.0, "ge_0.01": 0.01, "ge_0.02": 0.02, "ge_0.03": 0.03}
PRIMARY = "ge_0.02"
DRAWS = 5000
SEED = 20260929
Z_90 = 1.2816
P_BETTER_MIN = 0.90
V3_DECISION = 2.0
TIE_EDGES = (2, 4, 6, 8, 10, 12)
MAX_REVERSAL = 0.10
H_DEV_BINS = (0.0, 0.01, 0.02, 0.04, math.inf)
P_BINS = (0.0, 4.0, 8.0, math.inf)
COMPARATORS = ("H_pilot", "H_mean3")
TARGETS = ("H_formal", "css15_task_mean")


# ---------------------------------------------------------------- extract (node A)


def extract_model(
    entry: dict[str, Any], panel_root: Path, draws: int
) -> dict[str, Any]:
    from v2.eval import dev_readout, proxy_calibration

    run_dir = Path(entry["run_dir"])
    output = run_dir / "output"
    base = proxy_calibration.extract_model(
        {
            "key": entry["key"],
            "report": entry["formal_report"],
            "typed_dev": str(output / "typed-dev.predictions.jsonl"),
            "css_pilot": str(output / "css-pilot.predictions.jsonl"),
        },
        panel_root,
        draws,
    )
    readout = dev_readout.readout(
        Namespace(
            run_dir=run_dir,
            typed_dev=None,
            css_pilot=None,
            select=None,
            cal=None,
            label=entry["key"],
            panel_root=panel_root,
        )
    )
    feats = base["features"]
    for ours, theirs in (
        (feats["T_dev"], readout["typed_dev"]["T_dev"]),
        (feats["H_pilot"], readout["css_pilot"]["H_pilot"]),
    ):
        if abs(ours - theirs) > 1e-9:
            raise ValueError(
                f"{entry['key']}: proxy_calibration and dev_readout disagree"
            )
    pilot_tasks = readout["css_pilot"]["tasks"]
    htdev_path = run_dir / "REPORT-HTDEV.json"
    htdev = json.loads(htdev_path.read_text(encoding="utf-8"))
    report = json.loads(Path(entry["formal_report"]).read_text(encoding="utf-8"))
    css15 = report["panels"]["css15"]
    css_f1 = [task["macro_f1"] for task in css15["tasks"].values()]
    if abs(statistics.median(css_f1) - css15["H"]) > 1e-9:
        raise ValueError(
            f"{entry['key']}: css15 H is not the median of its task macro-F1"
        )
    t_dev, h_dev = feats["T_dev"], htdev["H_dev"]
    return {
        "key": entry["key"],
        "tier": entry["tier"],
        "group": entry["group"],
        "label": entry.get("label"),
        "formal": {
            "report": entry["formal_report"],
            "report_sha256": sha_file(Path(entry["formal_report"])),
            "H_formal": css15["H"],
            "css15_task_mean": statistics.fmean(css_f1),
            "css15_tasks": len(css_f1),
            "v3": report["v3"]["score"],
            "T_formal": report["v3"]["T"],
        },
        "htdev": {
            "report_sha256": sha_file(htdev_path),
            "H_dev": h_dev,
            "H_dev_mean": htdev["task_mean"],
            "by_type_mean": htdev["by_type_mean"],
            "score_qwk_mean": htdev.get("score_qwk_mean"),
            "tasks": {
                name: {
                    k: task.get(k) for k in ("type", "n", "valid", "macro_f1", "qwk")
                }
                for name, task in sorted(htdev["tasks"].items())
            },
            "items": htdev["items"],
            "valid": htdev["valid"],
            "invalid": htdev["items"] - htdev["valid"],
            "H_dev_sd": htdev["H_dev_bootstrap"]["sd"],
            "H_dev_ci95": htdev["H_dev_bootstrap"]["ci95"],
        },
        "development": {
            "T_dev": t_dev,
            "H_pilot": feats["H_pilot"],
            "H_mean3": feats["H_mean3"],
            "pilot_tasks": pilot_tasks,
            "typed_invalid": readout["typed_dev"]["invalid_or_missing"],
            "pilot_invalid": readout["css_pilot"]["invalid_or_missing"],
            "noise_sd": {
                k: base["noise_sd"][k] for k in ("T_dev", "H_pilot", "H_mean3", "P")
            },
            "inputs": base["inputs"],
        },
        "proxies": {
            "P": feats["P"],
            "P_HT": 100 * math.sqrt(t_dev * h_dev),
            "P_HT_mean": 100 * math.sqrt(t_dev * htdev["task_mean"]),
        },
    }


def extract(spec_path: Path, panel_root: Path, draws: int) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    hashes = panels.verify(panel_root, ["typed-dev", "css-pilot", "ht-dev"])
    return {
        "schema": FEATURES_SCHEMA,
        "created_utc": utc_now(),
        "scope": "per-model aggregates only (development readouts + stored formal REPORT.json fields)",
        "spec_sha256": sha_file(spec_path),
        "panels": hashes,
        "bootstrap": {
            "typed_pilot_draws": draws,
            "typed_pilot_unit": "typed DEV groups within family; CSS pilot items within task",
            "htdev": "v2.eval.htdev.score item bootstrap (2,000 draws, seed 20260929)",
        },
        "formal_H_item_sd": "not computed: it needs css15 gold and items (no v3 item is re-read)",
        "models": [extract_model(entry, panel_root, draws) for entry in spec["models"]],
    }


# ---------------------------------------------------------------- analyze (anywhere)


def flatten(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": model["key"],
        "tier": model["tier"],
        "group": model["group"],
        "H_formal": model["formal"]["H_formal"],
        "css15_task_mean": model["formal"]["css15_task_mean"],
        "v3": model["formal"]["v3"],
        "H_dev": model["htdev"]["H_dev"],
        "H_dev_mean": model["htdev"]["H_dev_mean"],
        "H_pilot": model["development"]["H_pilot"],
        "H_mean3": model["development"]["H_mean3"],
        "T_dev": model["development"]["T_dev"],
        **model["proxies"],
    }


def pair_score(dx: float, dy: float) -> float:
    if dx == 0:
        return 0.5
    return 1.0 if dx * dy > 0 else 0.0


def within_pairs(rows: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Index pairs in the same tier, skipping two draws of the same model."""
    return [
        (i, j)
        for i, j in itertools.combinations(range(len(rows)), 2)
        if rows[i]["tier"] == rows[j]["tier"] and rows[i]["key"] != rows[j]["key"]
    ]


def agreement(rows, pairs, x: str, y: str, threshold: float) -> dict[str, Any]:
    total, n = 0.0, 0
    for i, j in pairs:
        dy = rows[i][y] - rows[j][y]
        if dy == 0 or abs(dy) < threshold:
            continue
        total += pair_score(rows[i][x] - rows[j][x], dy)
        n += 1
    return {"n": n, "score": total, "rate": total / n if n else None}


def tier_r(rows, x: str, y: str) -> float:
    return within_tier_r(
        [r[x] for r in rows], [r[y] for r in rows], [r["tier"] for r in rows]
    )


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(ranks(xs), ranks(ys))


def correlations(rows, x: str, y: str) -> dict[str, float]:
    xs, ys = [r[x] for r in rows], [r[y] for r in rows]
    return {
        "within_tier_pearson": tier_r(rows, x, y),
        "cross_tier_spearman": spearman(xs, ys),
        "cross_tier_pearson": pearson(xs, ys),
    }


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[int(q * (len(ordered) - 1))]


def summary(deltas: list[float]) -> dict[str, Any]:
    finite = [d for d in deltas if d is not None and not math.isnan(d)]
    return {
        "draws": len(finite),
        "p_better": (sum(d > 0 for d in finite) + 0.5 * sum(d == 0 for d in finite))
        / len(finite),
        "delta_ci95": [quantile(finite, 0.025), quantile(finite, 0.975)],
        "delta_mean": statistics.fmean(finite),
    }


def paired_bootstrap(
    rows, draws: int, seed: int, target: str = "H_formal"
) -> dict[str, Any]:
    """Resample each tier's models with replacement; both statistics on the same resample."""
    rng = random.Random(seed)
    by_tier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_tier[row["tier"]].append(row)
    out: dict[str, dict[str, list[float]]] = {
        comp: {"primary": [], "within_r": []} for comp in COMPARATORS
    }
    empty = 0
    for _ in range(draws):
        sample = []
        for tier in sorted(by_tier):
            members = by_tier[tier]
            sample.extend(members[rng.randrange(len(members))] for _ in members)
        pairs = within_pairs(sample)
        dev = agreement(sample, pairs, "H_dev", target, THRESHOLDS[PRIMARY])
        r_dev = tier_r(sample, "H_dev", target)
        if dev["rate"] is None:
            empty += 1
        for comp in COMPARATORS:
            other = agreement(sample, pairs, comp, target, THRESHOLDS[PRIMARY])
            if dev["rate"] is not None:
                out[comp]["primary"].append(dev["rate"] - other["rate"])
            out[comp]["within_r"].append(r_dev - tier_r(sample, comp, target))
    return {
        "draws": draws,
        "seed": seed,
        "unit": "models within tiers (with replacement); two draws of one model form no pair",
        "draws_without_decidable_pairs": empty,
        **{
            comp: {stat: summary(values) for stat, values in stats.items()}
            for comp, stats in out.items()
        },
    }


def binned(
    rows, pairs, x: str, y: str, edges, reversal: float | None
) -> list[dict[str, Any]]:
    out = []
    for low, high in zip(edges, edges[1:]):
        chosen = [
            (i, j)
            for i, j in pairs
            if low <= abs(rows[i][x] - rows[j][x]) < high and rows[i][y] != rows[j][y]
        ]
        scores = [
            pair_score(rows[i][x] - rows[j][x], rows[i][y] - rows[j][y])
            for i, j in chosen
        ]
        row = {
            "bin": [low, None if math.isinf(high) else high],
            "n": len(chosen),
            "agree": statistics.fmean(scores) if scores else None,
        }
        if reversal is not None:
            rev = sum(
                (rows[i][x] - rows[j][x]) * (rows[i][y] - rows[j][y]) < 0
                and abs(rows[i][y] - rows[j][y]) >= reversal
                for i, j in chosen
            )
            row["reversal_rate"] = rev / len(chosen) if chosen else None
        out.append(row)
    return out


def proxy_block(rows, pairs, proxy: str) -> dict[str, Any]:
    xs, ys = [r[proxy] for r in rows], [r["v3"] for r in rows]
    intercept, slope = linear_1d(xs, ys)
    loo = error_summary(loo_1d(xs, ys), ys)
    normal_gap = Z_90 * math.sqrt(2) * loo["rmse"] / slope if slope > 0 else None
    grid = []
    for edge in TIE_EDGES:
        chosen = [
            (i, j) for i, j in pairs if abs(xs[i] - xs[j]) >= edge and ys[i] != ys[j]
        ]
        rev = sum(
            (xs[i] - xs[j]) * (ys[i] - ys[j]) < 0 and abs(ys[i] - ys[j]) >= V3_DECISION
            for i, j in chosen
        )
        grid.append(
            {
                "edge": edge,
                "n": len(chosen),
                "reversal_rate": rev / len(chosen) if chosen else None,
            }
        )
    empirical = next(
        (g["edge"] for g in grid if g["n"] and g["reversal_rate"] <= MAX_REVERSAL), None
    )
    candidates = [
        v
        for v in (math.ceil(normal_gap) if normal_gap else None, empirical)
        if v is not None
    ]
    return {
        "decision_pairs": agreement(rows, pairs, proxy, "v3", V3_DECISION),
        "all_within_tier_pairs": agreement(rows, pairs, proxy, "v3", 0.0),
        "cross_tier_spearman": spearman(xs, ys),
        "linear_map": {"intercept": intercept, "slope": slope},
        "loo": loo,
        "tie_band": {
            "normal_gap": normal_gap,
            "normal_gap_ceil": math.ceil(normal_gap) if normal_gap else None,
            "empirical_grid": grid,
            "empirical_edge": empirical,
            "band": max(candidates) if candidates else None,
            "rule": "max(ceil(1.2816*sqrt(2)*sigma_LOO/b), smallest edge in {2..12} whose within-tier pairs at or above it reverse by >= 2 v3 points <= 10%)",
        },
        "agreement_by_gap": binned(rows, pairs, proxy, "v3", P_BINS, V3_DECISION),
    }


def h_dev_tie_band(rows) -> dict[str, Any]:
    """formal H = tier effect + b * H_dev; gap = 1.2816*sqrt(2)*sigma/b rounded up to 0.005."""
    tiers = sorted({r["tier"] for r in rows})
    dummies = [[float(r["tier"] == t) for r in rows] for t in tiers[1:]]
    x = [r["H_dev"] for r in rows]
    y = [r["H_formal"] for r in rows]
    beta = ols([x, *dummies], y)
    fitted = [
        beta[0] + beta[1] * xi + sum(b * d[k] for b, d in zip(beta[2:], dummies))
        for k, xi in enumerate(x)
    ]
    dof = len(rows) - len(beta)
    sigma = math.sqrt(sum((a - b) ** 2 for a, b in zip(y, fitted)) / dof)
    slope = beta[1]
    gap = Z_90 * math.sqrt(2) * sigma / slope if slope > 0 else None
    return {
        "slope": slope,
        "residual_sd": sigma,
        "dof": dof,
        "gap": gap,
        "band": max(1, math.ceil(round(gap / 0.005, 9))) * 0.005 if gap else None,
    }


def panel_noise(models: list[dict[str, Any]]) -> dict[str, Any]:
    def p_ht_sd(m: dict[str, Any]) -> float:
        t, h = m["development"]["T_dev"], m["htdev"]["H_dev"]
        sd_t, sd_h = m["development"]["noise_sd"]["T_dev"], m["htdev"]["H_dev_sd"]
        return math.hypot(50 * math.sqrt(h / t) * sd_t, 50 * math.sqrt(t / h) * sd_h)

    rows = [
        {
            "key": m["key"],
            "H_dev_sd": m["htdev"]["H_dev_sd"],
            "H_pilot_sd": m["development"]["noise_sd"]["H_pilot"],
            "H_mean3_sd": m["development"]["noise_sd"]["H_mean3"],
            "P_sd": m["development"]["noise_sd"]["P"],
            "P_HT_sd_delta_method": p_ht_sd(m),
        }
        for m in models
    ]
    med = {k: statistics.median(r[k] for r in rows) for k in rows[0] if k != "key"}
    return {"median": med, "per_model": rows}


def analyze(data: dict[str, Any], draws: int = DRAWS) -> dict[str, Any]:
    models = [m for m in data["models"] if m["tier"] in TIERS]
    rows = [flatten(m) for m in models]
    pairs = within_pairs(rows)
    primary = {}
    for target in TARGETS:
        primary[target] = {
            x: {
                name: agreement(rows, pairs, x, target, thr)
                for name, thr in THRESHOLDS.items()
            }
            for x in ("H_dev", *COMPARATORS, "H_dev_mean")
        }
    secondary = {
        target: {
            x: correlations(rows, x, target)
            for x in ("H_dev", *COMPARATORS, "H_dev_mean")
        }
        for target in TARGETS
    }
    boot = paired_bootstrap(rows, draws, SEED)
    dev_rate = primary["H_formal"]["H_dev"][PRIMARY]["rate"]
    pilot_rate = primary["H_formal"]["H_pilot"][PRIMARY]["rate"]
    r_dev = secondary["H_formal"]["H_dev"]["within_tier_pearson"]
    r_pilot = secondary["H_formal"]["H_pilot"]["within_tier_pearson"]
    p_better = boot["H_pilot"]["primary"]["p_better"]
    cond_i = dev_rate > pilot_rate and p_better >= P_BETTER_MIN
    cond_ii = r_dev > r_pilot
    passes = cond_i and cond_ii
    proxies = {
        name: proxy_block(rows, pairs, name) for name in ("P_HT", "P_HT_mean", "P")
    }
    recommend = None
    if passes:
        recommend = (
            "P_HT"
            if proxies["P_HT"]["decision_pairs"]["rate"]
            >= proxies["P"]["decision_pairs"]["rate"]
            else "H_dev as a separate human-transfer screen; P stays the v3 shortlist proxy"
        )
    return {
        "schema": ANALYSIS_SCHEMA,
        "created_utc": utc_now(),
        "label": "development readout (HT-DEV v1 validation); never a release score",
        "features_sha256": data.get("_sha256"),
        "models": len(rows),
        "tiers": {t: sum(r["tier"] == t for r in rows) for t in TIERS},
        "within_tier_pairs": len(pairs),
        "section5": {
            "primary_pair_agreement": primary,
            "correlations": secondary,
            "paired_bootstrap_vs_H_formal": boot,
            "agreement_by_H_dev_gap": binned(
                rows, pairs, "H_dev", "H_formal", H_DEV_BINS, None
            ),
            "decision": {
                "H_dev_primary": dev_rate,
                "H_pilot_primary": pilot_rate,
                "p_better_primary": p_better,
                "H_dev_within_tier_r": r_dev,
                "H_pilot_within_tier_r": r_pilot,
                "condition_i": cond_i,
                "condition_ii": cond_ii,
                "tracks_better": passes,
                "rule": "tracks better iff (i) primary agreement higher with P(better) >= 0.90 and (ii) within-tier Pearson r higher (point estimate)",
            },
            "h_dev_tie_band": h_dev_tie_band(rows),
        },
        "section6": {
            "status": (
                "binding (section 5 passed)"
                if passes
                else "computed for the record only (section 5 did not pass)"
            ),
            "proxies": proxies,
            "recommendation": recommend,
        },
        "panel_noise": panel_noise(models),
        "table": [
            {
                k: r[k]
                for k in (
                    "key",
                    "tier",
                    "group",
                    "H_formal",
                    "H_dev",
                    "H_dev_mean",
                    "H_pilot",
                    "H_mean3",
                    "T_dev",
                    "v3",
                    "P",
                    "P_HT",
                )
            }
            for r in rows
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("extract")
    one.add_argument("--spec", type=Path, required=True)
    one.add_argument("--panel-root", type=Path, default=panels.DEFAULT_ROOT)
    one.add_argument("--draws", type=int, default=1000)
    one.add_argument("--output", type=Path, required=True)
    two = commands.add_parser("analyze")
    two.add_argument("--features", type=Path, required=True)
    two.add_argument("--draws", type=int, default=DRAWS)
    two.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "extract":
        write_json(args.output, extract(args.spec, args.panel_root, args.draws))
        return 0
    data = json.loads(args.features.read_text(encoding="utf-8"))
    data["_sha256"] = sha_file(args.features)
    result = analyze(data, args.draws)
    write_json(args.output, result)
    decision = result["section5"]["decision"]
    print(
        json.dumps(
            {
                k: decision[k]
                for k in (
                    "H_dev_primary",
                    "H_pilot_primary",
                    "p_better_primary",
                    "H_dev_within_tier_r",
                    "H_pilot_within_tier_r",
                    "tracks_better",
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
