"""HT-DEV v2 design check: how far can any development panel track formal within-tier ΔH?

    python3 -m v2.eval.htdev2.ceiling --spec <spec.json> --output <ceiling.json>

Reads, per model, the stored formal CSS15 predictions (hash-checked against the run's
SEAL.json), the CSS15 gold and the model's stored CSS-pilot predictions. No inference runs
and no development-panel item is read. Output holds per-model and per-pair aggregates only.

Split simulation: each replicate splits every CSS15 task's items, stratified by gold label,
into a "development" part (fraction f) and a disjoint "target" part. A parallel form drawn
from the formal items themselves is the most favourable development panel possible, so its
agreement with the target bounds what a held-out CSS15 panel of that size can reach. The
pilot three-task mean is scored against the same targets.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from transfer.build import EVALUATION_TASKS, PILOT_TASKS
from transfer.score import evaluate, read_jsonl
from v2.eval import panels
from v2.eval.same_panel import sha_file, utc_now, write_json

SCHEMA = "dev2-htdev2-ceiling/1"
THRESHOLD = 0.02
SEED = 20260930
FRACTIONS = (0.35, 0.5)


def coded(
    gold_rows: list[dict[str, Any]], predictions: dict[str, dict[str, Any]], tasks
):
    """task -> (gold index array, prediction index array with -1 for invalid, n labels)."""
    out = {}
    for task in tasks:
        rows = [row for row in gold_rows if row["task"] == task]
        labels = rows[0]["labels"]
        index = {label: i for i, label in enumerate(labels)}
        gold = np.array([index[row["gold"]] for row in rows], dtype=np.int64)
        pred = np.empty(len(rows), dtype=np.int64)
        for k, row in enumerate(rows):
            result = evaluate(row, predictions.get(row["id"]))
            pred[k] = index[result["choice"]] if result["valid"] else -1
        out[task] = (gold, pred, len(labels))
    return out


def macro_f1(gold: np.ndarray, pred: np.ndarray, n_labels: int) -> float:
    gold_count = np.bincount(gold, minlength=n_labels)
    valid = pred >= 0
    pred_count = np.bincount(pred[valid], minlength=n_labels)
    hit = valid & (pred == gold)
    tp = np.bincount(gold[hit], minlength=n_labels)
    denominator = gold_count + pred_count
    f1 = np.divide(2 * tp, denominator, out=np.zeros(n_labels), where=denominator > 0)
    return float(f1.mean())


def verified_predictions(run: Path, panel: str) -> Path:
    path = run / "output" / f"{panel}.predictions.jsonl"
    seal_path = run / "SEAL.json"
    if seal_path.is_file():
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        expected = seal["panels"][panel]["predictions_sha256"]
        if sha_file(path) != expected:
            raise ValueError(f"{run}: {panel} predictions differ from SEAL.json")
    return path


def stratified_masks(gold: np.ndarray, fraction: float, rng: np.random.Generator):
    dev = np.zeros(len(gold), dtype=bool)
    for label in np.unique(gold):
        idx = np.flatnonzero(gold == label)
        rng.shuffle(idx)
        dev[idx[: int(round(fraction * len(idx)))]] = True
    return dev, ~dev


def pair_matrices(values: np.ndarray, target: np.ndarray, same_tier: np.ndarray):
    """Per-pair score (1 agree, 0 disagree, 0.5 tie in x) and decidable mask."""
    dx = values[:, None] - values[None, :]
    dy = target[:, None] - target[None, :]
    score = np.where(dx == 0, 0.5, (dx * dy > 0).astype(float))
    decidable = same_tier & (dy != 0) & (np.abs(dy) >= THRESHOLD)
    return score, decidable


def agreement(score, decidable) -> float:
    upper = np.triu(decidable, 1)
    return float(score[upper].mean()) if upper.any() else float("nan")


def within_tier_r(values: np.ndarray, target: np.ndarray, tiers: list[str]) -> float:
    x, y = values.copy(), target.copy()
    for tier in set(tiers):
        mask = np.array([t == tier for t in tiers])
        x[mask] -= x[mask].mean()
        y[mask] -= y[mask].mean()
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap_p_better(score_a, score_b, decidable, tiers, draws, rng) -> float:
    """Paired model bootstrap within tiers; duplicate draws of one model form no pair."""
    n = len(tiers)
    by_tier = {t: np.array([i for i in range(n) if tiers[i] == t]) for t in set(tiers)}
    upper = np.triu(decidable, 1)
    wins = 0.0
    for _ in range(draws):
        counts = np.zeros(n)
        for members in by_tier.values():
            np.add.at(counts, rng.choice(members, size=len(members)), 1)
        weight = np.outer(counts, counts) * upper
        total = weight.sum()
        if total == 0:
            continue
        delta = (weight * score_a).sum() / total - (weight * score_b).sum() / total
        wins += 1.0 if delta > 0 else 0.5 if delta == 0 else 0.0
    return wins / draws


def run(
    spec_path: Path, panel_root: Path, replicates: int, draws: int
) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    hashes = panels.verify(panel_root, ["css15", "css-pilot"])
    css_gold = list(read_jsonl(panels.path(panel_root, "css15", "gold")).values())
    pilot_gold = list(read_jsonl(panels.path(panel_root, "css-pilot", "gold")).values())
    models, formal, pilot_mean3 = [], [], []
    for entry in spec["models"]:
        run_dir = Path(entry["formal_run"])
        preds = read_jsonl(verified_predictions(run_dir, "css15"))
        tasks = coded(css_gold, preds, EVALUATION_TASKS)
        report = json.loads((run_dir / "REPORT.json").read_text(encoding="utf-8"))
        h = statistics.median(macro_f1(*tasks[t]) for t in EVALUATION_TASKS)
        if abs(h - report["panels"]["css15"]["H"]) > 1e-9:
            raise ValueError(f"{entry['key']}: recomputed H differs from REPORT.json")
        mean3 = None
        if (
            entry.get("pilot_predictions")
            and Path(entry["pilot_predictions"]).is_file()
        ):
            pilot = coded(
                pilot_gold, read_jsonl(Path(entry["pilot_predictions"])), PILOT_TASKS
            )
            mean3 = statistics.fmean(macro_f1(*pilot[t]) for t in PILOT_TASKS)
        models.append(entry)
        formal.append(tasks)
        pilot_mean3.append(mean3)
    tiers = [m["tier"] for m in models]
    same_tier = np.array([[a == b for b in tiers] for a in tiers]) & ~np.eye(
        len(tiers), dtype=bool
    )
    h_full = np.array(
        [statistics.median(macro_f1(*f[t]) for t in EVALUATION_TASKS) for f in formal]
    )
    with_pilot = np.array([m is not None for m in pilot_mean3])
    mean3 = np.array([m if m is not None else np.nan for m in pilot_mean3])

    def restrict(mask2d):
        return mask2d & with_pilot[:, None] & with_pilot[None, :]

    s_pilot, d_full = pair_matrices(np.nan_to_num(mean3), h_full, same_tier)
    observed = {
        "pilot_mean3_vs_full_H": agreement(s_pilot, restrict(d_full)),
        "decidable_pairs_full": int(np.triu(d_full, 1).sum()),
        "decidable_pairs_with_pilot": int(np.triu(restrict(d_full), 1).sum()),
        "pilot_mean3_within_tier_r": within_tier_r(
            mean3[with_pilot],
            h_full[with_pilot],
            [t for t, w in zip(tiers, with_pilot) if w],
        ),
    }
    rng = np.random.default_rng(SEED)
    sims: dict[str, Any] = {}
    for fraction in FRACTIONS:
        rows = []
        boot = []
        for rep in range(replicates):
            dev_med, dev_mean, target = [], [], []
            masks = {
                t: stratified_masks(formal[0][t][0], fraction, rng)
                for t in EVALUATION_TASKS
            }
            for f in formal:
                dev_f1 = [
                    macro_f1(f[t][0][masks[t][0]], f[t][1][masks[t][0]], f[t][2])
                    for t in EVALUATION_TASKS
                ]
                tgt_f1 = [
                    macro_f1(f[t][0][masks[t][1]], f[t][1][masks[t][1]], f[t][2])
                    for t in EVALUATION_TASKS
                ]
                dev_med.append(statistics.median(dev_f1))
                dev_mean.append(statistics.fmean(dev_f1))
                target.append(statistics.median(tgt_f1))
            target = np.array(target)
            row = {}
            scores = {}
            for name, values in (
                ("dev_median", np.array(dev_med)),
                ("dev_mean", np.array(dev_mean)),
            ):
                score, decidable = pair_matrices(values, target, same_tier)
                scores[name] = score
                row[name] = agreement(score, restrict(decidable))
                row[f"{name}_all_models"] = agreement(score, decidable)
                row[f"{name}_r"] = within_tier_r(values, target, tiers)
            score_p, decidable = pair_matrices(np.nan_to_num(mean3), target, same_tier)
            row["pilot_mean3"] = agreement(score_p, restrict(decidable))
            rows.append(row)
            if rep < 20:
                pilot_tiers = [t for t, w in zip(tiers, with_pilot) if w]
                keep = np.flatnonzero(with_pilot)
                sub = np.ix_(keep, keep)
                boot.append(
                    {
                        name: bootstrap_p_better(
                            scores[name][sub],
                            score_p[sub],
                            decidable[sub],
                            pilot_tiers,
                            draws,
                            rng,
                        )
                        for name in ("dev_median", "dev_mean")
                    }
                )
        sims[str(fraction)] = {
            "replicates": replicates,
            "mean": {k: statistics.fmean(r[k] for r in rows) for k in rows[0]},
            "sd": {k: statistics.pstdev(r[k] for r in rows) for k in rows[0]},
            "share_dev_beats_pilot": {
                name: statistics.fmean(r[name] > r["pilot_mean3"] for r in rows)
                for name in ("dev_median", "dev_mean")
            },
            "bootstrap_p_better_vs_pilot_mean3": {
                name: {
                    "median": statistics.median(b[name] for b in boot),
                    "min": min(b[name] for b in boot),
                    "max": max(b[name] for b in boot),
                    "replicates": len(boot),
                    "draws": draws,
                }
                for name in ("dev_median", "dev_mean")
            },
        }
    sizes = {t: int(len(formal[0][t][0])) for t in EVALUATION_TASKS}
    return {
        "schema": SCHEMA,
        "created_utc": utc_now(),
        "label": "design check for HT-DEV v2 (stored formal predictions only; no inference, no development item)",
        "spec_sha256": sha_file(spec_path),
        "panels": hashes,
        "threshold": THRESHOLD,
        "seed": SEED,
        "models": len(models),
        "models_with_pilot": int(with_pilot.sum()),
        "tiers": {t: tiers.count(t) for t in sorted(set(tiers))},
        "css15_task_sizes": sizes,
        "observed": observed,
        "split_simulation": sims,
        "per_model": [
            {"key": m["key"], "tier": m["tier"], "H_formal": float(h), "pilot_mean3": p}
            for m, h, p in zip(models, h_full, pilot_mean3)
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path, default=panels.DEFAULT_ROOT)
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.spec, args.panel_root, args.replicates, args.draws)
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "observed": result["observed"],
                "split_simulation": {
                    f: {
                        "mean": s["mean"],
                        "p_better": s["bootstrap_p_better_vs_pilot_mean3"],
                    }
                    for f, s in result["split_simulation"].items()
                },
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
