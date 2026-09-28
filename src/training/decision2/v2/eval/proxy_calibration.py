"""Calibrate development proxies against post-key same-panel v3 across models.

Inputs are development predictions (typed DEV, CSS pilot) and each model's published
v3 aggregate from its REPORT.json; no v3 item, label or per-item output is read.

Subcommands:

* ``extract`` (node A, reads the frozen development gold): per-model development
  features and their bootstrap panel noise. The output holds aggregates only.
* ``analyze`` (anywhere, reads an ``extract`` output): compares the preregistered
  candidate proxies by rank correlation, leave-one-out linear v3 error and pairwise
  order agreement over all, same-tier, decision (same tier, at least one candidate) and
  finalist (same track, both candidates) pairs stratified by |dv3|, with paired
  stratified cluster-bootstrap intervals, tie bands and the preregistered selection rule
  (``v2/eval/records/m5-proxy-v2-prereg-2026-09-29.md``).
* ``v1``: the Milestone 2 calibration (features and analysis in one run).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from v2.eval import panels
from v2.eval.same_panel import read_jsonl, sha_file, utc_now, write_json

SCHEMA = "dev2-proxy-calibration/1"
FEATURES_SCHEMA = "dev2-proxy-features/1"
ANALYSIS_SCHEMA = "dev2-proxy-calibration/2"
BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 20260928
CLUSTER_DRAWS = 2000
CLUSTER_SEED = 20260929

BASELINE = "P"
CANDIDATES = (
    "P",
    "P_mean3",
    "P_type",
    "P_type_mean3",
    "P_CS_mean3",
    "A_med",
    "A_mean3",
    "L2",
)
REFERENCES = (
    "T_dev",
    "H_pilot",
    "H_mean3",
    "dev_choice",
    "dev_noul",
    "dev_score",
    "pilot_micro",
)
FITTED = {"L2": ("T_dev", "H_mean3")}
STRATA = (("tie_lt2", 0.0, 2.0), ("close_2to5", 2.0, 5.0), ("clear_ge5", 5.0, math.inf))
RESOLVABLE = 2.0
SELECTION = {
    "min_prob_better": 0.90,
    "max_rmse_increase": 0.25,
    "max_all_pairs_drop": 0.01,
}
TIE_GRID = tuple(range(1, 16))
TIE_RULE = {
    "min_agree": 0.90,
    "max_reversal": 0.05,
    "reversal_v3": 2.0,
    "min_pairs": 20,
}
Z_90 = 1.2815515655446004


def typed_outcomes(gold: dict[str, dict[str, Any]], preds: dict[str, dict[str, Any]]):
    """Per item: (family, group, {type: (correct, n)})."""
    from benchmark.score import evaluate_answer

    rows = []
    for item_id, item in gold.items():
        answers = (preds.get(item_id) or {}).get("answers")
        per_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for key, question in item["questions"].items():
            ok = isinstance(answers, dict) and set(answers) == set(item["questions"])
            correct = ok and bool(
                evaluate_answer(question, item["gold"][key], answers[key]).get(
                    "correct"
                )
            )
            per_type[question["type"]][0] += int(correct)
            per_type[question["type"]][1] += 1
        rows.append((item["family"], item["group_id"], dict(per_type)))
    return rows


def pilot_outcomes(gold: dict[str, dict[str, Any]], preds: dict[str, dict[str, Any]]):
    """Per task: list of (gold label, predicted label or None), label universe."""
    from transfer.score import evaluate

    tasks: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
    labels: dict[str, list[str]] = {}
    for item_id, row in gold.items():
        if row["role"] != "pilot":
            continue
        result = evaluate(row, preds.get(item_id))
        tasks[row["task"]].append(
            (row["gold"], result["choice"] if result["valid"] else None)
        )
        labels[row["task"]] = row["labels"]
    return dict(tasks), labels


def macro_f1(pairs: list[tuple[str, str | None]], labels: list[str]) -> float:
    scores = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in pairs)
        fp = sum(g != label and p == label for g, p in pairs)
        fn = sum(g == label and p != label for g, p in pairs)
        scores.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return statistics.mean(scores)


def features(typed_rows, pilot_tasks, pilot_labels) -> dict[str, float]:
    by_family: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for family, _group, per_type in typed_rows:
        for kind, (c, n) in per_type.items():
            by_family[family][0] += c
            by_family[family][1] += n
            by_type[kind][0] += c
            by_type[kind][1] += n
    t = statistics.mean(c / n for c, n in by_family.values())
    types = {k: c / n for k, (c, n) in by_type.items()}
    task_f1 = {
        task: macro_f1(pairs, pilot_labels[task]) for task, pairs in pilot_tasks.items()
    }
    h = statistics.median(task_f1.values())
    h3 = statistics.mean(task_f1.values())
    micro = sum(g == p for pairs in pilot_tasks.values() for g, p in pairs) / sum(
        len(pairs) for pairs in pilot_tasks.values()
    )
    choice, noul, score = (types.get(k, 0.0) for k in ("choice", "noul", "score"))
    type_mean = (choice + noul + score) / 3
    return {
        "P": 100 * math.sqrt(t * h),
        "T_dev": t,
        "H_pilot": h,
        "dev_choice": choice,
        "dev_noul": noul,
        "dev_score": score,
        "mean_T_H": (t + h) / 2,
        "P_type": 100 * math.sqrt(type_mean * h),
        "pilot_micro": micro,
        "H_mean3": h3,
        "P_mean3": 100 * math.sqrt(t * h3),
        "P_type_mean3": 100 * math.sqrt(type_mean * h3),
        "P_CS_mean3": 100 * math.sqrt((choice + score) / 2 * h3),
        "A_med": 100 * (t + h) / 2,
        "A_mean3": 100 * (t + h3) / 2,
        **{f"pilot_{task}": value for task, value in sorted(task_f1.items())},
    }


def bootstrap(
    typed_rows, pilot_tasks, pilot_labels, draws: int, seed: int
) -> dict[str, float]:
    rng = random.Random(seed)
    groups: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for row in typed_rows:
        groups[row[0]][row[1]].append(row)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(draws):
        rows = []
        for family, members in groups.items():
            keys = list(members)
            for _k in keys:
                rows.extend(members[keys[rng.randrange(len(keys))]])
        tasks = {
            task: [pairs[rng.randrange(len(pairs))] for _ in pairs]
            for task, pairs in pilot_tasks.items()
        }
        for name, value in features(rows, tasks, pilot_labels).items():
            samples[name].append(value)
    return {name: statistics.pstdev(values) for name, values in samples.items()}


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            result[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return result


def pearson(x: list[float], y: list[float]) -> float:
    mx, my = statistics.mean(x), statistics.mean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sxy / (sx * sy) if sx and sy else float("nan")


def kendall_tau_b(x: list[float], y: list[float]) -> float:
    concordant = discordant = tx = ty = 0
    for i, j in itertools.combinations(range(len(x)), 2):
        dx, dy = x[i] - x[j], y[i] - y[j]
        if dx == 0 and dy == 0:
            continue
        if dx == 0:
            tx += 1
        elif dy == 0:
            ty += 1
        elif dx * dy > 0:
            concordant += 1
        else:
            discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + tx) * (concordant + discordant + ty)
    )
    return (concordant - discordant) / denominator if denominator else float("nan")


def loo_linear(x: list[float], y: list[float]) -> dict[str, float]:
    errors = []
    for i in range(len(x)):
        xs = [v for k, v in enumerate(x) if k != i]
        ys = [v for k, v in enumerate(y) if k != i]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        var = sum((a - mx) ** 2 for a in xs)
        slope = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / var if var else 0.0
        errors.append(abs(my + slope * (x[i] - mx) - y[i]))
    return {
        "mae": statistics.mean(errors),
        "max": max(errors),
        "rmse": math.sqrt(statistics.mean(e * e for e in errors)),
    }


def pairwise(names, tiers, x, y, noise) -> dict[str, Any]:
    all_pairs, tier_pairs, by_gap = [], [], defaultdict(list)
    for i, j in itertools.combinations(range(len(x)), 2):
        if y[i] == y[j]:
            continue
        agree = (x[i] - x[j]) * (y[i] - y[j]) > 0
        all_pairs.append(agree)
        if tiers[i] == tiers[j]:
            tier_pairs.append((names[i], names[j], agree, x[i] - x[j], y[i] - y[j]))
        gap = (
            abs(x[i] - x[j]) / math.sqrt(noise[i] ** 2 + noise[j] ** 2)
            if (noise[i] or noise[j])
            else math.inf
        )
        by_gap[
            "<1 noise sd" if gap < 1 else "1-2 noise sd" if gap < 2 else ">=2 noise sd"
        ].append(agree)
    return {
        "all": {"n": len(all_pairs), "agree": sum(all_pairs)},
        "same_tier": {
            "n": len(tier_pairs),
            "agree": sum(p[2] for p in tier_pairs),
            "pairs": [
                {"a": a, "b": b, "agree": g, "proxy_delta": dx, "v3_delta": dy}
                for a, b, g, dx, dy in tier_pairs
            ],
        },
        "by_gap": {
            k: {"n": len(v), "agree": sum(v)} for k, v in sorted(by_gap.items())
        },
    }


def run(spec_path: Path, panel_root: Path, draws: int) -> dict[str, Any]:
    from benchmark.score import load_jsonl
    from transfer.score import read_jsonl as read_css

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    panels.verify(panel_root, ["typed-dev", "css-pilot"])
    typed_gold = load_jsonl(panels.path(panel_root, "typed-dev", "gold"))
    pilot_gold = read_css(panels.path(panel_root, "css-pilot", "gold"))
    models = []
    for entry in spec["models"]:
        report = json.loads(Path(entry["report"]).read_text(encoding="utf-8"))
        typed_rows = typed_outcomes(
            typed_gold, {r["id"]: r for r in read_jsonl(Path(entry["typed_dev"]))}
        )
        tasks, labels = pilot_outcomes(
            pilot_gold, {r["id"]: r for r in read_jsonl(Path(entry["css_pilot"]))}
        )
        models.append(
            {
                "key": entry["key"],
                "tier": report["model"]["tier"],
                "v3": report["v3"]["score"],
                "features": features(typed_rows, tasks, labels),
                "noise_sd": bootstrap(typed_rows, tasks, labels, draws, BOOTSTRAP_SEED),
            }
        )
    names = [m["key"] for m in models]
    tiers = [m["tier"] for m in models]
    y = [m["v3"] for m in models]
    proxies = {}
    for proxy in models[0]["features"]:
        x = [m["features"][proxy] for m in models]
        noise = [m["noise_sd"][proxy] for m in models]
        proxies[proxy] = {
            "spearman": pearson(ranks(x), ranks(y)),
            "kendall_tau_b": kendall_tau_b(x, y),
            "pearson": pearson(x, y),
            "loo_linear_v3_error": loo_linear(x, y),
            "pairwise": pairwise(names, tiers, x, y, noise),
            "median_noise_sd": statistics.median(noise),
        }
    return {
        "schema": SCHEMA,
        "created_utc": utc_now(),
        "scope": "development proxies vs post-key same-panel v3 aggregates; no v3 items used",
        "bootstrap": {
            "draws": draws,
            "seed": BOOTSTRAP_SEED,
            "unit": "typed DEV groups within family; CSS pilot items within task",
        },
        "models": models,
        "proxies": proxies,
    }


# ---------------------------------------------------------------- extract (node A)

_GOLD: dict[str, Any] = {}


def _load_gold(panel_root: Path) -> None:
    from benchmark.score import load_jsonl
    from transfer.score import read_jsonl as read_css

    if not _GOLD:
        _GOLD["typed"] = load_jsonl(panels.path(panel_root, "typed-dev", "gold"))
        _GOLD["pilot"] = read_css(panels.path(panel_root, "css-pilot", "gold"))


def extract_model(
    entry: dict[str, Any], panel_root: Path, draws: int
) -> dict[str, Any]:
    """Aggregate development features of one model (no items, labels or answers)."""
    _load_gold(panel_root)
    report_path = Path(entry["report"])
    typed_path, pilot_path = Path(entry["typed_dev"]), Path(entry["css_pilot"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    typed_preds = {r["id"]: r for r in read_jsonl(typed_path)}
    pilot_preds = {r["id"]: r for r in read_jsonl(pilot_path)}
    typed_rows = typed_outcomes(_GOLD["typed"], typed_preds)
    tasks, labels = pilot_outcomes(_GOLD["pilot"], pilot_preds)
    pilot_ids = {k for k, v in _GOLD["pilot"].items() if v["role"] == "pilot"}
    return {
        **{
            k: v
            for k, v in entry.items()
            if k not in ("report", "typed_dev", "css_pilot")
        },
        "tier": report["model"]["tier"],
        "label": report["model"].get("label"),
        "v3": report["v3"]["score"],
        "inputs": {
            "report": str(report_path),
            "report_sha256": sha_file(report_path),
            "typed_dev": str(typed_path),
            "typed_dev_sha256": sha_file(typed_path),
            "css_pilot": str(pilot_path),
            "css_pilot_sha256": sha_file(pilot_path),
            "typed_missing": len(set(_GOLD["typed"]) - set(typed_preds)),
            "pilot_missing": len(pilot_ids - set(pilot_preds)),
        },
        "features": features(typed_rows, tasks, labels),
        "noise_sd": bootstrap(typed_rows, tasks, labels, draws, BOOTSTRAP_SEED),
    }


def extract(
    spec_path: Path, panel_root: Path, draws: int, workers: int
) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    panel_hashes = panels.verify(panel_root, ["typed-dev", "css-pilot"])
    sensitivity = spec.get("sensitivity", {})
    entries = [("models", e) for e in spec["models"]]
    entries += [("controls", e) for e in sensitivity.get("S2_controls", [])]
    entries += [("v1_extra", e) for e in sensitivity.get("S4_v1_extra", [])]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(extract_model, entry, panel_root, draws) for _, entry in entries
        ]
        results = [future.result() for future in futures]
    out: dict[str, Any] = {
        "schema": FEATURES_SCHEMA,
        "created_utc": utc_now(),
        "scope": "aggregate development features and panel noise; no v3 items, labels or answers",
        "spec_sha256": sha_file(spec_path),
        "panels": panel_hashes,
        "bootstrap": {
            "draws": draws,
            "seed": BOOTSTRAP_SEED,
            "unit": "typed DEV groups within family; CSS pilot items within task",
        },
        "models": [],
        "controls": [],
        "v1_extra": [],
        "sensitivity": {
            k: v
            for k, v in sensitivity.items()
            if k not in ("S2_controls", "S4_v1_extra")
        },
    }
    for (section, _entry), result in zip(entries, results):
        out[section].append(result)
    return out


# ---------------------------------------------------------------- analyze (anywhere)


def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting for a small dense system."""
    n = len(vector)
    a = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ValueError("singular system")
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(n):
            if r != col:
                factor = a[r][col] / a[col][col]
                a[r] = [v - factor * w for v, w in zip(a[r], a[col])]
    return [a[i][n] / a[i][i] for i in range(n)]


def ols(columns: list[list[float]], y: list[float]) -> list[float]:
    """Least squares with intercept: returns [intercept, coef_1, ...]."""
    design = [[1.0, *values] for values in zip(*columns)]
    p = len(design[0])
    xtx = [[sum(r[i] * r[j] for r in design) for j in range(p)] for i in range(p)]
    xty = [sum(r[i] * v for r, v in zip(design, y)) for i in range(p)]
    return solve(xtx, xty)


def loo_predictions(columns: list[list[float]], y: list[float]) -> list[float]:
    predictions = []
    for i in range(len(y)):
        keep = [k for k in range(len(y)) if k != i]
        beta = ols([[col[k] for k in keep] for col in columns], [y[k] for k in keep])
        predictions.append(
            beta[0] + sum(b * col[i] for b, col in zip(beta[1:], columns))
        )
    return predictions


def linear_1d(x: list[float], y: list[float]) -> tuple[float, float]:
    """Intercept and slope of y on x (slope 0 for a constant x, as in v1)."""
    mx, my = statistics.mean(x), statistics.mean(y)
    var = sum((a - mx) ** 2 for a in x)
    slope = sum((a - mx) * (b - my) for a, b in zip(x, y)) / var if var else 0.0
    return my - slope * mx, slope


def loo_1d(x: list[float], y: list[float]) -> list[float]:
    predictions = []
    for i in range(len(x)):
        intercept, slope = linear_1d(x[:i] + x[i + 1 :], y[:i] + y[i + 1 :])
        predictions.append(intercept + slope * x[i])
    return predictions


def error_summary(predictions: list[float], y: list[float]) -> dict[str, float]:
    errors = [abs(p - v) for p, v in zip(predictions, y)]
    return {
        "mae": statistics.mean(errors),
        "rmse": math.sqrt(statistics.mean(e * e for e in errors)),
        "max": max(errors),
    }


def within_tier_r(x: list[float], y: list[float], tiers: list[str]) -> float:
    by_tier: dict[str, list[int]] = defaultdict(list)
    for i, tier in enumerate(tiers):
        by_tier[tier].append(i)
    dx, dy = [0.0] * len(x), [0.0] * len(y)
    for members in by_tier.values():
        mx = statistics.mean(x[i] for i in members)
        my = statistics.mean(y[i] for i in members)
        for i in members:
            dx[i], dy[i] = x[i] - mx, y[i] - my
    return pearson(dx, dy)


def stratum(dy: float) -> str:
    for name, low, high in STRATA:
        if low <= abs(dy) < high:
            return name
    raise ValueError(dy)


def model_pairs(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every unordered model pair with its v3 difference and pair classes."""
    pairs = []
    for i, j in itertools.combinations(range(len(models)), 2):
        a, b = models[i], models[j]
        dy = a["v3"] - b["v3"]
        if dy == 0:
            continue
        same_tier = a["tier"] == b["tier"]
        candidates = (a["group"] == "candidate") + (b["group"] == "candidate")
        pairs.append(
            {
                "i": i,
                "j": j,
                "dy": dy,
                "stratum": stratum(dy),
                "same_tier": same_tier,
                "decision": same_tier and candidates >= 1,
                "finalist": same_tier and candidates == 2 and a["track"] == b["track"],
            }
        )
    return pairs


PAIR_SETS = {
    "all": lambda p: True,
    "same_tier": lambda p: p["same_tier"],
    "decision": lambda p: p["decision"],
    "decision_resolvable": lambda p: p["decision"] and abs(p["dy"]) >= RESOLVABLE,
    "decision_close": lambda p: p["decision"] and p["stratum"] == "close_2to5",
    "decision_clear": lambda p: p["decision"] and p["stratum"] == "clear_ge5",
    "decision_tie": lambda p: p["decision"] and p["stratum"] == "tie_lt2",
    "finalist": lambda p: p["finalist"],
    "finalist_resolvable": lambda p: p["finalist"] and abs(p["dy"]) >= RESOLVABLE,
}
PRIMARY = "decision_resolvable"


def agreement(pairs: list[dict[str, Any]], x: list[float]) -> dict[str, Any]:
    agree = sum((x[p["i"]] - x[p["j"]]) * p["dy"] > 0 for p in pairs)
    return {
        "n": len(pairs),
        "agree": agree,
        "rate": agree / len(pairs) if pairs else None,
    }


def tie_band(pairs: list[dict[str, Any]], x: list[float]) -> dict[str, Any]:
    """Preregistered empirical tie band and the normal-residual check."""
    grid = []
    for gap in TIE_GRID:
        subset = [p for p in pairs if abs(x[p["i"]] - x[p["j"]]) >= gap]
        agree = sum((x[p["i"]] - x[p["j"]]) * p["dy"] > 0 for p in subset)
        reversal = sum(
            (x[p["i"]] - x[p["j"]]) * p["dy"] < 0
            and abs(p["dy"]) >= TIE_RULE["reversal_v3"]
            for p in subset
        )
        grid.append(
            {
                "gap": gap,
                "n": len(subset),
                "agree_rate": agree / len(subset) if subset else None,
                "reversal_rate": reversal / len(subset) if subset else None,
            }
        )

    def ok(row: dict[str, Any]) -> bool:
        return (
            row["n"] >= TIE_RULE["min_pairs"]
            and row["agree_rate"] >= TIE_RULE["min_agree"]
            and row["reversal_rate"] <= TIE_RULE["max_reversal"]
        )

    band = None
    for k, row in enumerate(grid):
        later = [r for r in grid[k:] if r["n"] >= TIE_RULE["min_pairs"]]
        if row["n"] >= TIE_RULE["min_pairs"] and all(ok(r) for r in later):
            band = row["gap"]
            break
    dxs = [x[p["i"]] - x[p["j"]] for p in pairs]
    dys = [p["dy"] for p in pairs]
    sxx = sum(d * d for d in dxs)
    beta = sum(a * b for a, b in zip(dxs, dys)) / sxx if sxx else 0.0
    sigma = (
        math.sqrt(statistics.mean((b - beta * a) ** 2 for a, b in zip(dxs, dys)))
        if pairs
        else 0.0
    )
    return {
        "rule": TIE_RULE,
        "grid": grid,
        "band": band,
        "model_check": {
            "beta_within": beta,
            "residual_sd": sigma,
            "band_10pct": Z_90 * sigma / beta if beta > 0 else None,
        },
    }


def cluster_bootstrap(
    models: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    vectors: dict[str, list[float]],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Stratified (by tier) model bootstrap of pair agreement, paired across proxies."""
    rng = random.Random(seed)
    by_tier: dict[str, list[int]] = defaultdict(list)
    for index, model in enumerate(models):
        by_tier[model["tier"]].append(index)
    sets = ("decision_resolvable", "decision_close", "finalist_resolvable", "same_tier")
    members = {name: [p for p in pairs if PAIR_SETS[name](p)] for name in sets}
    hits = {
        name: {
            proxy: [(x[p["i"]] - x[p["j"]]) * p["dy"] > 0 for p in members[name]]
            for proxy, x in vectors.items()
        }
        for name in sets
    }
    samples: dict[str, dict[str, list[float]]] = {
        name: {proxy: [] for proxy in vectors} for name in sets
    }
    for _ in range(draws):
        counts: Counter[int] = Counter()
        for indices in by_tier.values():
            counts.update(indices[rng.randrange(len(indices))] for _ in indices)
        for name in sets:
            weights = [counts[p["i"]] * counts[p["j"]] for p in members[name]]
            total = sum(weights)
            for proxy in vectors:
                if total:
                    samples[name][proxy].append(
                        sum(w for w, h in zip(weights, hits[name][proxy]) if h) / total
                    )
    summary: dict[str, Any] = {
        "draws": draws,
        "seed": seed,
        "unit": "models within tier",
    }
    for name in sets:
        base = samples[name][BASELINE]
        block = {}
        for proxy, values in samples[name].items():
            diffs = [v - b for v, b in zip(values, base)]
            block[proxy] = {
                "ci95": quantiles(values),
                "diff_vs_P_ci95": quantiles(diffs),
                "prob_better_than_P": (
                    sum(d > 0 for d in diffs) / len(diffs) if diffs else None
                ),
            }
        summary[name] = block
    return summary


def quantiles(values: list[float]) -> list[float] | None:
    if not values:
        return None
    ordered = sorted(values)

    def pick(q: float) -> float:
        return ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))]

    return [pick(0.025), pick(0.975)]


def proxy_vectors(
    models: list[dict[str, Any]], names: tuple[str, ...]
) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, Any]]:
    """Proxy values (LOO predictions for fitted proxies), noise SDs and fit details."""
    y = [m["v3"] for m in models]
    vectors, noise, fits = {}, {}, {}
    for name in names:
        if name in FITTED:
            columns = [[m["features"][f] for m in models] for f in FITTED[name]]
            beta = ols(columns, y)
            vectors[name] = loo_predictions(columns, y)
            noise[name] = [
                math.sqrt(
                    sum(
                        (b * m["noise_sd"][f]) ** 2
                        for b, f in zip(beta[1:], FITTED[name])
                    )
                )
                for m in models
            ]
            fits[name] = {"intercept": beta[0], **dict(zip(FITTED[name], beta[1:]))}
        else:
            vectors[name] = [m["features"][name] for m in models]
            noise[name] = [m["noise_sd"][name] for m in models]
    return vectors, noise, fits


def evaluate_set(
    models: list[dict[str, Any]],
    names: tuple[str, ...],
    cluster_draws: int = 0,
    keep_pairs: bool = True,
) -> dict[str, Any]:
    y = [m["v3"] for m in models]
    tiers = [m["tier"] for m in models]
    pairs = model_pairs(models)
    vectors, noise, fits = proxy_vectors(models, names)
    proxies = {}
    for name in names:
        x = vectors[name]
        if name in FITTED:
            slope, intercept, loo = 1.0, 0.0, x
        else:
            intercept, slope = linear_1d(x, y)
            loo = loo_1d(x, y)
        tiers_seen = sorted(set(tiers))
        decision = [p for p in pairs if PAIR_SETS[PRIMARY](p)]
        proxies[name] = {
            "fitted_parameters": 3 if name in FITTED else 2,
            "fit": fits.get(name, {"intercept": intercept, "slope": slope}),
            "spearman": pearson(ranks(x), ranks(y)),
            "kendall_tau_b": kendall_tau_b(x, y),
            "loo_v3_error": error_summary(loo, y),
            "within_tier_r": within_tier_r(x, y, tiers),
            "pairs": {
                set_name: agreement([p for p in pairs if test(p)], x)
                for set_name, test in PAIR_SETS.items()
            },
            "primary_by_tier": {
                tier: agreement(
                    [p for p in decision if models[p["i"]]["tier"] == tier], x
                )
                for tier in tiers_seen
            },
            "conflict_pairs": agreement(
                [
                    p
                    for p in decision
                    if (
                        models[p["i"]]["features"]["T_dev"]
                        - models[p["j"]]["features"]["T_dev"]
                    )
                    * (
                        models[p["i"]]["features"]["H_mean3"]
                        - models[p["j"]]["features"]["H_mean3"]
                    )
                    < 0
                ],
                x,
            ),
            "median_noise_sd": statistics.median(noise[name]),
            "median_noise_sd_v3": abs(slope) * statistics.median(noise[name]),
            "tie_band": (
                tie_band([p for p in pairs if p["decision"]], x)
                if name in CANDIDATES
                else None
            ),
        }
    result: dict[str, Any] = {
        "n_models": len(models),
        "n_candidates": sum(m["group"] == "candidate" for m in models),
        "tiers": dict(sorted(Counter(tiers).items())),
        "pair_counts": {
            k: sum(test(p) for p in pairs) for k, test in PAIR_SETS.items()
        },
        "proxies": proxies,
    }
    if cluster_draws:
        result["cluster_bootstrap"] = cluster_bootstrap(
            models,
            pairs,
            {n: vectors[n] for n in names if n in CANDIDATES},
            cluster_draws,
            CLUSTER_SEED,
        )
    if keep_pairs:
        result["decision_pairs"] = [
            {
                "a": models[p["i"]]["key"],
                "b": models[p["j"]]["key"],
                "tier": models[p["i"]]["tier"],
                "finalist": p["finalist"],
                "v3_delta": p["dy"],
                "proxy_delta": {
                    n: vectors[n][p["i"]] - vectors[n][p["j"]] for n in CANDIDATES
                },
            }
            for p in pairs
            if p["decision"]
        ]
    return result


def select(result: dict[str, Any]) -> dict[str, Any]:
    """Preregistered selection rule against the baseline P."""
    proxies, boot = result["proxies"], result["cluster_bootstrap"]
    base = proxies[BASELINE]
    checks = {}
    for name in CANDIDATES:
        if name == BASELINE:
            continue
        value = proxies[name]
        checks[name] = {
            "prob_better_primary": boot[PRIMARY][name]["prob_better_than_P"],
            "primary_rate": value["pairs"][PRIMARY]["rate"],
            "rmse_increase": value["loo_v3_error"]["rmse"]
            - base["loo_v3_error"]["rmse"],
            "all_pairs_drop": base["pairs"]["all"]["rate"]
            - value["pairs"]["all"]["rate"],
            "noise_v3_increase": value["median_noise_sd_v3"]
            - base["median_noise_sd_v3"],
        }
        c = checks[name]
        c["qualifies"] = (
            c["prob_better_primary"] >= SELECTION["min_prob_better"]
            and c["rmse_increase"] <= SELECTION["max_rmse_increase"]
            and c["all_pairs_drop"] <= SELECTION["max_all_pairs_drop"]
            and c["noise_v3_increase"] <= 0
        )
    qualified = sorted(
        (n for n, c in checks.items() if c["qualifies"]),
        key=lambda n: (proxies[n]["fitted_parameters"], -checks[n]["primary_rate"]),
    )
    return {
        "rule": SELECTION,
        "checks": checks,
        "recommended": qualified[0] if qualified else BASELINE,
        "qualified": qualified,
    }


def build_sets(data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    models = data["models"]
    sens = data.get("sensitivity", {})
    by_key = {m["key"]: m for m in models}
    sets = {"main": models}
    if sens.get("S1_exclude"):
        sets["S1"] = [m for m in models if m["key"] not in set(sens["S1_exclude"])]
    if data.get("controls"):
        replace = {c["replaces"]: c for c in data["controls"]}
        sets["S2"] = [replace.get(m["key"], m) for m in models]
    if sens.get("S3_exclude"):
        sets["S3"] = [m for m in models if m["key"] not in set(sens["S3_exclude"])]
    if sens.get("S4_v1_keys"):
        sets["S4"] = [by_key[k] for k in sens["S4_v1_keys"]] + list(
            data.get("v1_extra", [])
        )
    return sets


def analyze(data: dict[str, Any], cluster_draws: int) -> dict[str, Any]:
    names = CANDIDATES + REFERENCES
    sets = build_sets(data)
    results = {
        name: evaluate_set(
            models,
            names,
            cluster_draws if name == "main" else 0,
            keep_pairs=name == "main",
        )
        for name, models in sets.items()
    }
    main = results["main"]
    selection = select(main)
    recommended = selection["recommended"]
    return {
        "schema": ANALYSIS_SCHEMA,
        "created_utc": utc_now(),
        "scope": "development proxies vs post-key same-panel v3 composites; no v3 items used",
        "features_created_utc": data.get("created_utc"),
        "primary_metric": f"order agreement on {PRIMARY} pairs (same tier, >= 1 candidate, |dv3| >= {RESOLVABLE})",
        "selection": selection,
        "recommended": {
            "proxy": recommended,
            "fit": main["proxies"][recommended]["fit"],
            "tie_band": main["proxies"][recommended]["tie_band"],
        },
        "sets": results,
    }


def print_summary(result: dict[str, Any]) -> None:
    for set_name, block in result["sets"].items():
        print(
            f"== {set_name}: {block['n_models']} models ({block['n_candidates']} candidates), pairs {block['pair_counts']}"
        )
        for name, value in block["proxies"].items():
            pw, err = value["pairs"], value["loo_v3_error"]

            def fmt(key: str) -> str:
                return f"{pw[key]['agree']}/{pw[key]['n']}"

            band = value["tie_band"] or {"band": None, "model_check": {}}
            print(
                f"{name:13s} rho={value['spearman']:.3f} rw={value['within_tier_r']:.3f} "
                f"loo={err['mae']:.2f}/{err['rmse']:.2f}/{err['max']:.2f} all={fmt('all')} "
                f"tier={fmt('same_tier')} prim={fmt('decision_resolvable')} close={fmt('decision_close')} "
                f"fin={fmt('finalist_resolvable')} noise_v3={value['median_noise_sd_v3']:.2f} "
                f"band={band['band']} bmodel={band['model_check'].get('band_10pct')}"
            )
    print(json.dumps(result["selection"], indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    v1 = sub.add_parser("v1", help="Milestone 2 calibration (features and analysis)")
    v1.add_argument(
        "--spec",
        type=Path,
        required=True,
        help="JSON {models: [{key, report, typed_dev, css_pilot}]}",
    )
    v1.add_argument("--panel-root", type=Path, default=panels.DEFAULT_ROOT)
    v1.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    v1.add_argument("--output", type=Path, required=True)
    ext = sub.add_parser("extract", help="aggregate development features (node A)")
    ext.add_argument("--spec", type=Path, required=True)
    ext.add_argument("--panel-root", type=Path, default=panels.DEFAULT_ROOT)
    ext.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    ext.add_argument("--workers", type=int, default=8)
    ext.add_argument("--output", type=Path, required=True)
    ana = sub.add_parser(
        "analyze", help="compare candidate proxies on an extract output"
    )
    ana.add_argument("--features", type=Path, required=True)
    ana.add_argument("--cluster-draws", type=int, default=CLUSTER_DRAWS)
    ana.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "v1":
        result = run(args.spec, args.panel_root, args.draws)
        write_json(args.output, result, exclusive=False)
        for name, value in sorted(
            result["proxies"].items(), key=lambda kv: -kv[1]["kendall_tau_b"]
        ):
            pw = value["pairwise"]
            print(
                f"{name:12s} spearman={value['spearman']:.3f} tau={value['kendall_tau_b']:.3f} "
                f"loo_mae={value['loo_linear_v3_error']['mae']:.2f} max={value['loo_linear_v3_error']['max']:.2f} "
                f"pairs={pw['all']['agree']}/{pw['all']['n']} tier={pw['same_tier']['agree']}/{pw['same_tier']['n']} "
                f"noise={value['median_noise_sd']:.4f}"
            )
    elif args.command == "extract":
        result = extract(args.spec, args.panel_root, args.draws, args.workers)
        write_json(args.output, result, exclusive=False)
        for m in result["models"] + result["controls"] + result["v1_extra"]:
            f = m["features"]
            print(
                f"{m['key']:20s} {m['tier']:5s} v3={m['v3']:.3f} P={f['P']:.2f} "
                f"P_mean3={f['P_mean3']:.2f} T={f['T_dev']:.4f} H={f['H_pilot']:.4f} "
                f"H3={f['H_mean3']:.4f} missing={m['inputs']['typed_missing']}/{m['inputs']['pilot_missing']}"
            )
    else:
        data = json.loads(args.features.read_text(encoding="utf-8"))
        result = analyze(data, args.cluster_draws)
        write_json(args.output, result, exclusive=False)
        print_summary(result)


if __name__ == "__main__":
    main()
