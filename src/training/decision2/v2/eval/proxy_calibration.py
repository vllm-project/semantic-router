"""Calibrate development proxies against post-key same-panel v3 across models.

Inputs are development predictions (typed DEV, CSS pilot) and each model's published
v3 aggregate from its REPORT.json; no v3 item, label or per-item output is read.
For every candidate proxy the report gives Spearman and Kendall correlation with v3,
leave-one-out linear prediction error, pairwise order agreement (all pairs and
same-tier pairs) and agreement by proxy gap, plus per-model bootstrap noise of the
proxies from resampling the development panels.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from v2.eval import panels
from v2.eval.same_panel import read_jsonl, utc_now, write_json

SCHEMA = "dev2-proxy-calibration/1"
BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 20260928


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
    h = statistics.median(
        macro_f1(pairs, pilot_labels[task]) for task, pairs in pilot_tasks.items()
    )
    micro = sum(g == p for pairs in pilot_tasks.values() for g, p in pairs) / sum(
        len(pairs) for pairs in pilot_tasks.values()
    )
    type_mean = statistics.mean(types.get(k, 0.0) for k in ("choice", "noul", "score"))
    return {
        "P": 100 * math.sqrt(t * h),
        "T_dev": t,
        "H_pilot": h,
        "dev_choice": types.get("choice", 0.0),
        "dev_noul": types.get("noul", 0.0),
        "dev_score": types.get("score", 0.0),
        "mean_T_H": (t + h) / 2,
        "P_type": 100 * math.sqrt(type_mean * h),
        "pilot_micro": micro,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        required=True,
        help="JSON {models: [{key, report, typed_dev, css_pilot}]}",
    )
    parser.add_argument("--panel-root", type=Path, default=panels.DEFAULT_ROOT)
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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


if __name__ == "__main__":
    main()
