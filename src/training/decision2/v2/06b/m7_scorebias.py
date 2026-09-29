"""M7 (a) per-level Score bias: fit, development checks, replay and the S3 gate (stdlib only).

    python3 -m v2.06b.m7_scorebias fit --aho-dir AHO --probs PROBS.json --soup-dir SOUP
        [--readout-manifest DEV.predictions.jsonl.manifest.json] --output score_bias.json --report FIT.json
    python3 -m v2.06b.m7_scorebias check --score-bias score_bias.json --aho-dir AHO --probs PROBS.json
        --soup-dir SOUP --dev-predictions DEV.predictions.jsonl --output CHECK.json
    python3 -m v2.06b.m7_scorebias devcheck --aho-dir AHO --probs PROBS.json --label NAME --output BD1.json
    python3 -m v2.06b.m7_scorebias replay --logged DEV.predictions.jsonl --online DEV.predictions.jsonl
        --score-bias score_bias.json --output AD3.json
    python3 -m v2.06b.m7_scorebias gate --run RUN --gates RUN.gates --label NAME --output RUN.gates/score-s3.json

M7 prereg sections 0 (S3), 1.4 (fit), 1.5 (A-D1..A-D3) and 2.4 (B-D1). AHO is an
`m7_aho build` directory, PROBS.json an `m7_probs` manifest (trainer path, FP32), SOUP the
frozen soup's `full/` directory (cal.probs.jsonl, COMPLETE.json, best-export).

Logits are log p (p clamped at 1e-12). Fit: for each L in {3, 4, 5} with >= 150 FIT rows
(CAL698 Score rows + FIT_AHO), minimize the class-balanced NLL
sum_k (1/n_k) sum_{y_i = k} -log softmax(z_i + b)_{y_i} with b_0 = 0 by Newton's method in
float64 until the gradient norm is < 1e-9; shift to mean zero and round to 6 decimals.
Fewer rows keep zeros; a missing level stops the fit (report status FIT_FAILED, exit 1).
Checks apply the rounded file values. Predictions follow benchmark/score.py (unique argmax
of the probabilities; a tie or an invalid answer is no prediction and its own answer value).
Always-majority answers the most frequent gold level of the same items (smallest level on a
tie). The paired bootstrap resamples items (5,000 draws, random.Random(20260929), 2.5/97.5
percentiles); Wilson 95% intervals and Cohen's kappa (invalid as its own category) are
reported alongside.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from benchmark.score import evaluate_answer, unique_argmax
from training.model.infer import checkpoint_fingerprint, normalized_answer
from training.model.score_bias import (
    SCORE_BIAS_FORMAT,
    load_score_bias,
    validate_score_bias,
)

from .common import file_sha256, read_jsonl, write_json
from .contrast import percentile

LEVEL_COUNTS = (3, 4, 5)
MIN_FIT_ROWS = 150
GRAD_TOLERANCE = 1e-9
MAX_NEWTON_STEPS = 200
DECIMALS = 6
PROB_FLOOR = 1e-12
DRAWS = 5000
SEED = 20260929
TOP_SHARE = 0.9
DEV_SCORE_FLOOR = 101
REPLAY_TOLERANCE = 1e-4
CAL_GOLD = Path("/data/dev2/private/panels/gold/cal.jsonl")
DEV_GOLD = Path("/data/dev2/private/panels/gold/typed-dev.gold.jsonl")
PREREG = "v2/06b/records/m7-prereg-2026-09-29.md"


class FitError(ValueError):
    pass


# ------------------------------------------------------------------ math


def logits_of(probabilities: list[float]) -> list[float]:
    return [math.log(max(float(p), PROB_FLOOR)) for p in probabilities]


def softmax(values: list[float]) -> list[float]:
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return [v / total for v in exps]


def corrected(probabilities: list[float], offsets: list[float] | None) -> list[float]:
    if offsets is None:
        return list(probabilities)
    return softmax([z + b for z, b in zip(logits_of(probabilities), offsets)])


def predicted(probabilities: list[float]) -> int | None:
    winner = unique_argmax({str(k): float(p) for k, p in enumerate(probabilities)})
    return int(winner) if winner is not None else None


def balanced_nll(rows: list[tuple[list[float], int]], weights, b) -> float:
    total = 0.0
    for (z, y), w in zip(rows, weights):
        s = [a + c for a, c in zip(z, b)]
        top = max(s)
        total += w * (top + math.log(sum(math.exp(v - top) for v in s)) - s[y])
    return total


def gradient_hessian(rows, weights, b) -> tuple[list[float], list[list[float]]]:
    size = len(b)
    g = [0.0] * size
    h = [[0.0] * size for _ in range(size)]
    for (z, y), w in zip(rows, weights):
        p = softmax([a + c for a, c in zip(z, b)])
        for j in range(size):
            g[j] += w * (p[j] - (1.0 if j == y else 0.0))
            for k in range(size):
                h[j][k] += w * p[j] * ((1.0 if j == k else 0.0) - p[k])
    return g[1:], [row[1:] for row in h[1:]]


def solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting (small dense systems)."""
    n = len(rhs)
    a = [list(row) + [value] for row, value in zip(matrix, rhs)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-300:
            raise FitError("singular Hessian")
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(col + 1, n):
            factor = a[r][col] / a[col][col]
            for c in range(col, n + 1):
                a[r][c] -= factor * a[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (a[r][n] - sum(a[r][c] * x[c] for c in range(r + 1, n))) / a[r][r]
    return x


def fit_level(rows: list[tuple[list[float], int]], levels: int) -> dict[str, Any]:
    counts = Counter(y for _, y in rows)
    missing = [k for k in range(levels) if counts[k] == 0]
    if missing:
        raise FitError(f"L={levels}: FIT misses level(s) {missing}")
    weights = [1.0 / counts[y] for _, y in rows]
    b = [0.0] * levels
    value = start = balanced_nll(rows, weights, b)
    norm = math.inf
    for step in range(MAX_NEWTON_STEPS + 1):
        g, h = gradient_hessian(rows, weights, b)
        norm = math.sqrt(sum(x * x for x in g))
        if norm < GRAD_TOLERANCE:
            break
        if step == MAX_NEWTON_STEPS:
            raise FitError(f"L={levels}: Newton did not converge (|g| = {norm:.3g})")
        d = solve(h, [-x for x in g])
        t = 1.0
        while True:
            trial = [0.0] + [b[j + 1] + t * d[j] for j in range(levels - 1)]
            trial_value = balanced_nll(rows, weights, trial)
            slope = sum(gj * dj for gj, dj in zip(g, d))
            # Near the optimum take the pure Newton step: the objective is flat
            # to rounding there, so a sufficient-decrease test is meaningless.
            if norm < 1e-3 or trial_value <= value + 1e-4 * t * slope or t < 1e-8:
                break
            t /= 2
        b, value = trial, trial_value
    mean = sum(b) / levels
    return {
        "offsets": [round(v - mean, DECIMALS) + 0.0 for v in b],
        "raw_offsets": b,
        "newton_steps": step,
        "gradient_norm": norm,
        "objective_start": start,
        "objective_end": value,
        "rows": len(rows),
        "rows_by_level": {str(k): counts[k] for k in range(levels)},
    }


def wilson(correct: int, n: int) -> list[float]:
    from v2.eval.gates import wilson as gate_wilson

    return list(gate_wilson(correct, n))


def bootstrap(values: list[float]) -> list[float]:
    """Paired item bootstrap of the mean of per-item differences."""
    rng = random.Random(SEED)
    n = len(values)
    draws = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(DRAWS)]
    return [percentile(draws, 2.5), percentile(draws, 97.5)]


def label(value: int | None) -> str:
    return "invalid" if value is None else str(value)


def kappa(pred: list[int | None], gold: list[int]) -> float | None:
    n = len(gold)
    observed = sum(p == g for p, g in zip(pred, gold)) / n
    pc, gc = Counter(label(p) for p in pred), Counter(label(g) for g in gold)
    expected = sum(pc[c] * gc[c] for c in set(pc) | set(gc)) / (n * n)
    return None if expected >= 1 else (observed - expected) / (1 - expected)


def majority_level(gold: list[int]) -> int:
    counts = Counter(gold)
    best = max(counts.values())
    return min(k for k, v in counts.items() if v == best)


def summary(
    pred: list[int | None], gold: list[int], *, intervals: bool = False
) -> dict[str, Any]:
    n = len(gold)
    correct = [int(p == g) for p, g in zip(pred, gold)]
    majority = majority_level(gold)
    always = [int(g == majority) for g in gold]
    usage = Counter(label(p) for p in pred)
    top_value, top_count = usage.most_common(1)[0]
    levels = sorted(set(gold))
    out: dict[str, Any] = {
        "n": n,
        "correct": sum(correct),
        "accuracy": sum(correct) / n,
        "accuracy_ci95": wilson(sum(correct), n),
        "majority_level": majority,
        "majority_accuracy": sum(always) / n,
        "majority_accuracy_ci95": wilson(sum(always), n),
        "delta_vs_majority": (sum(correct) - sum(always)) / n,
        "top_value": top_value,
        "top_share": top_count / n,
        "predicted_distribution": dict(sorted(usage.items())),
        "gold_distribution": {str(k): v for k, v in sorted(Counter(gold).items())},
        "kappa": kappa(pred, gold),
        "recall_by_level": {
            str(k): sum(1 for p, g in zip(pred, gold) if g == k and p == k)
            / sum(1 for g in gold if g == k)
            for k in levels
        },
    }
    if intervals:
        out["delta_vs_majority_ci95"] = bootstrap(
            [c - a for c, a in zip(correct, always)]
        )
    return out


def confusion(pred: list[int | None], gold: list[int]) -> dict[str, dict[str, int]]:
    table: dict[str, Counter] = defaultdict(Counter)
    for p, g in zip(pred, gold):
        table[str(g)][label(p)] += 1
    return {g: dict(sorted(c.items())) for g, c in sorted(table.items())}


def paired_delta(before: list[int | None], after: list[int | None], gold) -> dict:
    diff = [int(a == g) - int(b == g) for b, a, g in zip(before, after, gold)]
    return {
        "delta_accuracy": sum(diff) / len(diff),
        "ci95": bootstrap(diff),
        "changed_answers": sum(b != a for b, a in zip(before, after)),
    }


# --------------------------------------------------------------- loaders


def save(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    return write_json(path, value, exclusive=True)


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def aho_manifest(aho_dir: Path) -> dict[str, Any]:
    manifest = load_json(aho_dir / "MANIFEST.json")
    for name, entry in manifest["outputs"].items():
        if file_sha256(aho_dir / name) != entry["sha256"]:
            raise ValueError(f"{aho_dir / name} changed after the AHO build")
    return manifest


def probs_output(probs_manifest: Path, input_path: Path) -> tuple[Path, dict]:
    manifest = load_json(probs_manifest)
    digest = file_sha256(input_path)
    for entry in manifest["outputs"]:
        if entry["input_sha256"] == digest:
            path = Path(entry["output"])
            if not path.is_absolute() or not path.is_file():
                path = probs_manifest.parent / path.name
            if file_sha256(path) != entry["output_sha256"]:
                raise ValueError(f"{path} differs from its probabilities manifest")
            return path, manifest
    raise ValueError(f"{probs_manifest} has no probabilities for {input_path}")


def aho_items(aho_dir: Path, name: str, probs_manifest: Path) -> list[dict[str, Any]]:
    index = {e["id"]: e for e in read_jsonl(aho_dir / "index.jsonl")}
    path, _ = probs_output(probs_manifest, aho_dir / name)
    probs = {r["id"]: r["probabilities"] for r in read_jsonl(path)}
    items = []
    for row in read_jsonl(aho_dir / name):
        p, levels = probs[row["id"]], len(row["options"])
        if row["task_type"] != "score" or len(p) != levels:
            raise ValueError(
                f"{row['id']}: not a Score row with {levels} probabilities"
            )
        items.append(
            {
                "id": row["id"],
                "source": index[row["id"]]["arm"],
                "levels": levels,
                "p": [float(v) for v in p],
                "y": row["label"],
            }
        )
    return items


def cal_items(aho_dir: Path, soup_dir: Path, cal_gold: Path) -> tuple[list, dict]:
    ids_file = load_json(aho_dir / "cal698.score.ids.json")
    complete = load_json(soup_dir / "COMPLETE.json")
    probs_path = soup_dir / "cal.probs.jsonl"
    if file_sha256(probs_path) != complete["cal"]["probabilities_sha256"]:
        raise ValueError("cal.probs.jsonl differs from the soup's COMPLETE.json")
    if file_sha256(cal_gold) != ids_file["cal700_sha256"]:
        raise ValueError("CAL gold differs from the one the AHO build checked")
    gold = {r["id"]: r for r in read_jsonl(cal_gold)}
    probs = {r["id"]: r["probabilities"] for r in read_jsonl(probs_path)}
    items = []
    for key in ids_file["ids"]:
        row, p = gold[key], probs[key]
        if row["task_type"] != "score" or len(p) != len(row["options"]):
            raise ValueError(f"{key}: CAL698 id is not a Score row with probabilities")
        items.append(
            {
                "id": key,
                "source": "CAL698",
                "levels": len(p),
                "p": [float(v) for v in p],
                "y": row["label"],
            }
        )
    return items, {
        "cal_probs_sha256": file_sha256(probs_path),
        "cal_gold_sha256": file_sha256(cal_gold),
        "cal698_ids_sha256": file_sha256(aho_dir / "cal698.score.ids.json"),
        "complete_sha256": file_sha256(soup_dir / "COMPLETE.json"),
        "best_export_manifest_sha256": complete["best_export_manifest_sha256"],
        "state_sha256": complete["state_sha256"],
    }


def soup_model_sha256(soup_dir: Path) -> str:
    return checkpoint_fingerprint(soup_dir / "best-export")["model_sha256"]


def fit_inputs(args: argparse.Namespace) -> tuple[list, dict]:
    aho = aho_manifest(args.aho_dir)
    cal, provenance = cal_items(args.aho_dir, args.soup_dir, args.cal_gold)
    path, probs = probs_output(args.probs, args.aho_dir / "fit_aho.jsonl")
    if probs["state_sha256"] != provenance["state_sha256"]:
        raise ValueError("FIT_AHO probabilities come from another state than the soup")
    parity = probs.get("cal_parity") or {}
    if parity.get("pass") is not True:
        raise ValueError("FIT_AHO probabilities lack a passing CAL parity check")
    items = cal + aho_items(args.aho_dir, "fit_aho.jsonl", args.probs)
    provenance.update(
        {
            "aho_manifest_sha256": file_sha256(args.aho_dir / "MANIFEST.json"),
            "fit_aho_sha256": aho["outputs"]["fit_aho.jsonl"]["sha256"],
            "fit_aho_probs_sha256": file_sha256(path),
            "probs_manifest_sha256": file_sha256(args.probs),
            "cal_parity_max_abs_drift": parity.get("max_abs_drift"),
        }
    )
    return items, provenance


def by_level(items: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list] = defaultdict(list)
    for item in items:
        out[item["levels"]].append(item)
    return out


# ------------------------------------------------------------------ fit


def fit(args: argparse.Namespace) -> int:
    for path in (args.output, args.report):
        if path.exists():
            raise FileExistsError(path)
    items, provenance = fit_inputs(args)
    model_sha = soup_model_sha256(args.soup_dir)
    if args.readout_manifest is not None:
        logged = load_json(args.readout_manifest)["model_sha256"]
        if logged != model_sha:
            raise ValueError("readout manifest model_sha256 differs from the export")
    groups = by_level(items)
    offsets, levels_report, status = {}, {}, "FIT"
    for levels in LEVEL_COUNTS:
        rows = [(logits_of(i["p"]), i["y"]) for i in groups.get(levels, [])]
        sources = dict(Counter(i["source"] for i in groups.get(levels, [])))
        if len(rows) < MIN_FIT_ROWS:
            offsets[str(levels)] = [0.0] * levels
            levels_report[str(levels)] = {
                "status": "too_few_rows",
                "rows": len(rows),
                "rows_by_source": sources,
            }
            continue
        try:
            result = fit_level(rows, levels)
        except FitError as exc:
            status = "FIT_FAILED"
            levels_report[str(levels)] = {"status": "failed", "error": str(exc)}
            continue
        offsets[str(levels)] = result.pop("offsets")
        levels_report[str(levels)] = {
            "status": "fit",
            "rows_by_source": sources,
            **result,
        }
    record = {
        "prereg": f"{PREREG} section 1.4",
        "objective": "class-balanced NLL, b_0 = 0, logits = log max(p, 1e-12)",
        "solver": f"Newton, float64, gradient norm < {GRAD_TOLERANCE}",
        "post": f"shift to mean zero, round to {DECIMALS} decimals",
        "min_fit_rows": MIN_FIT_ROWS,
        "levels": levels_report,
        "inputs": provenance,
    }
    report = {"schema": "dev2-06b-m7a-fit/1", "status": status, **record}
    if status != "FIT":
        save(args.report, report)
        print(json.dumps({"status": status, "levels": levels_report}, sort_keys=True))
        return 1
    bias = {
        "format": SCORE_BIAS_FORMAT,
        "model_sha256": model_sha,
        "offsets": offsets,
        "fit": record,
    }
    validate_score_bias(bias, model_sha)
    report["score_bias_sha256"] = save(args.output, bias)
    report["model_sha256"] = model_sha
    report["offsets"] = offsets
    save(args.report, report)
    print(
        json.dumps(
            {
                "status": status,
                "offsets": offsets,
                "sha256": report["score_bias_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


# ---------------------------------------------------------------- checks


def evaluate_items(
    items: list[dict[str, Any]], offsets: dict[int, list[float]] | None, gated
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for levels, group in sorted(by_level(items).items()):
        gold = [i["y"] for i in group]
        before = [predicted(i["p"]) for i in group]
        entry = {"before": summary(before, gold, intervals=gated)}
        if offsets is not None:
            row = offsets.get(levels)
            after = [predicted(corrected(i["p"], row)) for i in group]
            entry["after"] = summary(after, gold, intervals=gated)
            if gated:
                entry["paired_vs_uncorrected"] = paired_delta(before, after, gold)
        out[f"L{levels}"] = entry
    return out


def by_source(items, offsets) -> dict[str, Any]:
    groups: dict[str, list] = defaultdict(list)
    for item in items:
        groups[item["source"]].append(item)
    return {
        name: evaluate_items(g, offsets, False) for name, g in sorted(groups.items())
    }


def dev_items(predictions: Path, gold_path: Path, offsets) -> dict[str, Any]:
    from v2.eval import panels

    if file_sha256(gold_path) != panels.ALL["typed-dev"]["gold_sha256"]:
        raise ValueError("typed-DEV gold differs from the frozen panel")
    records = {r["id"]: r for r in read_jsonl(predictions)}
    gold, before, after, models = [], [], [], Counter()
    for item in read_jsonl(gold_path):
        record = records.get(item["id"]) or {}
        models[record.get("model_sha256")] += 1
        for key, question in item["questions"].items():
            if question["type"] != "score":
                continue
            truth = item["gold"][key]
            answer = (record.get("answers") or {}).get(key)
            first = evaluate_answer(question, truth, answer)
            before.append(first["point"] if first["status"] == "ok" else None)
            levels = len(question["criteria"])
            row = offsets.get(levels) if offsets else None
            fixed = answer
            if (
                row is not None
                and first["status"] == "ok"
                and isinstance(answer.get("probabilities"), dict)
            ):
                keys = [str(k) for k in range(levels)]
                logits = logits_of([answer["probabilities"][k] for k in keys])
                fixed = normalized_answer(
                    "score", keys, [z + b for z, b in zip(logits, row)], 1.0
                )
            second = evaluate_answer(question, truth, fixed)
            after.append(second["point"] if second["status"] == "ok" else None)
            gold.append(truth["value"])
    return {
        "gold": gold,
        "before": before,
        "after": after,
        "model_sha256": dict(models),
    }


def check(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise FileExistsError(args.output)
    model_sha = soup_model_sha256(args.soup_dir)
    raw, bias = load_score_bias(args.score_bias, model_sha)
    offsets = {k: v for k, v in raw.items()}
    fit_set, provenance = fit_inputs(args)
    path, probs = probs_output(args.probs, args.aho_dir / "chk.jsonl")
    if probs["state_sha256"] != provenance["state_sha256"]:
        raise ValueError("CHK probabilities come from another state than the soup")
    chk = aho_items(args.aho_dir, "chk.jsonl", args.probs)
    chk_eval = evaluate_items(chk, offsets, True)
    dev = dev_items(args.dev_predictions, args.dev_gold, offsets)
    if set(dev["model_sha256"]) != {model_sha}:
        raise ValueError("typed-DEV readout is not of this export")
    dev_before = summary(dev["before"], dev["gold"], intervals=True)
    dev_after = summary(dev["after"], dev["gold"], intervals=True)
    chk5 = chk_eval.get("L5", {}).get("after")
    ad1 = {
        "top_share_le_0.90": chk5 is not None and chk5["top_share"] <= TOP_SHARE,
        "delta_vs_majority_lower_bound_gt_0": chk5 is not None
        and chk5["delta_vs_majority_ci95"][0] > 0,
    }
    ad2 = {
        "top_share_le_0.90": dev_after["top_share"] <= TOP_SHARE,
        f"correct_ge_{DEV_SCORE_FLOOR}": dev_after["correct"] >= DEV_SCORE_FLOOR,
    }
    result = {
        "schema": "dev2-06b-m7a-check/1",
        "prereg": f"{PREREG} section 1.5",
        "score_bias": {
            "path": str(args.score_bias),
            "sha256": file_sha256(args.score_bias),
            "offsets": bias["offsets"],
        },
        "model_sha256": model_sha,
        "inputs": {
            **provenance,
            "chk_sha256": file_sha256(args.aho_dir / "chk.jsonl"),
            "chk_probs_sha256": file_sha256(path),
            "dev_predictions_sha256": file_sha256(args.dev_predictions),
            "dev_gold_sha256": file_sha256(args.dev_gold),
        },
        "fit": evaluate_items(fit_set, offsets, False),
        "fit_by_source": by_source(fit_set, offsets),
        "chk": chk_eval,
        "chk_by_arm": by_source(chk, offsets),
        "typed_dev_score": {
            "before": dev_before,
            "after": dev_after,
            "paired_vs_uncorrected": paired_delta(
                dev["before"], dev["after"], dev["gold"]
            ),
            "confusion_before": confusion(dev["before"], dev["gold"]),
            "confusion_after": confusion(dev["after"], dev["gold"]),
        },
        "A-D1": {"checks": ad1, "pass": all(ad1.values())},
        "A-D2": {"checks": ad2, "pass": all(ad2.values())},
    }
    save(args.output, result)
    print(
        json.dumps(
            {
                "A-D1": result["A-D1"],
                "A-D2": result["A-D2"],
                "chk5_after": chk5
                and {
                    k: chk5[k]
                    for k in ("accuracy", "top_share", "delta_vs_majority_ci95")
                },
                "dev_after": {
                    k: dev_after[k] for k in ("correct", "top_share", "accuracy")
                },
            },
            sort_keys=True,
        )
    )
    return 0


def devcheck(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise FileExistsError(args.output)
    aho_manifest(args.aho_dir)
    path, probs = probs_output(args.probs, args.aho_dir / "chk.jsonl")
    chk = aho_items(args.aho_dir, "chk.jsonl", args.probs)
    chk_eval = evaluate_items(chk, None, True)
    chk5 = chk_eval.get("L5", {}).get("before")
    bd1 = {
        "top_share_le_0.90": chk5 is not None and chk5["top_share"] <= TOP_SHARE,
        "delta_vs_majority_lower_bound_gt_0": chk5 is not None
        and chk5["delta_vs_majority_ci95"][0] > 0,
    }
    result = {
        "schema": "dev2-06b-m7-devcheck/1",
        "prereg": f"{PREREG} section 2.4 (B-D1 = A-D1 on uncorrected CHK_5)",
        "label": args.label,
        "state_sha256": probs["state_sha256"],
        "inputs": {
            "chk_sha256": file_sha256(args.aho_dir / "chk.jsonl"),
            "chk_probs_sha256": file_sha256(path),
            "probs_manifest_sha256": file_sha256(args.probs),
        },
        "chk": chk_eval,
        "chk_by_arm": by_source(chk, None),
        "B-D1": {"checks": bd1, "pass": all(bd1.values())},
    }
    save(args.output, result)
    print(json.dumps({"label": args.label, "B-D1": result["B-D1"]}, sort_keys=True))
    return 0


# ---------------------------------------------------------------- replay


def noul_point(p: float) -> bool | None:
    return None if p == 0.5 else p > 0.5


def replay(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise FileExistsError(args.output)
    logged = {r["id"]: r for r in read_jsonl(args.logged)}
    online = {r["id"]: r for r in read_jsonl(args.online)}
    models = {r["model_sha256"] for r in logged.values()}
    if len(models) != 1:
        raise ValueError("logged predictions mix model hashes")
    model_sha = models.pop()
    offsets, _ = load_score_bias(args.score_bias, model_sha)
    bias_sha = file_sha256(args.score_bias)
    manifest_path = args.online.with_name(args.online.name + ".manifest.json")
    manifest = load_json(manifest_path)
    binding = {
        "online_manifest_score_bias_sha256": manifest.get("score_bias_sha256")
        == bias_sha,
        "online_manifest_model_sha256": manifest.get("model_sha256") == model_sha,
        "online_rows_score_bias_sha256": all(
            r.get("score_bias_sha256") == bias_sha for r in online.values()
        ),
        "online_rows_model_sha256": all(
            r.get("model_sha256") == model_sha for r in online.values()
        ),
        "same_items": set(logged) == set(online),
    }
    counts: Counter = Counter()
    worst = {"score": 0.0, "choice": 0.0, "noul": 0.0}
    mismatches: list[dict[str, str]] = []
    for key in sorted(set(logged) & set(online)):
        left, right = logged[key]["answers"], online[key]["answers"]
        if set(left) != set(right):
            mismatches.append({"id": key, "reason": "question set"})
            continue
        for qid, answer in left.items():
            other = right[qid]
            kind = answer.get("type")
            if "error" in answer or "error" in other:
                counts["errors"] += 1
                if answer != other:
                    mismatches.append({"id": key, "question": qid, "reason": "error"})
                continue
            counts[kind] += 1
            if kind == "score":
                keys = [str(k) for k in range(len(answer["probabilities"]))]
                p = [answer["probabilities"][k] for k in keys]
                row = offsets.get(len(keys))
                expected = (
                    normalized_answer(
                        "score",
                        keys,
                        [z + b for z, b in zip(logits_of(p), row)],
                        1.0,
                    )
                    if row is not None
                    else answer
                )
                delta = max(
                    abs(expected["probabilities"][k] - other["probabilities"][k])
                    for k in keys
                )
                worst["score"] = max(worst["score"], delta)
                if unique_argmax(expected["probabilities"]) != unique_argmax(
                    other["probabilities"]
                ):
                    mismatches.append(
                        {"id": key, "question": qid, "reason": "score argmax"}
                    )
            elif kind == "choice":
                delta = max(
                    abs(answer["probabilities"][k] - other["probabilities"][k])
                    for k in answer["probabilities"]
                )
                worst["choice"] = max(worst["choice"], delta)
                counts["choice_bitwise_identical"] += answer == other
                if answer["choice"] != other["choice"]:
                    mismatches.append({"id": key, "question": qid, "reason": "choice"})
            else:
                delta = abs(answer["noul"] - other["noul"])
                worst["noul"] = max(worst["noul"], delta)
                counts["noul_bitwise_identical"] += answer == other
                if noul_point(answer["noul"]) != noul_point(other["noul"]):
                    mismatches.append({"id": key, "question": qid, "reason": "noul"})
    checks = {
        **binding,
        "no_answer_mismatch": not mismatches,
        "max_abs_dp_le_1e-4": max(worst.values()) <= REPLAY_TOLERANCE,
    }
    result = {
        "schema": "dev2-06b-m7a-replay/1",
        "prereg": f"{PREREG} section 1.5 (A-D3)",
        "logged": {"path": str(args.logged), "sha256": file_sha256(args.logged)},
        "online": {"path": str(args.online), "sha256": file_sha256(args.online)},
        "online_manifest_sha256": file_sha256(manifest_path),
        "score_bias_sha256": bias_sha,
        "model_sha256": model_sha,
        "counts": dict(counts),
        "max_abs_dp": worst,
        "mismatches": mismatches[:50],
        "mismatch_count": len(mismatches),
        "A-D3": {"checks": checks, "pass": all(checks.values())},
    }
    save(args.output, result)
    print(json.dumps({"A-D3": result["A-D3"], "max_abs_dp": worst}, sort_keys=True))
    return 0


# ------------------------------------------------------------------ gate


def gate(args: argparse.Namespace) -> int:
    from benchmark.score import load_jsonl
    from v2.eval import panels
    from v2.eval.gates import type_summary, verified

    if args.output.exists():
        raise FileExistsError(args.output)
    gold_path = panels.path(args.panel_root, "typed-final", "gold")
    if file_sha256(gold_path) != panels.ALL["typed-final"]["gold_sha256"]:
        raise ValueError("typed FINAL gold differs from the frozen panel")
    gold = list(load_jsonl(gold_path).values())
    predictions_path = verified(args.run, "typed-final")
    predictions = {r["id"]: r for r in read_jsonl(predictions_path)}
    truth, points = [], []
    for item in gold:
        answers = (predictions.get(item["id"]) or {}).get("answers")
        for key, question in item["questions"].items():
            if question["type"] != "score":
                continue
            answer = answers.get(key) if isinstance(answers, dict) else None
            result = (
                evaluate_answer(question, item["gold"][key], answer)
                if isinstance(answer, dict)
                else {"status": "invalid"}
            )
            points.append(result.get("point") if result["status"] == "ok" else None)
            truth.append(item["gold"][key]["value"])
    score = summary(points, truth, intervals=True)
    types = type_summary(gold, predictions)
    collapsed = {k: v["verdict"] for k, v in types.items() if v["verdict"] != "OK"}
    stored = args.gates / "types.json"
    stored_verdicts = (
        {k: v["verdict"] for k, v in load_json(stored)["types"].items()}
        if stored.is_file()
        else None
    )
    s3 = {
        "no_type_collapsed": not collapsed,
        "score_top_share_lt_0.90": score["top_share"] < TOP_SHARE,
        "score_delta_vs_majority_lower_bound_gt_0": score["delta_vs_majority_ci95"][0]
        > 0,
    }
    result: dict[str, Any] = {
        "schema": "dev2-06b-m7-gate-s3/1",
        "prereg": f"{PREREG} section 0 (S3)",
        "label": args.label,
        "run": str(args.run),
        "typed_final_predictions_sha256": file_sha256(predictions_path),
        "typed_final_gold_sha256": file_sha256(gold_path),
        "score": score,
        "type_verdicts": {k: v["verdict"] for k, v in types.items()},
        "types_json_agrees": (
            None
            if stored_verdicts is None
            else stored_verdicts == {k: v["verdict"] for k, v in types.items()}
        ),
        "S3": {"checks": s3, "pass": all(s3.values())},
    }
    summary_path = args.run / "M6-SUMMARY.json"
    if summary_path.is_file():
        checks = load_json(summary_path).get("successor", {}).get("checks")
        if checks:
            s1 = checks["v3_lower_bound_gt_0"]
            s2 = checks["H_upper_bound_ge_0"]
            result["successor"] = {
                "S1_v3_lower_bound_gt_0": s1,
                "S2_H_upper_bound_ge_0": s2,
                "S3": result["S3"]["pass"],
                "verdict": bool(s1 and s2 and result["S3"]["pass"]),
                "summary_sha256": file_sha256(summary_path),
            }
    save(args.output, result)
    print(
        json.dumps(
            {
                "S3": result["S3"],
                "score": {
                    k: score[k]
                    for k in (
                        "accuracy",
                        "top_value",
                        "top_share",
                        "delta_vs_majority_ci95",
                    )
                },
                "successor": result.get("successor", {}).get("verdict"),
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def fit_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--aho-dir", type=Path, required=True)
        p.add_argument("--probs", type=Path, required=True)
        p.add_argument("--soup-dir", type=Path, required=True)
        p.add_argument("--cal-gold", type=Path, default=CAL_GOLD)

    one = commands.add_parser("fit")
    fit_args(one)
    one.add_argument("--readout-manifest", type=Path)
    one.add_argument("--output", type=Path, required=True)
    one.add_argument("--report", type=Path, required=True)
    two = commands.add_parser("check")
    fit_args(two)
    two.add_argument("--score-bias", type=Path, required=True)
    two.add_argument("--dev-predictions", type=Path, required=True)
    two.add_argument("--dev-gold", type=Path, default=DEV_GOLD)
    two.add_argument("--output", type=Path, required=True)
    three = commands.add_parser("devcheck")
    three.add_argument("--aho-dir", type=Path, required=True)
    three.add_argument("--probs", type=Path, required=True)
    three.add_argument("--label", required=True)
    three.add_argument("--output", type=Path, required=True)
    four = commands.add_parser("replay")
    four.add_argument("--logged", type=Path, required=True)
    four.add_argument("--online", type=Path, required=True)
    four.add_argument("--score-bias", type=Path, required=True)
    four.add_argument("--output", type=Path, required=True)
    five = commands.add_parser("gate")
    five.add_argument("--run", type=Path, required=True)
    five.add_argument("--gates", type=Path, required=True)
    five.add_argument("--label", required=True)
    five.add_argument("--output", type=Path, required=True)
    five.add_argument(
        "--panel-root", type=Path, default=Path("/data/dev2/private/panels")
    )
    args = parser.parse_args(argv)
    return {
        "fit": fit,
        "check": check,
        "devcheck": devcheck,
        "replay": replay,
        "gate": gate,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
