"""mlx-diag aggregates for one decoder M5 formal run (node A; diagnostic only, prereg dec-m5-prereg-2026-09-29.md).

Reads the frozen `multilingual_panel score` output and, for the Noul (PAWS-X) items, the gold values and predictions
to derive the yes-bias figures of the prereg's diagnosis (the procedure of the earlier mlx_agg scoring): per
language predicted-yes rate, gold-yes rate, recall on gold-yes / gold-No; the non-English figures are the mean over
non-English languages (as `non_english_mean_accuracy`), pooled values are reported alongside. Aggregates only; no
item id is written.

usage: python3 m5-mlx-agg.py --panel <mlx-diag-v1 dir> --predictions <mlx-diag.predictions.jsonl>
    --score <mlx-diag.score.json> --output <mlx-agg.json>
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

TYPES = ("choice", "noul", "score")


def pred_bool(answer: dict | None) -> bool | None:
    if not isinstance(answer, dict):
        return None
    if isinstance(answer.get("noul"), (int, float)) and not isinstance(
        answer.get("noul"), bool
    ):
        return answer["noul"] >= 0.5
    value = answer.get("value")
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    probs = answer.get("probabilities") or {}
    if probs:
        return str(max(probs, key=probs.get)).lower() in ("true", "yes", "1")
    return None


def gold_bool(value) -> bool:
    return value is True or str(value) == "True"


def noul_bias(gold_rows: list[dict], predictions: dict[str, dict]) -> dict[str, dict]:
    counts: dict[str, list[int]] = defaultdict(
        lambda: [0, 0, 0, 0, 0]
    )  # n, pred_yes, gold_yes, hit_yes, hit_no
    for g in gold_rows:
        if g["type"] != "noul":
            continue
        p = pred_bool(((predictions.get(g["id"]) or {}).get("answers") or {}).get("q"))
        gv = gold_bool(g["value"])
        c = counts[g["language"]]
        c[0] += 1
        c[1] += p is True
        c[2] += gv
        c[3] += gv and p is True
        c[4] += (not gv) and p is False
    return {
        lang: {
            "n": n,
            "pred_yes": py / n,
            "gold_yes": gy / n,
            "recall_yes": hy / gy if gy else None,
            "recall_no": hn / (n - gy) if n - gy else None,
            "accuracy": (hy + hn) / n,
        }
        for lang, (n, py, gy, hy, hn) in sorted(counts.items())
    }


def aggregate(score: dict, bias: dict[str, dict]) -> dict:
    non_en = {k: v for k, v in bias.items() if k != "en"}
    n = sum(v["n"] for v in non_en.values())
    pooled = {
        "pred_yes": (
            sum(v["pred_yes"] * v["n"] for v in non_en.values()) / n if n else None
        ),
        "recall_no": (
            sum(
                v["recall_no"] * v["n"] * (1 - v["gold_yes"])
                for v in non_en.values()
                if v["recall_no"] is not None
            )
            / sum(v["n"] * (1 - v["gold_yes"]) for v in non_en.values())
            if n
            else None
        ),
    }
    by_type = score["by_type"]
    return {
        "schema": "dec-m5-mlx-agg/1",
        "scope": "mlx-diag multilingual diagnostic (post-v3-seal); drives no Milestone 5 decision; aggregates only",
        "predictions_sha256": score["predictions_sha256"],
        "gold_sha256": score["gold_sha256"],
        "items": score["items"],
        "invalid_or_missing": score["invalid_or_missing"],
        "overall": score["type_macro_accuracy"],
        "english_type_macro": score["english_type_macro_accuracy"],
        "non_english_type_macro": score["non_english_type_macro_accuracy"],
        "non_english_by_type": {
            t: by_type[t]["non_english_mean_accuracy"] for t in TYPES
        },
        "english_by_type": {t: by_type[t]["english_accuracy"] for t in TYPES},
        "per_language_mean_accuracy": score["per_language_mean_accuracy"],
        "accuracy_by_type_language": {
            t: {lang: v["accuracy"] for lang, v in by_type[t]["languages"].items()}
            for t in TYPES
        },
        "cross_language_consistency": score["cross_language_consistency"],
        "noul_by_language": bias,
        "non_english_noul": {
            "accuracy_mean": by_type["noul"]["non_english_mean_accuracy"],
            "pred_yes_rate_mean": (
                statistics.mean(v["pred_yes"] for v in non_en.values())
                if non_en
                else None
            ),
            "gold_no_recall_mean": (
                statistics.mean(
                    v["recall_no"]
                    for v in non_en.values()
                    if v["recall_no"] is not None
                )
                if non_en
                else None
            ),
            "pred_yes_rate_pooled": pooled["pred_yes"],
            "gold_no_recall_pooled": pooled["recall_no"],
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--score", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    gold = [
        json.loads(line)
        for line in (args.panel / "gold.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    predictions = {}
    for line in args.predictions.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            predictions[row["id"]] = row
    score = json.loads(args.score.read_text())
    result = aggregate(score, noul_bias(gold, predictions))
    args.output.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    ne = result["non_english_noul"]
    print(
        json.dumps(
            {
                "overall": round(result["overall"], 4),
                "non_english_by_type": {
                    k: round(v, 4) for k, v in result["non_english_by_type"].items()
                },
                "non_english_pred_yes": round(ne["pred_yes_rate_mean"], 3),
                "non_english_gold_no_recall": round(ne["gold_no_recall_mean"], 3),
            }
        )
    )


if __name__ == "__main__":
    main()
