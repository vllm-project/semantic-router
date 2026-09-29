"""Paired mlx-diag comparison on the card-eligible parts (Choice and Noul) of two runs.

    PYTHONPATH=<src>/training/decision2:<src>/training/decision2/v2/9b python3 -m \
        lux9b.mlx_paired --left RUN --right RUN --panel PANEL --left-name A --right-name B \
        --output OUT [--draws 5000 --seed 20260927]

RUN is an mlx-diag run directory (``output/mlx-diag.predictions.jsonl`` and
``mlx-diag.score.json``), PANEL the frozen mlx-diag v1 panel (``gold.jsonl``,
``prompts.jsonl``). The Score part (XNLI, CC BY-NC) is excluded. The unit is the panel
item, scored per item by ``v2.eval.overlap_effects.mlx_outcomes`` exactly as
``v2.eval.multilingual_panel.score`` counts it (a Noul item is one sentence pair). Every
type x language cell of both runs must reproduce the stored ``mlx-diag.score.json``.

overall = mean over Choice and Noul of the mean over languages (English included) of
per-language accuracy. The paired bootstrap resamples items within each type x language
stratum with the same draws for both runs (``strata_bootstrap``). The comparison is "not
significantly below" when the upper end of the 95% interval of left - right is >= 0.
Outputs are aggregates only: no item ids, text, gold values or answers.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from v2.eval import panels
from v2.eval.overlap_effects import mlx_outcomes, mlx_strata, strata_bootstrap
from v2.eval.same_panel import (
    PAIRED_REPLICATES,
    PAIRED_SEED,
    prediction_path,
    read_jsonl,
    sha_file,
    write_json,
)

SCHEMA = "dev2-9b-mlx-paired/1"
CARD_TYPES = ("choice", "noul")
FROZEN_GOLD_SHA256 = panels.DEVELOPMENT["mlx-diag"]["gold_sha256"]


def load_run(
    run: Path, gold: dict[str, Any], prompts: dict[str, Any], gold_sha256: str
) -> tuple[dict[str, dict[str, Any]], str]:
    path = prediction_path(run, "mlx-diag")
    stored = json.loads((run / "mlx-diag.score.json").read_text(encoding="utf-8"))
    digest = sha_file(path)
    if digest != stored["predictions_sha256"]:
        raise ValueError(f"{run}: mlx-diag predictions differ from the stored score")
    if stored["gold_sha256"] != gold_sha256:
        raise ValueError(f"{run}: stored score used a different mlx-diag gold")
    outcomes = mlx_outcomes(gold, prompts, {r["id"]: r for r in read_jsonl(path)})
    self_check(outcomes, stored, str(run))
    return outcomes, digest


def counts(outcomes: dict[str, dict[str, Any]]) -> dict[tuple[str, str], list[int]]:
    cells: dict[tuple[str, str], list[int]] = {}
    for row in outcomes.values():
        cell = cells.setdefault((row["type"], row["language"]), [0, 0])
        cell[0] += row["correct"]
        cell[1] += 1
    return cells


def self_check(outcomes: dict[str, dict[str, Any]], stored: dict[str, Any], label: str):
    ours = counts(outcomes)
    theirs = {
        (kind, lang): [v["correct"], v["n"]]
        for kind, part in stored["by_type"].items()
        for lang, v in part["languages"].items()
    }
    if ours != theirs:
        bad = sorted(k for k in set(ours) | set(theirs) if ours.get(k) != theirs.get(k))
        raise ValueError(
            f"{label}: per-item rescore differs from mlx-diag.score.json in {bad}"
        )
    for (kind, lang), (c, n) in ours.items():
        if c / n != stored["by_type"][kind]["languages"][lang]["accuracy"]:
            raise ValueError(
                f"{label}: {kind}/{lang} accuracy differs from the stored score"
            )


def macro(outcomes: dict[str, dict[str, Any]], languages=None) -> float:
    cells = counts(outcomes)
    by_type = [
        [
            c / n
            for (t, lang), (c, n) in cells.items()
            if t == kind and (languages is None or lang in languages)
        ]
        for kind in CARD_TYPES
    ]
    return statistics.mean(statistics.mean(v) for v in by_type if v)


def part(left, right, draws: int, seed: int, keep) -> dict[str, Any]:
    sub_l = {i: r for i, r in left.items() if keep(r)}
    sub_r = {i: right[i] for i in sub_l}
    strata, kinds = mlx_strata(sub_l, sub_r, set())
    a, b = macro(sub_l), macro(sub_r)
    return {
        "items": len(sub_l),
        "left": a,
        "right": b,
        "delta": a - b,
        "ci95": strata_bootstrap(strata, draws, seed, kinds=kinds),
    }


def compare(left, right, draws: int = PAIRED_REPLICATES, seed: int = PAIRED_SEED):
    if set(left) != set(right):
        raise ValueError("the two runs cover different mlx-diag items")
    left = {i: r for i, r in left.items() if r["type"] in CARD_TYPES}
    right = {i: right[i] for i in left}
    overall = part(left, right, draws, seed, lambda r: True)
    languages = sorted({r["language"] for r in left.values()})
    cells = counts(left)
    return {
        "overall": overall,
        "not_significantly_below": overall["ci95"]["high"] >= 0,
        "by_type": {
            kind: part(left, right, draws, seed, lambda r, k=kind: r["type"] == k)
            for kind in CARD_TYPES
        },
        "by_language": {
            lang: part(left, right, draws, seed, lambda r, g=lang: r["language"] == g)
            for lang in languages
        },
        "non_english": {
            kind: part(
                left,
                right,
                draws,
                seed,
                lambda r, k=kind: r["type"] == k and r["language"] != "en",
            )
            for kind in CARD_TYPES
        },
        "units": {f"{t}/{lang}": n for (t, lang), (_c, n) in sorted(cells.items())},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--left", type=Path, required=True)
    ap.add_argument("--right", type=Path, required=True)
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--left-name", required=True)
    ap.add_argument("--right-name", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=PAIRED_REPLICATES)
    ap.add_argument("--seed", type=int, default=PAIRED_SEED)
    args = ap.parse_args(argv)
    gold_path = args.panel / "gold.jsonl"
    gold_sha256 = sha_file(gold_path)
    if gold_sha256 != FROZEN_GOLD_SHA256:
        raise ValueError("mlx-diag gold differs from the frozen panel")
    gold = {g["id"]: g for g in read_jsonl(gold_path)}
    prompts = {p["id"]: p for p in read_jsonl(args.panel / "prompts.jsonl")}
    left, left_sha = load_run(args.left, gold, prompts, gold_sha256)
    right, right_sha = load_run(args.right, gold, prompts, gold_sha256)
    result = {
        "schema": SCHEMA,
        "label": "mlx-diag development diagnostic (card-eligible Choice + Noul); not a release score",
        "left": {
            "name": args.left_name,
            "run": str(args.left),
            "predictions_sha256": left_sha,
        },
        "right": {
            "name": args.right_name,
            "run": str(args.right),
            "predictions_sha256": right_sha,
        },
        "gold_sha256": gold_sha256,
        "types": list(CARD_TYPES),
        "excluded": "score (XNLI, CC BY-NC 4.0)",
        "self_check": "per type x language counts reproduce both mlx-diag.score.json files",
        "bootstrap": {
            "draws": args.draws,
            "seed": args.seed,
            "strata": "type x language",
        },
        **compare(left, right, args.draws, args.seed),
    }
    sha = write_json(args.output, result)
    o = result["overall"]
    print(
        json.dumps(
            {
                "delta": o["delta"],
                "ci95": o["ci95"],
                "not_significantly_below": result["not_significantly_below"],
                "sha256": sha,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
