"""Paired mlx-diag comparison of two prediction files on the same panel (decoder M6 successor rule 4).

Per-item outcomes come from `v2.eval.overlap_effects.mlx_outcomes` (the counting of
`v2.eval.multilingual_panel.score`); each side's full type macro is checked to equal the frozen scorer's
`type_macro_accuracy` exactly. The bootstrap is `v2.eval.overlap_effects.strata_bootstrap`: paired item draws
within each type x language stratum, 5,000 replicates, seed 20260927, statistic = the mean over types of the
mean over languages of accuracy, candidate minus reference. It is run for

- the card-eligible type macro over the Choice and Noul parts (the XNLI-based Score part is NC); rule 4 passes
  if its 95% upper bound is >= 0;
- the full type macro (Choice, Noul, Score), reported;
- each type (report) and each type x language cell (accuracy delta, report).

Every statistic has its own replicate loop with the same seed. Aggregates only: no item id, text, gold value or
answer is written.

    python3 -m v2.dec.mlx_paired --panel /data/dev2/private/panels/mlx-diag-v1 \
        --candidate CAND.predictions.jsonl --reference REF.predictions.jsonl \
        [--candidate-name A] [--reference-name B] --output OUT.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from v2.eval import panels
from v2.eval.multilingual_panel import score as frozen_score
from v2.eval.overlap_effects import (
    MLX_TYPES,
    mlx_outcomes,
    mlx_strata,
    mlx_value,
    strata_bootstrap,
)
from v2.eval.same_panel import PAIRED_REPLICATES, PAIRED_SEED, read_jsonl, sha_file

SCHEMA = "dec-m6-mlx-paired/1"
CARD_TYPES = ("choice", "noul")


def load(panel: Path, predictions: Path) -> dict[str, dict[str, Any]]:
    gold = {row["id"]: row for row in read_jsonl(panel / "gold.jsonl")}
    prompts = {row["id"]: row for row in read_jsonl(panel / "prompts.jsonl")}
    preds = {row["id"]: row for row in read_jsonl(predictions)}
    return mlx_outcomes(gold, prompts, preds)


def type_macro(outcomes: dict[str, dict[str, Any]], types: tuple[str, ...]) -> float:
    cells: dict[tuple[str, str], list[int]] = {}
    for row in outcomes.values():
        if row["type"] in types:
            cell = cells.setdefault((row["type"], row["language"]), [0, 0])
            cell[0] += row["correct"]
            cell[1] += 1
    return statistics.mean(
        statistics.mean(c / n for (t, _lang), (c, n) in cells.items() if t == kind)
        for kind in types
        if any(t == kind for t, _lang in cells)
    )


def restrict(
    strata: list[list[tuple[int, int]]], kinds: list[str], types: tuple[str, ...]
) -> tuple[list[list[tuple[int, int]]], list[str]]:
    keep = [i for i, kind in enumerate(kinds) if kind in types]
    return [strata[i] for i in keep], [kinds[i] for i in keep]


def statistic(
    candidate: dict[str, dict[str, Any]],
    reference: dict[str, dict[str, Any]],
    strata: list[list[tuple[int, int]]],
    kinds: list[str],
    types: tuple[str, ...],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    sub, sub_kinds = restrict(strata, kinds, types)
    left, right = type_macro(candidate, types), type_macro(reference, types)
    return {
        "types": list(types),
        "candidate": left,
        "reference": right,
        "delta": left - right,
        "ci95": strata_bootstrap(sub, replicates, seed, sub_kinds),
        "strata": len(sub),
        "items": sum(len(rows) for rows in sub),
    }


def compare(
    panel: Path,
    candidate_path: Path,
    reference_path: Path,
    replicates: int = PAIRED_REPLICATES,
    seed: int = PAIRED_SEED,
    names: tuple[str, str] = ("candidate", "reference"),
) -> dict[str, Any]:
    sides = {
        "candidate": load(panel, candidate_path),
        "reference": load(panel, reference_path),
    }
    checks = {}
    for side, path in (("candidate", candidate_path), ("reference", reference_path)):
        stored = frozen_score(panel, path)
        mine = mlx_value(sides[side], set())
        checks[side] = {
            "frozen_type_macro": stored["type_macro_accuracy"],
            "type_macro": mine["type_macro_accuracy"],
            "equal": stored["type_macro_accuracy"] == mine["type_macro_accuracy"]
            and stored["per_language_mean_accuracy"] == mine["per_language"],
        }
    cand, ref = sides["candidate"], sides["reference"]
    strata, kinds = mlx_strata(cand, ref, set())
    keys = sorted({(r["type"], r["language"]) for r in cand.values()})
    cells: dict[str, dict[str, Any]] = {}
    for (kind, lang), rows in zip(keys, strata):
        n = len(rows)
        a, b = sum(x for x, _y in rows) / n, sum(y for _x, y in rows) / n
        cells.setdefault(kind, {})[lang] = {
            "n": n,
            "candidate": a,
            "reference": b,
            "delta": a - b,
            "ci95": strata_bootstrap([rows], replicates, seed, [kind]),
        }
    gold_sha = sha_file(panel / "gold.jsonl")
    card = statistic(cand, ref, strata, kinds, CARD_TYPES, replicates, seed)
    return {
        "schema": SCHEMA,
        "scope": "mlx-diag multilingual diagnostic; paired candidate minus reference; aggregates only",
        "candidate_name": names[0],
        "reference_name": names[1],
        "gold_sha256": gold_sha,
        "frozen_panel": gold_sha == panels.DEVELOPMENT["mlx-diag"]["gold_sha256"],
        "predictions_sha256": {
            "candidate": sha_file(candidate_path),
            "reference": sha_file(reference_path),
        },
        "bootstrap": {
            "replicates": replicates,
            "seed": seed,
            "method": "paired item draws within type x language (v2.eval.overlap_effects.strata_bootstrap); "
            "statistic = mean over types of the mean over languages of accuracy",
        },
        "reproduces_frozen_score": checks,
        "card_eligible": card,
        "full": statistic(cand, ref, strata, kinds, MLX_TYPES, replicates, seed),
        "per_type": {
            kind: statistic(cand, ref, strata, kinds, (kind,), replicates, seed)
            for kind in MLX_TYPES
            if kind in kinds
        },
        "per_type_language": cells,
        "rule4_pass": card["ci95"]["high"] >= 0,
        "problems": [
            f"{side}: type macro differs from multilingual_panel.score"
            for side, check in checks.items()
            if not check["equal"]
        ],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--candidate-name", default="candidate")
    p.add_argument("--reference-name", default="reference")
    p.add_argument("--replicates", type=int, default=PAIRED_REPLICATES)
    p.add_argument("--seed", type=int, default=PAIRED_SEED)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare(
        args.panel,
        args.candidate,
        args.reference,
        args.replicates,
        args.seed,
        (args.candidate_name, args.reference_name),
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=1, sort_keys=True)
        stream.write("\n")
    card, full = result["card_eligible"], result["full"]
    print(
        json.dumps(
            {
                "card_eligible_delta": round(card["delta"], 4),
                "card_eligible_ci95": [
                    round(card["ci95"]["low"], 4),
                    round(card["ci95"]["high"], 4),
                ],
                "full_delta": round(full["delta"], 4),
                "full_ci95": [
                    round(full["ci95"]["low"], 4),
                    round(full["ci95"]["high"], 4),
                ],
                "rule4_pass": result["rule4_pass"],
                "frozen_panel": result["frozen_panel"],
                "problems": len(result["problems"]),
            }
        )
    )
    return 1 if result["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
