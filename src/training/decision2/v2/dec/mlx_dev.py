"""Decoder Milestone 5 MLX-DEV panel: build it from the ``m5_block`` selection, score readouts.

``build`` copies the selected superset rows with their training rendering and
gold and only ``split`` / ``evaluation_role`` set to ``select`` (the input hash
covers neither), so ``v2.dec.eval_rows`` reads the panel exactly as it reads
SELECT700 (raw probabilities; Noul "yes" when p(true) > 0.5, Choice / Score by
argmax). An index file maps each row to its cell.

``score`` reads one ``eval_rows`` predictions file: **Noul-ML** = balanced
accuracy (mean of recall on gold-yes and gold-No) per language over both Noul
cells, macro over languages, with predicted-yes rate and gold-No recall;
**Choice-ML** / **Score-ML** = accuracy macro over the panel's source cells
(Choice: JCommonsenseQA, MTOP; Score: A7q, A7k, A7s, H8 Spanglish), Score also
within one level; **M_dev** = mean of the three. Unanswered rows count wrong.

``compare`` is the paired group-clustered bootstrap of B - A (groups resampled
with replacement within each cell, identically for both files; 2,000
replicates, seed 20260929), reporting the point difference and the 2.5 / 97.5
percentiles for every headline metric.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from training.model.data import file_sha256, load_partition

from .m5_block import MLX_CELLS, MLX_PREFIX, component_slices, read_lines

REPS = 2000
SEED = 20260929
NOUL_CELLS = tuple(c for c, spec in MLX_CELLS.items() if spec[1] == "noul")
CHOICE_CELLS = tuple(c for c, spec in MLX_CELLS.items() if spec[1] == "choice")
SCORE_CELLS = tuple(c for c, spec in MLX_CELLS.items() if spec[1] == "score")


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    selection = json.loads(args.selection.read_text())
    if selection["prefix"] != MLX_PREFIX:
        raise ValueError("selection was made with another hash prefix")
    superset = read_lines(args.superset)
    manifest = json.loads(
        args.superset.with_name(args.superset.name + ".manifest.json").read_text()
    )
    components = dict(component_slices(superset, manifest))
    out_rows, index = [], []
    for cell in MLX_CELLS:
        component = MLX_CELLS[cell][0]
        wanted = set(selection["cells"][cell])
        for _, row in components[component]:
            if row["group_id"] in wanted:
                out_rows.append(dict(row, split="select", evaluation_role="select"))
                index.append(
                    {
                        "id": row["id"],
                        "cell": cell,
                        "group_id": row["group_id"],
                        "language": row["language"],
                        "source": row["source"],
                        "task_type": row["task_type"],
                        "gold_key": row["options"][row["label"]]["key"],
                    }
                )
        found = {r["group_id"] for r in index if r["cell"] == cell}
        if found != wanted:
            raise ValueError(
                f"{cell}: {len(wanted - found)} selected groups missing from the superset"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for row in out_rows:
            stream.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    load_partition(args.output, "select")
    index_path = args.output.with_name(args.output.name + ".index.jsonl")
    with index_path.open("x", encoding="utf-8") as stream:
        for entry in index:
            stream.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "schema_version": "dec-m5-mlxdev/1",
        "selection_sha256": file_sha256(args.selection),
        "superset_sha256": file_sha256(args.superset),
        "rows": len(out_rows),
        "groups": len({r["group_id"] for r in index}),
        "cells": {
            cell: {
                "rows": sum(r["cell"] == cell for r in index),
                "groups": len(selection["cells"][cell]),
                "rows_by_language": dict(
                    sorted(
                        Counter(
                            r["language"] for r in index if r["cell"] == cell
                        ).items()
                    )
                ),
                "gold_by_key": dict(
                    sorted(
                        Counter(
                            r["gold_key"] for r in index if r["cell"] == cell
                        ).items()
                    )
                ),
            }
            for cell in MLX_CELLS
        },
        "output_sha256": file_sha256(args.output),
        "index_sha256": file_sha256(index_path),
    }
    args.output.with_name(args.output.name + ".manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def read_index(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def outcomes(index: list[dict[str, Any]], predictions: Path) -> list[dict[str, Any]]:
    """Per panel row: its index entry plus predicted key (None if unanswered)."""
    predicted: dict[str, Any] = {}
    with predictions.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            predicted[record["id"]] = record.get("prediction_key")
    out = []
    for entry in index:
        out.append(dict(entry, predicted=predicted.get(entry["id"])))
    return out


def row_stats(row: dict[str, Any]) -> dict[tuple[str, ...], int]:
    """Additive counts contributed by one row (the bootstrap sums these)."""
    stats: dict[tuple[str, ...], int] = {}
    predicted, gold = row["predicted"], row["gold_key"]
    if row["cell"] in NOUL_CELLS:
        side = "yes" if gold == "true" else "no"
        lang = row["language"]
        stats[("noul", lang, side, "n")] = 1
        stats[("noul", lang, side, "correct")] = int(predicted == gold)
        stats[("noul", lang, "pred_yes")] = int(predicted == "true")
        stats[("noul_src", row["source"], side, "n")] = 1
        stats[("noul_src", row["source"], side, "correct")] = int(predicted == gold)
        stats[("noul_src", row["source"], "pred_yes")] = int(predicted == "true")
    else:
        cell = row["cell"]
        stats[("cell", cell, "n")] = 1
        stats[("cell", cell, "correct")] = int(predicted == gold)
        if row["cell"] in SCORE_CELLS:
            within = predicted is not None and abs(int(predicted) - int(gold)) <= 1
            stats[("cell", cell, "within1")] = int(within)
        stats[("src", row["source"], "n")] = 1
        stats[("src", row["source"], "correct")] = int(predicted == gold)
    return stats


def _noul(
    totals: dict[tuple[str, ...], float], kind: str
) -> dict[str, dict[str, float]]:
    out = {}
    keys = {k[1] for k in totals if k[0] == kind}
    for key in sorted(keys):
        n_yes, n_no = totals.get((kind, key, "yes", "n"), 0), totals.get(
            (kind, key, "no", "n"), 0
        )
        recalls = []
        entry: dict[str, float] = {
            "n": n_yes + n_no,
            "gold_yes": n_yes,
            "gold_no": n_no,
        }
        if n_yes:
            entry["gold_yes_recall"] = totals[(kind, key, "yes", "correct")] / n_yes
            recalls.append(entry["gold_yes_recall"])
        if n_no:
            entry["gold_no_recall"] = totals[(kind, key, "no", "correct")] / n_no
            recalls.append(entry["gold_no_recall"])
        entry["balanced_accuracy"] = statistics.mean(recalls)
        entry["pred_yes_rate"] = totals.get((kind, key, "pred_yes"), 0) / (n_yes + n_no)
        out[key] = entry
    return out


def metrics(
    totals: dict[tuple[str, ...], float], detail: bool = True
) -> dict[str, Any]:
    by_language = _noul(totals, "noul")
    noul_ml = statistics.mean(v["balanced_accuracy"] for v in by_language.values())

    def cell_acc(cell: str, what: str = "correct") -> float:
        return totals.get(("cell", cell, what), 0) / totals[("cell", cell, "n")]

    choice = {c: cell_acc(c) for c in CHOICE_CELLS}
    score = {c: cell_acc(c) for c in SCORE_CELLS}
    within = {c: cell_acc(c, "within1") for c in SCORE_CELLS}
    out: dict[str, Any] = {
        "noul_ml": noul_ml,
        "choice_ml": statistics.mean(choice.values()),
        "score_ml": statistics.mean(score.values()),
        "score_ml_within1": statistics.mean(within.values()),
        "noul_pred_yes_rate_macro": statistics.mean(
            v["pred_yes_rate"] for v in by_language.values()
        ),
        "noul_gold_no_recall_macro": statistics.mean(
            v["gold_no_recall"] for v in by_language.values() if "gold_no_recall" in v
        ),
    }
    out["m_dev"] = (out["noul_ml"] + out["choice_ml"] + out["score_ml"]) / 3
    if detail:
        out["noul_by_language"] = by_language
        out["noul_by_source"] = _noul(totals, "noul_src")
        out["choice_by_cell"] = choice
        out["score_by_cell"] = score
        out["score_within1_by_cell"] = within
        out["by_source"] = {
            k[1]: totals[("src", k[1], "correct")] / totals[("src", k[1], "n")]
            for k in sorted(totals)
            if k[0] == "src" and k[2] == "n"
        }
    return out


def sum_stats(rows: list[dict[str, Any]]) -> dict[tuple[str, ...], float]:
    totals: dict[tuple[str, ...], float] = defaultdict(float)
    for row in rows:
        for key, value in row_stats(row).items():
            totals[key] += value
    return totals


HEADLINE = (
    "noul_ml",
    "choice_ml",
    "score_ml",
    "score_ml_within1",
    "m_dev",
    "noul_pred_yes_rate_macro",
    "noul_gold_no_recall_macro",
)


def paired_bootstrap(
    a: list[dict[str, Any]], b: list[dict[str, Any]], reps: int = REPS, seed: int = SEED
) -> dict[str, Any]:
    if [r["id"] for r in a] != [r["id"] for r in b]:
        raise ValueError("compared readouts cover different panel rows")
    groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(a):
        groups[row["cell"]][row["group_id"]].append(i)
    stats_a = [row_stats(r) for r in a]
    stats_b = [row_stats(r) for r in b]
    group_stats = {}
    for cell, members in groups.items():
        for g, idx in members.items():
            sa: dict[tuple[str, ...], int] = defaultdict(int)
            sb: dict[tuple[str, ...], int] = defaultdict(int)
            for i in idx:
                for k, v in stats_a[i].items():
                    sa[k] += v
                for k, v in stats_b[i].items():
                    sb[k] += v
            group_stats[(cell, g)] = (dict(sa), dict(sb))
    point_a, point_b = metrics(sum_stats(a), False), metrics(sum_stats(b), False)
    rng = random.Random(seed)
    cells = sorted(groups)
    draws: dict[str, list[float]] = {k: [] for k in HEADLINE}
    for _ in range(reps):
        ta: dict[tuple[str, ...], float] = defaultdict(float)
        tb: dict[tuple[str, ...], float] = defaultdict(float)
        for cell in cells:
            names = sorted(groups[cell])
            for g, w in Counter(rng.choices(names, k=len(names))).items():
                sa, sb = group_stats[(cell, g)]
                for k, v in sa.items():
                    ta[k] += w * v
                for k, v in sb.items():
                    tb[k] += w * v
        try:
            ma, mb = metrics(ta, False), metrics(tb, False)
        except (ZeroDivisionError, statistics.StatisticsError, KeyError):
            continue
        for k in HEADLINE:
            draws[k].append(mb[k] - ma[k])
    out = {}
    for k in HEADLINE:
        values = sorted(draws[k])
        n = len(values)
        out[k] = {
            "a": point_a[k],
            "b": point_b[k],
            "diff": point_b[k] - point_a[k],
            "ci95": (
                [values[int(0.025 * (n - 1))], values[int(round(0.975 * (n - 1)))]]
                if n
                else None
            ),
            "replicates": n,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    b = commands.add_parser("build")
    b.add_argument("--superset", type=Path, required=True)
    b.add_argument("--selection", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    s = commands.add_parser("score")
    s.add_argument("--index", type=Path, required=True)
    s.add_argument("--predictions", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    c = commands.add_parser("compare")
    c.add_argument("--index", type=Path, required=True)
    c.add_argument("--a", type=Path, required=True, help="reference predictions (A)")
    c.add_argument(
        "--b", type=Path, required=True, help="compared predictions (B); reports B - A"
    )
    c.add_argument("--reps", type=int, default=REPS)
    c.add_argument("--seed", type=int, default=SEED)
    c.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        report = build(args)
        print(json.dumps({k: report[k] for k in ("rows", "groups", "output_sha256")}))
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    index = read_index(args.index)
    if args.command == "score":
        rows = outcomes(index, args.predictions)
        result = metrics(sum_stats(rows))
        result["unanswered"] = sum(r["predicted"] is None for r in rows)
        result["predictions_sha256"] = file_sha256(args.predictions)
    else:
        result = paired_bootstrap(
            outcomes(index, args.a), outcomes(index, args.b), args.reps, args.seed
        )
        result = {
            "metrics": result,
            "a_sha256": file_sha256(args.a),
            "b_sha256": file_sha256(args.b),
            "reps": args.reps,
            "seed": args.seed,
            "clusters": "group_id within MLX-DEV cell",
        }
    result["index_sha256"] = file_sha256(args.index)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {k: result.get(k) for k in ("noul_ml", "choice_ml", "score_ml", "m_dev")}
            if args.command == "score"
            else {k: v["diff"] for k, v in result["metrics"].items()}
        )
    )


if __name__ == "__main__":
    main()
