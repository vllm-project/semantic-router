"""Development readout for checkpoint selection (never a release score).

Inputs are native predictions on the frozen development panels:

* typed DEV (1,600 items) and the three-task CSS pilot (1,430 items): standard
  ``id``/``answers`` JSONL from the same native adapter the formal runner uses
  (``same_panel collect --panels typed-dev,css-pilot`` writes them to
  ``<run-dir>/output/``);
* SELECT and CAL (700 rows each, private training-dataset partitions): JSONL rows
  ``{"id": ..., "probabilities": [p_0, ...]}`` aligned with the row's ``options``.

SELECT/CAL use the trainers' definitions (unique-argmax correctness, multiclass
Brier divided by two, family-macro accuracy and Brier over the six families) with
one difference: missing or malformed rows count as wrong with Brier 1 instead of
being dropped. The development proxy is ``100*sqrt(T_dev*H_pilot)``; within a tier,
two checkpoints less than 8 proxy points apart are a tie that only the formal paired
v3 interval can decide (``v2/eval/records/m5-proxy-v2-calibration-2026-09-29.md``).

A run directory with ``output/score5-dev.predictions.jsonl`` also gets a ``score5`` block
(5-level Score level usage, accuracy with CI, macro-F1, QWK and COLLAPSE / WARN /
NO-SIGNAL flags; ``v2/eval/score5.py``). With ``output/ht-dev.predictions.jsonl`` it gets
``htdev_empathy_levels``: gold-free level usage on HT-DEV's 5-level empathy task, context
only (no flags).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from v2.eval import panels
from v2.eval.same_panel import read_jsonl, sha_file, source_hashes, utc_now, write_json

SCHEMA = "dev2-development-readout/1"
NLL_FLOOR = 1e-12


def select_row(
    row: dict[str, Any], prediction: dict[str, Any] | None
) -> dict[str, Any]:
    options, label = row["options"], row["label"]
    probabilities = (prediction or {}).get("probabilities")
    valid = (
        isinstance(probabilities, list)
        and len(probabilities) == len(options)
        and all(
            type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1
            for p in probabilities
        )
        and abs(sum(probabilities) - 1.0) <= 0.02
    )
    if not valid:
        return {
            "family": row["family"],
            "task_type": row["task_type"],
            "valid": False,
            "correct": False,
            "brier": 1.0,
            "nll": -math.log(NLL_FLOOR),
        }
    total = sum(probabilities)
    p = [value / total for value in probabilities]
    best = max(p)
    winners = [i for i, value in enumerate(p) if abs(value - best) <= 1e-8]
    chosen = winners[0] if len(winners) == 1 else None
    return {
        "family": row["family"],
        "task_type": row["task_type"],
        "valid": True,
        "correct": chosen == label,
        "brier": sum((value - float(i == label)) ** 2 for i, value in enumerate(p)) / 2,
        "nll": -math.log(max(p[label], NLL_FLOOR)),
    }


def select_summary(
    rows: list[dict[str, Any]], predictions_path: Path
) -> dict[str, Any]:
    predictions = {r["id"]: r for r in read_jsonl(predictions_path)}
    unknown = set(predictions) - {row["id"] for row in rows}
    if unknown:
        raise ValueError(f"{len(unknown)} prediction ids are not in the partition")
    records = [select_row(row, predictions.get(row["id"])) for row in rows]
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_family[record["family"]].append(record)
        by_type[record["task_type"]].append(record)

    def block(subset: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(subset),
            "correct": sum(r["correct"] for r in subset),
            "accuracy": sum(r["correct"] for r in subset) / len(subset),
            "brier": sum(r["brier"] for r in subset) / len(subset),
            "nll": sum(r["nll"] for r in subset) / len(subset),
        }

    families = {name: block(subset) for name, subset in sorted(by_family.items())}
    return {
        "predictions_sha256": sha_file(predictions_path),
        "n": len(records),
        "correct": sum(r["correct"] for r in records),
        "invalid_or_missing": sum(not r["valid"] for r in records),
        "micro_accuracy": sum(r["correct"] for r in records) / len(records),
        "family_macro_accuracy": sum(v["accuracy"] for v in families.values())
        / len(families),
        "family_macro_brier": sum(v["brier"] for v in families.values())
        / len(families),
        "by_family": families,
        "by_type": {name: block(subset) for name, subset in sorted(by_type.items())},
    }


def score5_block(panel_root: Path, predictions_path: Path) -> dict[str, Any]:
    from v2.eval import score5

    panels.verify(panel_root, ["score5-dev"])
    gold = read_jsonl(panels.path(panel_root, "score5-dev", "gold"))
    predictions = {r["id"]: r for r in read_jsonl(predictions_path)}
    return {
        "predictions_sha256": sha_file(predictions_path),
        **score5.summary(gold, predictions),
    }


def htdev_empathy_levels(panel_root: Path, predictions_path: Path) -> dict[str, Any]:
    """Level usage on HT-DEV's empathy task from prompts and predictions only."""
    from benchmark.score import evaluate_answer
    from v2.eval import score5
    from v2.eval.htdev.sources.empathic_reactions import QUESTION, TASK

    panels.verify(panel_root, ["ht-dev"])
    items = {
        row["id"]: row["questions"]["decision"]
        for row in read_jsonl(panels.path(panel_root, "ht-dev", "prompts"))
        if row["questions"]["decision"] == QUESTION
    }
    predictions = {r["id"]: r for r in read_jsonl(predictions_path)}
    points = []
    for item_id, question in items.items():
        answer = ((predictions.get(item_id) or {}).get("answers") or {}).get("decision")
        # A placeholder gold: only the gold-independent `point` is read.
        result = evaluate_answer(question, {"value": 0}, answer)
        points.append(result.get("point") if result.get("status") == "ok" else None)
    return {
        "task": TASK,
        "scope": "context only (gold-free level usage; no flags)",
        "n": len(points),
        **score5.level_usage(points),
    }


def readout(args: argparse.Namespace) -> dict[str, Any]:
    from benchmark.score import score_suite
    from transfer.score import score as score_css

    out: dict[str, Any] = {
        "schema": SCHEMA,
        "scope": "development readout for checkpoint selection; never a release or formal score",
        "label": args.label,
        "created_utc": utc_now(),
        "sources": source_hashes(),
    }
    output_dir = args.run_dir / "output" if args.run_dir else None
    typed_path = args.typed_dev or (
        output_dir / "typed-dev.predictions.jsonl" if output_dir else None
    )
    css_path = args.css_pilot or (
        output_dir / "css-pilot.predictions.jsonl" if output_dir else None
    )
    if typed_path and typed_path.is_file():
        panels.verify(args.panel_root, ["typed-dev"])
        typed = score_suite(
            panels.path(args.panel_root, "typed-dev", "gold"),
            typed_path,
            args.label,
            "development",
            "native",
        )
        out["typed_dev"] = {
            "predictions_sha256": typed["predictions_sha256"],
            "T_dev": typed["macro_family_accuracy"],
            "by_type": {
                k: {"correct": v["correct_n"], "n": v["n"]}
                for k, v in typed["by_type"].items()
            },
            "by_family": {k: v["accuracy_all"] for k, v in typed["by_family"].items()},
            "invalid_or_missing": typed["overall"]["invalid_or_missing_n"],
            "brier": typed["overall"]["brier"],
            "ece_10": typed["overall"]["ece_10"],
            "order_invariance": typed["pairs"]["order_invariance"][
                "relation_consistency_all"
            ],
        }
    if css_path and css_path.is_file():
        panels.verify(args.panel_root, ["css-pilot"])
        css = score_css(panels.path(args.panel_root, "css-pilot", "gold"), css_path)
        pilot = css["roles"]["pilot"]
        out["css_pilot"] = {
            "predictions_sha256": css["predictions_sha256"],
            "H_pilot": pilot["median_task_macro_f1_all"],
            "micro_accuracy": pilot["micro_accuracy_all"],
            "invalid_or_missing": pilot["items"] - pilot["valid_items"],
            "tasks": {
                k: v["macro_f1_all"]
                for k, v in css["tasks"].items()
                if v["role"] == "pilot"
            },
        }
    if "typed_dev" in out and "css_pilot" in out:
        t, h = out["typed_dev"]["T_dev"], out["css_pilot"]["H_pilot"]
        out["development_proxy"] = 100 * math.sqrt(t * h)
    score5_path = args.score5 or (
        output_dir / "score5-dev.predictions.jsonl" if output_dir else None
    )
    if score5_path and score5_path.is_file():
        out["score5"] = score5_block(args.panel_root, score5_path)
    htdev_path = output_dir / "ht-dev.predictions.jsonl" if output_dir else None
    if htdev_path and htdev_path.is_file():
        out["htdev_empathy_levels"] = htdev_empathy_levels(args.panel_root, htdev_path)
    for name, path in (("select", args.select), ("cal", args.cal)):
        if path is not None:
            panels.verify(args.panel_root, [name])
            rows = read_jsonl(panels.path(args.panel_root, name, "gold"))
            out[name] = select_summary(rows, path)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--panel-root", type=Path, default=panels.DEFAULT_ROOT)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="reads output/typed-dev and output/css-pilot predictions",
    )
    parser.add_argument("--typed-dev", type=Path)
    parser.add_argument("--css-pilot", type=Path)
    parser.add_argument("--score5", type=Path)
    parser.add_argument("--select", type=Path)
    parser.add_argument("--cal", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = readout(args)
    write_json(args.output, result)
    summary = {k: result.get(k) for k in ("development_proxy",)}
    for key, field in (
        ("typed_dev", "T_dev"),
        ("css_pilot", "H_pilot"),
        ("select", "correct"),
        ("select", "family_macro_accuracy"),
    ):
        if key in result:
            summary[f"{key}.{field}"] = result[key][field]
    if "score5" in result:
        for field in ("accuracy", "modal_share", "rare_levels", "flags"):
            summary[f"score5.{field}"] = result["score5"][field]
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
