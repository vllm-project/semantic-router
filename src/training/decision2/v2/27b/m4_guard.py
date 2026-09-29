"""Milestone 4 soup-readout guard for the ~27B track (host CPU, stdlib).

Inputs: the ``READOUT.json`` of every M4 soup's kernel-path readout
(``run_finalist.sh STAGES=readout``: 32,768 tokens, a fresh copy of the frozen
readout cache) and F1's, read on the same path, limit and cache. Next to each
``READOUT.json`` the guard reads ``output/typed-dev.predictions.jsonl``,
``output/css-pilot.predictions.jsonl``, ``cal698.summary.json`` and
``cal/cal.probs.jsonl``. Rules of the preregistration
(``records/m4-prereg-2026-09-29.md``, "Soups, readouts and finalists"):

* collapse check: a soup is not a finalist if a typed-DEV type is at or below
  chance (Choice .267, Noul .50, Score .20, the formal gate's levels), if one
  answer category takes a whole type (semantic values for Choice, levels for
  Score, true / false for Noul, as ``v2.eval.gates types`` counts them), or if
  invalid answers exceed 1% on typed DEV or on the CSS pilot;
* proxy drop (proxy v2): a soup is dropped if its P_dev = 100*sqrt(T_dev*H_pilot)
  is at least 8 below the best of F1's P_dev and the M4 soups';
* every other soup is a finalist;
* retention against F1 is report-only: a typed-DEV type down more than 3 points,
  H_pilot down more than 1.5, CAL698 Brier worse by more than 0.010, or more
  invalid answers.

Per-item outcomes come from the canonical scorers (``benchmark.score``'s
``evaluate_answer`` under ``score_suite``'s rule that an answer object holds
exactly the item's question ids; ``transfer.score.evaluate``). The recomputed
per-type counts, invalid answers, T_dev, H_pilot and P_dev must equal the values
in ``READOUT.json``. Per-seed SELECT700 at BEST (``--seed-run``, the trainer's
own metrics) is reported only. Development readouts: never release or formal
scores.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from v2.eval import panels as panel_registry

contrast = importlib.import_module("v2.27b.contrast")

SCHEMA = "decision2-27b-m4-guard/1"
TYPES = contrast.TYPES
CHANCE = {"choice": 0.267, "noul": 0.50, "score": 0.20}
INVALID_MAX = 0.01
PROXY_DROP = 8.0
TYPE_DROP = 0.03
H_PILOT_DROP = 0.015
CAL_BRIER_WORSE = 0.010
TOLERANCE = 1e-9
EPSILON = 1e-12


class Inputs:
    """SHA-256 of every file read, keyed by path."""

    def __init__(self) -> None:
        self.files: dict[str, str] = {}

    def sha(self, path: Path) -> str:
        key = str(path)
        if key not in self.files:
            self.files[key] = panel_registry.sha_file(path)
        return self.files[key]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def agree(name: str, got: float, expected: float) -> None:
    if not math.isclose(got, expected, rel_tol=0, abs_tol=TOLERANCE):
        raise ValueError(
            f"{name}: recomputed {got!r} differs from recorded {expected!r}"
        )


def typed_cells(dev: list[dict[str, Any]], predictions: Path) -> dict[str, Any]:
    """Per type: slots, correct, valid, undecided and answer categories; per family accuracy."""
    from benchmark.score import evaluate_answer

    by_id = {row["id"]: row for row in contrast.read_jsonl(predictions)}
    cells: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "n": 0,
            "correct": 0,
            "valid": 0,
            "undecided": 0,
            "chance_sum": 0.0,
            "categories": Counter(),
        }
    )
    families: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for item in dev:
        prediction = by_id.get(item["id"])
        answers = prediction.get("answers") if isinstance(prediction, dict) else None
        exact = isinstance(answers, dict) and set(answers) == set(item["questions"])
        for key, question in item["questions"].items():
            cell = cells[question["type"]]
            cell["n"] += 1
            options = 2 if question["type"] == "noul" else len(question["criteria"])
            cell["chance_sum"] += 1.0 / options
            families[item["family"]][1] += 1
            if not exact:
                continue
            result = evaluate_answer(question, item["gold"][key], answers[key])
            if result.get("status") != "ok":
                continue
            cell["valid"] += 1
            if result.get("correct"):
                cell["correct"] += 1
                families[item["family"]][0] += 1
            point = result.get("semantic_point")
            if point is None:
                cell["undecided"] += 1
            else:
                cell["categories"][str(point)] += 1
    return {"cells": cells, "families": families}


class Readout:
    """One kernel-path readout directory, recomputed and checked against its READOUT.json."""

    def __init__(self, name: str, path: Path, panels: Any, inputs: Inputs) -> None:
        self.name = name
        self.path = path.resolve()
        directory = self.path.parent
        record = read_json(self.path)
        inputs.sha(self.path)
        typed = directory / "output" / "typed-dev.predictions.jsonl"
        css = directory / "output" / "css-pilot.predictions.jsonl"
        for key, prediction in (("typed_dev", typed), ("css_pilot", css)):
            if record[key]["predictions_sha256"] != inputs.sha(prediction):
                raise ValueError(f"{name}: READOUT.json {key} scored other predictions")
        outcome = typed_cells(panels.dev, typed)
        cells, families = outcome["cells"], outcome["families"]
        by_type = {}
        for kind in TYPES:
            cell = cells.get(kind) or {
                "n": 0,
                "correct": 0,
                "valid": 0,
                "undecided": 0,
                "chance_sum": 0.0,
                "categories": Counter(),
            }
            recorded = record["typed_dev"]["by_type"].get(kind, {"correct": 0, "n": 0})
            if (cell["correct"], cell["n"]) != (recorded["correct"], recorded["n"]):
                raise ValueError(f"{name}: typed-DEV {kind} differs from READOUT.json")
            categories = dict(cell["categories"].most_common())
            top = max(categories.values()) if categories else 0
            by_type[kind] = {
                "n": cell["n"],
                "correct": cell["correct"],
                "accuracy": cell["correct"] / cell["n"] if cell["n"] else 0.0,
                "chance": CHANCE[kind],
                "panel_chance_report_only": (
                    cell["chance_sum"] / cell["n"] if cell["n"] else None
                ),
                "valid": cell["valid"],
                "invalid": cell["n"] - cell["valid"],
                "undecided": cell["undecided"],
                "answer_categories": categories,
                "distinct_categories": len(categories),
                "top_share": top / cell["n"] if cell["n"] else 0.0,
            }
        slots = sum(c["n"] for c in by_type.values())
        typed_invalid = sum(c["invalid"] for c in by_type.values())
        if typed_invalid != record["typed_dev"]["invalid_or_missing"]:
            raise ValueError(
                f"{name}: typed-DEV invalid answers differ from READOUT.json"
            )
        t_dev = statistics.fmean(c / n for c, n in families.values())
        agree(f"{name} T_dev", t_dev, record["typed_dev"]["T_dev"])

        choices = panels.css_choices(css)
        css_point = contrast.css_macro(panels, choices, panels.task_items)
        agree(f"{name} H_pilot", css_point["H_pilot"], record["css_pilot"]["H_pilot"])
        css_invalid = sum(c is None for c in choices)
        if css_invalid != record["css_pilot"]["invalid_or_missing"]:
            raise ValueError(
                f"{name}: CSS-pilot invalid answers differ from READOUT.json"
            )
        p_dev = 100 * math.sqrt(t_dev * css_point["H_pilot"])
        agree(f"{name} P_dev", p_dev, record["development_proxy"])
        tasks = {}
        for task, items in panels.task_items.items():
            predicted = Counter(
                str(choices[i]) for i in items if choices[i] is not None
            )
            tasks[task] = {
                "macro_f1": css_point[f"css:{task}"],
                "predicted": dict(predicted.most_common()),
            }

        summary_path = directory / "cal698.summary.json"
        summary = read_json(summary_path)
        inputs.sha(summary_path)
        if summary["predictions_sha256"] != inputs.sha(
            directory / "cal" / "cal.probs.jsonl"
        ):
            raise ValueError(f"{name}: cal698.summary.json scored other probabilities")
        cal_types = summary["by_type"].values()
        self.values = {
            "readout": str(self.path),
            "label": record.get("label"),
            "P_dev": p_dev,
            "T_dev": t_dev,
            "H_pilot": css_point["H_pilot"],
            "typed_dev": {
                "slots": slots,
                "invalid": typed_invalid,
                "invalid_rate": typed_invalid / slots if slots else 1.0,
                "by_type": by_type,
                "by_family": {f: c / n for f, (c, n) in sorted(families.items())},
            },
            "css_pilot": {
                "items": len(choices),
                "invalid": css_invalid,
                "invalid_rate": css_invalid / len(choices) if choices else 1.0,
                "tasks": tasks,
            },
            "cal698": {
                "n": summary["n"],
                "brier": sum(b["brier"] * b["n"] for b in cal_types)
                / sum(b["n"] for b in cal_types),
                "family_macro_brier": summary["family_macro_brier"],
                "family_macro_accuracy": summary["family_macro_accuracy"],
            },
            "invalid_total": typed_invalid + css_invalid,
        }


def collapse(values: dict[str, Any]) -> list[str]:
    flags = []
    for kind in TYPES:
        cell = values["typed_dev"]["by_type"][kind]
        if cell["accuracy"] <= CHANCE[kind]:
            flags.append(
                f"typed-DEV {kind} accuracy {cell['accuracy']:.4f} is at or below chance "
                f"{CHANCE[kind]}"
            )
        if cell["distinct_categories"] <= 1:
            flags.append(
                f"typed-DEV {kind} uses {cell['distinct_categories']} answer category "
                f"({cell['answer_categories']}) for the whole type"
            )
    for panel in ("typed_dev", "css_pilot"):
        rate = values[panel]["invalid_rate"]
        if rate > INVALID_MAX:
            flags.append(f"{panel} invalid answers {rate:.4f} exceed {INVALID_MAX}")
    return flags


def retention(values: dict[str, Any], incumbent: dict[str, Any]) -> dict[str, Any]:
    delta = {
        f"{kind}_accuracy": values["typed_dev"]["by_type"][kind]["accuracy"]
        - incumbent["typed_dev"]["by_type"][kind]["accuracy"]
        for kind in TYPES
    }
    delta["H_pilot"] = values["H_pilot"] - incumbent["H_pilot"]
    delta["cal_brier"] = values["cal698"]["brier"] - incumbent["cal698"]["brier"]
    delta["invalid"] = values["invalid_total"] - incumbent["invalid_total"]
    checks = {
        "no_type_drop_over_3_points": all(
            delta[f"{kind}_accuracy"] >= -TYPE_DROP - EPSILON for kind in TYPES
        ),
        "h_pilot_drop_at_most_1.5_points": delta["H_pilot"] >= -H_PILOT_DROP - EPSILON,
        "cal_brier_worse_at_most_0.010": delta["cal_brier"]
        <= CAL_BRIER_WORSE + EPSILON,
        "invalid_not_increased": delta["invalid"] <= 0,
    }
    return {
        "report_only": True,
        "delta": delta,
        "checks": checks,
        "all_pass": all(checks.values()),
    }


def decide(
    incumbent_name: str, incumbent: dict[str, Any], values: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Collapse check, proxy drop against the best of the incumbent and the soups, finalists."""
    pool = {
        incumbent_name: incumbent["P_dev"],
        **{n: v["P_dev"] for n, v in values.items()},
    }
    best = max(pool, key=lambda n: pool[n])
    candidates = {}
    for name, value in values.items():
        flags = collapse(value)
        gap = pool[best] - value["P_dev"]
        dropped = gap >= PROXY_DROP
        candidates[name] = {
            **value,
            "collapse": {"flags": flags, "collapsed": bool(flags)},
            "proxy": {"best": best, "gap": gap, "dropped": dropped},
            "retention_vs_incumbent": retention(value, incumbent),
            "finalist": not flags and not dropped,
        }
    return {
        "proxy_pool": {"P_dev": pool, "best": best, "best_P_dev": pool[best]},
        "candidates": candidates,
        "finalists": [n for n, c in candidates.items() if c["finalist"]],
    }


def trainer_select(run_dir: Path, inputs: Inputs) -> dict[str, Any]:
    best_path = run_dir / "BEST.json"
    best = read_json(best_path)["checkpoint"]
    inputs.sha(best_path)
    step = re.fullmatch(r"checkpoint-([0-9]{7})", best)
    if step is None:
        raise ValueError(f"{run_dir}: unexpected BEST checkpoint {best}")
    metrics_path = run_dir / f"select-step-{step.group(1)}-metrics.json"
    metrics = read_json(metrics_path)
    inputs.sha(metrics_path)
    keys = ("n", "correct", "family_macro_accuracy", "family_macro_brier")
    return {"run": str(run_dir), "best": best, **{k: metrics[k] for k in keys}}


def pairs(specs: list[str]) -> dict[str, Path]:
    out = {}
    for spec in specs:
        name, _, path = spec.partition("=")
        if not name or not path or name in out:
            raise ValueError(f"expected distinct NAME=PATH, got {spec!r}")
        out[name] = Path(path)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    inputs = Inputs()
    if args.dev_gold or args.css_gold:
        dev_gold, css_gold = args.dev_gold, args.css_gold
    else:
        panel_registry.verify(args.panel_root, ["typed-dev", "css-pilot"])
        dev_gold = panel_registry.path(args.panel_root, "typed-dev", "gold")
        css_gold = panel_registry.path(args.panel_root, "css-pilot", "gold")
    inputs.sha(dev_gold)
    inputs.sha(css_gold)
    panels = contrast.Panels(dev_gold, css_gold)
    ((incumbent_name, incumbent_path),) = pairs([args.incumbent]).items()
    candidates = pairs(args.candidate)
    if incumbent_name in candidates or set(args.absent) & set(candidates):
        raise ValueError("candidate, incumbent and absent names must differ")
    incumbent = Readout(incumbent_name, incumbent_path, panels, inputs).values
    values = {n: Readout(n, p, panels, inputs).values for n, p in candidates.items()}
    decision = decide(incumbent_name, incumbent, values)
    seeds = {n: trainer_select(p, inputs) for n, p in pairs(args.seed_run).items()}
    for source in (globals().get("__file__"), contrast.__file__):
        if source and Path(source).is_file():
            inputs.sha(Path(source))
    return {
        "schema": SCHEMA,
        "label": "development readout guard (kernel path); never a release or formal score",
        "rules": {
            "chance": CHANCE,
            "collapse": "a typed-DEV type at or below chance, one answer category for a whole "
            f"type, or invalid answers above {INVALID_MAX:.0%} on typed DEV or the CSS pilot",
            "proxy_drop": f"P_dev at least {PROXY_DROP} below the best of the incumbent and the soups",
            "retention_report_only": {
                "type_drop": TYPE_DROP,
                "h_pilot_drop": H_PILOT_DROP,
                "cal_brier_worse": CAL_BRIER_WORSE,
                "invalid": "not increased",
            },
        },
        "incumbent": {"name": incumbent_name, **incumbent},
        **decision,
        "absent": sorted(args.absent),
        "seeds_select700_at_best": seeds,
        "inputs_sha256": dict(sorted(inputs.files.items())),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--panel-root", type=Path, default=panel_registry.DEFAULT_ROOT)
    parser.add_argument("--dev-gold", type=Path, help="override (no panel check)")
    parser.add_argument("--css-gold", type=Path, help="override (no panel check)")
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="NAME=READOUT.json of a soup",
    )
    parser.add_argument("--incumbent", required=True, help="NAME=READOUT.json of F1")
    parser.add_argument(
        "--seed-run",
        action="append",
        default=[],
        help="NAME=TRAINER_RUN_DIR (report only)",
    )
    parser.add_argument(
        "--absent", action="append", default=[], help="an arm with no soup (recorded)"
    )
    parser.add_argument("--output", type=Path, help="written once; default stdout")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    result = run(args)
    text = json.dumps(result, indent=1, sort_keys=True) + "\n"
    summary = sys.stdout
    if args.output is None:
        print(text, end="", flush=True)
        summary = sys.stderr
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    for name, entry in result["candidates"].items():
        line = {
            "candidate": name,
            "P_dev": round(entry["P_dev"], 3),
            "gap": round(entry["proxy"]["gap"], 3),
            "collapse": entry["collapse"]["flags"],
            "retention_all_pass": entry["retention_vs_incumbent"]["all_pass"],
            "finalist": entry["finalist"],
        }
        print(json.dumps(line), file=summary, flush=True)
    print(json.dumps({"finalists": result["finalists"]}), file=summary, flush=True)


if __name__ == "__main__":
    main()
