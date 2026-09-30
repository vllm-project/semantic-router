"""Milestone 4 typed-FINAL contrasts, successor rule and attribution (~27B track; host CPU, stdlib).

Inputs are sealed post-key same-panel formal runs (``same_panel`` run directories whose
``SEAL.json`` covers their typed-FINAL predictions) and the frozen typed-FINAL gold,
read from the panel root as ``v2.eval.gates`` reads it. Preregistration
``records/m4-prereg-2026-09-29.md``, "Comparisons, successor rule and attribution":

* point accuracy per family (constraint_competition, evidence_join, exception_stack,
  resource_ledger), per type (choice, noul, score) and T, checked against each run's
  ``REPORT.json``; an answer counts only if its answer object holds exactly the item's
  question ids (``benchmark.score.score_suite``'s rule);
* for every LEFT:RIGHT pair, the paired bootstrap of LEFT minus RIGHT (5,000 draws,
  seed 20260927, 95% percentile intervals of ``transfer.compare.interval95``). ``ci95``
  resamples items within family, the same items for both runs (the preregistered
  item level). ``ci95_group`` resamples the panel's independent four-variant groups
  within family, the unit of the v3 T interval (``jev_arena.compare_v3``): the four
  variants of a group are not independent, so the item-level interval is too narrow;
* Score level usage of every finalist (``v2.eval.gates.type_summary``: predicted vs gold
  distribution, recall by level, the ``types`` verdicts);
* with ``--gates``: the successor rule (v3 minus F1 lower bound above 0, human transfer
  not significantly below F1, v3 >= 64.92, human transfer not significantly below the
  three peers, every type ``OK`` in ``v2.eval.gates types``, no overlap exposure), its
  tie-break, and the attribution claims (dose = M4-A20 minus M4-Ar, capacity = M4-A20r
  minus M4-A20: claimed only if the lower bound of Delta T is above 0), from the files
  that ``m4/m4-tail.sh gates`` writes and the ``v2.eval.overlap_effects exposure``
  receipts.

Aggregates only: no item ids, gold values or answers.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmark.generate import FINAL_FAMILIES
from transfer.compare import interval95
from v2.eval import panels as panel_registry
from v2.eval.gates import type_summary, verified
from v2.eval.same_panel import PAIRED_REPLICATES, PAIRED_SEED, read_jsonl

SCHEMA = "decision2-27b-m4-contrast/1"
TYPES = ("choice", "noul", "score")
UNITS = ("item", "group")
TOLERANCE = 1e-9
V3_FLOOR = 64.92
PEERS = {
    "autojev27": "AutoJev-27B",
    "eikos27b": "Eikos-27B",
    "jebadiah27b": "Jebadiah-27B",
}
CLAIMS = {
    "dose": ("contrast-dose.json", "new A7 content beats repeating M3-A's rows"),
    "capacity": ("contrast-capacity.json", "rank 32 adds typed reasoning at this dose"),
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def agree(name: str, got: float, expected: float) -> None:
    if not math.isclose(got, expected, rel_tol=0, abs_tol=TOLERANCE):
        raise ValueError(
            f"{name}: recomputed {got!r} differs from recorded {expected!r}"
        )


def outcomes(
    gold: dict[str, dict[str, Any]], predictions: dict[str, dict[str, Any]]
) -> dict[str, list[tuple[str, bool]]]:
    """Per item: (type, correct) for each question."""
    from benchmark.score import evaluate_answer

    out = {}
    for item_id, item in gold.items():
        prediction = predictions.get(item_id)
        answers = prediction.get("answers") if isinstance(prediction, dict) else None
        exact = isinstance(answers, dict) and set(answers) == set(item["questions"])
        rows = []
        for key, question in item["questions"].items():
            correct = False
            if exact:
                result = evaluate_answer(question, item["gold"][key], answers[key])
                correct = result.get("status") == "ok" and bool(result.get("correct"))
            rows.append((question["type"], correct))
        out[item_id] = rows
    return out


def point(
    gold: dict[str, dict[str, Any]], outcome: dict[str, list[tuple[str, bool]]]
) -> dict[str, float]:
    family = {f: [0, 0] for f in FINAL_FAMILIES}
    kind = {t: [0, 0] for t in TYPES}
    for item_id, item in gold.items():
        for question_type, correct in outcome[item_id]:
            family[item["family"]][0] += correct
            family[item["family"]][1] += 1
            kind[question_type][0] += correct
            kind[question_type][1] += 1
    out = {f"family:{f}": c / n for f, (c, n) in family.items()}
    out.update({f"type:{t}": c / n for t, (c, n) in kind.items() if n})
    out["T"] = statistics.fmean(c / n for c, n in family.values())
    return out


def paired_units(
    gold: dict[str, dict[str, Any]],
    left: dict[str, list[tuple[str, bool]]],
    right: dict[str, list[tuple[str, bool]]],
    unit: str,
) -> dict[str, list[tuple[tuple[int, int, int, int], ...]]]:
    """Per family, its resampling units as (type index, left correct, right correct, slots) cells."""
    grouped: dict[str, dict[str, dict[int, list[int]]]] = {
        f: {} for f in FINAL_FAMILIES
    }
    for item_id, item in gold.items():
        key = item_id if unit == "item" else item["group_id"]
        cells = grouped[item["family"]].setdefault(key, {})
        for (kind, left_correct), (_, right_correct) in zip(
            left[item_id], right[item_id]
        ):
            cell = cells.setdefault(TYPES.index(kind), [0, 0, 0])
            cell[0] += left_correct
            cell[1] += right_correct
            cell[2] += 1
    return {
        family: [
            tuple((t, *counts) for t, counts in sorted(cells.items()))
            for _, cells in sorted(units.items())
        ]
        for family, units in grouped.items()
    }


def bootstrap(
    units: dict[str, list[tuple[tuple[int, int, int, int], ...]]], draws: int, seed: int
) -> dict[str, dict[str, float]]:
    """Paired draws of the units within each family: LEFT minus RIGHT per family, type and T."""
    rng = random.Random(seed)
    deltas: dict[str, list[float]] = defaultdict(list)
    count = len(TYPES)
    for _ in range(draws):
        type_left, type_right, type_slots = [0] * count, [0] * count, [0] * count
        lefts, rights = [], []
        for family in FINAL_FAMILIES:
            pool = units[family]
            size = len(pool)
            fam_left = fam_right = fam_slots = 0
            for _ in range(size):
                for t, a, b, s in pool[rng.randrange(size)]:
                    fam_left += a
                    fam_right += b
                    fam_slots += s
                    type_left[t] += a
                    type_right[t] += b
                    type_slots[t] += s
            lefts.append(fam_left / fam_slots)
            rights.append(fam_right / fam_slots)
            deltas[f"family:{family}"].append(lefts[-1] - rights[-1])
        deltas["T"].append(statistics.fmean(lefts) - statistics.fmean(rights))
        for t in range(count):
            if type_slots[t]:
                deltas[f"type:{TYPES[t]}"].append(
                    (type_left[t] - type_right[t]) / type_slots[t]
                )
    return {key: interval95(values) for key, values in sorted(deltas.items())}


class Run:
    """One sealed formal run: typed-FINAL outcomes, point values and Score level usage."""

    def __init__(
        self, name: str, directory: Path, gold: dict[str, dict[str, Any]]
    ) -> None:
        self.name, self.directory = name, directory
        path = verified(directory, "typed-final")
        self.predictions_sha256 = panel_registry.sha_file(path)
        predictions = {row["id"]: row for row in read_jsonl(path)}
        self.outcome = outcomes(gold, predictions)
        self.point = point(gold, self.outcome)
        report = read_json(directory / "REPORT.json")
        typed = report["panels"]["typed-final"]
        agree(f"{name} T", self.point["T"], typed["T"])
        for family, value in typed["by_family"].items():
            agree(f"{name} {family}", self.point[f"family:{family}"], value)
        for kind, value in typed["by_type"].items():
            agree(f"{name} {kind}", self.point[f"type:{kind}"], value["accuracy"])
        self.loaded_parameters = report["parameters"]["loaded"]
        self.v3 = report["v3"]["score"]
        self.types = type_summary(list(gold.values()), predictions)

    def summary(self) -> dict[str, Any]:
        return {
            "run": str(self.directory),
            "typed_final_predictions_sha256": self.predictions_sha256,
            "v3": self.v3,
            "loaded_parameters": self.loaded_parameters,
            "point": self.point,
        }


def contrast_pair(
    gold: dict[str, dict[str, Any]], left: Run, right: Run, draws: int, seed: int
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "left": left.name,
        "right": right.name,
        "delta": {k: left.point[k] - right.point[k] for k in left.point},
    }
    for unit in UNITS:
        units = paired_units(gold, left.outcome, right.outcome, unit)
        key = "ci95" if unit == "item" else "ci95_group"
        entry[key] = bootstrap(units, draws, seed)
        entry[f"{unit}_units_per_family"] = {f: len(u) for f, u in units.items()}
    return entry


def score_levels(run: Run) -> dict[str, Any]:
    score = run.types["score"]
    keys = (
        "accuracy",
        "accuracy_ci95",
        "chance",
        "predicted_distinct",
        "predicted_top_share",
        "predicted_distribution",
        "gold_distribution",
        "recall_by_level",
        "verdict",
    )
    return {
        "score": {k: score[k] for k in keys},
        "verdicts": {k: v["verdict"] for k, v in run.types.items()},
    }


def interval(paired: dict[str, Any], axis: str) -> dict[str, float]:
    return paired["ci95"] if axis == "v3" else paired["axis_ci95"][axis]["delta"]


def gate_summary(paired: dict[str, Any]) -> dict[str, Any]:
    return {
        "left": paired["models"]["left"],
        "right": paired["models"]["right"],
        "delta": paired["point"]["delta"],
        "ci95": {axis: interval(paired, axis) for axis in ("v3", "T", "H")},
    }


def successor(
    gates: Path,
    finalists: list[str],
    runs: dict[str, Run],
    exposures: dict[str, Path],
    pairs: dict[str, Any],
    incumbent: str,
) -> dict[str, Any]:
    entries = {}
    for name in finalists:
        folder = gates / name
        vs_f1 = read_json(folder / "paired-vs-F1.json")
        peers = {
            label: read_json(folder / f"paired-vs-{key}.json")
            for key, label in PEERS.items()
        }
        types = read_json(folder / "types.json")["types"]
        exposure = read_json(exposures[name]) if name in exposures else None
        v3 = vs_f1["point"]["left"]["score"]
        agree(f"{name} paired v3", v3, runs[name].v3)
        checks = {
            "1_v3_minus_F1_lower_bound_above_0": vs_f1["ci95"]["low"] > 0,
            "2_H_not_significantly_below_F1": vs_f1["axis_ci95"]["H"]["delta"]["high"]
            >= 0,
            "3a_v3_at_least_64.92": v3 >= V3_FLOOR,
            "3b_H_not_significantly_below_peers": all(
                p["axis_ci95"]["H"]["delta"]["high"] >= 0 for p in peers.values()
            ),
            "3c_no_type_collapsed": all(types[k]["verdict"] == "OK" for k in TYPES),
            "4_no_overlap_exposure": exposure is not None and not exposure["groups"],
        }
        entries[name] = {
            "checks": checks,
            "passes": all(checks.values()),
            "v3": v3,
            "vs_F1": gate_summary(vs_f1),
            "vs_peers": {label: gate_summary(p) for label, p in peers.items()},
            "types": {
                k: {
                    "accuracy": types[k]["accuracy"],
                    "accuracy_ci95": types[k]["accuracy_ci95"],
                    "chance": types[k]["chance"],
                    "predicted_distinct": types[k]["predicted_distinct"],
                    "predicted_top_share": types[k]["predicted_top_share"],
                    "verdict": types[k]["verdict"],
                }
                for k in TYPES
            },
            "overlap_exposure_groups": (
                None if exposure is None else len(exposure["groups"])
            ),
            "loaded_parameters": runs[name].loaded_parameters,
        }
    passing = [n for n in finalists if entries[n]["passes"]]
    order = sorted(
        passing,
        key=lambda n: (
            -entries[n]["vs_F1"]["ci95"]["v3"]["low"],
            -entries[n]["vs_F1"]["delta"]["score"],
            entries[n]["loaded_parameters"],
        ),
    )
    attribution: dict[str, Any] = {}
    for key, (filename, claim) in CLAIMS.items():
        path = gates / filename
        if not path.is_file():
            attribution[key] = {
                "run": False,
                "reason": f"no {filename} (an arm is not a finalist)",
            }
            continue
        paired = gate_summary(read_json(path))
        attribution[key] = {
            "run": True,
            **paired,
            "claim": claim,
            "claimed": paired["ci95"]["T"]["low"] > 0,
            "families": pairs.get(f"{paired['left']} - {paired['right']}"),
        }
    attribution["descriptive_vs_F1"] = {
        name: {
            **entries[name]["vs_F1"],
            "families": pairs.get(f"{name} - {incumbent}"),
        }
        for name in finalists
    }
    return {
        "rule": "preregistered successor rule (coordinator condition 4) and tie-break: highest v3 "
        "lower bound vs F1, then the larger Delta v3, then the smaller adapter (loaded parameters)",
        "finalists": entries,
        "passing": passing,
        "tie_break_order": order,
        "successor": order[0] if order else None,
        "outcome": (
            f"{order[0]} replaces F1: hand it to the coordinator for a release step"
            if order
            else "F1 stays; M4 is attribution only"
        ),
        "attribution": attribution,
    }


def named(specs: list[str]) -> dict[str, Path]:
    out = {}
    for spec in specs:
        name, _, path = spec.partition("=")
        if not name or not path or name in out:
            raise ValueError(f"expected distinct NAME=PATH, got {spec!r}")
        out[name] = Path(path)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.gold:
        gold_path = args.gold
    else:
        panel_registry.verify(args.panel_root, ["typed-final"])
        gold_path = panel_registry.path(args.panel_root, "typed-final", "gold")
    from benchmark.score import load_jsonl

    gold = load_jsonl(gold_path)
    runs = {name: Run(name, path, gold) for name, path in named(args.run).items()}
    specs = [tuple(spec.split(":")) for spec in args.pair]
    if any(
        len(s) != 2 or s[0] not in runs or s[1] not in runs or s[0] == s[1]
        for s in specs
    ):
        raise ValueError(f"pairs must name two different --run names: {args.pair}")
    unknown = set(args.finalist) - set(runs)
    if unknown:
        raise ValueError(f"finalists without a --run: {sorted(unknown)}")
    pairs = {
        f"{left} - {right}": contrast_pair(
            gold, runs[left], runs[right], args.draws, args.seed
        )
        for left, right in specs
    }
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "label": "post-key same-panel; typed FINAL",
        "draws": args.draws,
        "seed": args.seed,
        "units": {
            "ci95": "items resampled within family, paired (preregistered item level)",
            "ci95_group": "independent four-variant groups resampled within family, paired "
            "(the v3 T interval's unit)",
        },
        "gold_sha256": panel_registry.sha_file(gold_path),
        "runs": {name: r.summary() for name, r in runs.items()},
        "pairs": pairs,
        "score_levels": {name: score_levels(runs[name]) for name in args.finalist},
    }
    if args.gates:
        result["successor_rule"] = successor(
            args.gates, args.finalist, runs, named(args.exposure), pairs, args.incumbent
        )
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--panel-root", type=Path, default=panel_registry.DEFAULT_ROOT)
    parser.add_argument(
        "--gold", type=Path, help="typed-FINAL gold override (no panel check)"
    )
    parser.add_argument(
        "--run", action="append", required=True, help="NAME=FORMAL_RUN_DIR"
    )
    parser.add_argument(
        "--pair", action="append", default=[], help="LEFT:RIGHT (LEFT minus RIGHT)"
    )
    parser.add_argument(
        "--finalist", action="append", default=[], help="a finalist's run NAME"
    )
    parser.add_argument("--incumbent", default="DEV2.0-27B (F1)", help="F1's run NAME")
    parser.add_argument("--gates", type=Path, help="m4-tail.sh gates output directory")
    parser.add_argument(
        "--exposure", action="append", default=[], help="NAME=overlap exposure receipt"
    )
    parser.add_argument("--draws", type=int, default=PAIRED_REPLICATES)
    parser.add_argument("--seed", type=int, default=PAIRED_SEED)
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
    for key, entry in result["pairs"].items():
        for stat in (
            "T",
            *(f"family:{f}" for f in FINAL_FAMILIES),
            *(f"type:{t}" for t in TYPES),
        ):
            item, group = entry["ci95"][stat], entry["ci95_group"][stat]
            print(
                f"{key}  {stat:30s} {entry['delta'][stat]:+.4f}  item [{item['low']:+.4f}, "
                f"{item['high']:+.4f}]  group [{group['low']:+.4f}, {group['high']:+.4f}]",
                file=summary,
            )
    for name, levels in result["score_levels"].items():
        score = levels["score"]
        print(
            f"{name} Score predicted {score['predicted_distribution']} gold "
            f"{score['gold_distribution']} recall {score['recall_by_level']}",
            file=summary,
        )
    if "successor_rule" in result:
        rule = result["successor_rule"]
        print(
            json.dumps({"passing": rule["passing"], "outcome": rule["outcome"]}),
            file=summary,
        )
        for key in CLAIMS:
            claim = rule["attribution"][key]
            print(
                json.dumps({key: claim.get("claimed"), "run": claim["run"]}),
                file=summary,
            )


if __name__ == "__main__":
    main()
