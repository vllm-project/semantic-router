"""Milestone 3 development contrasts for ~27B kernel-path readouts (CPU, stdlib).

Per candidate (an arm-seed BEST or a soup), one kernel-path readout directory of
``v2/27b/run_dev_readout.sh``: ``output/typed-dev|css-pilot.predictions.jsonl``,
``READOUT.json``, ``cal698.summary.json``, ``cal/cal.probs.jsonl`` (and
``cal/calibration.json``), and for arm-seeds ``aho/aho-<SLICE>-predictions.jsonl``.
An arm-seed also names its trainer run (``BEST.json``, SELECT700 metrics).

Per-item outcomes, the paired bootstrap (typed-DEV item groups, CSS-pilot items
within task, AHO groups; 10,000 draws) and the retention checks are Milestone 2's
(``v2/27b/contrast.py``). CAL698 Brier is bootstrapped over CAL698 groups with the
same pairing (every arm resamples the same groups), from a separate seeded stream.
Point values must agree with ``READOUT.json``, ``cal698.summary.json`` and
``aho-summary.json``. Retention floors (Milestone 3 preregistration, pooled M3-A vs
M3-S) are report-only; the soup rule and the proxy screen follow the same record.
Development readouts only: never release scores.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from v2.eval import panels as panel_registry
from v2.eval.dev_readout import select_row

contrast = importlib.import_module("v2.27b.contrast")

SCHEMA = "decision2-27b-m3-contrast/1"
FLOORS = (
    "no_type_drop_over_3_points",
    "h_pilot_drop_at_most_1.5_points",
    "cal_brier_worse_at_most_0.010",
    "invalid_not_increased",
)
PROXY_DROP = 8.0
TOLERANCE = 1e-9


class Inputs:
    """SHA-256 of every file read, keyed by path."""

    def __init__(self) -> None:
        self.files: dict[str, str] = {}

    def __call__(self, path: Path) -> Path:
        key = str(path)
        if key not in self.files:
            self.files[key] = panel_registry.sha_file(path)
        return path

    def sha(self, path: Path) -> str:
        self(path)
        return self.files[str(path)]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def agree(name: str, got: float, expected: float) -> None:
    if not math.isclose(got, expected, rel_tol=0, abs_tol=TOLERANCE):
        raise ValueError(
            f"{name}: recomputed {got!r} differs from recorded {expected!r}"
        )


class CalRows:
    def __init__(self, path: Path) -> None:
        self.rows = contrast.read_jsonl(path)
        groups: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(self.rows):
            groups[row["group_id"]].append(index)
        self.groups = [groups[g] for g in sorted(groups)]


def ece_10(
    records: list[dict[str, Any]], probabilities: list[list[float] | None]
) -> float:
    """``training.model.calibration.metrics``'s ECE on stored probabilities (invalid rows skipped)."""
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(10)]
    for record, probs in zip(records, probabilities):
        if record["valid"]:
            top = max(probs)
            bins[min(9, int(top * 10))].append((top, record["correct"]))
    n = sum(len(bucket) for bucket in bins)
    return sum(
        len(bucket)
        / n
        * abs(
            sum(c for c, _ in bucket) / len(bucket)
            - sum(h for _, h in bucket) / len(bucket)
        )
        for bucket in bins
        if bucket
    )


class Candidate:
    """One readout directory; ``dev``/``css``/``aho`` follow ``contrast.Arm``."""

    def __init__(
        self,
        name: str,
        directory: Path,
        panels: Any,
        cal: CalRows,
        aho: dict[str, Path],
        inputs: Inputs,
        train_run: Path | None = None,
    ) -> None:
        self.name, self.directory, self.train_run = name, directory, train_run
        typed = inputs(directory / "output" / "typed-dev.predictions.jsonl")
        css = inputs(directory / "output" / "css-pilot.predictions.jsonl")
        self.dev = panels.dev_outcomes(typed)
        self.css = panels.css_choices(css)
        self.readout = read_json(inputs(directory / "READOUT.json"))
        for key, path in (("typed_dev", typed), ("css_pilot", css)):
            if self.readout[key]["predictions_sha256"] != inputs.sha(path):
                raise ValueError(f"{name}: READOUT.json {key} scored other predictions")

        probs_path = inputs(directory / "cal" / "cal.probs.jsonl")
        predictions = {r["id"]: r for r in contrast.read_jsonl(probs_path)}
        records = [select_row(row, predictions.get(row["id"])) for row in cal.rows]
        self.cal = [r["brier"] for r in records]
        self.cal_summary = read_json(inputs(directory / "cal698.summary.json"))
        if self.cal_summary["predictions_sha256"] != inputs.sha(probs_path):
            raise ValueError(f"{name}: cal698.summary.json scored other probabilities")
        by_type = self.cal_summary["by_type"].values()
        agree(
            f"{name} CAL698 Brier",
            statistics.fmean(self.cal),
            sum(b["brier"] * b["n"] for b in by_type) / sum(b["n"] for b in by_type),
        )
        self.cal_ece = ece_10(
            records,
            [
                (predictions.get(row["id"]) or {}).get("probabilities")
                for row in cal.rows
            ],
        )
        calibration = directory / "cal" / "calibration.json"
        self.temperatures = (
            read_json(inputs(calibration))["temperature_by_type"]
            if calibration.is_file()
            else None
        )

        self.aho: dict[str, dict[str, list[bool]]] = {}
        summary_path = directory / "aho" / "aho-summary.json"
        aho_summary = (
            read_json(inputs(summary_path))["slices"] if summary_path.is_file() else {}
        )
        for slice_name, rows in aho.items():
            records_path = directory / "aho" / f"aho-{slice_name}-predictions.jsonl"
            if not records_path.is_file():
                continue
            groups = contrast.aho_outcomes(inputs(records_path), inputs(rows))[1]
            self.aho[slice_name] = dict(groups)
            flat = [v for g in sorted(groups) for v in groups[g]]
            agree(
                f"{name} AHO-{slice_name}",
                sum(flat) / len(flat),
                aho_summary[slice_name]["micro_accuracy"],
            )
        self.select = trainer_select(train_run, inputs) if train_run else None

    def point(self, panels: Any) -> dict[str, float]:
        dev = contrast.dev_metrics(panels, self.dev, list(range(len(self.dev))))
        css = contrast.css_macro(panels, self.css, panels.task_items)
        typed_invalid = sum(not q[3] for item in self.dev for q in item)
        css_invalid = sum(c is None for c in self.css)
        out = {
            **dev,
            "H_pilot": css["H_pilot"],
            "P_dev": 100 * math.sqrt(dev["T_dev"] * css["H_pilot"]),
            "cal_brier": statistics.fmean(self.cal),
            "cal_family_macro_brier": self.cal_summary["family_macro_brier"],
            "cal_ece_10": self.cal_ece,
            "invalid": typed_invalid + css_invalid,
            "invalid_typed": typed_invalid,
            "invalid_css": css_invalid,
            **{f"css:{t}": css[f"css:{t}"] for t in panels.tasks},
        }
        for slice_name, groups in self.aho.items():
            flat = [v for g in sorted(groups) for v in groups[g]]
            out[f"aho:{slice_name}"] = sum(flat) / len(flat)
        agree(f"{self.name} T_dev", out["T_dev"], self.readout["typed_dev"]["T_dev"])
        agree(
            f"{self.name} H_pilot", out["H_pilot"], self.readout["css_pilot"]["H_pilot"]
        )
        agree(f"{self.name} P_dev", out["P_dev"], self.readout["development_proxy"])
        return out

    def details(self) -> dict[str, Any]:
        select = self.readout.get("select") or {}
        return {
            "readout_dir": str(self.directory),
            "label": self.readout.get("label"),
            "train_run": str(self.train_run) if self.train_run else None,
            "cal698_temperature_by_type": self.temperatures,
            "cal698_family_macro_accuracy": self.cal_summary["family_macro_accuracy"],
            "readout_invalid_or_missing": {
                "typed_dev": self.readout["typed_dev"]["invalid_or_missing"],
                "css_pilot": self.readout["css_pilot"]["invalid_or_missing"],
            },
            "readout_select700": {
                k: select.get(k)
                for k in ("correct", "family_macro_accuracy", "family_macro_brier")
            },
            "trainer_select700": self.select,
        }


def trainer_select(run_dir: Path, inputs: Inputs) -> dict[str, Any]:
    best = read_json(inputs(run_dir / "BEST.json"))["checkpoint"]
    step = re.fullmatch(r"checkpoint-([0-9]{7})", best)
    if step is None:
        raise ValueError(f"{run_dir}: unexpected BEST checkpoint {best}")
    metrics = read_json(inputs(run_dir / f"select-step-{step.group(1)}-metrics.json"))
    return {
        "best": best,
        **{
            k: metrics[k]
            for k in ("n", "correct", "family_macro_accuracy", "family_macro_brier")
        },
    }


def paired_cal(
    cal: CalRows, left: list[Candidate], right: list[Candidate], draws: int, seed: str
) -> dict[str, Any]:
    rng = random.Random(seed)
    values = []
    for _ in range(draws):
        sample = [
            i for _ in cal.groups for i in cal.groups[rng.randrange(len(cal.groups))]
        ]
        total = 0.0
        for sign, arms in ((1, left), (-1, right)):
            for arm in arms:
                total += (
                    sign * sum(arm.cal[i] for i in sample) / len(sample) / len(arms)
                )
        values.append(total)
    return {
        "ci95": [
            contrast.percentile(values, 0.025),
            contrast.percentile(values, 0.975),
        ],
        "draws": draws,
    }


def compare(
    panels: Any,
    cal: CalRows,
    candidates: dict[str, Candidate],
    points: dict[str, dict[str, float]],
    spec: str,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    treat, control = spec.split(":")
    left = [candidates[n] for n in treat.split("+")]
    right = [candidates[n] for n in control.split("+")]
    boot = contrast.paired(panels, left, right, draws, seed)
    boot["cal_brier"] = paired_cal(cal, left, right, draws, f"{seed}/cal698")
    rows = [points[a.name] for a in left + right]
    common = set.intersection(*(set(r) for r in rows))
    trim = lambda arms: [  # noqa: E731
        {k: v for k, v in points[a.name].items() if k in common} for a in arms
    ]
    decision = contrast.decide(trim(left), trim(right), boot, "T_dev", 0.0)
    floors = {k: decision["checks"][k] for k in FLOORS}
    return {
        "treatment": [a.name for a in left],
        "control": [a.name for a in right],
        "delta": decision["delta"],
        "bootstrap": boot,
        "retention_floors": {
            **floors,
            "all_pass": all(floors.values()),
            "gates": False,
        },
    }


def soup_rule(
    points: dict[str, dict[str, float]],
    candidates: dict[str, Candidate],
    soup: str,
    seeds: list[str],
) -> dict[str, Any]:
    """Soup if P_dev(soup) >= mean P_dev(seeds); else the seed with the better SELECT700."""
    mean_seeds = statistics.fmean(points[s]["P_dev"] for s in seeds)
    out: dict[str, Any] = {
        "soup": soup,
        "seeds": seeds,
        "P_dev_soup": points[soup]["P_dev"],
        "P_dev_seed_mean": mean_seeds,
    }
    if points[soup]["P_dev"] >= mean_seeds:
        return {**out, "artifact": soup, "reason": "soup P_dev >= mean seed P_dev"}
    selects = {}
    for s in seeds:
        if candidates[s].select is None:
            raise ValueError(f"{s}: the soup rule needs its trainer run's SELECT700")
        selects[s] = candidates[s].select
    key = lambda s: (  # noqa: E731
        -selects[s]["family_macro_accuracy"],
        selects[s]["family_macro_brier"],
    )
    ranked = sorted(seeds, key=key)
    return {
        **out,
        "artifact": ranked[0],
        "reason": "soup P_dev < mean seed P_dev; seed with the higher SELECT700 "
        "(family-macro accuracy, then Brier)",
        "select700": selects,
        "select700_tie": key(ranked[0]) == key(ranked[1]),
    }


def proxy_screen(
    points: dict[str, dict[str, float]], pool: list[str]
) -> dict[str, Any]:
    best = max(pool, key=lambda n: points[n]["P_dev"])
    top = points[best]["P_dev"]
    return {
        "pool": pool,
        "best": best,
        "best_P_dev": top,
        "drop_if_at_least_below": PROXY_DROP,
        "candidates": {
            n: {
                "P_dev": points[n]["P_dev"],
                "gap": top - points[n]["P_dev"],
                "kept": top - points[n]["P_dev"] < PROXY_DROP,
            }
            for n in pool
        },
    }


def pairs(specs: list[str]) -> dict[str, Path]:
    return {s.split("=", 1)[0]: Path(s.split("=", 1)[1]) for s in specs}


def run(args: argparse.Namespace) -> dict[str, Any]:
    inputs = Inputs()
    if args.dev_gold or args.css_gold:
        dev_gold, css_gold = args.dev_gold, args.css_gold
    else:
        panel_registry.verify(args.panel_root, ["typed-dev", "css-pilot"])
        dev_gold = panel_registry.path(args.panel_root, "typed-dev", "gold")
        css_gold = panel_registry.path(args.panel_root, "css-pilot", "gold")
    panels = contrast.Panels(inputs(dev_gold), inputs(css_gold))
    if inputs.sha(args.cal_rows) != args.cal_sha256:
        raise ValueError("CAL rows differ from their pinned SHA-256")
    cal = CalRows(args.cal_rows)
    aho = pairs(args.aho)
    runs = pairs(args.train_run)
    candidates = {
        name: Candidate(name, directory, panels, cal, aho, inputs, runs.get(name))
        for name, directory in pairs(args.candidate).items()
    }
    points = {name: c.point(panels) for name, c in candidates.items()}
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "label": "development readout; never a release or formal score",
        "draws": args.draws,
        "seed": args.seed,
        "points": points,
        "details": {name: c.details() for name, c in candidates.items()},
        "contrasts": {
            spec: compare(panels, cal, candidates, points, spec, args.draws, args.seed)
            for spec in args.contrast
        },
        "soup_rule": {},
    }
    for spec in args.soup:
        arm, rest = spec.split("=", 1)
        soup, seeds = rest.split(":")
        result["soup_rule"][arm] = soup_rule(points, candidates, soup, seeds.split("+"))
    pool = [r["artifact"] for r in result["soup_rule"].values()] or sorted(candidates)
    result["proxy_screen"] = proxy_screen(points, pool)
    inputs(Path(__file__))
    inputs(Path(contrast.__file__))
    result["inputs_sha256"] = dict(sorted(inputs.files.items()))
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--panel-root", type=Path, default=panel_registry.DEFAULT_ROOT)
    parser.add_argument("--dev-gold", type=Path, help="override (no panel check)")
    parser.add_argument("--css-gold", type=Path, help="override (no panel check)")
    parser.add_argument("--cal-rows", type=Path, required=True, help="CAL698 rows")
    parser.add_argument("--cal-sha256", required=True)
    parser.add_argument(
        "--candidate", action="append", required=True, help="NAME=READOUT_DIR"
    )
    parser.add_argument(
        "--train-run", action="append", default=[], help="NAME=TRAINER_RUN_DIR"
    )
    parser.add_argument("--aho", action="append", default=[], help="SLICE=ROWS")
    parser.add_argument(
        "--contrast",
        action="append",
        default=[],
        help="TREAT[+TREAT2]:CONTROL[+CONTROL2]",
    )
    parser.add_argument(
        "--soup", action="append", default=[], help="ARM=SOUP:SEED1+SEED2"
    )
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    result = run(args)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=1, sort_keys=True) + "\n")
    for spec, entry in result["contrasts"].items():
        print(
            json.dumps(
                {
                    "contrast": spec,
                    "delta_P_dev": entry["delta"]["P_dev"],
                    "ci95_P_dev": entry["bootstrap"]["P_dev"]["ci95"],
                    "floors_pass": entry["retention_floors"]["all_pass"],
                }
            ),
            flush=True,
        )
    print(json.dumps({"proxy_screen": result["proxy_screen"]["candidates"]}))


if __name__ == "__main__":
    main()
