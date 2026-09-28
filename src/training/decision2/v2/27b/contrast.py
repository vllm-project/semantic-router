"""Paired development contrasts for the ~27B Milestone 2 data arms (CPU, stdlib).

Per arm run directory (``<arm>/full``): typed DEV and CSS-pilot predictions from
the readout, CAL report, and arm held-out (AHO) predictions. Per-item outcomes
come from the frozen scorers (``benchmark.score.evaluate_answer``,
``transfer.score.evaluate`` and ``macro_f1``'s definition). The paired bootstrap
resamples typed-DEV item groups, CSS-pilot items within task and AHO groups
(10,000 draws). Development readouts only: never release scores.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

TYPES = ("choice", "noul", "score")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


class Panels:
    """Gold-side structure shared by every arm."""

    def __init__(self, dev_gold: Path, css_gold: Path) -> None:
        self.dev = read_jsonl(dev_gold)
        self.css = [row for row in read_jsonl(css_gold) if row["role"] == "pilot"]
        self.families = sorted({row["family"] for row in self.dev})
        groups: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(self.dev):
            groups[row["group_id"]].append(index)
        self.dev_groups = [groups[g] for g in sorted(groups)]
        self.tasks = sorted({row["task"] for row in self.css})
        self.task_items = {
            t: [i for i, r in enumerate(self.css) if r["task"] == t] for t in self.tasks
        }

    def dev_outcomes(
        self, predictions: Path
    ) -> list[list[tuple[str, str, bool, bool]]]:
        """Per typed-DEV item, one (family, type, correct, valid) tuple per question."""
        from benchmark.score import evaluate_answer

        by_id = {row["id"]: row for row in read_jsonl(predictions)}
        out = []
        for item in self.dev:
            answers = (by_id.get(item["id"]) or {}).get("answers")
            questions = []
            for key, question in item["questions"].items():
                if not isinstance(answers, dict) or set(answers) != set(
                    item["questions"]
                ):
                    questions.append((item["family"], question["type"], False, False))
                    continue
                result = evaluate_answer(question, item["gold"][key], answers[key])
                questions.append(
                    (
                        item["family"],
                        question["type"],
                        bool(result.get("correct")),
                        result.get("status") == "ok",
                    )
                )
            out.append(questions)
        return out

    def css_choices(self, predictions: Path) -> list[str | None]:
        from transfer.score import evaluate

        by_id = {row["id"]: row for row in read_jsonl(predictions)}
        choices = []
        for row in self.css:
            result = evaluate(row, by_id.get(row["id"]))
            choices.append(result["choice"] if result["valid"] else None)
        return choices


def dev_metrics(panels: Panels, outcomes: list, sample: list[int]) -> dict[str, float]:
    fam = defaultdict(lambda: [0, 0])
    typ = defaultdict(lambda: [0, 0])
    for index in sample:
        for family, kind, correct, _ in outcomes[index]:
            fam[family][0] += correct
            fam[family][1] += 1
            typ[kind][0] += correct
            typ[kind][1] += 1
    result = {
        "T_dev": statistics.fmean(
            c / n for c, n in (fam[f] for f in panels.families) if n
        )
    }
    for kind in TYPES:
        c, n = typ[kind]
        result[f"{kind}_accuracy"] = c / n if n else 0.0
    return result


def css_macro(
    panels: Panels, choices: list, sample_by_task: dict[str, list[int]]
) -> dict[str, float]:
    f1 = {}
    for task, items in sample_by_task.items():
        labels = panels.css[panels.task_items[task][0]]["labels"]
        tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
        for index in items:
            gold, pred = panels.css[index]["gold"], choices[index]
            if pred == gold:
                tp[gold] += 1
            else:
                fn[gold] += 1
                if pred is not None:
                    fp[pred] += 1
        scores = [
            (
                2 * tp[l] / (2 * tp[l] + fp[l] + fn[l])
                if (2 * tp[l] + fp[l] + fn[l])
                else 0.0
            )
            for l in labels
        ]
        f1[task] = statistics.fmean(scores)
    return {
        "H_pilot": statistics.median(f1.values()),
        **{f"css:{t}": v for t, v in f1.items()},
    }


def aho_outcomes(
    records: Path, rows: Path
) -> tuple[list[list[bool]], dict[str, list[bool]]]:
    correct = {r["id"]: bool(r["correct"]) for r in read_jsonl(records)}
    groups: dict[str, list[bool]] = defaultdict(list)
    for row in read_jsonl(rows):
        groups[row["group_id"]].append(correct.get(row["id"], False))
    return [groups[g] for g in sorted(groups)], groups


class Arm:
    def __init__(
        self, name: str, directory: Path, panels: Panels, aho: dict[str, Path]
    ) -> None:
        self.name = name
        self.dev = panels.dev_outcomes(directory / "dev.predictions.jsonl")
        self.css = panels.css_choices(directory / "css-pilot.predictions.jsonl")
        calibration = json.loads(
            (directory / "calibration.json").read_text(encoding="utf-8")
        )
        after = {k: v["after"] for k, v in calibration["by_type"].items()}
        self.cal_brier = sum(v["brier"] * v["n"] for v in after.values()) / sum(
            v["n"] for v in after.values()
        )
        self.invalid = sum(not q[3] for item in self.dev for q in item) + sum(
            c is None for c in self.css
        )
        self.aho = {}
        for slice_name, rows in aho.items():
            records = directory / f"aho-{slice_name}-predictions.jsonl"
            if records.is_file():
                self.aho[slice_name] = {
                    g: v for g, v in aho_outcomes(records, rows)[1].items()
                }

    def point(self, panels: Panels) -> dict[str, float]:
        dev = dev_metrics(panels, self.dev, list(range(len(self.dev))))
        css = css_macro(panels, self.css, panels.task_items)
        out = {**dev, **css, "P_dev": 100 * math.sqrt(dev["T_dev"] * css["H_pilot"])}
        out["cal_brier"] = self.cal_brier
        out["invalid"] = self.invalid
        for slice_name, groups in self.aho.items():
            flat = [v for g in sorted(groups) for v in groups[g]]
            out[f"aho:{slice_name}"] = sum(flat) / len(flat)
        return out


def paired(
    panels: Panels, left: list[Arm], right: list[Arm], draws: int, seed: int
) -> dict[str, Any]:
    """Mean over seeds of (left_s - right_s), resampling the same groups for every arm."""
    rng = random.Random(seed)
    metrics: dict[str, list[float]] = defaultdict(list)
    slices = (
        sorted(set.intersection(*(set(a.aho) for a in left + right)))
        if left[0].aho
        else []
    )
    for _ in range(draws):
        groups = [
            panels.dev_groups[rng.randrange(len(panels.dev_groups))]
            for _ in panels.dev_groups
        ]
        dev_sample = [i for g in groups for i in g]
        css_sample = {
            t: [items[rng.randrange(len(items))] for _ in items]
            for t, items in panels.task_items.items()
        }
        aho_samples = {}
        for name in slices:
            keys = sorted(left[0].aho[name])
            aho_samples[name] = [keys[rng.randrange(len(keys))] for _ in keys]
        values = defaultdict(list)
        for sign, arms in ((1, left), (-1, right)):
            for arm in arms:
                dev = dev_metrics(panels, arm.dev, dev_sample)
                css = css_macro(panels, arm.css, css_sample)
                sample = {
                    **dev,
                    "H_pilot": css["H_pilot"],
                    "P_dev": 100 * math.sqrt(dev["T_dev"] * css["H_pilot"]),
                }
                for name, keys in aho_samples.items():
                    flat = [v for k in keys for v in arm.aho[name][k]]
                    sample[f"aho:{name}"] = sum(flat) / len(flat)
                for key, value in sample.items():
                    values[key].append(sign * value / len(arms))
        for key, parts in values.items():
            metrics[key].append(sum(parts))
    return {
        key: {"ci95": [percentile(v, 0.025), percentile(v, 0.975)], "draws": draws}
        for key, v in sorted(metrics.items())
    }


def decide(
    point_left: list[dict],
    point_right: list[dict],
    boot: dict,
    target: str,
    sigma: float,
) -> dict[str, Any]:
    mean = lambda rows, key: statistics.fmean(r[key] for r in rows)  # noqa: E731
    delta = {
        k: mean(point_left, k) - mean(point_right, k)
        for k in point_left[0]
        if k in point_right[0]
    }
    checks = {
        "effect_lower_bound_positive": boot[target]["ci95"][0] > 0,
        "effect_above_2_sigma": delta[target] > 2 * sigma,
        "no_type_drop_over_3_points": all(
            delta[f"{k}_accuracy"] >= -0.03 for k in TYPES
        ),
        "h_pilot_drop_at_most_1.5_points": delta["H_pilot"] >= -0.015,
        "cal_brier_worse_at_most_0.010": delta["cal_brier"] <= 0.010,
        "invalid_not_increased": delta["invalid"] <= 0,
    }
    return {
        "target": target,
        "sigma_seed": sigma,
        "delta": delta,
        "checks": checks,
        "passes": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-gold", type=Path, required=True)
    parser.add_argument("--css-gold", type=Path, required=True)
    parser.add_argument(
        "--arm", action="append", required=True, help="NAME=RUN_FULL_DIR"
    )
    parser.add_argument("--aho", action="append", default=[], help="SLICE=ROWS")
    parser.add_argument(
        "--contrast",
        action="append",
        default=[],
        help="TREAT[+TREAT2]:CONTROL[+CONTROL2]:TARGET:SIGMA",
    )
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panels = Panels(args.dev_gold, args.css_gold)
    aho = {spec.split("=", 1)[0]: Path(spec.split("=", 1)[1]) for spec in args.aho}
    arms = {}
    for spec in args.arm:
        name, directory = spec.split("=", 1)
        arms[name] = Arm(name, Path(directory), panels, aho)
    points = {name: arm.point(panels) for name, arm in arms.items()}
    result: dict[str, Any] = {
        "schema": "decision2-27b-m2-contrast/1",
        "label": "development readout",
        "points": points,
        "contrasts": {},
    }
    for spec in args.contrast:
        treat, control, target, sigma = spec.split(":")
        left, right = [arms[n] for n in treat.split("+")], [
            arms[n] for n in control.split("+")
        ]
        boot = paired(panels, left, right, args.draws, args.seed)
        decision = decide(
            [points[n] for n in treat.split("+")],
            [points[n] for n in control.split("+")],
            boot,
            target,
            float(sigma),
        )
        result["contrasts"][f"{treat}-vs-{control}"] = {**decision, "bootstrap": boot}
        print(
            json.dumps(
                {
                    "contrast": f"{treat}-vs-{control}",
                    "target": target,
                    "delta": decision["delta"][target],
                    "ci95": boot[target]["ci95"],
                    "passes": decision["passes"],
                }
            ),
            flush=True,
        )
    args.output.write_text(
        json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
