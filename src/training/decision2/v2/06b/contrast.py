"""Paired development contrast between two checkpoints (experiment matrix v1.1 rule).

Typed-DEV groups are resampled within family and CSS-pilot items within task, with the
same draws for both checkpoints, so each proxy's difference (treatment minus control)
gets a paired 95% interval. Per-item scoring is the eval track's
(`v2.eval.proxy_calibration`); only aggregates are written.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

DRAWS = 10000
SEED = 20260928


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def outcomes(typed_gold: Any, css_gold: Any, typed: Path, css: Path) -> tuple[Any, ...]:
    from v2.eval.proxy_calibration import pilot_outcomes, typed_outcomes
    from v2.eval.same_panel import read_jsonl

    typed_rows = typed_outcomes(typed_gold, {r["id"]: r for r in read_jsonl(typed)})
    tasks, labels = pilot_outcomes(css_gold, {r["id"]: r for r in read_jsonl(css)})
    return typed_rows, tasks, labels


def paired(
    treatment: tuple[Any, ...], control: tuple[Any, ...], draws: int, seed: int
) -> dict[str, Any]:
    from v2.eval.proxy_calibration import features

    typed_t, tasks_t, labels = treatment
    typed_c, tasks_c, _ = control
    if [r[:2] for r in typed_t] != [r[:2] for r in typed_c] or {
        k: len(v) for k, v in tasks_t.items()
    } != {k: len(v) for k, v in tasks_c.items()}:
        raise ValueError("Treatment and control readouts are not item-aligned")
    groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, (family, group, _) in enumerate(typed_t):
        groups[family][group].append(index)
    point_t = features(typed_t, tasks_t, labels)
    point_c = features(typed_c, tasks_c, labels)
    rng = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(draws):
        rows: list[int] = []
        for members in groups.values():
            keys = list(members)
            for _key in keys:
                rows.extend(members[keys[rng.randrange(len(keys))]])
        picks = {
            task: [rng.randrange(len(pairs)) for _ in pairs]
            for task, pairs in tasks_t.items()
        }
        ft = features(
            [typed_t[i] for i in rows],
            {task: [tasks_t[task][j] for j in js] for task, js in picks.items()},
            labels,
        )
        fc = features(
            [typed_c[i] for i in rows],
            {task: [tasks_c[task][j] for j in js] for task, js in picks.items()},
            labels,
        )
        for key in point_t:
            samples[key].append(ft[key] - fc[key])
    return {
        "treatment": point_t,
        "control": point_c,
        "delta": {key: point_t[key] - point_c[key] for key in point_t},
        "ci95": {
            key: [percentile(v, 2.5), percentile(v, 97.5)] for key, v in samples.items()
        },
        "draws": draws,
        "seed": seed,
        "unit": "typed-DEV groups within family; CSS pilot items within task",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--typed-gold", type=Path, required=True)
    parser.add_argument("--css-gold", type=Path, required=True)
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        help="label=treatment_readout_dir=control_readout_dir",
    )
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from benchmark.score import load_jsonl
    from transfer.score import read_jsonl as read_css

    typed_gold = load_jsonl(args.typed_gold)
    css_gold = read_css(args.css_gold)
    report: dict[str, Any] = {"pairs": {}}
    for item in args.pair:
        label, treatment, control = item.split("=", 2)
        sides = [
            outcomes(
                typed_gold,
                css_gold,
                Path(d) / "dev.predictions.jsonl",
                Path(d) / "css-pilot.predictions.jsonl",
            )
            for d in (treatment, control)
        ]
        report["pairs"][label] = paired(sides[0], sides[1], args.draws, SEED)
        print(json.dumps({label: report["pairs"][label]["delta"]}), flush=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
