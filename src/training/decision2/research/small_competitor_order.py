"""Freeze and summarize a Choice option-order check from the smoke packet."""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import statistics
from pathlib import Path

from small_decision_competitors import input_digest, json_bytes, read_rows, sha_file

SMOKE_SHA = "a8518ae10cec3fb4ba531413bac485b66711f6dbf5143a9863252ff9f330964f"
QUOTAS = {"dev": 16, "css": 8, "public": 8}


def rank(item_id: str) -> str:
    return hashlib.sha256(("small-competitor-order-v1/" + item_id).encode()).hexdigest()


def freeze(smoke: Path, output: Path) -> dict:
    if output.exists() or output.with_suffix(".manifest.json").exists():
        raise FileExistsError(output)
    if sha_file(smoke) != SMOKE_SHA:
        raise ValueError("smoke packet changed")
    rows = read_rows(smoke)
    if len(rows) != 100:
        raise ValueError("smoke cardinality changed")
    strata = {"dev": rows[:40], "css": rows[40:70], "public": rows[70:]}
    selected = []
    mapping = []
    for name, panel in strata.items():
        choices = [
            row
            for row in panel
            if next(iter(row["questions"].values()))["type"] == "choice"
        ]
        if len(choices) < QUOTAS[name]:
            raise ValueError(f"too few Choice rows in {name}")
        for row in sorted(choices, key=lambda value: rank(value["id"]))[: QUOTAS[name]]:
            variant = copy.deepcopy(row)
            variant["id"] = "order/" + row["id"]
            question = next(iter(variant["questions"].values()))
            question["criteria"] = dict(reversed(list(question["criteria"].items())))
            selected.append(variant)
            mapping.append(
                {
                    "panel": name,
                    "original_id": row["id"],
                    "variant_id": variant["id"],
                    "original_input_sha256": input_digest(row),
                    "variant_input_sha256": input_digest(variant),
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as destination:
        for row in selected:
            destination.write(json_bytes(row))
    result = {
        "selection": "small-competitor-order-v1",
        "smoke_sha256": SMOKE_SHA,
        "prompt_sha256": sha_file(output),
        "counts": QUOTAS,
        "mapping": mapping,
    }
    output.with_suffix(".manifest.json").write_bytes(json_bytes(result))
    return {key: value for key, value in result.items() if key != "mapping"}


def read_predictions(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in map(json.loads, path.read_text().splitlines())}


def analyze(
    smoke_a: Path,
    order_a: Path,
    smoke_b: Path,
    order_b: Path,
    order_manifest: Path,
    output: Path,
) -> dict:
    if output.exists():
        raise FileExistsError(output)
    mapping = json.loads(order_manifest.read_text())["mapping"]
    reports = {}
    for arm, source, variant in (("a", smoke_a, order_a), ("b", smoke_b, order_b)):
        left, right = read_predictions(source), read_predictions(variant)
        counts: collections.Counter[str] = collections.Counter()
        drift: list[float] = []
        by_panel: dict[str, collections.Counter[str]] = collections.defaultdict(
            collections.Counter
        )
        for pair in mapping:
            original, reversed_row = (
                left[pair["original_id"]],
                right[pair["variant_id"]],
            )
            if (
                original["source_input_sha256"] != pair["original_input_sha256"]
                or reversed_row["source_input_sha256"] != pair["variant_input_sha256"]
            ):
                raise ValueError("option-order prediction input changed")
            qid = next(iter(original["answers"]), None)
            a = original["answers"].get(qid) if qid else None
            b = reversed_row["answers"].get(qid) if qid else None
            panel = pair["panel"]
            counts["items"] += 1
            by_panel[panel]["items"] += 1
            if a is None or b is None:
                counts["invalid_either"] += 1
                by_panel[panel]["invalid_either"] += 1
                continue
            if set(a["probabilities"]) != set(b["probabilities"]):
                raise ValueError("option labels changed")
            counts["valid_both"] += 1
            by_panel[panel]["valid_both"] += 1
            if a["choice"] == b["choice"]:
                counts["same_choice"] += 1
                by_panel[panel]["same_choice"] += 1
            drift.append(
                max(
                    abs(a["probabilities"][label] - b["probabilities"][label])
                    for label in a["probabilities"]
                )
            )
        reports[arm] = {
            "counts": dict(counts),
            "same_choice_rate_all": counts["same_choice"] / counts["items"],
            "mean_max_probability_drift_valid": (
                statistics.mean(drift) if drift else None
            ),
            "max_probability_drift_valid": max(drift) if drift else None,
            "by_panel": {key: dict(value) for key, value in by_panel.items()},
            "smoke_predictions_sha256": sha_file(source),
            "order_predictions_sha256": sha_file(variant),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(json_bytes(reports))
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--smoke", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    analyze_parser = sub.add_parser("analyze")
    for name in (
        "smoke_a",
        "order_a",
        "smoke_b",
        "order_b",
        "order_manifest",
        "output",
    ):
        analyze_parser.add_argument(
            "--" + name.replace("_", "-"), type=Path, required=True
        )
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(args.smoke, args.output)
    else:
        result = analyze(
            args.smoke_a,
            args.order_a,
            args.smoke_b,
            args.order_b,
            args.order_manifest,
            args.output,
        )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
