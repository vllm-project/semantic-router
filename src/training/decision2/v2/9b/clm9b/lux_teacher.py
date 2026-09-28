"""Own Decision 1.0 Lux 9B soft distributions from frozen joint features.

Lux's native prompt equals the unchanged Decision 2.0 segmented prompt, and its
head is the same shared candidate head. Applying the published head and its
single CAL temperature to final-norm joint features reproduces Lux's
distribution without the qualified FLA image; the published package example
is the parity reference. Teacher targets are produced for TRAIN rows only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from . import pins

NOUL_DEFAULTS = {
    "false": "The answer to the question is no.",
    "true": "The answer to the question is yes.",
}
PARITY_MAX_DRIFT = 0.05


def example_rows(example: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the published example requests with Lux's own question semantics."""
    rows = []
    for request_id, request in example["requests"].items():
        for name, question in request["questions"].items():
            kind = question["type"]
            criteria = question.get("criteria")
            if kind == "noul":
                criteria = criteria or {}
                options = [
                    {"key": key, "description": criteria.get(key, NOUL_DEFAULTS[key])}
                    for key in ("false", "true")
                ]
            elif kind == "score":
                options = [
                    {"key": str(i), "description": value}
                    for i, value in enumerate(criteria)
                ]
            else:
                options = [
                    {"key": key, "description": value}
                    for key, value in criteria.items()
                ]
            rows.append(
                {
                    "id": f"{request_id}/{name}",
                    "state": request["state"],
                    "instructions": question.get("instructions"),
                    "options": options,
                    "task_type": kind,
                    "family": "lux-release-example",
                }
            )
    return rows


def distribution(logits: list[float], temperature: float) -> list[float]:
    scaled = [value / temperature for value in logits]
    top = max(scaled)
    weights = [math.exp(value - top) for value in scaled]
    total = sum(weights)
    return [value / total for value in weights]


def published_slots(example: dict[str, Any]) -> dict[str, dict[str, Any]]:
    slots = {}
    for request_id, response in example["actual_responses"].items():
        for name, answer in response["answers"].items():
            slots[f"{request_id}/{name}"] = answer
    return slots


def compare(
    row: dict[str, Any], probabilities: list[float], published: dict[str, Any]
) -> dict[str, Any]:
    keys = [option["key"] for option in row["options"]]
    if row["task_type"] == "noul":
        observed = probabilities[keys.index("true")]
        drift = abs(observed - published["noul"])
        same = (observed >= 0.5) == (published["noul"] >= 0.5)
    else:
        expected = published["probabilities"]
        drift = max(abs(p - expected[key]) for key, p in zip(keys, probabilities))
        mine = keys[max(range(len(keys)), key=probabilities.__getitem__)]
        theirs = max(expected, key=expected.get)
        same = mine == theirs
    return {"id": row["id"], "max_drift": drift, "same_category": same}


def load_lux_head(root: Path):
    import torch
    from safetensors.torch import load_file

    from training.model.decision_model import CandidateHead

    head = CandidateHead(4096, 256)
    head.load_state_dict(
        load_file(str(root / "decision_head.safetensors")), strict=True
    )
    temperature = json.loads((root / "temperature.json").read_text(encoding="utf-8"))[
        "temperatures"
    ]
    return head.eval().to(torch.float32), temperature


def teacher_distributions(feature_dir: Path, head: Any, temperatures: dict[str, float]):
    import torch
    from safetensors.torch import load_file

    joint = load_file(str(feature_dir / "joint.safetensors"))["L32"]
    rows = [
        json.loads(line)
        for line in (feature_dir / "rows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    output = []
    with torch.inference_mode():
        for row in rows:
            if not row["j_valid"]:
                output.append({"id": row["id"], "valid": False})
                continue
            block = joint[row["j_offset"] : row["j_offset"] + row["j_count"]]
            logits = head(block[None, :-1], block[None, -1]).squeeze(0).tolist()
            probabilities = distribution(logits, temperatures[row["task_type"]])
            output.append(
                {
                    "id": row["id"],
                    "valid": True,
                    "task_type": row["task_type"],
                    "keys": row["keys"],
                    "logits": logits,
                    "probabilities": probabilities,
                }
            )
    return rows, output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    rows_cmd = commands.add_parser("example-rows")
    rows_cmd.add_argument("--lux-root", type=Path, required=True)
    rows_cmd.add_argument("--output", type=Path, required=True)
    teach = commands.add_parser("teacher")
    teach.add_argument("--lux-root", type=Path, required=True)
    teach.add_argument("--example-features", type=Path, required=True)
    teach.add_argument("--train-features", type=Path, required=True)
    teach.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "example-rows":
        pins.verify_source("lux1-9b", args.lux_root)
        example = json.loads(
            (args.lux_root / "model-card-example.json").read_text(encoding="utf-8")
        )
        with args.output.open("x", encoding="utf-8") as stream:
            for row in example_rows(example):
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        return

    pins.verify_source("lux1-9b", args.lux_root)
    head, temperatures = load_lux_head(args.lux_root)
    example = json.loads(
        (args.lux_root / "model-card-example.json").read_text(encoding="utf-8")
    )
    published = published_slots(example)
    example_meta, example_out = teacher_distributions(
        args.example_features, head, temperatures
    )
    flattened = {row["id"]: row for row in example_rows(example)}
    checks = [
        compare(flattened[item["id"]], item["probabilities"], published[item["id"]])
        for item in example_out
    ]
    parity = {
        "slots": len(checks),
        "all_same_category": all(check["same_category"] for check in checks),
        "max_drift": max(check["max_drift"] for check in checks),
        "gate_max_drift": PARITY_MAX_DRIFT,
        "checks": checks,
    }
    parity["passed"] = (
        parity["slots"] == len(published)
        and parity["all_same_category"]
        and parity["max_drift"] <= PARITY_MAX_DRIFT
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "parity.json").write_text(
        json.dumps(parity, indent=2) + "\n", encoding="utf-8"
    )
    if not parity["passed"]:
        raise SystemExit(f"Lux teacher parity failed: {json.dumps(parity)}")
    train_meta, train_out = teacher_distributions(
        args.train_features, head, temperatures
    )
    valid = [item for item in train_out if item["valid"]]
    if len(valid) != len(train_out) or any(
        not all(math.isfinite(p) and p >= 0 for p in item["probabilities"])
        or abs(sum(item["probabilities"]) - 1) > 1e-6
        for item in valid
    ):
        raise SystemExit("Lux teacher coverage or probability validity failed")
    with (args.output / "train-teacher.jsonl").open("x", encoding="utf-8") as stream:
        for item in train_out:
            stream.write(json.dumps(item, separators=(",", ":")) + "\n")
    summary = {
        "rows": len(train_out),
        "by_type": {
            kind: sum(item["task_type"] == kind for item in valid)
            for kind in ("choice", "noul", "score")
        },
        "temperatures": temperatures,
        "teacher_file_sha256": pins.file_sha256(args.output / "train-teacher.jsonl"),
        "parity_sha256": pins.file_sha256(args.output / "parity.json"),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
