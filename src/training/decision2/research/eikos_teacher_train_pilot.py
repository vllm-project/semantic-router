"""Bounded, TRAIN-only native teacher screening; never a release evaluation.

The output contains aggregate metrics and hashes, not source rows or answers.
It does not produce distillation targets or change any training partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from inference.eikos import REVISION, load_native, verify_release
from training.model.data import canonical, file_sha256, load_partition

KINDS = ("choice", "noul", "score")
PER_KIND = 32


def roster(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Choose one row per source group, using only gold-free identities."""
    picked = []
    seen_groups = set()
    for kind in KINDS:
        eligible = sorted(
            (row for row in rows if row["task_type"] == kind),
            key=lambda row: hashlib.sha256(
                f"{kind}\x00{row['group_id']}\x00{row['id']}".encode()
            ).digest(),
        )
        matching = []
        for row in eligible:
            if row["group_id"] in seen_groups:
                continue
            seen_groups.add(row["group_id"])
            matching.append(row)
            if len(matching) == PER_KIND:
                break
        if len(matching) != PER_KIND:
            raise ValueError(f"Too few independent TRAIN groups for {kind}")
        picked.extend(matching)
    return picked


def roster_sha256(rows: list[dict[str, Any]]) -> str:
    identities = [
        {"id": row["id"], "input_sha256": row["input_sha256"]} for row in rows
    ]
    return hashlib.sha256(canonical(identities).encode()).hexdigest()


def question(row: dict[str, Any]) -> dict[str, Any]:
    """Render the same native decision semantics without exposing gold labels."""
    options = row["options"]
    common = {"type": row["task_type"], "instructions": row["instructions"]}
    if row["task_type"] == "choice":
        common["criteria"] = {item["key"]: item["description"] for item in options}
    elif row["task_type"] == "noul":
        mapping = {item["key"]: item["description"] for item in options}
        common["criteria"] = {"true": mapping["true"], "false": mapping["false"]}
    else:
        common["criteria"] = [
            item["description"]
            for item in sorted(options, key=lambda item: int(item["key"]))
        ]
    return common


def probabilities(row: dict[str, Any], answer: dict[str, Any]) -> dict[str, float]:
    keys = [item["key"] for item in row["options"]]
    if answer.get("type") != row["task_type"]:
        raise ValueError("Native teacher returned a different question type")
    if row["task_type"] == "noul":
        yes = answer.get("noul")
        if type(yes) not in (int, float) or not math.isfinite(yes):
            raise ValueError("Invalid native Noul probability")
        result = {"false": 1.0 - float(yes), "true": float(yes)}
    else:
        result = answer.get("probabilities")
        if not isinstance(result, dict) or set(result) != set(keys):
            raise ValueError("Native teacher option keys differ")
    if (
        set(result) != set(keys)
        or any(
            type(value) not in (int, float) or not math.isfinite(value) or value < 0
            for value in result.values()
        )
        or abs(sum(result.values()) - 1.0) > 1e-5
    ):
        raise ValueError("Invalid native teacher distribution")
    return {key: float(result[key]) for key in keys}


def aggregate(rows: list[dict[str, Any]], native: Any) -> dict[str, Any]:
    results: dict[str, dict[str, float | int]] = {
        kind: {
            "valid": 0,
            "invalid": 0,
            "ties": 0,
            "correct": 0,
            "brier_sum": 0.0,
            "gold_probability_sum": 0.0,
        }
        for kind in KINDS
    }
    for row in rows:
        kind = row["task_type"]
        stats = results[kind]
        try:
            response = native.decide_all(
                state=row["state"], questions={"decision": question(row)}
            )
            if set(response) != {"decision"}:
                raise ValueError("Native teacher question roster differs")
            answer, _ = response["decision"]
            probs = probabilities(row, answer)
        except ValueError as error:
            if "tokens > 16000" not in str(error):
                raise
            stats["invalid"] += 1
            continue
        gold = row["options"][row["label"]]["key"]
        maximum = max(probs.values())
        winners = [key for key, value in probs.items() if abs(value - maximum) <= 1e-8]
        stats["valid"] += 1
        stats["ties"] += int(len(winners) != 1)
        stats["correct"] += int(len(winners) == 1 and winners[0] == gold)
        stats["gold_probability_sum"] += probs[gold]
        stats["brier_sum"] += (
            sum((value - int(key == gold)) ** 2 for key, value in probs.items()) / 2
        )
    return results


def write_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or not path.parent.is_dir():
        raise ValueError("Output must be a fresh file in an existing private directory")
    pending = path.with_name(path.name + ".pending")
    descriptor = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload, stream, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(pending, path)
        pending.unlink()
    except Exception:
        pending.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--train-sha256", required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--roster-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if file_sha256(args.train) != args.train_sha256:
        raise ValueError("Frozen TRAIN bytes differ")
    selected = roster(load_partition(args.train, "train"))
    identity = roster_sha256(selected)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "rows": len(selected),
                    "groups": len({row["group_id"] for row in selected}),
                    "roster_sha256": identity,
                }
            )
        )
        return
    if (
        args.revision != REVISION
        or args.roster_sha256 != identity
        or args.model_path is None
        or args.output is None
    ):
        raise ValueError("Teacher source, roster, model path and output must be frozen")
    release = verify_release(args.model_path, args.revision)
    native = load_native(args.model_path, "cuda:0")
    stats = aggregate(selected, native)
    payload = {
        "schema": "decision2-eikos-teacher-train-screen/1",
        "source": f"caiovicentino1/Eikos-4B@{REVISION}",
        **release,
        "train_sha256": args.train_sha256,
        "roster_sha256": identity,
        "script_sha256": file_sha256(__file__),
        "sampled_rows": len(selected),
        "sampled_groups": len({row["group_id"] for row in selected}),
        "by_type": stats,
    }
    write_once(args.output, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
