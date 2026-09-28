"""Convert native own-1.0 teacher outputs on RP-v1 into R2 replay rows.

Every teacher receipt must match the RP-v1 prompt it answered (the collector's
``source_input_sha256`` over ``{state, questions}``), carry the expected model
identity and an attested revision, and report the validated runtime. Rows the
teacher could not answer natively (for example over its input cap) get no
target; they are counted, never filled in.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from training.model.data import canonical, validate_row
from v2.data.build_a0_variants import native_prompt


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def collector_digest(prompt: dict[str, Any]) -> str:
    # Must equal inference.run.digest: key order as stored in the prompt file, unsorted.
    payload = {"state": prompt["state"], "questions": prompt["questions"]}
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def teacher_distribution(
    row: dict[str, Any], answer: dict[str, Any]
) -> dict[str, float]:
    keys = [option["key"] for option in row["options"]]
    kind = row["task_type"]
    if answer.get("type") != kind:
        raise ValueError(f"{row['id']}: teacher answered a different question type")
    if kind == "noul":
        p_true = float(answer["noul"])
        raw = {"false": 1.0 - p_true, "true": p_true}
    else:
        raw = {str(key): float(value) for key, value in answer["probabilities"].items()}
    if set(raw) != set(keys):
        raise ValueError(f"{row['id']}: teacher keys differ from the option keys")
    if any(not math.isfinite(value) or value < 0 for value in raw.values()):
        raise ValueError(
            f"{row['id']}: teacher probability is not finite and nonnegative"
        )
    total = sum(raw.values())
    if abs(total - 1.0) > 1e-4:
        raise ValueError(f"{row['id']}: teacher probabilities sum to {total}")
    return {key: raw[key] / total for key in keys}


def _entropy(probs: list[float]) -> float:
    if len(probs) < 2:
        return 0.0
    return -sum(p * math.log(p) for p in probs if p > 0) / math.log(len(probs))


def convert(
    train_rows: list[dict[str, Any]],
    prompt_rows: list[dict[str, Any]],
    teacher_rows: list[dict[str, Any]],
    *,
    teacher: str,
    model_id: str,
    revision: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id = {row["id"]: row for row in teacher_rows}
    prompts = {row["id"]: row for row in prompt_rows}
    ids = {row["id"] for row in train_rows}
    if len(by_id) != len(teacher_rows) or len(prompts) != len(prompt_rows):
        raise ValueError("duplicate teacher receipt or prompt id")
    if set(by_id) != ids or set(prompts) != ids:
        raise ValueError(
            "teacher receipts or prompts do not cover exactly the RP-v1 ids"
        )
    replay: list[dict[str, Any]] = []
    stats: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    sums: dict[str, collections.defaultdict[str, float]] = collections.defaultdict(
        lambda: collections.defaultdict(float)
    )
    for row in sorted(train_rows, key=lambda item: item["id"]):
        receipt = by_id[row["id"]]
        if (
            receipt.get("model_id") != model_id
            or receipt.get("model_revision") != revision
        ):
            raise ValueError(f"{row['id']}: teacher identity differs")
        if receipt.get("revision_attested") is not True:
            raise ValueError(f"{row['id']}: teacher revision is not attested")
        if receipt.get("runtime_matches_validated") is not True:
            raise ValueError(
                f"{row['id']}: teacher runtime differs from its validated runtime"
            )
        prompt = prompts[row["id"]]
        if canonical(prompt) != canonical(native_prompt(row)):
            raise ValueError(f"{row['id']}: prompt file differs from its training row")
        expected = collector_digest(prompt)
        if receipt.get("source_input_sha256") != expected:
            raise ValueError(f"{row['id']}: teacher answered a different prompt")
        kind = row["task_type"]
        stats[kind]["rows"] += 1
        answer = receipt["answers"].get("decision")
        if answer is None:
            stats[kind]["no_native_answer"] += 1
            continue
        probs = teacher_distribution(row, answer)
        ordered = [probs[option["key"]] for option in row["options"]]
        gold = ordered[row["label"]]
        stats[kind]["targets"] += 1
        stats[kind]["argmax_correct"] += int(
            max(range(len(ordered)), key=lambda index: (ordered[index], -index))
            == row["label"]
        )
        sums[kind]["gold_probability"] += gold
        sums[kind]["normalized_entropy"] += _entropy(ordered)
        sums[kind]["brier"] += sum(
            (p - (1.0 if index == row["label"] else 0.0)) ** 2
            for index, p in enumerate(ordered)
        )
        new = dict(row)
        new["teacher_probs"] = probs
        meta = dict(row["audit_metadata"])
        meta["replay_teacher"] = {
            "teacher": teacher,
            "model_id": model_id,
            "model_revision": revision,
            "source_input_sha256": expected,
            "temperature": receipt.get("temperature"),
            "adapter_version": receipt.get("adapter_version"),
        }
        new["audit_metadata"] = meta
        validate_row(new, "train", replay=True)
        replay.append(new)
    report: dict[str, Any] = {
        "teacher": teacher,
        "model_id": model_id,
        "model_revision": revision,
    }
    for kind in sorted(stats):
        counts = stats[kind]
        n = counts["targets"]
        report[kind] = {
            "rows": counts["rows"],
            "targets": n,
            "no_native_answer": counts["no_native_answer"],
            "argmax_accuracy_vs_gold": (
                round(counts["argmax_correct"] / n, 4) if n else None
            ),
            "mean_gold_probability": (
                round(sums[kind]["gold_probability"] / n, 4) if n else None
            ),
            "mean_normalized_entropy": (
                round(sums[kind]["normalized_entropy"] / n, 4) if n else None
            ),
            "mean_brier": round(sums[kind]["brier"] / n, 4) if n else None,
        }
    return replay, report


def repeat_max_abs_diff(
    full: list[dict[str, Any]], repeat: list[dict[str, Any]]
) -> dict[str, Any]:
    first = {row["id"]: row for row in full}
    worst, compared, mismatched = 0.0, 0, 0
    for row in repeat:
        a, b = first[row["id"]]["answers"]["decision"], row["answers"]["decision"]
        if (a is None) != (b is None):
            mismatched += 1
            continue
        if a is None:
            continue
        compared += 1
        if a.get("type") == "noul":
            worst = max(worst, abs(float(a["noul"]) - float(b["noul"])))
        else:
            for key, value in a["probabilities"].items():
                worst = max(worst, abs(float(value) - float(b["probabilities"][key])))
    return {
        "compared": compared,
        "validity_mismatches": mismatched,
        "max_abs_diff": worst,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--teacher-output", type=Path, required=True)
    parser.add_argument("--repeat-output", type=Path)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    train_rows = load_jsonl(args.train)
    teacher_rows = load_jsonl(args.teacher_output)
    replay, report = convert(
        train_rows,
        load_jsonl(args.prompts),
        teacher_rows,
        teacher=args.teacher,
        model_id=args.model_id,
        revision=args.revision,
    )
    if args.repeat_output:
        report["repeat"] = repeat_max_abs_diff(
            teacher_rows, load_jsonl(args.repeat_output)
        )
    data = "".join(canonical(row) + "\n" for row in replay).encode("utf-8")
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
    report["rows"] = len(replay)
    report["content_sha256"] = hashlib.sha256(data).hexdigest()
    fd = os.open(args.report, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=1, sort_keys=True)
    print(json.dumps(report, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
