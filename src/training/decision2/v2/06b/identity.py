"""Compare two native receipt files answer-by-answer (latency and paths ignored)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, write_json


def _winner(answer: dict[str, Any]) -> Any:
    if "error" in answer:
        return ("invalid", answer["error"])
    if answer["type"] == "noul":
        return answer["noul"] > 0.5
    return (
        answer.get("choice")
        if answer["type"] == "choice"
        else max(answer["probabilities"], key=answer["probabilities"].get)
    )


def _values(answer: dict[str, Any]) -> list[float]:
    if "error" in answer:
        return []
    if answer["type"] == "noul":
        return [answer["noul"]]
    return [answer["probabilities"][k] for k in sorted(answer["probabilities"])] + (
        [answer["score"]] if "score" in answer else []
    )


def compare(reference: Path, candidate: Path) -> dict[str, Any]:
    left = {row["id"]: row for row in read_jsonl(reference)}
    right = {row["id"]: row for row in read_jsonl(candidate)}
    if set(left) != set(right):
        raise ValueError("Receipt files cover different item IDs")
    answers = identical = changed = invalid_mismatch = input_mismatch = 0
    drift = 0.0
    for item_id, a in left.items():
        b = right[item_id]
        input_mismatch += a.get("source_input_sha256") != b.get("source_input_sha256")
        if set(a["answers"]) != set(b["answers"]):
            raise ValueError(f"{item_id}: question IDs differ")
        for qid, x in a["answers"].items():
            y = b["answers"][qid]
            answers += 1
            identical += x == y
            invalid_mismatch += ("error" in x) != ("error" in y)
            changed += _winner(x) != _winner(y)
            vx, vy = _values(x), _values(y)
            if len(vx) == len(vy):
                drift = max([drift] + [abs(p - q) for p, q in zip(vx, vy)])
    return {
        "items": len(left),
        "answers": answers,
        "identical_answers": identical,
        "winner_changes": changed,
        "validity_mismatches": invalid_mismatch,
        "input_hash_mismatches": input_mismatch,
        "max_abs_value_drift": drift,
        "exact_match": identical == answers and input_mismatch == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        **compare(args.reference, args.candidate),
    }
    write_json(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
