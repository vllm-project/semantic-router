"""Compare official-Qwen zero-step native probabilities to archived control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.model.data import file_sha256

CONTROL_SHA256 = "e62731df7a8b9f35c6ecda17aa763b1dd30700522b3cd8c082f8063f2829f5c3"


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 700 or len({row["id"] for row in rows}) != 700:
        raise ValueError("Expected 700 unique SELECT predictions")
    return rows


def compare(control: list[dict[str, Any]], treatment: list[dict[str, Any]]) -> dict:
    if [row["id"] for row in control] != [row["id"] for row in treatment]:
        raise ValueError("SELECT identity or order differs")
    changes = 0
    drift = 0.0
    for before, after in zip(control, treatment, strict=True):
        if (
            before["prompt_sha256"] != after["prompt_sha256"]
            or before["token_ids_sha256"] != after["token_ids_sha256"]
            or before["answer"]["type"] != after["answer"]["type"]
        ):
            raise ValueError("Native SELECT request differs")
        changes += before["prediction_key"] != after["prediction_key"]
        left, right = before["answer"], after["answer"]
        if left["type"] == "noul":
            drift = max(drift, abs(left["noul"] - right["noul"]))
        else:
            if set(left["probabilities"]) != set(right["probabilities"]):
                raise ValueError("Candidate option domain differs")
            drift = max(
                drift,
                *(
                    abs(left["probabilities"][key] - right["probabilities"][key])
                    for key in left["probabilities"]
                ),
            )
    return {
        "schema": "decision2-autojev-kl06-zero-parity/1",
        "rows": len(control),
        "category_changes": changes,
        "max_probability_drift": drift,
        "status": "PASS" if changes == 0 and drift <= 1e-4 else "HOLD",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--treatment", required=True, type=Path)
    args = parser.parse_args()
    if file_sha256(args.control) != CONTROL_SHA256:
        raise ValueError("Archived control baseline bytes differ")
    result = compare(read_rows(args.control), read_rows(args.treatment))
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
