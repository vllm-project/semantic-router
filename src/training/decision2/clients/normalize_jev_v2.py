"""Rebind legacy Jev DEV predictions to verified gold-free prompt payloads.

The v1 collector stored the full API-body hash (including model) in the
benchmark's source_input_sha256 field. This converter verifies that hash and
the original response against immutable receipts, then emits a separate file
with the state/questions payload hash required by the v2 scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from benchmark.generate import digest
from clients.jev_api import canonical

NORMALIZATION_VERSION = "jev-dev-input-binding/2"


def read_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank line")
            row = json.loads(line)
            item_id = row.get("id") if isinstance(row, dict) else None
            if not isinstance(item_id, str) or not item_id or item_id in rows:
                raise ValueError(f"{path}:{line_number}: missing or duplicate ID")
            rows[item_id] = row
    if not rows:
        raise ValueError(f"{path}: empty JSONL")
    return rows


def normalize(
    prompts_path: Path,
    receipts_path: Path,
    legacy_predictions_path: Path,
    output_path: Path,
    expected_model: str = "jev-1.13.0",
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(output_path)
    prompts = read_rows(prompts_path)
    receipts = read_rows(receipts_path)
    legacy = read_rows(legacy_predictions_path)
    if set(prompts) != set(receipts) or set(prompts) != set(legacy):
        raise ValueError("Prompt, receipt, and legacy prediction IDs differ")

    converted = []
    for item_id, prompt in prompts.items():
        if set(prompt) != {"id", "state", "questions"} or not isinstance(
            prompt["questions"], dict
        ):
            raise ValueError(
                f"{item_id}: prompt is not a gold-free state/questions record"
            )
        receipt, old = receipts[item_id], legacy[item_id]
        payload = {"state": prompt["state"], "questions": prompt["questions"]}
        body = {
            "state": prompt["state"],
            "model": expected_model,
            "questions": prompt["questions"],
        }
        api_hash = hashlib.sha256(canonical(body)).hexdigest()
        if (
            receipt.get("input_sha256") != api_hash
            or receipt.get("requested_model") != expected_model
            or receipt.get("returned_model") != expected_model
            or receipt.get("http_status") != 200
        ):
            raise ValueError(
                f"{item_id}: receipt does not bind the prompt and expected API model"
            )
        response = receipt.get("response")
        answers = response.get("answers") if isinstance(response, dict) else None
        if not isinstance(answers, dict) or set(answers) != set(prompt["questions"]):
            raise ValueError(
                f"{item_id}: receipt answer IDs differ from prompt questions"
            )
        if (
            old.get("source_input_sha256") != api_hash
            or old.get("answers") != answers
            or old.get("model") != expected_model
            or old.get("http_status") != 200
            or old.get("usage") != response.get("usage")
        ):
            raise ValueError(
                f"{item_id}: legacy prediction differs from verified receipt"
            )
        converted.append(
            {
                **old,
                "source_input_sha256": digest(payload),
                "api_body_sha256": api_hash,
                "normalization_version": NORMALIZATION_VERSION,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as target:
        for row in converted:
            target.write(
                json.dumps(
                    row, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            )
    return {
        "normalization_version": NORMALIZATION_VERSION,
        "items": len(converted),
        "prompts_sha256": hashlib.sha256(prompts_path.read_bytes()).hexdigest(),
        "receipts_sha256": hashlib.sha256(receipts_path.read_bytes()).hexdigest(),
        "legacy_predictions_sha256": hashlib.sha256(
            legacy_predictions_path.read_bytes()
        ).hexdigest(),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--legacy-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-model", default="jev-1.13.0")
    args = parser.parse_args()
    result = normalize(
        args.prompts,
        args.receipts,
        args.legacy_predictions,
        args.output,
        args.expected_model,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
