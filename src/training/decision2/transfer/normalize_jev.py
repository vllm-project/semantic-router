"""Verify official Jev HTTP receipts and convert them to CSS scoring records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .build import sha_value


def api_body_sha256(state, questions, model: str) -> str:
    body = {"state": state, "model": model, "questions": questions}
    encoded = json.dumps(
        body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize(
    prompts_path: Path, receipts_path: Path, output_path: Path, expected_model: str
) -> dict[str, int]:
    if output_path.exists():
        raise FileExistsError(output_path)
    prompts = {}
    with prompts_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            row = json.loads(line)
            item_id = row["id"]
            if item_id in prompts or set(row) != {"id", "state", "questions"}:
                raise ValueError(
                    f"{prompts_path}:{line_number}: duplicate ID or unexpected prompt fields"
                )
            prompts[item_id] = row
    if not prompts:
        raise ValueError("empty prompt file")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with receipts_path.open(encoding="utf-8") as source, output_path.open(
        "x", encoding="utf-8"
    ) as target:
        for line_number, line in enumerate(source, 1):
            raw = json.loads(line)
            item_id = raw.get("id")
            if item_id not in prompts or item_id in seen:
                raise ValueError(
                    f"{receipts_path}:{line_number}: unknown or duplicate receipt ID"
                )
            seen.add(item_id)
            prompt = prompts[item_id]
            expected_body = api_body_sha256(
                prompt["state"], prompt["questions"], expected_model
            )
            if (
                raw.get("input_sha256") != expected_body
                or raw.get("requested_model") != expected_model
                or raw.get("returned_model") != expected_model
                or raw.get("http_status") != 200
            ):
                raise ValueError(
                    f"{receipts_path}:{line_number}: API body, model, or HTTP status mismatch"
                )
            response = raw.get("response")
            if not isinstance(response, dict) or not isinstance(
                response.get("answers"), dict
            ):
                raise ValueError(
                    f"{receipts_path}:{line_number}: missing typed answers"
                )
            if response["answers"].keys() != prompt["questions"].keys():
                raise ValueError(
                    f"{receipts_path}:{line_number}: answer IDs differ from prompt"
                )
            payload = {"state": prompt["state"], "questions": prompt["questions"]}
            record = {
                "id": item_id,
                "answers": response["answers"],
                "source_input_sha256": sha_value(payload),
                "api_body_sha256": raw["input_sha256"],
                "model": raw["returned_model"],
                "latency_ms": raw.get("latency_seconds", 0) * 1000,
                "usage": response.get("usage"),
            }
            target.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    if not seen:
        raise ValueError("empty receipt file")
    return {
        "normalized": len(seen),
        "expected": len(prompts),
        "missing": len(prompts) - len(seen),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-model", default="jev-1.13.0")
    args = parser.parse_args()
    print(
        json.dumps(
            normalize(args.prompts, args.receipts, args.output, args.expected_model),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
