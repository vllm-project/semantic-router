"""Convert raw official Jev receipts to the benchmark prediction contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def normalize(input_path: Path, output_path: Path, expected_model: str) -> None:
    if output_path.exists():
        raise FileExistsError(output_path)
    seen: set[str] = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(encoding="utf-8") as source, output_path.open(
        "x", encoding="utf-8"
    ) as target:
        for line_number, line in enumerate(source, 1):
            raw = json.loads(line)
            item_id = raw["id"]
            if item_id in seen:
                raise ValueError(
                    f"Duplicate receipt id at line {line_number}: {item_id}"
                )
            seen.add(item_id)
            response = raw.get("response") or {}
            valid = (
                raw.get("http_status") == 200
                and raw.get("returned_model") == expected_model
            )
            answers = response.get("answers") if valid else None
            if not isinstance(answers, dict):
                answers = {}
            record = {
                "id": item_id,
                "answers": answers,
                "latency_ms": raw.get("latency_seconds", 0) * 1000,
                "usage": response.get("usage") if valid else None,
                "source_input_sha256": raw.get("input_sha256"),
                "model": raw.get("returned_model"),
                "http_status": raw.get("http_status"),
            }
            target.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    if not seen:
        raise ValueError("Empty receipt file")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-model", default="jev-1.13.0")
    args = parser.parse_args()
    normalize(args.input, args.output, args.expected_model)


if __name__ == "__main__":
    main()
