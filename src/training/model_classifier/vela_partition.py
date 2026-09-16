"""Partition immutable JSONL by measured token budget without truncating text."""

import argparse
import hashlib
import json
from pathlib import Path


def partition(rows, count_tokens, budget):
    if budget <= 0:
        raise ValueError("Token budget must be positive")
    within, over, ids = [], [], set()
    for row in rows:
        if not row.get("id") or row["id"] in ids or not row.get("group_id"):
            raise ValueError("Rows require unique IDs and source groups")
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError("Rows require nonempty text")
        ids.add(row["id"])
        measured = {**row, "actual_tokens": count_tokens(row["text"])}
        (within if measured["actual_tokens"] <= budget else over).append(measured)
    return within, over


def main():
    from transformers import AutoTokenizer  # noqa: PLC0415

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing overwrite of an immutable partition")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = [
        json.loads(line) for line in args.input.read_text().split("\n") if line.strip()
    ]
    within, over = partition(
        rows,
        lambda text: len(tokenizer(text, truncation=False, padding=False)["input_ids"]),
        args.max_length,
    )
    args.output.mkdir(parents=True)
    manifest = {
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "max_length": args.max_length,
        "truncation": False,
        "files": {},
    }
    for name, items in [("within-budget", within), ("over-budget", over)]:
        path = args.output / f"{name}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in items)
        )
        manifest["files"][name] = {
            "rows": len(items),
            "max_tokens": max((row["actual_tokens"] for row in items), default=0),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
