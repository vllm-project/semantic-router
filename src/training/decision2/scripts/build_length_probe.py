"""Extract a long eligible TRAIN row for a one-step backbone memory probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path, role: str) -> list[dict]:
    value = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not value or any(row.get("evaluation_role") != role for row in value):
        raise ValueError(f"{path}: wrong or empty partition")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--select", type=Path, required=True)
    parser.add_argument("--cal", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or args.max_length < 1:
        raise ValueError("Fresh output directory and positive max length required")
    from training.model.decision_model import segments
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    source = rows(args.train, "train")
    eligible = []
    for row in source:
        prefix, options, suffix = segments(row)
        length = sum(
            len(tokenizer.encode(part, add_special_tokens=False))
            for part in (prefix, *options, suffix)
        )
        if length <= args.max_length:
            eligible.append((length, row["id"], row))
    if not eligible:
        raise ValueError("No eligible row")
    length, _, longest = max(eligible)
    selected, calibration = (
        rows(args.select, "select")[:12],
        rows(args.cal, "calibrate")[:12],
    )
    if len({longest["group_id"]} & {row["group_id"] for row in selected + calibration}):
        raise ValueError("TRAIN probe group overlaps SELECT/CAL")
    args.output_dir.mkdir(parents=True)
    products = {"train": [longest], "select": selected, "cal": calibration}
    checksums = {}
    for split, data in products.items():
        target = args.output_dir / f"{split}.jsonl"
        target.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                for row in data
            ),
            encoding="utf-8",
        )
        checksums[split] = digest(target)
    manifest = {
        "version": "decision2-length-probe/1",
        "train_id": longest["id"],
        "train_tokens": length,
        "max_length": args.max_length,
        "source_sha256": {name: digest(getattr(args, name)) for name in products},
        "split_sha256": checksums,
        "note": "One-step memory feasibility only; no accuracy conclusion",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
