"""Check every frozen Kai recovery TRAIN row with the native 1,024-token packer."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
from pathlib import Path

from training.data.build_kai06_recovery1024 import sha_file

TRAIN_SHA = "58b94ac987b2fe134a392310a8d9d63f3d0e996a6016884f6dafadf3ae5449da"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if sha_file(args.train) != TRAIN_SHA:
        raise ValueError("Frozen recovery TRAIN bytes changed")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.native / "tokenizer", local_files_only=True
    )
    spec = importlib.util.spec_from_file_location(
        "kai_native_packing", args.native / "packing.py"
    )
    if spec is None or spec.loader is None:
        raise ValueError("Native packing module unavailable")
    packing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(packing)
    collator = packing.MarkerCollator(
        tokenizer, max_length=1024, state_truncation="error"
    )
    lengths = []
    failures = []
    with args.train.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            try:
                encoded = collator.encode(row, labeled=True)
                lengths.append(encoded["input_tokens"])
            except (KeyError, TypeError, ValueError) as error:
                failures.append({"id": row.get("id"), "reason": str(error)})
    report = {
        "schema": "decision2-kai06-recovery1024-length/1",
        "train_sha256": TRAIN_SHA,
        "packing_sha256": sha_file(args.native / "packing.py"),
        "tokenizer_sha256": sha_file(args.native / "tokenizer" / "tokenizer.json"),
        "rows": len(lengths) + len(failures),
        "valid": len(lengths),
        "failures": failures,
        "max_tokens": max(lengths) if lengths else None,
        "p95_tokens": (
            sorted(lengths)[int(0.95 * (len(lengths) - 1))] if lengths else None
        ),
        "median_tokens": statistics.median(lengths) if lengths else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output.chmod(0o600)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "rows",
                    "valid",
                    "max_tokens",
                    "p95_tokens",
                    "median_tokens",
                )
            },
            sort_keys=True,
        )
    )
    if failures:
        raise ValueError(f"{len(failures)} native rows fail full-length packing")


if __name__ == "__main__":
    main()
