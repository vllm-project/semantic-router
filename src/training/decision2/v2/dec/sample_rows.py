"""Deterministic stratified sample of a labeled slice (for example A7 AHO).

Takes at most ``--per-cell`` rows per (family, task type) cell in a seed-keyed
hash order, skipping rows above ``--max-tokens`` native tokens, and writes them
unchanged with a manifest of cell counts and hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from training.model.data import file_sha256, load_partition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--partition", default="select")
    parser.add_argument("--per-cell", type=int, default=80)
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--seed", default="dec-sample-v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    from transformers import AutoTokenizer

    from training.model.decision_model import encode

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    cells = defaultdict(list)
    for row in load_partition(args.input, args.partition):
        cells[(row["family"], row["task_type"])].append(row)
    chosen = []
    for key in sorted(cells):
        ordered = sorted(
            cells[key],
            key=lambda r: hashlib.sha256(
                f"{args.seed}\0{r['id']}".encode()
            ).hexdigest(),
        )
        taken = 0
        for row in ordered:
            if taken >= args.per_cell:
                break
            if len(encode(row, tokenizer, 8192)["ids"]) <= args.max_tokens:
                chosen.append(row)
                taken += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for row in chosen:
            stream.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    manifest = {
        "schema_version": "dec-sample/1",
        "input_sha256": file_sha256(args.input),
        "seed": args.seed,
        "per_cell": args.per_cell,
        "max_tokens": args.max_tokens,
        "rows": len(chosen),
        "cells": {
            f"{f}|{t}": n
            for (f, t), n in sorted(
                Counter((r["family"], r["task_type"]) for r in chosen).items()
            )
        },
        "output_sha256": file_sha256(args.output),
    }
    args.output.with_name(args.output.name + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"rows": manifest["rows"], "cells": len(manifest["cells"])}))


if __name__ == "__main__":
    main()
