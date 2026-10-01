"""Decoder M11 label-prompt lengths (prereg "NT max length"), CPU, in the decoder image.

Encodes every row of each named file with the label-token prompt (``v2.dec.label_token.encode_label``, the real
tokenizer of the NT arm's 1.0 start, no length limit) and reports the longest prompt, the rows above 8,192 / 8,448
and the NT ``--max-length``: the smallest multiple of 256 that is >= 8,448 and >= the longest prompt.

usage: m11_lengths.py --tokenizer DIR --file NAME=PATH:PARTITION [--file ...] --workers N --output OUT.json
"""

from __future__ import annotations

import argparse
import json
import math
from multiprocessing import Pool
from pathlib import Path
from typing import Any

FLOOR = 8448
STEP = 256
_TOKENIZER: Any = None


def nt_max_length(longest: int) -> int:
    return max(FLOOR, STEP * math.ceil(longest / STEP))


def _init(tokenizer_dir: str) -> None:
    global _TOKENIZER
    from transformers import AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)


def _length(row: dict[str, Any]) -> tuple[str, int]:
    from v2.dec.label_token import encode_label

    return row["id"], len(encode_label(row, _TOKENIZER, 1 << 30)["ids"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--file", action="append", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from training.model.data import file_sha256, load_partition

    report: dict[str, Any] = {"tokenizer": args.tokenizer, "files": {}}
    longest = 0
    with Pool(args.workers, initializer=_init, initargs=(args.tokenizer,)) as pool:
        for spec in args.file:
            name, rest = spec.split("=", 1)
            path, partition = rest.rsplit(":", 1)
            rows = load_partition(path, partition)
            lengths = pool.map(_length, rows, chunksize=64)
            top = sorted(lengths, key=lambda x: -x[1])[:10]
            report["files"][name] = {
                "path": path,
                "sha256": file_sha256(path),
                "rows": len(rows),
                "max": top[0][1],
                "over_8192": sum(n > 8192 for _, n in lengths),
                "over_8448": sum(n > FLOOR for _, n in lengths),
                "longest": [{"id": i, "tokens": n} for i, n in top],
            }
            longest = max(longest, top[0][1])
            print(
                name,
                report["files"][name]["max"],
                report["files"][name]["over_8192"],
                flush=True,
            )
    report["longest"] = longest
    report["nt_max_length"] = nt_max_length(longest)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"longest": longest, "nt_max_length": report["nt_max_length"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
