"""Convert pinned parquet source files to JSONL with identical field names.

Runs inside the pinned runtime image (pyarrow). Writes ``<name>.jsonl`` next to
each input as ``train.jsonl`` for ``train-*.parquet`` files and prints input and
output SHA-256 values; refuses to overwrite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(parquet: Path, target: Path) -> dict[str, str | int]:
    import pyarrow.parquet as pq

    rows = pq.read_table(parquet).to_pylist()
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return {
        "input": str(parquet),
        "input_sha256": sha_file(parquet),
        "output": str(target),
        "output_sha256": sha_file(target),
        "rows": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.parquet:
        stem = "train" if path.name.startswith("train-") else path.stem
        print(
            json.dumps(convert(path, path.with_name(f"{stem}.jsonl")), sort_keys=True)
        )


if __name__ == "__main__":
    main()
