"""Convert pinned parquet source files to JSONL named after each file's stem.

Runs inside the pinned runtime image (pyarrow). ``a/train-00000-of-00002.parquet``
becomes ``a/train-00000-of-00002.jsonl``; existing outputs are verified by hash
against a fresh conversion instead of being overwritten. Prints one JSON receipt
per file with input and output SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode(parquet: Path) -> bytes:
    import pyarrow.parquet as pq

    rows = pq.read_table(parquet).to_pylist()
    return "".join(
        json.dumps(row, ensure_ascii=False, default=str, sort_keys=True) + "\n"
        for row in rows
    ).encode("utf-8")


def convert(parquet: Path) -> dict[str, object]:
    data = encode(parquet)
    target = parquet.with_suffix(".jsonl")
    if target.exists():
        if target.read_bytes() != data:
            raise FileExistsError(f"{target} exists with different content")
        status = "verified"
    else:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        status = "written"
    return {
        "input": str(parquet),
        "input_sha256": sha_bytes(parquet.read_bytes()),
        "output": str(target),
        "output_sha256": sha_bytes(data),
        "rows": data.count(b"\n"),
        "status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("parquet", type=Path, nargs="+")
    for path in parser.parse_args().parquet:
        print(json.dumps(convert(path), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
