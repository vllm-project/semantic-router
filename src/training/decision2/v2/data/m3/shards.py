"""Split a teacher prompt file into GPU shards and merge the shard outputs (M3a).

    python3 -m v2.data.m3.shards split --prompts P.jsonl --shards 3 --out-prefix DIR/aj-m
    python3 -m v2.data.m3.shards merge --prompts P.jsonl --output S0.jsonl ... --out merged.jsonl

Shard k holds the prompts with ``int(sha256("m3a-shard:" + id), 16) % n == k`` (input
order, byte-identical lines). A merge requires every prompt id exactly once across the
shard outputs and writes the receipt lines unchanged, sorted by id.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

SALT = "m3a-shard:"


def shard_of(ident: str, shards: int) -> int:
    return int(hashlib.sha256((SALT + ident).encode("utf-8")).hexdigest(), 16) % shards


def _ids_and_lines(path: Path) -> list[tuple[str, bytes]]:
    out = []
    with path.open("rb") as stream:
        for line in stream:
            if line.strip():
                out.append(
                    (
                        json.loads(line)["id"],
                        line if line.endswith(b"\n") else line + b"\n",
                    )
                )
    return out


def _write(path: Path, data: bytes) -> str:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
    return hashlib.sha256(data).hexdigest()


def split(prompts: Path, shards: int, prefix: str) -> dict:
    parts: list[list[bytes]] = [[] for _ in range(shards)]
    for ident, line in _ids_and_lines(prompts):
        parts[shard_of(ident, shards)].append(line)
    return {
        f"{prefix}.shard{k}.prompts.jsonl": {
            "rows": len(lines),
            "sha256": _write(Path(f"{prefix}.shard{k}.prompts.jsonl"), b"".join(lines)),
        }
        for k, lines in enumerate(parts)
    }


def merge(prompts: Path, outputs: list[Path], out: Path) -> dict:
    wanted = {ident for ident, _ in _ids_and_lines(prompts)}
    merged: dict[str, bytes] = {}
    for path in outputs:
        for ident, line in _ids_and_lines(path):
            if ident in merged or ident not in wanted:
                raise ValueError(f"{path}: duplicate or unknown id")
            merged[ident] = line
    if set(merged) != wanted:
        raise ValueError(f"{len(wanted - set(merged))} prompts have no receipt")
    return {
        "rows": len(merged),
        "sha256": _write(out, b"".join(merged[i] for i in sorted(merged))),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("split")
    s.add_argument("--prompts", type=Path, required=True)
    s.add_argument("--shards", type=int, required=True)
    s.add_argument("--out-prefix", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--prompts", type=Path, required=True)
    m.add_argument("--output", type=Path, action="append", required=True)
    m.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "split":
        result = split(args.prompts, args.shards, args.out_prefix)
    else:
        result = merge(args.prompts, args.output, args.out)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
