"""Teacher prompt waves for XL rows without targets (M3b prereg §3).

    python3 -m v2.data.m3.xl_prompts --pools xl-pools.json \\
        --missing mx-xl-full.lux1.missing.jsonl --missing mx-xl-short.lux1.missing.jsonl \\
        --wave-size 60000 --prefix OUT/lux-xl

Ids of the first ``--missing`` file come first, then new ids of each later file, each part
sorted by id; the sequence is cut into waves of at most ``--wave-size``. Writes
``<prefix>-w<k>.rows.jsonl`` (canonical rows) and ``<prefix>-w<k>.prompts.jsonl`` (native
prompts) per wave and prints counts and hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from training.model.data import canonical
from v2.data.build_a0_variants import native_prompt
from v2.data.m2.common import read_jsonl


def _write(path: Path, lines: list[str]) -> str:
    data = "".join(lines).encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pools", type=Path, required=True)
    parser.add_argument("--missing", type=Path, action="append", required=True)
    parser.add_argument("--wave-size", type=int, default=60000)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args(argv)
    order: list[str] = []
    seen: set[str] = set()
    for path in args.missing:
        part = sorted({r["id"] for r in read_jsonl(path)} - seen)
        order += part
        seen |= set(part)
    rows = {}
    for spec in json.loads(args.pools.read_text()).values():
        for path in spec["rows"]:
            for row in read_jsonl(Path(path)):
                if row["id"] in seen:
                    rows.setdefault(row["id"], row)
    if set(rows) != seen:
        raise ValueError(f"{len(seen - set(rows))} ids not found in the pools")
    out = {}
    for k in range(0, len(order), args.wave_size):
        wave = order[k : k + args.wave_size]
        name = f"{args.prefix}-w{k // args.wave_size + 1}"
        out[name] = {
            "rows": len(wave),
            "rows_sha256": _write(
                Path(name + ".rows.jsonl"), [canonical(rows[i]) + "\n" for i in wave]
            ),
            "prompts_sha256": _write(
                Path(name + ".prompts.jsonl"),
                [
                    json.dumps(native_prompt(rows[i]), ensure_ascii=False) + "\n"
                    for i in wave
                ],
            ),
        }
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
