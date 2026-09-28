"""Per-row token counts for recipe sampling (native decoder encode and raw Kai).

    python3 -m v2.data.m2.row_tokens --tokenizers tokenizers.json \\
        --native qwen3.5-0.8b-base@dc7cdfe2 --raw kai-0.6b@7185f514 \\
        --rows a.jsonl [--rows b.jsonl ...] --out tokens.jsonl

Uses the same tokenizers and ``encode`` path as ``v2.data.freeze``; writes one
``{"id", "native", "kai"}`` line per row, sorted by id.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from v2.data.freeze import _native_encode, load_tokenizer, raw_text
from v2.data.m2.common import read_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokenizers", type=Path, required=True)
    parser.add_argument("--native", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--rows", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    specs = {s["name"]: s for s in json.loads(args.tokenizers.read_text())}
    native_tok = load_tokenizer(specs[args.native])
    raw_tok = load_tokenizer(specs[args.raw])
    encode = _native_encode()
    lines = []
    for path in args.rows:
        for row in read_jsonl(path):
            native = len(encode(row, native_tok, sys.maxsize)["ids"])
            kai = len(raw_tok.encode(raw_text(row), add_special_tokens=False))
            lines.append((row["id"], native, kai))
    lines.sort()
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        for ident, native, kai in lines:
            stream.write(json.dumps({"id": ident, "native": native, "kai": kai}) + "\n")
    print(json.dumps({"rows": len(lines), "native": sum(n for _, n, _ in lines)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
