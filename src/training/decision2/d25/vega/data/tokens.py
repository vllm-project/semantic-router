"""Prompt token lengths (Qwen3.8 tokenizer, d25-vega prompt) for every row of a row set, cached per file.

    python -m d25.vega.data.tokens --tokenizer /models/Qwen3.8-27B --rows DATA_ROOT/rows/tasksource/part-*.jsonl.gz

Writes <file>.ntok.jsonl.gz next to each rows file: {"id": ..., "n": prompt tokens incl. chat template}
(n = -1 when the row cannot be rendered).
"""

from __future__ import annotations

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

from d25.vega.common import decision_format as df
from d25.vega.data.util import read_jsonl, write_jsonl

_TOK = None
_CODES = None


def _init(path: str) -> None:
    global _TOK, _CODES
    from transformers import AutoTokenizer

    _TOK = AutoTokenizer.from_pretrained(path)
    _CODES, _ = df.answer_codes(_TOK)


def count(rows: list[dict]) -> list[dict]:
    assert _TOK is not None
    texts = []
    for row in rows:
        try:
            texts.append(df.render(_TOK, row["state"], row["question"], _CODES))
        except ValueError:
            texts.append(None)
    encoded = _TOK([t or "" for t in texts], add_special_tokens=False)["input_ids"]
    return [
        {"id": row["id"], "n": len(ids) if text is not None else -1}
        for row, text, ids in zip(rows, texts, encoded)
    ]


def ntok_path(path: str) -> str:
    return path.replace(".jsonl.gz", ".ntok.jsonl.gz")


def load_lengths(paths: list[str]) -> dict[str, int]:
    lengths = {}
    for path in paths:
        for item in read_jsonl(ntok_path(path)):
            lengths[item["id"]] = item["n"]
    return lengths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="/models/Qwen3.8-27B")
    parser.add_argument("--rows", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    args = parser.parse_args()
    files = sorted(
        {str(Path(p)) for p in args.rows if not p.endswith(".ntok.jsonl.gz")}
    )
    with Pool(args.workers, initializer=_init, initargs=(args.tokenizer,)) as pool:
        for path in files:
            out = ntok_path(path)
            if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(path):
                print("cached", out, flush=True)
                continue
            rows = list(read_jsonl(path))
            chunks = [rows[i : i + 512] for i in range(0, len(rows), 512)]
            result = [
                item for part in pool.map(count, chunks, chunksize=1) for item in part
            ]
            write_jsonl(out, result)
            print(out, len(result), flush=True)


if __name__ == "__main__":
    main()
