"""13-gram decontamination of training rows against evaluation texts.

Evaluation side: every string value inside the given JSONL / JSONL.GZ files (Index suite rows, JevArena and JevBench
prompt panels, our dev panels). Candidate side: the state, instructions and option descriptions of each row. A row is
flagged when it shares a lower-cased alphanumeric 13-gram with any evaluation text, unless that 13-gram occurs in
more than ``--boilerplate-df`` evaluation records (instruction boilerplate, not item content).
"""

from __future__ import annotations

import gzip
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

N = 13
_TOKEN = re.compile(r"[a-z0-9]+")


def strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            if key in ("id", "group_id", "input_sha256", "_evaluation"):
                continue
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def grams(text: str) -> set[int]:
    tokens = _TOKEN.findall(text.lower())
    return {hash(" ".join(tokens[i : i + N])) for i in range(len(tokens) - N + 1)}


def _records(path: Path) -> Iterator[Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def build_index(paths: Iterable[Path]) -> tuple[set[int], int]:
    index: set[int] = set()
    count = 0
    for path in paths:
        for record in _records(path):
            count += 1
            for text in strings(record):
                index |= grams(text)
    return index, count


def row_grams(row: dict[str, Any]) -> set[int]:
    out: set[int] = set()
    for text in strings(
        {"s": row["state"], "i": row["instructions"], "o": row["options"]}
    ):
        out |= grams(text)
    return out


def document_frequency(paths: Iterable[Path], wanted: set[int]) -> Counter:
    df: Counter = Counter()
    for path in paths:
        for record in _records(path):
            seen: set[int] = set()
            for text in strings(record):
                seen |= grams(text) & wanted
            df.update(seen)
    return df


def scan(
    rows: list[dict[str, Any]], eval_paths: list[Path], boilerplate_df: int = 25
) -> dict[str, Any]:
    """Flag rows; returns {"flagged": set(ids), "eval_records": n, "hits": n, "boilerplate": n}."""
    index, count = build_index(eval_paths)
    hits_by_row = {}
    all_hits: set[int] = set()
    for row in rows:
        hits = row_grams(row) & index
        if hits:
            hits_by_row[row["id"]] = hits
            all_hits |= hits
    del index
    df = document_frequency(eval_paths, all_hits) if all_hits else Counter()
    boiler = {g for g, n in df.items() if n > boilerplate_df}
    flagged = {rid for rid, hits in hits_by_row.items() if hits - boiler}
    return {
        "flagged": flagged,
        "eval_records": count,
        "rows_with_hits": len(hits_by_row),
        "boilerplate_grams": len(boiler),
        "flagged_rows": len(flagged),
    }
