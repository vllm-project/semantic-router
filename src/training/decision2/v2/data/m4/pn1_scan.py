"""PN1 short-text sentence scan (prereg A1 / A3).

The house scans skip text leaves under 20-40 characters; PN1 sentences are short. Every PN1
sentence is compared with sentence segments of protected text:

- E: the normalized sentence (>= 6 characters) equals a protected segment;
- C: the normalized sentence (>= 12 characters) is a substring of a protected leaf;
- N: character 4-gram containment >= 0.8 of the sentence in a segment, or of a segment
  (>= 12 characters) in the sentence.

Protected inputs are JSONL files (every string leaf is read), TSV files (every cell) or
parquet files (every string column). Receipts carry ids, roles and methods only, never text.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from v2.data.m4.pn1_text import norm

LAYOUT_PREFIXES = (
    "Sentence A: ",
    "Sentence B: ",
    "First sentence: ",
    "Second sentence: ",
    "Text 1: ",
    "Text 2: ",
    "(1) ",
    "(2) ",
    "A: ",
    "B: ",
)
SEGMENT_SPLIT = re.compile(r"[\n\r]+|(?<=[。．.!?！？;；])\s*")
MIN_EXACT = 6
MIN_CONTAIN = 12
THRESHOLD = 0.8
N = 4


def sentences_of(row: dict[str, Any]) -> list[str]:
    lines = [line for line in str(row["state"]).split("\n") if line.strip()]
    if len(lines) != 2:
        raise ValueError(f"{row['id']}: expected two state lines")
    out = []
    for line in lines:
        for prefix in LAYOUT_PREFIXES:
            if line.startswith(prefix):
                line = line[len(prefix) :]
                break
        else:
            raise ValueError(f"{row['id']}: unknown layout")
        out.append(line)
    return out


def grams(text: str) -> set[str]:
    return {text[i : i + N] for i in range(len(text) - N + 1)}


def _leaves(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _leaves(item)
    elif isinstance(value, list):
        for item in value:
            yield from _leaves(item)


def protected_leaves(path: Path) -> Iterator[str]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        for column in table.column_names:
            for value in table.column(column).to_pylist():
                if isinstance(value, str):
                    yield value
    elif path.suffix in (".tsv", ".txt"):
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                yield from line.rstrip("\n").split("\t")
    else:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield from _leaves(json.loads(line))


def segments(leaf: str) -> Iterable[str]:
    for piece in SEGMENT_SPLIT.split(leaf):
        value = norm(piece)
        if value:
            yield value


def scan(
    candidates: list[dict[str, Any]], roles: dict[str, list[Path]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return (hits, counts). ``candidates``: dicts with id, group_id, split, sentences."""
    items = []
    for row in candidates:
        for index, sentence in enumerate(row["sentences"]):
            value = norm(sentence)
            if len(value) >= MIN_EXACT:
                items.append((row, index, value, grams(value)))
    exact = collections.defaultdict(list)
    posting = collections.defaultdict(set)
    for k, (_, _, value, gs) in enumerate(items):
        exact[value].append(k)
        for g in gs:
            posting[g].add(k)
    long_items = {k for k, it in enumerate(items) if len(it[2]) >= MIN_CONTAIN}
    hits: dict[tuple[int, str, str], None] = {}
    counts: dict[str, Any] = {}
    for role, paths in sorted(roles.items()):
        leaves = segs = 0
        for path in paths:
            for leaf in protected_leaves(path):
                leaves += 1
                whole = norm(leaf)
                for seg in segments(leaf):
                    segs += 1
                    for k in exact.get(seg, ()):
                        hits[(k, role, "E")] = None
                    seg_grams = grams(seg)
                    shared = collections.Counter()
                    for g in seg_grams:
                        for k in posting.get(g, ()):
                            shared[k] += 1
                    for k, n in shared.items():
                        own = len(items[k][3])
                        if own and n / own >= THRESHOLD:
                            hits[(k, role, "N")] = None
                        elif (
                            len(seg) >= MIN_CONTAIN
                            and seg_grams
                            and n / len(seg_grams) >= THRESHOLD
                        ):
                            hits[(k, role, "N")] = None
                if len(whole) >= MIN_CONTAIN:
                    shared = collections.Counter()
                    for g in grams(whole):
                        for k in posting.get(g, ()):
                            shared[k] += 1
                    for k, n in shared.items():
                        if (
                            k in long_items
                            and n == len(items[k][3])
                            and items[k][2] in whole
                        ):
                            hits[(k, role, "C")] = None
        counts[role] = {"files": len(paths), "leaves": leaves, "segments": segs}
    out = []
    for k, role, method in hits:
        row, index, _, _ = items[k]
        out.append(
            {
                "id": row["id"],
                "group_id": row["group_id"],
                "split": row["split"],
                "sentence": index,
                "role": role,
                "method": method,
            }
        )
    out.sort(key=lambda h: (h["role"], h["id"], h["sentence"], h["method"]))
    return out, counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, action="append", required=True)
    parser.add_argument(
        "--role",
        action="append",
        required=True,
        help="NAME=PATH[,PATH...] protected text for one role",
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--drop-groups", type=Path)
    args = parser.parse_args(argv)
    candidates = []
    inputs = {}
    for path in args.rows:
        inputs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    candidates.append(
                        {
                            "id": row["id"],
                            "group_id": row["group_id"],
                            "split": row["split"],
                            "sentences": sentences_of(row),
                        }
                    )
    roles: dict[str, list[Path]] = {}
    for spec in args.role:
        name, _, paths = spec.partition("=")
        roles[name] = [Path(p) for p in paths.split(",") if p]
        for p in roles[name]:
            inputs[f"{name}:{p.name}"] = hashlib.sha256(p.read_bytes()).hexdigest()
    hits, counts = scan(candidates, roles)
    by_role = collections.Counter((h["role"], h["split"], h["method"]) for h in hits)
    receipt = {
        "rule": {
            "exact_min_chars": MIN_EXACT,
            "contain_min_chars": MIN_CONTAIN,
            "gram": N,
            "threshold": THRESHOLD,
        },
        "inputs_sha256": inputs,
        "rows": len(candidates),
        "protected": counts,
        "hits": hits,
        "hit_counts": {"|".join(k): v for k, v in sorted(by_role.items())},
        "groups_hit": sorted({h["group_id"] for h in hits}),
    }
    args.receipt.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    if args.drop_groups is not None:
        args.drop_groups.write_text(
            "".join(g + "\n" for g in receipt["groups_hit"]), encoding="utf-8"
        )
    print(json.dumps({"rows": len(candidates), "hit_counts": receipt["hit_counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
