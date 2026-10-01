"""IX1 row-level contamination audit: Index panel rows against each model's training files.

    PYTHONHASHSEED=0 python3 -m v2.eval.ix1.contamination --panel <panel dir> \
        --train NAME=FILE[,FILE...] [--train ...] --workers 24 --out <private dir>

CPU only. Text is NFKC-normalized and casefolded, then split into ``\\w+`` tokens. Index units are the
string leaves of each row's state, question instructions and option descriptions; training units are
every string value of every training row, recursively.

- Exact: a whole leaf of at least 4 tokens, or a sentence of at least 8 tokens, equal to a whole
  training leaf or a training sentence.
- 13-gram: every word 13-gram of every leaf, matched exhaustively (no sampling).
- Template filter: a unit or 13-gram present in at least 50 rows of one benchmark is that
  benchmark's template and is ignored for its rows.
- Per row: ``coverage`` = share of its distinct 13-grams found in the training data, and
  ``state_exact`` = its longest state leaf matched exactly. Classes: ``duplicate`` (coverage >= 0.5
  or state_exact; familiar text, e.g. a premise or evidence passage), ``partial`` (any other exact
  or 13-gram hit), ``clean``.
- ``item``: the stricter overlap used for the override. The row's item units, i.e. every
  non-template sentence of at least 8 tokens and every whole leaf of 4 to 30 tokens (claims,
  hypotheses, utterances, positions), are all found in the training data.

A planted control appends the states of 200 seeded panel rows to a synthetic training file; every
planted row must be classed ``duplicate``. Outputs are private: ``audit.json`` (per training set and
benchmark: class counts, rows with exact hits, maximum coverage) and ``duplicates.json`` (run IDs of
``duplicate`` rows) and ``items.json`` (run IDs of ``item`` rows).
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import multiprocessing
import os
import random
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterator

TOKEN = re.compile(r"\w+")
SENTENCE = re.compile(r"(?<=[.!?\u3002\uff01\uff1f])\s+|\n+")
GRAM = 13
LEAF_MIN = 4
ITEM_LEAF_MAX = 30
SENTENCE_MIN = 8
TEMPLATE_ROWS = 50
DUPLICATE_COVERAGE = 0.5
PLANTED = 200
SEED = 20261001

GRAMS: frozenset[int] = frozenset()
UNITS: frozenset[int] = frozenset()


def tokens(text: str) -> list[str]:
    return TOKEN.findall(unicodedata.normalize("NFKC", text).casefold())


def leaves(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from leaves(child)
    elif isinstance(value, list):
        for child in value:
            yield from leaves(child)


def grams_of(words: list[str]) -> set[int]:
    return {hash(tuple(words[i : i + GRAM])) for i in range(len(words) - GRAM + 1)}


def sentence_units(text: str) -> set[int]:
    units = set()
    for sentence in SENTENCE.split(text):
        part = tokens(sentence)
        if len(part) >= SENTENCE_MIN:
            units.add(hash(" ".join(part)))
    return units


def units_of(text: str, words: list[str]) -> set[int]:
    units = {hash(" ".join(words))} if len(words) >= LEAF_MIN else set()
    return units | sentence_units(text)


def item_units_of(text: str, words: list[str]) -> set[int]:
    units = sentence_units(text)
    if LEAF_MIN <= len(words) <= ITEM_LEAF_MAX:
        units.add(hash(" ".join(words)))
    return units


def row_texts(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    state = list(leaves(row["state"]))
    other = []
    for question in row["questions"].values():
        other.extend(leaves(question.get("instructions")))
        criteria = question.get("criteria")
        if isinstance(criteria, dict):
            other.extend(leaves(list(criteria.values())))
    return state, other


def index_rows(panel: Path) -> list[dict[str, Any]]:
    report = json.loads((panel / "panel.json").read_text())
    rows = []
    for shard in report["shards"]:
        with gzip.open(panel / shard["file"], "rt", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                state, other = row_texts(row)
                grams: set[int] = set()
                units: set[int] = set()
                items: set[int] = set()
                for text in state + other:
                    words = tokens(text)
                    grams |= grams_of(words)
                    units |= units_of(text, words)
                    items |= item_units_of(text, words)
                main = max(state, key=len, default="")
                main_words = tokens(main)
                rows.append(
                    {
                        "run_id": row["_evaluation"]["run_id"],
                        "benchmark": row["_evaluation"]["catalog_id"],
                        "grams": grams,
                        "units": units,
                        "items": items,
                        "state_unit": (
                            hash(" ".join(main_words))
                            if len(main_words) >= LEAF_MIN
                            else None
                        ),
                        "state": row["state"],
                    }
                )
    return rows


def drop_templates(rows: list[dict[str, Any]]) -> dict[str, int]:
    by_benchmark = collections.defaultdict(list)
    for row in rows:
        by_benchmark[row["benchmark"]].append(row)
    dropped = {}
    for benchmark, members in by_benchmark.items():
        counts = collections.Counter()
        for row in members:
            counts.update(row["grams"])
            counts.update(row["units"])
        template = {key for key, count in counts.items() if count >= TEMPLATE_ROWS}
        for row in members:
            row["grams"] -= template
            row["units"] -= template
            row["items"] -= template
            if row["state_unit"] in template:
                row["state_unit"] = None
        dropped[str(benchmark)] = len(template)
    return dropped


def chunks(path: Path, size: int = 32 << 20) -> list[tuple[str, int, int]]:
    total = path.stat().st_size
    bounds, start = [], 0
    with path.open("rb") as stream:
        while start < total:
            stream.seek(min(start + size, total))
            stream.readline()
            end = min(stream.tell(), total)
            bounds.append((str(path), start, end))
            start = end
    return bounds


def scan_chunk(task: tuple[str, int, int]) -> tuple[set[int], set[int], int]:
    path, start, end = task
    found_grams: set[int] = set()
    found_units: set[int] = set()
    lines = 0
    with open(path, "rb") as stream:
        stream.seek(start)
        data = stream.read(end - start)
    for raw in data.splitlines():
        if not raw.strip():
            continue
        lines += 1
        for text in leaves(json.loads(raw)):
            words = tokens(text)
            if len(words) >= GRAM:
                found_grams.update(g for g in grams_of(words) if g in GRAMS)
            found_units.update(u for u in units_of(text, words) if u in UNITS)
    return found_grams, found_units, lines


def classify(row: dict[str, Any], grams: set[int], units: set[int]) -> dict[str, Any]:
    hit_grams = len(row["grams"] & grams)
    coverage = hit_grams / len(row["grams"]) if row["grams"] else 0.0
    exact = bool(row["units"] & units)
    state_exact = row["state_unit"] is not None and row["state_unit"] in units
    if coverage >= DUPLICATE_COVERAGE or state_exact:
        kind = "duplicate"
    elif hit_grams or exact:
        kind = "partial"
    else:
        kind = "clean"
    item = bool(row["items"]) and row["items"] <= units
    return {"class": kind, "coverage": coverage, "exact": exact, "item": item}


def audit(rows, files: list[Path], workers: int) -> dict[str, Any]:
    tasks = [task for path in files for task in chunks(path)]
    grams: set[int] = set()
    units: set[int] = set()
    lines = 0
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        for found_grams, found_units, count in pool.imap_unordered(scan_chunk, tasks):
            grams |= found_grams
            units |= found_units
            lines += count
    per_benchmark = collections.defaultdict(
        lambda: {
            "rows": 0,
            "duplicate": 0,
            "partial": 0,
            "clean": 0,
            "exact_rows": 0,
            "max_coverage": 0.0,
            "undetectable": 0,
            "item": 0,
        }
    )
    duplicates, items = [], []
    for row in rows:
        verdict = classify(row, grams, units)
        entry = per_benchmark[str(row["benchmark"])]
        entry["rows"] += 1
        entry[verdict["class"]] += 1
        entry["undetectable"] += not row["grams"] and not row["units"]
        entry["exact_rows"] += verdict["exact"]
        entry["item"] += verdict["item"]
        entry["max_coverage"] = max(
            entry["max_coverage"], round(verdict["coverage"], 4)
        )
        if verdict["class"] == "duplicate":
            duplicates.append(row["run_id"])
        if verdict["item"]:
            items.append(row["run_id"])
    return {
        "training_lines": lines,
        "files": [str(path) for path in files],
        "benchmarks": dict(sorted(per_benchmark.items(), key=lambda kv: int(kv[0]))),
        "duplicate_rows": len(duplicates),
        "item_rows": len(items),
        "duplicates": sorted(duplicates),
        "items": sorted(items),
    }


def main() -> None:
    global GRAMS, UNITS
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument(
        "--train", action="append", required=True, help="NAME=FILE[,FILE...]"
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit(
            "run with PYTHONHASHSEED=0 (hashes are shared by forked workers)"
        )
    sets = {}
    for spec in args.train:
        name, _, files = spec.partition("=")
        sets[name] = [Path(f) for f in files.split(",") if f]
    rows = index_rows(args.panel)
    templates = drop_templates(rows)
    GRAMS = frozenset(g for row in rows for g in row["grams"])
    UNITS = frozenset(u for row in rows for u in row["units"])
    args.out.mkdir(parents=True, exist_ok=True)
    detectable = [row for row in rows if row["state_unit"] is not None]
    planted_rows = random.Random(SEED).sample(detectable, PLANTED)
    keep = {row["run_id"] for row in planted_rows}
    for row in rows:
        if row["run_id"] not in keep:
            row["state"] = None
    planted = args.out / "planted.jsonl"
    with planted.open("w", encoding="utf-8") as stream:
        for row in planted_rows:
            stream.write(json.dumps({"text": row["state"]}, ensure_ascii=False) + "\n")
    control = audit(rows, [planted], args.workers)
    found = set(control["duplicates"])
    planted_items = sum(row["run_id"] in set(control["items"]) for row in planted_rows)
    missed = [row["run_id"] for row in planted_rows if row["run_id"] not in found]
    report = {
        "schema": "ix1-contamination/1",
        "method": {
            "normalization": "NFKC, casefold, \\w+ tokens",
            "gram": GRAM,
            "leaf_min_tokens": LEAF_MIN,
            "sentence_min_tokens": SENTENCE_MIN,
            "template_rows": TEMPLATE_ROWS,
            "duplicate_coverage": DUPLICATE_COVERAGE,
        },
        "index_rows": len(rows),
        "index_grams": len(GRAMS),
        "index_units": len(UNITS),
        "template_units_dropped": templates,
        "planted_control": {
            "planted": PLANTED,
            "found": PLANTED - len(missed),
            "missed": missed,
            "state_only_rows_classed_item": planted_items,
        },
        "training_sets": {},
    }
    duplicates, items = {}, {}
    for name, files in sets.items():
        result = audit(rows, files, args.workers)
        duplicates[name] = result.pop("duplicates")
        items[name] = result.pop("items")
        report["training_sets"][name] = result
        print(
            json.dumps(
                {
                    "set": name,
                    "lines": result["training_lines"],
                    "duplicates": result["duplicate_rows"],
                    "items": result["item_rows"],
                }
            ),
            flush=True,
        )
    (args.out / "items.json").write_text(
        json.dumps(items, indent=2, sort_keys=True) + "\n"
    )
    (args.out / "audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (args.out / "duplicates.json").write_text(
        json.dumps(duplicates, indent=2, sort_keys=True) + "\n"
    )
    if missed:
        print(f"planted control missed {len(missed)} rows", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
