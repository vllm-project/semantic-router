#!/usr/bin/env python3
"""Which toxicchat join texts also occur in a model's training sources? (issue #4669)

Reads the suite's jailbreak and safety ``hold-toxicchat`` files, keeps the texts that have gold
on both tasks, and scans every string field of every record under ``--sources`` (.json, .jsonl
and .parquet files) for the model card's overlap rule: the same text, or a shared run of 13
words. Text is lowercased and split into word tokens before matching. A file whose path names a
test split is reported but does not count, since a model does not train on it.

    python3 tools/guard_overlap_scan.py --suite SUITE/text --sources TRAIN_SOURCES --out overlap.json

The report lists, per overlapping text, the files and match kinds that hit it, and ``texts``,
the ones that hit a training file. Drop those texts from the suite rows and from each arm's
predictions before running ``guard_miss_correlation.py`` on the rest.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

N = 13
WORD = re.compile(r"\w+")
FILES = ("toxicchat/jailbreak/hold-toxicchat", "toxicchat/safety/hold-toxicchat")
SUFFIXES = (".json", ".jsonl", ".parquet")


def tokens(text: str) -> list[str]:
    """Split text into lowercased word tokens.

    Args:
        text: Any text.

    Returns:
        The tokens, in order.
    """
    return WORD.findall(text.lower())


def grams(toks: list[str]) -> set[tuple[str, ...]]:
    """List every run of ``N`` consecutive tokens.

    Args:
        toks: The tokens of one text.

    Returns:
        The runs; empty when the text is shorter than ``N`` tokens.
    """
    return {tuple(toks[i : i + N]) for i in range(len(toks) - N + 1)}


def strings(value: Any) -> Iterator[str]:
    """Yield every string inside a decoded JSON value.

    Args:
        value: A string, dict, list or any other JSON value.

    Yields:
        Each string, depth first.
    """
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from strings(v)


def records(path: Path) -> Iterator[str]:
    """Yield every string field of a training file.

    Args:
        path: A .parquet file, a .json file holding one array, or a JSON lines file.

    Yields:
        Each string field; a JSON lines file's undecodable lines are skipped.
    """
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(batch_size=20000):
            for column in batch.columns:
                for value in column.to_pylist():
                    yield from strings(value)
        return
    with path.open(encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if path.suffix == ".json" and first == "[":
            for record in json.load(f):
                yield from strings(record)
            return
        for line in f:
            if line.strip():
                try:
                    yield from strings(json.loads(line))
                except json.JSONDecodeError:
                    continue


def joint_texts(suite: Path) -> list[str]:
    """Read the texts that have gold on both the jailbreak and the safety file.

    Args:
        suite: The suite's ``text/`` directory.

    Returns:
        The texts, sorted.
    """
    tasks: dict[str, int] = collections.Counter()
    for stem in FILES:
        seen = set()
        for line in (suite / f"{stem}.jsonl").open(encoding="utf-8"):
            if line.strip():
                seen.add(json.loads(line)["text"])
        tasks.update(seen)
    return sorted(t for t, n in tasks.items() if n == len(FILES))


def is_test_file(path: str) -> bool:
    """Say whether a training source path names a test split.

    Args:
        path: The file's path relative to the sources directory.

    Returns:
        True for a path with a ``test`` part or file name, such as ``en/test.jsonl``.
    """
    return any(
        re.search(r"(^|[^a-z])test([^a-z]|$)", part) for part in Path(path).parts
    )


def scan(texts: list[str], sources: Path) -> dict[str, set[str]]:
    """Find which texts share an exact text or a run of ``N`` words with a training source.

    Args:
        texts: The texts to look for.
        sources: A directory of training files, searched recursively (``.cache`` skipped).

    Returns:
        For each text that overlaps, the set of ``<file>:<exact|13-gram>`` hits.
    """
    exact = {" ".join(tokens(t)): t for t in texts}
    owners: dict[tuple[str, ...], set[str]] = {}
    for t in texts:
        for g in grams(tokens(t)):
            owners.setdefault(g, set()).add(t)
    hits: dict[str, set[str]] = {}
    paths = sorted(p for p in sources.rglob("*") if p.suffix in SUFFIXES)
    for path in paths:
        if ".cache" in path.parts:
            continue
        name = str(path.relative_to(sources))
        for s in records(path):
            toks = tokens(s)
            match = exact.get(" ".join(toks))
            if match is not None:
                hits.setdefault(match, set()).add(f"{name}:exact")
            for g in grams(toks):
                for t in owners.get(g, ()):
                    hits.setdefault(t, set()).add(f"{name}:{N}-gram")
        print(f"{name}: {len(hits)} texts hit so far", flush=True)
    return hits


def main(argv: list[str] | None = None) -> int:
    """Run the scan and write its report.

    Args:
        argv: The command line, without the program name; ``sys.argv`` when None.

    Returns:
        The exit code.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--suite", required=True, type=Path, help="the suite's text/ directory"
    )
    p.add_argument("--sources", required=True, type=Path, help="the training files")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args(argv)
    texts = joint_texts(args.suite)
    hits = scan(texts, args.sources)
    train = sorted(
        t
        for t, found in hits.items()
        if any(not is_test_file(h.rsplit(":", 1)[0]) for h in found)
    )
    report = {
        "joint_texts": len(texts),
        "overlap_texts": len(train),
        "texts": train,
        "hits": {t: sorted(found) for t, found in sorted(hits.items())},
    }
    args.out.write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"{len(train)} of {len(texts)} texts overlap a training file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
