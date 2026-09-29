"""Row containment of derived training mixtures in already-screened files (count-only).

    python3 -m v2.eval.sealed.containment --screened LABEL=MANIFEST ... \
        --reference PATH ... --mixture LABEL=PATH ... \
        --output <receipt.json> --novel <private rows.jsonl>

Every `--reference` file (or every supported file under a directory) must have its
SHA-256 in one of the `--screened` corpus manifests (`c1-corpora/1`, e.g. an earlier
recheck's training manifest), so its rows already went through the overlap scan; the
manifest's LABEL is recorded as the check that screened it. The significant strings of
a row are its string leaves (JSON strings decoded, as in `overlap`), normalised as the
scanner does, that the scanner can match: at least 5 tokens (they yield shingles) or at
least 20 characters (exact matching). Each mixture row is

    identical   equal as canonical JSON to a reference row
    no_text     not identical and without a significant string (cannot match)
    contained   every significant string equals a significant string of one reference
                row with the same `id` or `input_sha256`: its shingles and exact-match
                strings are a subset of that row's, so against any protected row it
                scores no higher than a row that was already scanned
    novel       otherwise (also a line that is not a JSON object); written verbatim to
                --novel, so that only these rows need a new overlap scan

Rows: jsonl and jsonl.gz lines, the items of a .json list (a dict: its list value if it
has exactly one, else the dict), parquet rows. The receipt holds counts and hashes only.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

from v2.eval.sealed.overlap import Params, string_leaves, tokens
from v2.eval.sealed.schema import normalized

SCHEMA = "dev2-c1-containment/1"
SUFFIXES = (".jsonl", ".jsonl.gz", ".json", ".parquet")
KEYS = ("id", "input_sha256")
COUNTS = ("rows", "identical", "contained", "no_text", "novel")
PARAMS = Params()


def files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        p
        for p in path.rglob("*")
        if p.is_file()
        and p.name.endswith(SUFFIXES)
        and not any(part.startswith(".") for part in p.relative_to(path).parts)
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> Iterator[Any]:
    name = path.name
    if name.endswith((".jsonl", ".jsonl.gz")):
        opener = gzip.open if name.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8", errors="surrogateescape") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except ValueError:
                    yield line.rstrip("\n")
    elif name.endswith(".json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            lists = [v for v in data.values() if isinstance(v, list)]
            data = lists[0] if len(lists) == 1 else [data]
        yield from data if isinstance(data, list) else [data]
    elif name.endswith(".parquet"):
        import pyarrow.parquet as pq

        yield from pq.read_table(path).to_pylist()


def canonical(row: Any) -> bytes:
    text = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return text.encode("utf-8", "surrogatepass")


def key(data: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def significant(row: Any) -> frozenset[int]:
    found = set()
    for text in string_leaves(row):
        if len(text) < 5:
            continue
        norm = normalized(text)
        if len(norm) >= PARAMS.min_chars or len(tokens(norm)) >= PARAMS.min_tokens:
            found.add(key(norm.encode("utf-8", "surrogatepass")))
    return frozenset(found)


def screened(specs: list[str]) -> dict[str, str]:
    known: dict[str, str] = {}
    for spec in specs:
        label, _, path = spec.partition("=")
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        for entry in manifest["labels"].values():
            for item in entry["files"]:
                known.setdefault(item["sha256"], label)
    return known


def run(args: argparse.Namespace) -> dict[str, Any]:
    known = screened(args.screened)
    exact: set[int] = set()
    by_key: dict[str, list[frozenset[int]]] = defaultdict(list)
    references = []
    for root in args.reference:
        for path in files(Path(root)):
            digest = sha256(path)
            if digest not in known:
                raise SystemExit(f"reference {path} ({digest[:12]}) is not screened")
            count = 0
            for row in rows(path):
                count += 1
                exact.add(key(canonical(row)))
                if isinstance(row, dict):
                    leaves = significant(row)
                    for name in KEYS:
                        if isinstance(row.get(name), str):
                            by_key[f"{name}\0{row[name]}"].append(leaves)
            references.append(
                {
                    "path": str(path),
                    "sha256": digest,
                    "rows": count,
                    "screened_by": known[digest],
                }
            )
    novel = bytearray()
    mixtures: dict[str, list[dict[str, Any]]] = {}
    totals: Counter = Counter()
    for spec in args.mixture:
        label, _, root = spec.partition("=")
        entries = []
        for path in files(Path(root)):
            counts: Counter = Counter()
            for row in rows(path):
                counts["rows"] += 1
                data = canonical(row)
                if key(data) in exact:
                    counts["identical"] += 1
                    continue
                if isinstance(row, dict):
                    leaves = significant(row)
                    if not leaves:
                        counts["no_text"] += 1
                        continue
                    if any(
                        leaves <= other
                        for name in KEYS
                        if isinstance(row.get(name), str)
                        for other in by_key.get(f"{name}\0{row[name]}", ())
                    ):
                        counts["contained"] += 1
                        continue
                counts["novel"] += 1
                novel += data + b"\n"
            entries.append({"path": str(path), "sha256": sha256(path), **counts})
            totals.update(counts)
        mixtures[label] = entries
    descriptor = os.open(args.novel, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(novel)
    return {
        "schema": SCHEMA,
        "params": {
            "min_tokens": PARAMS.min_tokens,
            "min_chars": PARAMS.min_chars,
            "keys": list(KEYS),
        },
        "screened": sorted({s.partition("=")[0] for s in args.screened}),
        "references": references,
        "mixtures": mixtures,
        "totals": {k: totals[k] for k in COUNTS},
        "novel_sha256": hashlib.sha256(bytes(novel)).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--screened", action="append", required=True, metavar="LABEL=MANIFEST"
    )
    parser.add_argument("--reference", action="append", required=True)
    parser.add_argument(
        "--mixture", action="append", required=True, metavar="LABEL=PATH"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--novel", type=Path, required=True)
    args = parser.parse_args(argv)
    for path in (args.output, args.novel):
        if path.exists():
            parser.error(f"refusing to overwrite {path}")
    receipt = run(args)
    text = json.dumps(receipt, indent=1, sort_keys=True) + "\n"
    args.output.write_text(text, encoding="utf-8")
    summary = {"totals": receipt["totals"], "novel_sha256": receipt["novel_sha256"]}
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
