"""Dataset-level source independence of C1 sources from training data (count-only).

    python3 -m v2.eval.sealed.independence names --terms <terms.json> --root DIR ... --output <names.json>
    python3 -m v2.eval.sealed.independence rows --sources-dir <dir> --source KEY ... --output <protected.jsonl>
    python3 -m v2.eval.sealed.independence manifest --label LABEL=DIR ... --output <manifest.json>

`names` searches registry, manifest, readme, audit and code files (json, md, txt, py, yaml)
for each source's terms. It also searches the provenance fields of every row file
(`source`, `family`, `id`, `group_id`, `render_template`, `audit_metadata`). It reports
counts per source, term and file, never text. `rows` turns every row of every data file
of each C1 source snapshot (all splits) into the overlap scanner's protected shape.
`manifest` writes an overlap-scanner corpus manifest for local directories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

PROVENANCE = (
    "source",
    "family",
    "id",
    "group_id",
    "render_template",
    "audit_metadata",
    "dataset",
)
TEXT_SUFFIXES = (".json", ".md", ".txt", ".py", ".yaml", ".yml")
ROW_SUFFIXES = (".jsonl", ".jsonl.gz")
DATA_SUFFIXES = (".jsonl", ".json", ".csv", ".tsv", ".parquet")
MIN_CHARS = 20


def strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def iter_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and ".cache" not in path.parts and ".git" not in path.parts:
            yield path


def match_terms(text: str, terms: dict[str, list[str]]) -> Counter:
    low = text.casefold()
    found: Counter = Counter()
    for source, words in terms.items():
        for word in words:
            if word.casefold() in low:
                found[(source, word)] += 1
    return found


def open_rows(path: Path) -> Iterator[dict[str, Any]]:
    if path.name.endswith(".gz"):
        import gzip

        stream = gzip.open(path, "rt", encoding="utf-8")
    else:
        stream = path.open(encoding="utf-8")
    with stream:
        for line in stream:
            line = line.strip()
            if line.startswith("{"):
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def names(args: argparse.Namespace) -> int:
    terms = json.loads(args.terms.read_text(encoding="utf-8"))
    hits: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    scanned = Counter()
    for root in args.root:
        for path in iter_files(root):
            name = path.name
            found: Counter = Counter()
            if name.endswith(ROW_SUFFIXES):
                scanned["row_files"] += 1
                for row in open_rows(path):
                    scanned["rows"] += 1
                    fields = " ".join(
                        text
                        for key in PROVENANCE
                        if key in row
                        for text in strings(row[key])
                    )
                    found.update(match_terms(fields, terms))
            elif name.endswith(TEXT_SUFFIXES) and path.stat().st_size < 64 << 20:
                scanned["text_files"] += 1
                found.update(
                    match_terms(
                        path.read_text(encoding="utf-8", errors="ignore"), terms
                    )
                )
            for (source, word), count in found.items():
                hits[source][str(path)][word] = count
    out = {
        "schema": "dev2-c1-independence-names/1",
        "roots": [str(r) for r in args.root],
        "terms": terms,
        "scanned": dict(scanned),
        "hits": {s: dict(v) for s, v in hits.items()},
        "sources_found": sorted(hits),
    }
    args.output.write_text(
        json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"scanned": dict(scanned), "sources_found": out["sources_found"]}))
    return 0


def data_rows(path: Path) -> Iterator[Any]:
    name = path.name
    if name.endswith(".jsonl"):
        yield from open_rows(path)
    elif name.endswith(".json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            yield from data
        elif isinstance(data, dict):
            lists = [v for v in data.values() if isinstance(v, list)]
            yield from (lists[0] if len(lists) == 1 else [data])
    elif name.endswith((".csv", ".tsv")):
        csv.field_size_limit(1 << 30)
        with path.open(encoding="utf-8", errors="ignore", newline="") as stream:
            yield from csv.DictReader(
                stream, delimiter="\t" if name.endswith(".tsv") else ","
            )
    elif name.endswith(".parquet"):
        import pyarrow.parquet as pq

        yield from pq.read_table(path).to_pylist()


def rows(args: argparse.Namespace) -> int:
    counts = Counter()
    lines = []
    for key in args.source:
        root = args.sources_dir / key
        for path in iter_files(root):
            if not path.name.endswith(DATA_SUFFIXES) or path.name in (
                "dataset_info.json",
            ):
                continue
            relative = str(path.relative_to(root))
            for index, row in enumerate(data_rows(path)):
                texts = sorted({t for t in strings(row) if len(t.strip()) >= MIN_CHARS})
                if not texts:
                    continue
                counts[key] += 1
                lines.append(
                    json.dumps(
                        {
                            "source": key,
                            "task": f"{key}/{relative}",
                            "source_item_id": str(index),
                            "state": {},
                            "overlap_texts": texts,
                        },
                        ensure_ascii=False,
                    )
                )
    data = ("\n".join(lines) + "\n").encode("utf-8")
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
    print(
        json.dumps({"rows": dict(counts), "sha256": hashlib.sha256(data).hexdigest()})
    )
    return 0


def manifest(args: argparse.Namespace) -> int:
    labels = {}
    for item in args.label:
        label, _, directory = item.partition("=")
        files = []
        for path in iter_files(Path(directory)):
            if not path.name.endswith((".jsonl", ".jsonl.gz", ".json", ".parquet")):
                continue
            if path.name.endswith(".ids.jsonl"):
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            files.append(
                {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}
            )
        labels[label] = {"kind": "training", "files": files}
    args.output.write_text(
        json.dumps({"schema": "c1-corpora/1", "labels": labels}, indent=1) + "\n"
    )
    print(json.dumps({label: len(v["files"]) for label, v in labels.items()}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("names")
    one.add_argument("--terms", type=Path, required=True)
    one.add_argument("--root", type=Path, action="append", required=True)
    one.add_argument("--output", type=Path, required=True)
    two = commands.add_parser("rows")
    two.add_argument("--sources-dir", type=Path, required=True)
    two.add_argument("--source", action="append", required=True)
    two.add_argument("--output", type=Path, required=True)
    three = commands.add_parser("manifest")
    three.add_argument("--label", action="append", required=True)
    three.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    return {"names": names, "rows": rows, "manifest": manifest}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
