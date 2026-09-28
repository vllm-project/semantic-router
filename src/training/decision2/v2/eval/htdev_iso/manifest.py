"""Freeze the training scope: TRAINING-MANIFEST.json plus an overlap-scanner corpus manifest.

    python3 -m v2.eval.htdev_iso.manifest --label LABEL=DIR ... \
        [--merge-old <c1-corpora manifest.json> --merge-glob 'train-*'] \
        --output <TRAINING-MANIFEST.json> --corpora <training-corpora.json> --workers N

Every file under each labelled directory (dot directories skipped) is listed in the
training manifest with path, sha256, bytes and rows (jsonl lines, parquet rows, json
list items; -1 when not a row format). The corpus manifest (schema c1-corpora/1, as
`independence manifest` writes it) lists the row-format files the overlap scanner
reads, without `*.ids.jsonl`. Labels merged from an older manifest keep its hashes.
"""

from __future__ import annotations

import argparse
import fnmatch
import gzip
import hashlib
import json
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Any

ROW_SUFFIXES = (".jsonl", ".jsonl.gz", ".json", ".parquet", ".csv", ".tsv")


def _rows(path: Path) -> int:
    name = path.name
    try:
        if name.endswith(".jsonl"):
            with path.open("rb") as stream:
                return sum(1 for line in stream if line.strip())
        if name.endswith(".jsonl.gz"):
            with gzip.open(path, "rb") as stream:
                return sum(1 for line in stream if line.strip())
        if name.endswith(".parquet"):
            import pyarrow.parquet as pq

            return pq.ParquetFile(path).metadata.num_rows
        if name.endswith(".json") and path.stat().st_size < 256 << 20:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return len(data)
            lists = (
                [v for v in data.values() if isinstance(v, list)]
                if isinstance(data, dict)
                else []
            )
            return len(lists[0]) if len(lists) == 1 else 1
        if name.endswith((".csv", ".tsv")):
            with path.open("rb") as stream:
                return max(sum(1 for _ in stream) - 1, 0)
    except (OSError, ValueError, UnicodeDecodeError):
        return -1
    return -1


def describe(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    if not path.is_file():
        return {"path": path_text, "sha256": "missing", "bytes": 0, "rows": -1}
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            digest.update(chunk)
    return {
        "path": path_text,
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
        "rows": _rows(path),
    }


def walk(directory: Path) -> list[str]:
    found = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        found.extend(
            str(Path(root) / f) for f in sorted(files) if not f.startswith(".")
        )
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--merge-old", type=Path)
    parser.add_argument("--merge-glob", default="train-*")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpora", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args(argv)
    for path in (args.output, args.corpora):
        if path.exists():
            parser.error(f"refusing to overwrite {path}")
    listing = {}
    for item in args.label:
        label, _, directory = item.partition("=")
        listing[label] = (directory, walk(Path(directory)))
    old_labels: dict[str, Any] = {}
    if args.merge_old:
        old = json.loads(args.merge_old.read_text(encoding="utf-8"))["labels"]
        old_labels = {
            k: v
            for k, v in old.items()
            if fnmatch.fnmatch(k, args.merge_glob) and k not in listing
        }
        for label, entry in old_labels.items():
            listing[label] = ("(merged)", [f["path"] for f in entry["files"]])
    everything = sorted({p for _, paths in listing.values() for p in paths})
    with Pool(args.workers) as pool:
        described = dict(zip(everything, pool.map(describe, everything, chunksize=4)))
    labels, corpora = {}, {}
    for label, (directory, paths) in listing.items():
        files = [described[p] for p in paths]
        changed = None
        if label in old_labels:
            want = {f["path"]: f["sha256"] for f in old_labels[label]["files"]}
            changed = sum(want[f["path"]] != f["sha256"] for f in files)
        labels[label] = {
            "changed_since_merged_manifest": changed,
            "root": directory,
            "files": files,
            "file_count": len(files),
            "bytes": sum(f["bytes"] for f in files),
            "rows": sum(max(f["rows"], 0) for f in files),
        }
        corpora[label] = {
            "kind": "training",
            "files": [
                {k: f[k] for k in ("path", "sha256", "bytes")}
                for f in files
                if f["path"].endswith(ROW_SUFFIXES)
                and not f["path"].endswith(".ids.jsonl")
                and f["sha256"] != "missing"
            ],
        }
    manifest = {
        "schema": "htdev-training-manifest/1",
        "labels": labels,
        "totals": {
            "labels": len(labels),
            "files": sum(v["file_count"] for v in labels.values()),
            "unique_files": len(everything),
            "bytes": sum(v["bytes"] for v in labels.values()),
            "rows": sum(v["rows"] for v in labels.values()),
            "corpus_files": sum(len(v["files"]) for v in corpora.values()),
        },
    }
    for path, payload in (
        (args.output, manifest),
        (args.corpora, {"schema": "c1-corpora/1", "labels": corpora}),
    ):
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=1)
    print(json.dumps(manifest["totals"]))
    print(
        json.dumps(
            {k: [v["file_count"], v["bytes"], v["rows"]] for k, v in labels.items()}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
