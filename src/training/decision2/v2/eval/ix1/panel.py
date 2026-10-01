"""IX1 panel: the 38 Index benchmarks of edition 0.2.1 as gold-free, sharded runner rows.

Runs on the node host with the port and the hash-pinned kit 19ad28ec on PYTHONPATH:

    PYTHONPATH=<decision2>:<kit-19ad28ec> python3 -m v2.eval.ix1.panel \
        --suite-dir <suite-0.2> --shards 8 --out <private dir> [--compat compat-86.jsonl.gz]

Writes ``shard-<k>-of-<n>.jsonl.gz`` holding only ``_evaluation``, ``state`` and ``questions`` (no
gold, no scoring fields), and ``panel.json`` with per-benchmark counts, the run-ID digest and every
shard's SHA-256. A row goes to shard ``sha256(run_id) mod n``. ``--compat`` also writes a gold-free
copy of the 86-request compatibility sample. The output directory is private: rows carry
restricted benchmark text.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Iterable

RUNNER_FIELDS = ("_evaluation", "state", "questions")


def index_benchmarks() -> set[int]:
    from external_index021.protocol import spec

    return {number for area in spec()["areas"] for number in area["benchmarks"]}


def shard_of(run_id: str, shards: int) -> int:
    return int.from_bytes(hashlib.sha256(run_id.encode()).digest()[:8], "big") % shards


def gold_free(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in RUNNER_FIELDS}


def digest(run_ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(run_ids)).encode()).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        text = io.TextIOWrapper(gz, encoding="utf-8")
        for row in rows:
            text.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            count += 1
        text.flush()
        text.detach()
    return count


def census(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_benchmark = collections.Counter(r["_evaluation"]["catalog_id"] for r in rows)
    kinds = collections.Counter(
        q["type"] for r in rows for q in r["questions"].values()
    )
    return {
        "rows": len(rows),
        "questions": dict(sorted(kinds.items())),
        "benchmarks": {str(k): per_benchmark[k] for k in sorted(per_benchmark)},
        "run_ids_sha256": digest(r["_evaluation"]["run_id"] for r in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compat", type=Path)
    args = parser.parse_args()
    from external_index021.score import selected_rows, verified_suite

    if args.shards < 1:
        raise SystemExit("--shards must be positive")
    suite = verified_suite(args.suite_dir)
    rows, selection = selected_rows(suite)
    if selection.status != "upstream_021_row_ids_matched":
        raise SystemExit(f"selection status {selection.status}")
    wanted = index_benchmarks()
    panel = [r for r in rows if r["_evaluation"]["catalog_id"] in wanted]
    if {r["_evaluation"]["catalog_id"] for r in panel} != wanted:
        raise SystemExit("an Index benchmark has no selected rows")
    args.out.mkdir(parents=True, exist_ok=True)
    buckets: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in panel:
        buckets[shard_of(row["_evaluation"]["run_id"], args.shards)].append(
            gold_free(row)
        )
    shards = []
    for k in range(args.shards):
        path = args.out / f"shard-{k}-of-{args.shards}.jsonl.gz"
        count = write_rows(path, buckets[k])
        shards.append({"file": path.name, "rows": count, "sha256": sha_file(path)})
    report = {
        "schema": "ix1-panel/1",
        "edition": "0.2.1",
        "suite": {
            "rows_sha256": suite.edition["rows_sha256"],
            "added_sha256": suite.edition["added_sha256"],
        },
        "selection": {
            "status": selection.status,
            "scoreable": selection.scoreable,
            "keep_ids_sha256": selection.keep_ids_sha256,
        },
        "index_benchmarks": len(wanted),
        **census(panel),
        "shards": shards,
    }
    if args.compat:
        with gzip.open(args.compat, "rt", encoding="utf-8") as stream:
            sample = [json.loads(line) for line in stream if line.strip()]
        path = args.out / "compat-86.gold-free.jsonl.gz"
        write_rows(path, (gold_free(r) for r in sample))
        report["compat"] = {
            "source_sha256": sha_file(args.compat),
            "file": path.name,
            "sha256": sha_file(path),
            **census(sample),
        }
    (args.out / "panel.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({k: report[k] for k in ("rows", "questions", "run_ids_sha256")}))


if __name__ == "__main__":
    main()
