"""Complement of a stored IX1 run: the scoreable 0.2.1 requests its results lack.

    PYTHONPATH=<decision2>:<kit-87d4650b> python3 -m v2.eval.ix1.complement \
        --suite-dir <suite-0.2> --results <run>/merged/results.jsonl --shards N --out <private dir>

The kit (``decision_index.pipeline.score_run_v02``) marks a 0.2.1 run complete only when every
scoreable request has a result: the edition's rows minus the 442 exclusions, plus the added rows
(119,898 + 30,419). IX1 panels hold only the 38 Index benchmarks, so an IX1 run lacks the requests
of the benchmarks the board shows but does not count. This writes exactly those requests as
gold-free runner rows (``_evaluation``, ``state``, ``questions``) in ``shard-<k>-of-<n>.jsonl.gz``
(shard ``sha256(run_id) mod n``) plus ``panel.json``, the layout ``launch.sh run --rows-dir`` reads.
The output directory is private: rows carry restricted benchmark text.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Iterable

from v2.eval.ix1.panel import census, digest, gold_free, sha_file, shard_of, write_rows

EDITION = "0.2.1"


def stored_run_ids(lines: Iterable[str]) -> set[str]:
    """Run IDs with a final result; refuses duplicate or errored final records."""
    seen: dict[str, str] = {}
    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        if record["run_id"] in seen:
            raise ValueError(f"run ID {record['run_id']} appears twice")
        seen[record["run_id"]] = record["status"]
    errors = sorted(r for r, s in seen.items() if s == "error")
    if errors:
        raise ValueError(f"{len(errors)} stored results are errors")
    return set(seen)


def complement(
    rows: Iterable[dict[str, Any]], stored: set[str]
) -> tuple[list[dict[str, Any]], set[str]]:
    """Scoreable rows missing from ``stored``, and the full scoreable run-ID set."""
    missing, expected = [], set()
    for row in rows:
        run_id = row["_evaluation"]["run_id"]
        if run_id in expected:
            raise ValueError(f"suite repeats run ID {run_id}")
        expected.add(run_id)
        if run_id not in stored:
            missing.append(row)
    extra = stored - expected
    if extra:
        raise ValueError(
            f"{len(extra)} stored run IDs are not scoreable 0.2.1 requests"
        )
    return missing, expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from decision_index import editions
    from decision_index.suite.io import Suite

    if args.shards < 1:
        raise SystemExit("--shards must be positive")
    suite = Suite(args.suite_dir, EDITION)
    verified = suite.verify(strict=True)
    edition = editions.get(EDITION)
    with args.results.open(encoding="utf-8") as stream:
        stored = stored_run_ids(stream)
    missing, expected = complement(suite.rows(apply_exclusions=True), stored)
    if len(expected) != edition["scoreable"] + edition["added_requests"]:
        raise SystemExit(
            f"{len(expected)} scoreable requests, kit expects {edition['scoreable'] + edition['added_requests']}"
        )
    args.out.mkdir(parents=True, exist_ok=True)
    buckets: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in missing:
        buckets[shard_of(row["_evaluation"]["run_id"], args.shards)].append(
            gold_free(row)
        )
    shards = []
    for k in range(args.shards):
        path = args.out / f"shard-{k}-of-{args.shards}.jsonl.gz"
        count = write_rows(path, buckets[k])
        shards.append({"file": path.name, "rows": count, "sha256": sha_file(path)})
    tokens = collections.Counter()
    for row in missing:
        tokens[str(row["_evaluation"]["catalog_id"])] += int(
            row["_evaluation"].get("proxy_tokens") or 0
        )
    report = {
        "schema": "ix1-complement/1",
        "edition": EDITION,
        "suite": {
            k: verified.get(k)
            for k in (
                "sha256",
                "uncompressed_sha256",
                "added_sha256",
                "exclusions_sha256",
                "match",
            )
        },
        "expected_scoreable": len(expected),
        "expected_run_ids_sha256": digest(expected),
        "stored_results_sha256": sha_file(args.results),
        "stored_rows": len(stored),
        "stored_run_ids_sha256": digest(stored),
        **census(missing),
        "proxy_tokens": dict(sorted(tokens.items(), key=lambda kv: int(kv[0]))),
        "shards": shards,
    }
    (args.out / "panel.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "expected_scoreable",
                    "stored_rows",
                    "rows",
                    "benchmarks",
                    "run_ids_sha256",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
