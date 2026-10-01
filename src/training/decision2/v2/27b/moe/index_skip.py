"""Device-aborting requests in a sharded MoE Index run (IX1's override rule; host CPU, stdlib).

    python3 -m v2.27b.moe.index_skip skip --shard-dir RUN/shard-K --rows PANEL/shard-K-of-N.jsonl.gz
    python3 -m v2.27b.moe.index_skip abort --extra-dir RUN/extra-TAG --rows RUN/shard-K/extra-rows/TAG.jsonl.gz

``skip`` runs after a kit runner exited non-zero. The culprit is the request whose final record is a
device error (``OutOfMemoryError``, ``AcceleratorError`` or a device-side assert, which make the kit
halt) or, when the process aborted before writing one, the first row of the shard in file order without
a final record. It joins ``skipped.json``; ``rows.override.jsonl.gz`` becomes the shard's rows minus
every skipped request (the shard resumes over it) and ``extra-rows/<n>.jsonl.gz`` holds the culprit
alone, to be rerun by itself. Exit 3 when there is no culprit (any other failure: stop for a person).

``abort`` runs after such a rerun failed: if the rerun wrote no final record for its request, it
appends an ``error`` record (the request's ``_evaluation`` fields, the failure), so the merge accounts
for every row and both scorers count it as a failure, as IX1 recorded its device-aborting request.

``presplit`` applies the same rule before a run, from ``index_scan``'s request sizes: every request
of at least ``--min-padded-tokens`` padded tokens (the measured memory limit of one GPU) is skipped in
its shard and rerun alone afterwards, so it cannot halt the shard. It changes which process answers a
request, never how: each request still goes once through the native path in one batch.

    python3 -m v2.27b.moe.index_skip presplit --scan SCAN.json --panel P/panel-N/panel.json \
        --run P/runs/NAME --min-padded-tokens T
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEVICE_EXCEPTIONS = {"OutOfMemoryError", "AcceleratorError"}


def read_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        text = io.TextIOWrapper(gz, encoding="utf-8")
        for row in rows:
            text.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
        text.flush()
        text.detach()


def final_records(path: Path) -> dict[str, dict[str, Any]]:
    final: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line cut by the abort
                final[record["run_id"]] = record
    return final


def device_error(record: dict[str, Any]) -> bool:
    return record.get("status") == "error" and (
        record.get("exception") in DEVICE_EXCEPTIONS
        or "device-side assert" in str(record.get("error", ""))
    )


def culprit(rows: list[dict[str, Any]], final: dict[str, dict[str, Any]]) -> str | None:
    pending = {row["_evaluation"]["run_id"] for row in rows}
    errors = [r for r in final.values() if device_error(r) and r["run_id"] in pending]
    if errors:
        return errors[-1]["run_id"]
    for row in rows:
        if row["_evaluation"]["run_id"] not in final:
            return row["_evaluation"]["run_id"]
    return None


def skip(shard_dir: Path, rows_path: Path) -> dict[str, Any]:
    rows = read_rows(rows_path)
    skipped_path = shard_dir / "skipped.json"
    skipped = json.loads(skipped_path.read_text()) if skipped_path.exists() else []
    remaining = [r for r in rows if r["_evaluation"]["run_id"] not in set(skipped)]
    found = culprit(remaining, final_records(shard_dir / "results.jsonl"))
    if found is None or found in skipped:
        raise SystemExit(3)
    skipped.append(found)
    skipped_path.write_text(json.dumps(skipped, indent=1) + "\n")
    write_rows(
        shard_dir / "rows.override.jsonl.gz",
        [r for r in rows if r["_evaluation"]["run_id"] not in set(skipped)],
    )
    extra = shard_dir / "extra-rows"
    extra.mkdir(exist_ok=True)
    write_rows(
        extra / f"{len(skipped)}.jsonl.gz",
        [r for r in rows if r["_evaluation"]["run_id"] == found],
    )
    return {"skipped": found, "total_skipped": len(skipped)}


def abort(extra_dir: Path, rows_path: Path, reason: str) -> dict[str, Any]:
    (row,) = read_rows(rows_path)
    run_id = row["_evaluation"]["run_id"]
    extra_dir.mkdir(parents=True, exist_ok=True)
    results = extra_dir / "results.jsonl"
    if run_id in final_records(results):
        return {"run_id": run_id, "recorded": False}
    text = results.read_text(encoding="utf-8") if results.exists() else ""
    if text and not text.endswith("\n"):
        results.write_text(text[: text.rfind("\n") + 1], encoding="utf-8")
    now = datetime.now(timezone.utc).isoformat()
    record = {
        **row["_evaluation"],
        "started_utc": now,
        "completed_utc": now,
        "status": "error",
        "error": reason,
        "exception": "ProcessAbort",
        "total_wall_ms": 0.0,
        "model_request_wall_ms": 0.0,
    }
    with results.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"run_id": run_id, "recorded": True}


def presplit(
    scan: dict[str, Any], panel_path: Path, run: Path, threshold: int
) -> dict[str, Any]:
    panel = json.loads(panel_path.read_text())
    if scan["panel_run_ids_sha256"] != panel["run_ids_sha256"]:
        raise SystemExit("the scan belongs to another panel")
    big = {r["run_id"] for r in scan["requests"] if r["padded_tokens"] >= threshold}
    out: dict[str, Any] = {"min_padded_tokens": threshold, "shards": {}}
    for k, shard in enumerate(panel["shards"]):
        directory = run / f"shard-{k}"
        if (directory / "results.jsonl").exists() or (
            directory / "skipped.json"
        ).exists():
            raise SystemExit(f"{directory} has already started or been split")
        rows = read_rows(panel_path.parent / shard["file"])
        skipped = [r for r in rows if r["_evaluation"]["run_id"] in big]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "extra-rows").mkdir(exist_ok=True)
        for n, row in enumerate(skipped, 1):
            write_rows(directory / "extra-rows" / f"{n}.jsonl.gz", [row])
        write_rows(
            directory / "rows.override.jsonl.gz",
            [r for r in rows if r["_evaluation"]["run_id"] not in big],
        )
        (directory / "skipped.json").write_text(
            json.dumps([r["_evaluation"]["run_id"] for r in skipped], indent=1) + "\n"
        )
        out["shards"][str(k)] = {"rows": len(rows), "skipped": len(skipped)}
    (run / "presplit.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("skip")
    p.add_argument("--shard-dir", type=Path, required=True)
    p.add_argument("--rows", type=Path, required=True)
    p = sub.add_parser("presplit")
    p.add_argument("--scan", type=Path, required=True)
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--min-padded-tokens", type=int, required=True)
    p = sub.add_parser("abort")
    p.add_argument("--extra-dir", type=Path, required=True)
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument(
        "--reason",
        default="the request aborted the device when rerun alone (recorded as a final error)",
    )
    args = parser.parse_args(argv)
    if args.command == "skip":
        out = skip(args.shard_dir, args.rows)
    elif args.command == "presplit":
        out = presplit(
            json.loads(args.scan.read_text()),
            args.panel,
            args.run,
            args.min_padded_tokens,
        )
    else:
        out = abort(args.extra_dir, args.rows, args.reason)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
