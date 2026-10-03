"""Board submission of an IX1 run made kit-complete by a complement run (see ``complement``).

    # private: stored IX1 results + complement results -> one kit results file in suite order
    PYTHONPATH=<decision2>:<kit-87d4650b> python3 -m v2.eval.ix1.submission merge \
        --suite-dir <suite-0.2> --stored <IX1 run> --complement <complement run> --out <private dir>
    # public run directory: payload-free rows, path-free runner records, public receipt
    python3 -m v2.eval.ix1.submission public --merged <private dir> --stored <IX1 run> \
        --complement <complement run> --out <public dir>/runs/<name>
    # after ``decision_index score`` of both: identical outputs, or the same Index (--index-only)
    python3 -m v2.eval.ix1.submission compare --a <scored dir> --b <scored dir> [--index-only]

``merge`` refuses unless the stored results are byte-identical to their IX1 receipt, the two result
sets are disjoint, together they hold exactly the edition's scoreable run IDs, no final record is an
error, and every runner reports the same model source. ``public`` drops each row's ``payload`` and
``raw_output`` (keeping ``payload_sha256``) and keeps only file names of rows paths; it refuses
output that still names a node path.
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

from v2.eval.ix1.merge import final_records

EDITION = "0.2.1"
DROPPED_ROW_KEYS = ("payload", "raw_output")
VOLATILE = {"generated_utc", "out", "results", "results_path", "time", "scored_utc"}
PRIVATE_MARKERS = ("/data/", "/root/", "/home/", "/mnt/", "/tmp/")


def _dumps(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _runner_dirs(run: Path, prefix: str) -> list[Path]:
    return sorted(run.glob(f"{prefix}-*"), key=lambda p: (len(p.name), p.name))


def _gpu_seconds(runner: Path) -> int:
    total = 0
    for start in runner.glob("start_epoch*"):
        end = runner / start.name.replace("start", "end", 1)
        total += int(end.read_text()) - int(start.read_text())
    return total


def combine(
    expected: list[tuple[str, int]],
    stored: dict[str, dict[str, Any]],
    added: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Final records in suite order plus row accounting; raises on any accounting defect."""
    wanted = [run_id for run_id, _ in expected]
    if len(set(wanted)) != len(wanted):
        raise ValueError("the suite repeats run IDs")
    overlap = set(stored) & set(added)
    if overlap:
        raise ValueError(
            f"{len(overlap)} run IDs answered by both the stored and the complement run"
        )
    union = set(stored) | set(added)
    if union != set(wanted):
        raise ValueError(
            f"row accounting failed: {len(set(wanted) - union)} missing, {len(union - set(wanted))} extra"
        )
    rows = [stored.get(run_id) or added[run_id] for run_id in wanted]
    statuses = collections.Counter(r["status"] for r in rows)
    if statuses["error"]:
        raise ValueError(f"{statuses['error']} final errors")
    per_benchmark: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    for (run_id, catalog), record in zip(expected, rows):
        per_benchmark[str(catalog)][record["status"]] += 1
    reasons = collections.Counter(
        r.get("error", "") for r in rows if r["status"] != "ok"
    )
    engines = {r["engine"] for r in rows}
    if len(engines) != 1:
        raise ValueError(f"rows come from {len(engines)} engines")
    accounting = {
        "rows": len(rows),
        "stored_rows": len(stored),
        "complement_rows": len(added),
        "statuses": dict(sorted(statuses.items())),
        "non_ok_reasons": dict(sorted(reasons.items())),
        "per_benchmark_statuses": {
            k: dict(sorted(v.items()))
            for k, v in sorted(per_benchmark.items(), key=lambda kv: int(kv[0]))
        },
        "engine": engines.pop(),
    }
    return rows, accounting


def public_row(record: dict[str, Any]) -> dict[str, Any]:
    if not record.get("payload_sha256"):
        raise ValueError(f"{record['run_id']} has no payload_sha256")
    return {k: v for k, v in record.items() if k not in DROPPED_ROW_KEYS}


def public_environment(env: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(env))
    rows = out.get("rows_path")
    if isinstance(rows, list):
        out["rows_path"] = [Path(p).name for p in rows]
    elif isinstance(rows, str):
        out["rows_path"] = Path(rows).name
    return out


def private_strings(value: Any) -> list[str]:
    """Strings that look like node filesystem paths, anywhere in a JSON value."""
    if isinstance(value, dict):
        return [s for v in value.values() for s in private_strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in private_strings(v)]
    if isinstance(value, str) and any(marker in value for marker in PRIVATE_MARKERS):
        return [value]
    return []


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def index_view(index: dict[str, Any]) -> dict[str, Any]:
    """The parts of a kit index.json that the 38 Index benchmarks determine."""
    return {
        "index": index["index"],
        "raw_index": index["raw_index"],
        "scores": index["scores"],
        "areas": index["areas"],
        "coverage": index.get("coverage"),
        "benchmarks": {
            k: v for k, v in index["benchmarks"].items() if v.get("in_index")
        },
    }


def _read_lines(path: Path) -> list[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return stream.read().splitlines()


def _records(lines: Iterable[str]) -> dict[str, dict[str, Any]]:
    out = {}
    for line in lines:
        if line.strip():
            record = json.loads(line)
            if record["run_id"] in out:
                raise ValueError(f"run ID {record['run_id']} appears twice")
            out[record["run_id"]] = record
    return out


def cmd_merge(args: argparse.Namespace) -> None:
    from decision_index.suite.io import Suite

    receipt = json.loads((args.stored / "merged" / "receipt.json").read_text())
    stored_path = args.stored / "merged" / "results.jsonl"
    if _sha_file(stored_path) != receipt["results_sha256"]:
        raise SystemExit("stored results differ from their IX1 receipt")
    stored = _records(_read_lines(stored_path))
    added: dict[str, dict[str, Any]] = {}
    superseded = 0
    runners = _runner_dirs(args.complement, "shard") + _runner_dirs(
        args.complement, "extra"
    )
    if not runners:
        raise SystemExit("the complement run has no runner directories")
    for runner in runners:
        records, retried = final_records(_read_lines(runner / "results.jsonl"))
        superseded += retried
        for run_id, record in records.items():
            if run_id in added and added[run_id]["status"] != "error":
                raise SystemExit(f"{runner.name} answers {run_id} a second time")
            added[run_id] = record
    sources = set()
    for runner in _runner_dirs(args.stored, "shard") + runners:
        env = json.loads((runner / "environment.json").read_text())
        sources.add(json.dumps(env["model_source"], sort_keys=True))
    if len(sources) != 1:
        raise SystemExit(
            "the stored and complement runners report different model sources"
        )
    suite = Suite(args.suite_dir, EDITION)
    suite.verify(strict=True)
    expected = [
        (r["_evaluation"]["run_id"], r["_evaluation"]["catalog_id"])
        for r in suite.rows(apply_exclusions=True)
    ]
    try:
        rows, accounting = combine(expected, stored, added)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.out.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with (args.out / "results.jsonl").open("x", encoding="utf-8") as stream:
        for record in rows:
            line = _dumps(record)
            stream.write(line)
            digest.update(line.encode("utf-8"))
    out = {
        "schema": "ix1-submission-merge/1",
        "edition": EDITION,
        **accounting,
        "model_source": json.loads(sources.pop()),
        "stored": {
            "results_sha256": receipt["results_sha256"],
            "runners": len(_runner_dirs(args.stored, "shard")),
            "gpu_hours": receipt["gpu_hours"],
        },
        "complement": {
            "runners": len(runners),
            "superseded_error_records": superseded,
            "gpu_hours": round(sum(_gpu_seconds(r) for r in runners) / 3600, 4),
        },
        "results_sha256": digest.hexdigest(),
    }
    (args.out / "receipt.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "rows",
                    "stored_rows",
                    "complement_rows",
                    "statuses",
                    "results_sha256",
                )
            }
        )
    )


def cmd_public(args: argparse.Namespace) -> None:
    receipt = json.loads((args.merged / "receipt.json").read_text())
    source = args.merged / "results.jsonl"
    if _sha_file(source) != receipt["results_sha256"]:
        raise SystemExit("merged results differ from their receipt")
    args.out.mkdir(parents=True, exist_ok=True)
    stripped = 0
    rows = 0
    with (args.out / "results.jsonl.gz").open("xb") as raw, gzip.GzipFile(
        fileobj=raw, mode="wb", mtime=0
    ) as gz:
        text = io.TextIOWrapper(gz, encoding="utf-8")
        with source.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                stripped += any(k in record for k in DROPPED_ROW_KEYS)
                text.write(_dumps(public_row(record)))
                rows += 1
        text.flush()
        text.detach()
    if rows != receipt["rows"]:
        raise SystemExit(f"wrote {rows} rows, receipt has {receipt['rows']}")
    harness = args.out / "harness"
    written: list[Path] = []
    for label, run, prefix in (
        ("index", args.stored, "shard"),
        ("complement", args.complement, "shard"),
        ("complement-extra", args.complement, "extra"),
    ):
        for runner in _runner_dirs(run, prefix):
            target = harness / "runners" / f"{label}-{runner.name}"
            target.mkdir(parents=True, exist_ok=True)
            env = public_environment(
                json.loads((runner / "environment.json").read_text())
            )
            status = json.loads((runner / "status.json").read_text())
            for name, value in (("environment.json", env), ("status.json", status)):
                (target / name).write_text(
                    json.dumps(value, indent=2, ensure_ascii=False) + "\n"
                )
                written.append(target / name)
    public_receipt = {k: v for k, v in receipt.items() if k != "schema"}
    public_receipt["schema"] = "ix1-submission-receipt/1"
    public_receipt["public_results_sha256"] = _sha_file(args.out / "results.jsonl.gz")
    public_receipt["rows_with_dropped_fields"] = stripped
    (harness / "receipt.json").write_text(
        json.dumps(public_receipt, indent=2, sort_keys=True) + "\n"
    )
    written.append(harness / "receipt.json")
    if args.panel:
        panel = json.loads(args.panel.read_text())
        keep = (
            "schema",
            "edition",
            "expected_scoreable",
            "expected_run_ids_sha256",
            "stored_rows",
            "stored_run_ids_sha256",
            "rows",
            "benchmarks",
            "questions",
            "run_ids_sha256",
            "suite",
        )
        (harness / "complement-panel.json").write_text(
            json.dumps(
                {k: panel[k] for k in keep if k in panel}, indent=2, sort_keys=True
            )
            + "\n"
        )
        written.append(harness / "complement-panel.json")
    leaks = [
        (str(p.relative_to(args.out)), s)
        for p in written
        for s in private_strings(json.loads(p.read_text()))
    ]
    if leaks:
        raise SystemExit(f"node paths left in public files: {leaks[:5]}")
    print(
        json.dumps(
            {
                "rows": rows,
                "rows_with_dropped_fields": stripped,
                "runners": len(written) // 2,
                "public_results_sha256": public_receipt["public_results_sha256"],
            }
        )
    )


def cmd_compare(args: argparse.Namespace) -> None:
    ok = True
    report = {}
    if args.index_only:
        a = index_view(json.loads((args.a / "index.json").read_text()))
        b = index_view(json.loads((args.b / "index.json").read_text()))
        report["index.json (Index benchmarks)"] = a == b
        ok = a == b
    else:
        for name in ("scores.json", "index.json", "benchmark-summary.json"):
            same = clean(json.loads((args.a / name).read_text())) == clean(
                json.loads((args.b / name).read_text())
            )
            report[name] = same
            ok &= same
    for name, same in report.items():
        print(f"{name}: {'IDENTICAL' if same else 'DIFFERENT'}")
    print("MATCH" if ok else "MISMATCH")
    if not ok:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--suite-dir", type=Path, required=True)
    m.add_argument("--stored", type=Path, required=True)
    m.add_argument("--complement", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    m.set_defaults(func=cmd_merge)
    p = sub.add_parser("public")
    p.add_argument("--merged", type=Path, required=True)
    p.add_argument("--stored", type=Path, required=True)
    p.add_argument("--complement", type=Path, required=True)
    p.add_argument(
        "--panel",
        type=Path,
        help="the complement panel.json (counts and digests are copied)",
    )
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_public)
    c = sub.add_parser("compare")
    c.add_argument("--a", type=Path, required=True)
    c.add_argument("--b", type=Path, required=True)
    c.add_argument("--index-only", action="store_true")
    c.set_defaults(func=cmd_compare)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
