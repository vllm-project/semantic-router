"""Verdict of the event-3 confirmatory overlap scan against the event-2 scan (ids only, never text).

    python3 -m v2.eval.sealed.scanverdict compare --hits NEW --baseline OLD --receipt RECEIPT \
        --manifest MANIFEST --protected-sha SHA --output SCAN-VERDICT.json
    python3 -m v2.eval.sealed.scanverdict check --verdict SCAN-VERDICT.json --manifest-sha SHA \
        --protected-sha SHA

``compare`` reads two ``overlap scan --hits`` files (one row per protected candidate) and fails when
any candidate is OVERLAP, when a candidate is non-CLEAN now but was CLEAN at event 2, when a recurring
non-CLEAN candidate has a higher containment than at event 2, or when the two scans cover different
candidates. It exits 1 on FAIL. ``check`` is the event-3 interlock: it exits 0 only for a PASS verdict
of the pinned manifest and protected rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "dev2-c1-event3-scan-verdict/1"


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_hits(path: Path) -> dict[str, dict[str, Any]]:
    hits = {}
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                hits[row["id"]] = row
    return hits


def judge(
    new: dict[str, dict[str, Any]], old: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    counts = Counter(h["verdict"] for h in new.values())
    flagged = sorted(k for k, h in new.items() if h["verdict"] != "CLEAN")
    before = {k for k, h in old.items() if h["verdict"] != "CLEAN"}
    problems = []
    if set(new) != set(old):
        problems.append("the scans cover different protected candidates")
    overlap = [k for k in flagged if new[k]["verdict"] == "OVERLAP"]
    if overlap:
        problems.append(f"{len(overlap)} OVERLAP")
    fresh = [k for k in flagged if k not in before]
    if fresh:
        problems.append(f"{len(fresh)} non-CLEAN ids that were CLEAN at event 2")
    higher = [
        k
        for k in flagged
        if k in before and new[k]["containment"] > old[k]["containment"]
    ]
    if higher:
        problems.append(f"{len(higher)} recurring ids with higher containment")
    return {
        "candidates": len(new),
        "counts": dict(sorted(counts.items())),
        "non_clean_ids": flagged,
        "event2_non_clean": len(before),
        "overlap_ids": overlap,
        "new_non_clean_ids": fresh,
        "higher_containment_ids": higher,
        "problems": problems,
        "verdict": "FAIL" if problems else "PASS",
    }


def compare(args: argparse.Namespace) -> int:
    result = judge(read_hits(args.hits), read_hits(args.baseline))
    result = {
        "schema": SCHEMA,
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest_sha256": sha_file(args.manifest),
        "protected_sha256": args.protected_sha,
        "receipt_sha256": sha_file(args.receipt),
        "hits_sha256": sha_file(args.hits),
        "baseline_hits_sha256": sha_file(args.baseline),
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(
        json.dumps(
            {k: result[k] for k in ("verdict", "counts", "problems", "hits_sha256")}
        )
    )
    return 0 if result["verdict"] == "PASS" else 1


def check(args: argparse.Namespace) -> int:
    try:
        value = json.loads(args.verdict.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"no readable scan verdict: {exc}", file=sys.stderr)
        return 1
    problems = []
    if value.get("schema") != SCHEMA:
        problems.append("not a scan verdict")
    if value.get("verdict") != "PASS":
        problems.append(f"verdict {value.get('verdict')!r}")
    if value.get("manifest_sha256") != args.manifest_sha:
        problems.append("scan used another training manifest")
    if value.get("protected_sha256") != args.protected_sha:
        problems.append("scan used other protected rows")
    if problems:
        print("scan verdict refused: " + "; ".join(problems), file=sys.stderr)
        return 1
    print(f"scan verdict PASS (hits {value['hits_sha256'][:12]}, {value['utc']})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("compare")
    for flag in ("--hits", "--baseline", "--receipt", "--manifest", "--output"):
        a.add_argument(flag, type=Path, required=True)
    a.add_argument("--protected-sha", required=True)
    b = sub.add_parser("check")
    b.add_argument("--verdict", type=Path, required=True)
    b.add_argument("--manifest-sha", required=True)
    b.add_argument("--protected-sha", required=True)
    args = parser.parse_args(argv)
    return {"compare": compare, "check": check}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
