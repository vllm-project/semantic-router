"""Remove quarantined groups from an arm using overlap and embedding receipts.

A group is quarantined when any private overlap receipt flags it against a
role outside ``--report-only-role`` (A0 TRAIN for new arms), or when an
embedding receipt lists it at or above its quarantine threshold. Output rows
keep their canonical order; the report lists removed counts, never text.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from training.model.data import canonical


def overlap_groups(
    receipt: dict[str, Any], report_only: set[str]
) -> dict[str, list[str]]:
    flagged: dict[str, list[str]] = {}
    for group, record in receipt.get("groups", {}).items():
        roles = sorted(set(record.get("roles", [])) - report_only)
        if roles:
            flagged[group] = roles
    return flagged


def embed_groups(receipt: dict[str, Any]) -> dict[str, list[str]]:
    return {
        item["group_id"]: [item["protected_role"]]
        for item in receipt.get("quarantined", [])
    }


def apply(
    rows: list[dict[str, Any]], quarantine: dict[str, list[str]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept = [row for row in rows if row["group_id"] not in quarantine]
    removed = [row for row in rows if row["group_id"] in quarantine]
    by_role: collections.Counter[str] = collections.Counter()
    for group in {row["group_id"] for row in removed}:
        by_role.update(quarantine[group])
    return kept, {
        "rows_in": len(rows),
        "rows_kept": len(kept),
        "rows_removed": len(removed),
        "groups_removed": len({row["group_id"] for row in removed}),
        "removed_by_source": dict(
            sorted(collections.Counter(row["source"] for row in removed).items())
        ),
        "removed_groups_by_role": dict(sorted(by_role.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--overlap-receipt", type=Path, action="append", default=[])
    parser.add_argument("--embed-receipt", type=Path, action="append", default=[])
    parser.add_argument("--report-only-role", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    quarantine: dict[str, list[str]] = {}
    sources = []
    for path in args.overlap_receipt:
        data = path.read_bytes()
        sources.append({"overlap_receipt_sha256": hashlib.sha256(data).hexdigest()})
        for group, roles in overlap_groups(
            json.loads(data), set(args.report_only_role)
        ).items():
            quarantine.setdefault(group, []).extend(roles)
    for path in args.embed_receipt:
        data = path.read_bytes()
        sources.append({"embed_receipt_sha256": hashlib.sha256(data).hexdigest()})
        for group, roles in embed_groups(json.loads(data)).items():
            quarantine.setdefault(group, []).extend(roles)
    rows = [
        json.loads(line) for line in args.rows.open(encoding="utf-8") if line.strip()
    ]
    kept, report = apply(rows, {g: sorted(set(r)) for g, r in quarantine.items()})
    data = "".join(
        canonical(row) + "\n" for row in sorted(kept, key=lambda r: r["id"])
    ).encode()
    for path, payload in (
        (args.out, data),
        (
            args.report,
            (
                json.dumps(
                    {
                        **report,
                        "receipts": sources,
                        "report_only_roles": sorted(args.report_only_role),
                        "input_sha256": hashlib.sha256(
                            args.rows.read_bytes()
                        ).hexdigest(),
                        "output_sha256": hashlib.sha256(data).hexdigest(),
                    },
                    indent=1,
                    sort_keys=True,
                )
                + "\n"
            ).encode(),
        ),
    ):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
