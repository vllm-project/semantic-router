"""Decoder M15 MLX-DEV-M15 panel for one tier (prereg dec-m15-prereg-2026-10-01.md, "Development readouts").

The decoder MLX-DEV panel minus every group that shares a group id, row id or input hash with any listed TRAIN file
(the tier's released TRAIN, IB1-r3 and IB2), as the 9B track's MLX-DEV-9B. Panel lines are kept byte for byte, in
order, with their index entries. A cell left empty fails the build.

usage: m15_mlxpanel.py --panel P --panel-sha S --index I --index-sha S --train F=SHA [...] --name NAME --output DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA = "dec-m15-mlxpanel/1"
FIELDS = ("id", "input_sha256", "group_id")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 23), b""):
            h.update(block)
    return h.hexdigest()


def verified(path: Path, want: str) -> Path:
    if sha256(path) != want:
        raise ValueError(f"{path} is not {want}")
    return path


def build(args: argparse.Namespace) -> dict[str, Any]:
    panel_lines = (
        verified(args.panel, args.panel_sha).read_bytes().splitlines(keepends=True)
    )
    index = [
        json.loads(x)
        for x in verified(args.index, args.index_sha).read_text().splitlines()
    ]
    panel = [json.loads(x) for x in panel_lines]
    if [e["id"] for e in index] != [r["id"] for r in panel]:
        raise ValueError("MLX-DEV index and panel rows differ")
    seen: dict[str, set[str]] = {k: set() for k in FIELDS}
    trains = {}
    for spec in args.train:
        path, want = spec.rsplit("=", 1)
        n = 0
        with verified(Path(path), want).open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                for k in FIELDS:
                    seen[k].add(row[k])
                n += 1
        trains[path] = {"sha256": want, "rows": n}
    dropped: dict[str, set[str]] = {k: set() for k in FIELDS}
    for row in panel:
        for k in FIELDS:
            if row[k] in seen[k]:
                dropped[k].add(row["group_id"])
    drop = set().union(*dropped.values())
    keep = [i for i, e in enumerate(index) if e["group_id"] not in drop]
    cells = {e["cell"] for e in index}
    empty = sorted(cells - {index[i]["cell"] for i in keep})
    if empty:
        raise ValueError(f"MLX-DEV-M15 would leave cells empty: {empty}")
    out = args.output
    out.mkdir(parents=True)
    with (out / "panel.jsonl").open("wb") as stream:
        for i in keep:
            stream.write(panel_lines[i])
    with (out / "panel.jsonl.index.jsonl").open("w", encoding="utf-8") as stream:
        for i in keep:
            stream.write(json.dumps(index[i], ensure_ascii=False) + "\n")
    per_cell = {}
    for cell in sorted(cells):
        entries = [index[i] for i in keep if index[i]["cell"] == cell]
        per_cell[cell] = {
            "rows": len(entries),
            "rows_before": sum(e["cell"] == cell for e in index),
            "groups": len({e["group_id"] for e in entries}),
            "rows_by_language": dict(
                sorted(Counter(e["language"] for e in entries).items())
            ),
        }
    report = {
        "schema": SCHEMA,
        "name": args.name,
        "panel_sha256": args.panel_sha,
        "index_sha256": args.index_sha,
        "train": trains,
        "rows_before": len(index),
        "groups_before": len({e["group_id"] for e in index}),
        "rows": len(keep),
        "groups": len({index[i]["group_id"] for i in keep}),
        "dropped_groups": {k: len(v) for k, v in dropped.items()},
        "dropped_groups_total": len(drop),
        "cells": per_cell,
        "output_panel_sha256": sha256(out / "panel.jsonl"),
        "output_index_sha256": sha256(out / "panel.jsonl.index.jsonl"),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--panel-sha", required=True)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--index-sha", required=True)
    p.add_argument("--train", action="append", required=True, help="FILE=SHA256")
    p.add_argument("--name", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    r = build(args)
    print(
        json.dumps(
            {
                k: r[k]
                for k in (
                    "name",
                    "rows",
                    "groups",
                    "dropped_groups_total",
                    "output_panel_sha256",
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
