"""Attribute C1 rescan hits to corpus groups (analysis only; ids, paths and numbers, never text).

    python3 -m v2.eval.sealed.hitgroups subset --protected P --ids IDS.txt --output SUB.jsonl
    python3 -m v2.eval.sealed.hitgroups manifest --manifest M --coverage-receipt R --work DIR \
        --output GROUPED.json
    python3 -m v2.eval.sealed.hitgroups table --hits HITS.jsonl [--hits ...] --output TABLE.json

``subset`` keeps the protected rows whose id is listed. ``manifest`` relabels a rescan manifest
(``coverage build``) so that each label is one corpus group: a derived file in WORK is mapped back to
its source path (an archive member to its archive), then grouped by directory — raw sources by source,
Hugging Face dataset snapshots by repo (the private training data by its directory up to four levels),
the content-addressed blob store as one group, run and data directories by their first levels.
``overlap scan`` over the grouped manifest gives each protected row its best match per group, and
``table`` merges one or more such hits files into, per row, every group it is non-CLEAN in (verdict,
containment, exact-match tokens, shingles of the row).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

SCHEMA = "dev2-c1-hitgroups/1"
RANK = ("CLEAN", "REVIEW", "OVERLAP")
TRAINING = "datasets--llm-semantic-router--decision-2.0-training-data"
DELTA = re.compile(
    r"^/data/dev2/private/c1-rescan-hf-delta/[^/]+/files/[0-9a-f]{12}/(.+)$"
)
SNAPSHOT = re.compile(
    r"^/(?:data/dev2/hf-cache|root/\.cache/huggingface/hub)/([^/]+)/snapshots/[^/]+/(.+)$"
)


def source_path(path: str, work: str, roots: dict[str, str]) -> str:
    """The walked path a manifest entry stands for (WORK copies mapped back to their origin)."""
    prefix = work.rstrip("/") + "/"
    if not path.startswith(prefix):
        return path
    label, _, rest = path[len(prefix) :].partition("/")
    origin = roots[label].rstrip("/") + "/" + rest
    if ".d/" in origin:
        origin = origin[: origin.index(".d/")]
    return (
        re.sub(r"\.jsonl$", "", origin)
        if origin.endswith((".txt.jsonl", ".md.jsonl"))
        else origin
    )


def group(path: str) -> str:
    parts = path.strip("/").split("/")
    match = DELTA.match(path)
    if match:
        return "training-data:" + "/".join(match.group(1).split("/")[:-1][:4])
    match = SNAPSHOT.match(path)
    if match:
        repo, rest = match.groups()
        if repo == TRAINING:
            return "training-data:" + "/".join(rest.split("/")[:-1][:4])
        return "hf:" + repo
    if path.startswith("/data/dev2/hf-cache/blobs/"):
        return "hf-blobs"
    if path.startswith("/data/dev2/private/sources/m3b/"):
        return "/" + "/".join(parts[:6])
    if path.startswith(
        ("/data/dev2/private/sources/", "/data/dev2/private/a7/sources/")
    ):
        return "/" + "/".join(parts[: 5 if parts[3] == "sources" else 6])
    return "/" + "/".join(parts[: min(len(parts) - 1, 5)])


def subset(args: argparse.Namespace) -> int:
    wanted = {
        line.strip() for line in args.ids.read_text().splitlines() if line.strip()
    }
    kept = []
    for line in args.protected.read_bytes().splitlines():
        row = json.loads(line)
        if f"{row['task']}|{row['source_item_id']}" in wanted:
            kept.append(line)
    if len(kept) != len(wanted):
        raise ValueError(f"{len(wanted) - len(kept)} listed ids are not protected rows")
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(b"\n".join(kept) + b"\n")
    print(json.dumps({"rows": len(kept)}))
    return 0


def manifest(args: argparse.Namespace) -> int:
    value = json.loads(args.manifest.read_text())
    roots = json.loads(args.coverage_receipt.read_text())["roots"]
    labels: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in value["labels"].values():
        for item in entry["files"]:
            labels[group(source_path(item["path"], str(args.work), roots))].append(item)
    grouped = {
        "schema": "c1-corpora/1",
        "labels": {
            k: {"kind": "training", "files": v} for k, v in sorted(labels.items())
        },
    }
    args.output.write_text(json.dumps(grouped) + "\n")
    print(json.dumps({"groups": len(labels), "files": sum(map(len, labels.values()))}))
    return 0


def table(args: argparse.Namespace) -> int:
    rows: dict[str, dict[str, Any]] = {}
    for path in args.hits:
        for line in path.read_text().splitlines():
            hit = json.loads(line)
            row = rows.setdefault(
                hit["id"], {"shingles": hit.get("shingles"), "groups": {}}
            )
            for label, cell in (hit.get("labels") or {}).items():
                if cell.get("verdict", "CLEAN") == "CLEAN":
                    continue
                old = row["groups"].get(label)
                new = {k: cell.get(k) for k in ("verdict", "containment", "exact")}
                if old is None or (RANK.index(new["verdict"]), new["containment"]) > (
                    RANK.index(old["verdict"]),
                    old["containment"],
                ):
                    row["groups"][label] = new
    out = {"schema": SCHEMA, "rows": dict(sorted(rows.items()))}
    args.output.write_text(json.dumps(out, sort_keys=True) + "\n")
    flagged = sum(bool(r["groups"]) for r in rows.values())
    print(json.dumps({"rows": len(rows), "non_clean_rows": flagged}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("subset")
    for flag in ("--protected", "--ids", "--output"):
        a.add_argument(flag, type=Path, required=True)
    b = sub.add_parser("manifest")
    for flag in ("--manifest", "--coverage-receipt", "--work", "--output"):
        b.add_argument(flag, type=Path, required=True)
    c = sub.add_parser("table")
    c.add_argument("--hits", type=Path, action="append", required=True)
    c.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    return {"subset": subset, "manifest": manifest, "table": table}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
