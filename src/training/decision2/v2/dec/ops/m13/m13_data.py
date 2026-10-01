"""Decoder M13 family-upweighted TRAIN (prereg dec-m13-prereg-2026-10-01.md, "Data"): 08b-RA-AG.

The output keeps an M12 arm's TRAIN file whole (every row, byte for byte, in file order) and then appends copies
k = 2..W of every row of the released TRAIN whose family is in --families, so those rows weigh W in the arm. A copy
keeps its line except the id, which gets the suffix `~a<k>` (distinct from M12's `~c<k>` IB copies). Every upweighted
row must be a released TRAIN row present in the arm file, and every listed family must have rows.

usage: m13_data.py --arm-train F --arm-sha S --released F --released-sha S --families a,b,... --weight W
         --name NAME --output DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA = "dec-m13-data/1"
UPWEIGHT_SUFFIX = "~a"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 23), b""):
            h.update(block)
    return h.hexdigest()


def upweight_line(line: bytes, k: int) -> bytes:
    row = json.loads(line)
    row["id"] = f"{row['id']}{UPWEIGHT_SUFFIX}{k}"
    return json.dumps(row, ensure_ascii=False).encode() + b"\n"


def upweighted(
    arm_lines: list[bytes], released_ids: set[str], families: set[str], weight: int
) -> tuple[list[bytes], dict[str, Any]]:
    """The arm's lines, then copies 2..weight of its released rows in `families`."""
    if weight < 2:
        raise ValueError("weight must be an integer >= 2")
    picked: list[bytes] = []
    per_family: Counter[str] = Counter()
    ids: set[str] = set()
    for line in arm_lines:
        row = json.loads(line)
        if row["id"] in ids:
            raise ValueError(f"repeated id {row['id']}")
        ids.add(row["id"])
        if UPWEIGHT_SUFFIX in row["id"]:
            raise ValueError(f"an arm id already contains {UPWEIGHT_SUFFIX!r}")
        if row["family"] in families:
            if row["id"] not in released_ids:
                raise ValueError(
                    f"{row['id']}: family {row['family']} row is not released TRAIN"
                )
            picked.append(line)
            per_family[row["family"]] += 1
    missing = families - set(per_family)
    if missing:
        raise ValueError(f"families without rows: {sorted(missing)}")
    extra = [upweight_line(line, k) for k in range(2, weight + 1) for line in picked]
    return arm_lines + extra, {
        "weight": weight,
        "upweighted_rows": len(picked),
        "per_family": dict(sorted(per_family.items())),
        "added_rows": len(extra),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm-train", type=Path, required=True)
    p.add_argument("--arm-sha", required=True)
    p.add_argument("--released", type=Path, required=True)
    p.add_argument("--released-sha", required=True)
    p.add_argument("--families", required=True)
    p.add_argument("--weight", type=int, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    for path, want in ((a.arm_train, a.arm_sha), (a.released, a.released_sha)):
        if sha256(path) != want:
            raise ValueError(f"{path}: sha256 differs from {want}")
    released_ids = {json.loads(line)["id"] for line in a.released.open("rb")}
    arm_lines = list(a.arm_train.open("rb"))
    families = set(a.families.split(","))
    lines, info = upweighted(arm_lines, released_ids, families, a.weight)
    out = a.output / a.name
    out.mkdir(parents=True)
    with (out / "train.jsonl").open("xb") as sink:
        sink.writelines(lines)
    report = {
        "schema": SCHEMA,
        "name": a.name,
        "arm_train": {
            "path": str(a.arm_train),
            "sha256": a.arm_sha,
            "rows": len(arm_lines),
        },
        "released": {"path": str(a.released), "sha256": a.released_sha},
        "families": sorted(families),
        **info,
        "rows": len(lines),
        "train_sha256": sha256(out / "train.jsonl"),
    }
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "name",
                    "rows",
                    "upweighted_rows",
                    "added_rows",
                    "train_sha256",
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
