"""Write the preregistered shortcut-gate cells of an arm's TRAIN file.

    python3 -m v2.data.m2.audit_cells --rows h1.train.jsonl --out-dir CELLS

One JSONL per (source, family) with at least 30 rows (families never span
sources); cells above 20,000 rows keep whole groups in
``sha256("sc-v2:" + group_id)`` order until 20,000 rows. ``shortcut.py`` then
gates each task type inside each cell.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

from training.model.data import canonical
from v2.data.m2.common import _write_new, read_jsonl, sha

MIN_ROWS = 30
MAX_ROWS = 20000


def cells(rows: list[dict]) -> dict[str, list[dict]]:
    by_family: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_family[f"{row['source']}__{row['family']}"].append(row)
    out = {}
    for key in sorted(by_family):
        members = by_family[key]
        if len(members) < MIN_ROWS:
            continue
        if len(members) > MAX_ROWS:
            groups: dict[str, list[dict]] = collections.defaultdict(list)
            for row in members:
                groups[row["group_id"]].append(row)
            chosen: list[dict] = []
            for group in sorted(groups, key=lambda g: sha("sc-v2:" + g)):
                if len(chosen) + len(groups[group]) > MAX_ROWS:
                    continue
                chosen.extend(groups[group])
            members = chosen
        out[key] = sorted(members, key=lambda row: row["id"])
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    summary = {}
    for key, members in cells(list(read_jsonl(args.rows))).items():
        data = "".join(canonical(row) + "\n" for row in members).encode("utf-8")
        summary[key] = {
            "rows": len(members),
            "sha256": _write_new(args.out_dir / f"{key}.jsonl", data),
        }
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
