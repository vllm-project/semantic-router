"""Drop held-out groups whose inputs also occur in the arm's TRAIN file.

    python3 -m v2.data.m2.dedup_heldout --train H1.train.jsonl \\
        --heldout H1.aho.jsonl --out H1.aho.dedup.jsonl --report report.json

Distinct source items can render to byte-identical inputs; the isolation rule
forbids such an input in both TRAIN and a held-out slice, so the whole held-out
group is removed (keeping twins and coverage groups intact).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from training.model.data import canonical
from v2.data.m2.common import _write_new, read_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    seen = {row["input_sha256"] for row in read_jsonl(args.train)}
    rows = list(read_jsonl(args.heldout))
    bad = {row["group_id"] for row in rows if row["input_sha256"] in seen}
    kept = [row for row in rows if row["group_id"] not in bad]
    digest = _write_new(
        args.out, "".join(canonical(r) + "\n" for r in kept).encode("utf-8")
    )
    report = {
        "rows_in": len(rows),
        "rows_kept": len(kept),
        "groups_removed": len(bad),
        "output_sha256": digest,
    }
    _write_new(args.report, (json.dumps(report, sort_keys=True) + "\n").encode())
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
