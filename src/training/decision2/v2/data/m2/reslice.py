"""Move the sealed slice (SHO) out of a generator arm's TRAIN file.

    python3 -m v2.data.m2.reslice --train a2.train.jsonl --aho a2.aho.jsonl \\
        --arm G2 --out-dir OUT

Generator arms already hold AHO (``split=select``); SHO =
``sha256("sho-v2:" + group_id) % 50 == 0`` among TRAIN groups. Row ids get an
arm prefix (generator ids are index-based and would repeat v1 arm ids); rows are
not otherwise changed. Writes ``<arm>.{train,aho,sho}.jsonl`` and a manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from training.model.data import file_sha256, validate_row
from v2.data.m2.common import _write_new, counts, read_jsonl, sha
from training.model.data import canonical


def is_sho(group: str) -> bool:
    return int(sha("sho-v2:" + group), 16) % 50 == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--aho", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--id-prefix", required=True)
    args = parser.parse_args(argv)
    prefix = f"{args.id_prefix}:"
    train, sho = [], []
    for row in read_jsonl(args.train):
        row = dict(row, id=prefix + row["id"])
        if is_sho(row["group_id"]):
            sho.append(
                validate_row(
                    dict(row, split="select", evaluation_role="select"), "select"
                )
            )
        else:
            train.append(validate_row(row, "train"))
    aho = [
        validate_row(dict(row, id=prefix + row["id"]), "select")
        for row in read_jsonl(args.aho)
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest = {
        "arm": args.arm,
        "inputs": {"train": file_sha256(args.train), "aho": file_sha256(args.aho)},
        "rule": "SHO = sha256('sho-v2:' + group_id) % 50 == 0 among TRAIN groups",
        "id_prefix": prefix,
    }
    for name, rows in (("train", train), ("aho", aho), ("sho", sho)):
        data = "".join(canonical(r) + "\n" for r in sorted(rows, key=lambda r: r["id"]))
        manifest[name] = {
            **counts(rows),
            "sha256": _write_new(
                args.out_dir / f"{args.arm}.{name}.jsonl", data.encode()
            ),
        }
    _write_new(
        args.out_dir / f"{args.arm}.build.json",
        (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps({k: manifest[k]["rows"] for k in ("train", "aho", "sho")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
