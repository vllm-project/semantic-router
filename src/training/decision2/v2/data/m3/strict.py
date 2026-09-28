"""A0s-strict: pk1 A0s without the excluded shortcut families, with matching target subsets.

    python3 -m v2.data.m3.strict --rows m3/pk1/A0s/train.jsonl \\
        --targets lux1=A0-train.canonical.jsonl --targets autojev27=A0s-train.targets.jsonl \\
        --targets autojev27-attestation=A0s-train.attestation.jsonl --out-dir DIR

Drops every row of the families in ``EXCLUDED`` (known shortcut failures; the A7 default
mixtures exclude them too) and writes ``DIR/train.jsonl`` plus, for each target file,
``DIR/<name>.jsonl`` restricted to the kept ids (lines unchanged, input order). Every kept row
must have exactly one line in every target file, with the row's ``input_sha256``.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from pathlib import Path

EXCLUDED = ("natural_cosmos_qa", "natural_squad2_answerability")


def _write(path: Path, data: bytes) -> str:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--targets", action="append", default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lines = args.rows.read_bytes().splitlines(keepends=True)
    kept, dropped = [], collections.Counter()
    hashes = {}
    for line in lines:
        row = json.loads(line)
        if row["family"] in EXCLUDED:
            dropped[f"{row['family']}|{row['task_type']}"] += 1
            continue
        kept.append(line)
        hashes[row["id"]] = row["input_sha256"]
    receipt = {
        "schema": "decision2-m3b-a0s-strict/1",
        "excluded_families": list(EXCLUDED),
        "source_sha256": hashlib.sha256(b"".join(lines)).hexdigest(),
        "rows": len(kept),
        "dropped": dict(sorted(dropped.items())),
        "sha256": _write(args.out_dir / "train.jsonl", b"".join(kept)),
        "targets": {},
    }
    for spec in args.targets:
        name, _, path = spec.partition("=")
        raw = Path(path).read_bytes().splitlines(keepends=True)
        chosen = []
        for line in raw:
            record = json.loads(line)
            if record["id"] in hashes:
                if record["input_sha256"] != hashes[record["id"]]:
                    raise ValueError(
                        f"{name}: {record['id']} input hash differs from the row"
                    )
                chosen.append(line)
        if len(chosen) != len(hashes) or len(
            {json.loads(c)["id"] for c in chosen}
        ) != len(hashes):
            raise ValueError(f"{name}: does not cover every kept row exactly once")
        receipt["targets"][name] = {
            "source_sha256": hashlib.sha256(b"".join(raw)).hexdigest(),
            "rows": len(chosen),
            "sha256": _write(args.out_dir / f"{name}.jsonl", b"".join(chosen)),
        }
    _write(
        args.out_dir / "receipt.json",
        (json.dumps(receipt, indent=1, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
