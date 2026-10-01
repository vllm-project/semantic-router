"""9B M9 stage-2 TRAIN builds (amendment 2): the released 9B mixture x60 plus the release-safe IB1-r3 and IB2 blocks.

Each build writes ``train.jsonl`` = every x60 line byte for byte, then every kept IB1 line, then every kept IB2 line
(file order), and ``teacher.jsonl`` = x60's own-Lux targets byte for byte (IB rows carry no teacher target, so they
train on gold only under ``--teacher-partial``). ``--exclude-family`` drops IB families (the transfer-only ablation
drops the in-distribution ones). The build refuses (no silent drop) any IB row whose id, lineage group or canonical
input hash also occurs in x60 or earlier in the build. Token counts come from the IB releases' ``*.tokens.jsonl``
files and x60's manifest.

usage: m9_data.py --x60-dir D --ib1 F --ib2 F [--exclude-family NAME ...] --output DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 23), b""):
            h.update(block)
    return h.hexdigest()


def tokens_by_id(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {}
    out = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                out[row["id"]] = int(row["native"])
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x60-dir", type=Path, required=True)
    parser.add_argument("--ib1", type=Path, required=True)
    parser.add_argument("--ib2", type=Path, required=True)
    parser.add_argument("--exclude-family", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    x60_train, x60_teacher = (
        args.x60_dir / "train.jsonl",
        args.x60_dir / "teacher.jsonl",
    )
    seen: dict[str, set[str]] = {"id": set(), "group_id": set(), "input_sha256": set()}
    x60_rows = 0
    with x60_train.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            x60_rows += 1
            for key in seen:
                seen[key].add(row[key])
    kept: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    ib_tokens = 0
    sources: dict[str, Any] = {}
    out_train = args.output / "train.jsonl"
    with out_train.open("xb") as out:
        with x60_train.open("rb") as stream:
            shutil.copyfileobj(stream, out)
        for name, path in (("ib1", args.ib1), ("ib2", args.ib2)):
            counts = tokens_by_id(
                path.with_name(path.name.replace(".jsonl", ".tokens.jsonl"))
            )
            sources[name] = {"file": str(path), "sha256": sha256(path)}
            with path.open("rb") as stream:
                for raw in stream:
                    row = json.loads(raw)
                    fam = f"{name}/{row['family']}"
                    if row["family"] in args.exclude_family:
                        dropped[fam] += 1
                        continue
                    for key in seen:
                        if row[key] in seen[key]:
                            raise ValueError(
                                f"{row['id']}: {key} {row[key]} already in the build"
                            )
                    for key in seen:
                        seen[key].add(row[key])
                    if not raw.endswith(b"\n"):
                        raw += b"\n"
                    out.write(raw)
                    kept[fam] += 1
                    ib_tokens += counts.get(row["id"], 0)
    shutil.copyfile(x60_teacher, args.output / "teacher.jsonl")
    x60_manifest = json.loads((args.x60_dir / "manifest.json").read_text())
    manifest = {
        "schema": "lux9b-m9-stage2-train/1",
        "x60": {
            "train_sha256": sha256(x60_train),
            "teacher_sha256": sha256(x60_teacher),
            "rows": x60_rows,
            "recipe": x60_manifest.get("recipe"),
        },
        "ib_sources": sources,
        "excluded_families": sorted(args.exclude_family),
        "ib_rows_kept": dict(sorted(kept.items())),
        "ib_rows_dropped": dict(sorted(dropped.items())),
        "ib_native_tokens_kept": ib_tokens,
        "rows": x60_rows + sum(kept.values()),
        "train_sha256": sha256(out_train),
        "teacher_sha256": sha256(args.output / "teacher.jsonl"),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
