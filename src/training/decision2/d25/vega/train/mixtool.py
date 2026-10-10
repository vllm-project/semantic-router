"""Derived training mixtures from a VERIFIED mixture, selected by ``meta.part``.

    # ablation: M2T-v5 without the in-distribution train splits
    python -m d25.vega.train.mixtool --src /data/d25/shared/data/v1/M2T-v5 \
        --out /data/d25/vega/train/mix/M2T-v5-noindist --drop-part indist

    # continuation mix: every row of the parts M3T-a adds to M2T-v5 (SYN1) + 3x as many other rows
    python -m d25.vega.train.mixtool --src /data/d25/shared/data/v1/M3T-a \
        --out /data/d25/vega/train/mix/M3Ta-cont --keep-new-parts-vs /data/d25/shared/data/v1/M2T-v5 \
        --others-ratio 3 --seed 20261019 --wait-h 30

Rows keep the source's (already shuffled) order; sampling of the other rows is a deterministic hash
of (seed, row id). The output has the source ``dev.jsonl.gz`` (unchanged unless ``--filter-dev``),
train shards of 50,000 rows, ``manifest.json`` (source manifest sha256, selection, counts by part,
output sha256s) and ``VERIFIED`` written last (built in ``<out>.partial`` and renamed), so a runner
arm can point ``train`` at it and wait. Re-running with the same arguments is a no-op.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SHARD_ROWS = 50_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def train_files(src: Path) -> list[Path]:
    files = sorted(
        p
        for p in src.iterdir()
        if p.name.startswith("train") and p.name.endswith((".jsonl", ".jsonl.gz"))
    )
    if not files:
        raise SystemExit(f"no train*.jsonl(.gz) files in {src}")
    return files


def rows(files: list[Path]):
    for path in files:
        opener = gzip.open if path.name.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield line


def part_of(row: dict) -> str:
    return str((row.get("meta") or {}).get("part", ""))


def part_counts(src: Path) -> Counter:
    manifest = src / "manifest.json"
    if manifest.exists():
        parts = (
            (json.loads(manifest.read_text()).get("composition") or {})
            .get("train", {})
            .get("part")
        )
        if parts:
            return Counter({str(k): int(v) for k, v in parts.items()})
    counts: Counter = Counter()
    for line in rows(train_files(src)):
        counts[part_of(json.loads(line))] += 1
    return counts


def keep_fraction(seed: int, row_id: str) -> float:
    digest = hashlib.sha256(f"{seed}:{row_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--src", required=True, help="VERIFIED mixture directory")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--drop-part", action="append", default=[], help="Drop rows with this meta.part"
    )
    parser.add_argument(
        "--keep-part",
        action="append",
        default=[],
        help="Always keep rows with this meta.part",
    )
    parser.add_argument(
        "--keep-new-parts-vs",
        help="Also keep every part that this base mixture does not have",
    )
    parser.add_argument(
        "--others-ratio",
        type=float,
        help="With keep parts: sample other rows to this x kept count",
    )
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument(
        "--filter-dev", action="store_true", help="Apply the drop filter to dev too"
    )
    parser.add_argument(
        "--wait-h",
        type=float,
        default=0.0,
        help="Wait for <src>/VERIFIED up to this long",
    )
    args = parser.parse_args()
    src, out = Path(args.src), Path(args.out)
    deadline = time.time() + args.wait_h * 3600
    while not (src / "VERIFIED").exists():
        if time.time() > deadline:
            raise SystemExit(f"{src}/VERIFIED not present")
        time.sleep(60)
    selection = {
        "drop_parts": sorted(args.drop_part),
        "keep_parts": sorted(args.keep_part),
        "keep_new_parts_vs": args.keep_new_parts_vs,
        "others_ratio": args.others_ratio,
        "seed": args.seed,
        "filter_dev": args.filter_dev,
    }
    src_manifest = src / "manifest.json"
    src_id = {
        "path": str(src),
        "verified": (src / "VERIFIED").read_text().strip(),
        "manifest_sha256": sha256(src_manifest) if src_manifest.exists() else None,
    }
    if (out / "VERIFIED").exists():
        done = json.loads((out / "manifest.json").read_text())
        if done.get("selection") == selection and done.get("source") == src_id:
            print(json.dumps({"out": str(out), "already": True, "rows": done["rows"]}))
            return 0
        raise SystemExit(
            f"{out} exists with a different selection/source; choose a new --out"
        )
    counts = part_counts(src)
    keep = set(args.keep_part)
    if args.keep_new_parts_vs:
        base_parts = set(part_counts(Path(args.keep_new_parts_vs)))
        new = {p for p in counts if p not in base_parts}
        if not new:
            raise SystemExit(
                f"{src} has no parts beyond {args.keep_new_parts_vs}: {sorted(counts)}"
            )
        keep |= new
    drop = set(args.drop_part)
    if keep & drop:
        raise SystemExit(f"parts both kept and dropped: {sorted(keep & drop)}")
    probability = 1.0
    if keep and args.others_ratio is not None:
        kept = sum(v for p, v in counts.items() if p in keep)
        others = sum(v for p, v in counts.items() if p not in keep and p not in drop)
        probability = min(1.0, args.others_ratio * kept / max(1, others))
    partial = out.with_name(out.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    began = time.time()
    written: Counter = Counter()
    seen: Counter = Counter()
    shard, shard_rows, files = None, 0, []

    def open_shard():
        index = len(files)
        path = partial / f"train-{index:05d}.jsonl.gz"
        files.append(path)
        return gzip.open(path, "wt", encoding="utf-8", compresslevel=6)

    for line in rows(train_files(src)):
        row = json.loads(line)
        part = part_of(row)
        seen[part] += 1
        if part in drop:
            continue
        if (
            keep
            and part not in keep
            and keep_fraction(args.seed, str(row["id"])) >= probability
        ):
            continue
        if shard is None or shard_rows >= SHARD_ROWS:
            if shard is not None:
                shard.close()
            shard, shard_rows = open_shard(), 0
        shard.write(line if line.endswith("\n") else line + "\n")
        shard_rows += 1
        written[part] += 1
    if shard is not None:
        shard.close()
    total = sum(written.values())
    if total == 0:
        raise SystemExit("selection is empty")
    final_names = {}
    for i, path in enumerate(files):
        name = f"train-{i:05d}-of-{len(files):05d}.jsonl.gz"
        path.rename(partial / name)
        final_names[name] = partial / name
    dev_src = src / "dev.jsonl.gz"
    dev_rows = None
    if dev_src.exists():
        if args.filter_dev and drop:
            dev_rows = 0
            with gzip.open(dev_src, "rt", encoding="utf-8") as fin, gzip.open(
                partial / "dev.jsonl.gz", "wt", encoding="utf-8"
            ) as fout:
                for line in fin:
                    if line.strip() and part_of(json.loads(line)) not in drop:
                        fout.write(line)
                        dev_rows += 1
        else:
            shutil.copyfile(dev_src, partial / "dev.jsonl.gz")
    manifest = {
        "name": out.name,
        "kind": "d25-vega derived mixture (ws-train mixtool)",
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": src_id,
        "selection": selection,
        "kept_parts": sorted(keep),
        "other_rows_probability": probability,
        "rows": total,
        "rows_by_part": dict(sorted(written.items())),
        "source_rows_by_part": dict(sorted(seen.items())),
        "dev": (
            "filtered"
            if dev_rows is not None
            else ("source copy" if dev_src.exists() else None)
        ),
        "dev_rows": dev_rows,
        "files": {
            name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for name, path in final_names.items()
        },
        "seconds": round(time.time() - began, 1),
    }
    if (partial / "dev.jsonl.gz").exists():
        manifest["files"]["dev.jsonl.gz"] = {
            "sha256": sha256(partial / "dev.jsonl.gz"),
            "bytes": (partial / "dev.jsonl.gz").stat().st_size,
        }
    (partial / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for path in partial.iterdir():
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    (partial / "VERIFIED").write_text(f"mixtool {manifest['created']} rows {total}\n")
    os.replace(partial, out)
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "name",
                    "rows",
                    "rows_by_part",
                    "kept_parts",
                    "other_rows_probability",
                    "seconds",
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
