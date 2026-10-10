"""Image training rows for d3-edge with d3 soft labels.

    python -m d25.family.edge_data mix --rows mm-v1a/rows-*.jsonl.gz --out DIR \
        [--probs teacher/d3-mmv1a/probs.jsonl] [--gold-weight 0.5]

Writes one file per input file (same name) under ``DIR`` with ``target = g * gold + (1 - g) * d3``
(the rule that built M2T-d3; rows d3 could not score keep gold), the gold target in
``meta.gold_target`` and d3's probabilities in ``meta.teachers.d3``. Image references become absolute
paths, because they are relative to the directory of the source file. Without ``--probs`` (labels not
there yet) the targets stay gold. ``DIR/manifest.json`` records the inputs, counts and rule.
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

from d25.omni.eval.shards import read_jsonl
from d25.omni.model import checkpoint


def load_probs(path: Path | None) -> dict[str, list[float] | None]:
    if path is None:
        return {}
    return {
        str(r["id"]): r["probs"] if r.get("status") == "ok" else None
        for r in read_jsonl(path)
    }


def absolute(ref: str, root: Path) -> str:
    return (
        ref if ref.startswith("data:") or Path(ref).is_absolute() else str(root / ref)
    )


def mix(
    rows: list[str], out: Path, probs_path: Path | None, gold_weight: float
) -> dict:
    teacher = load_probs(probs_path)
    out.mkdir(parents=True, exist_ok=True)
    counts = {"rows": 0, "mixed": 0, "gold_only": 0}
    for name in rows:
        source = Path(name)
        target = out / source.name
        tmp = target.with_name(target.name + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as stream:
            for row in read_jsonl(source):
                gold = [float(v) for v in row["target"]]
                probs = teacher.get(str(row["id"]))
                meta = dict(row.get("meta") or {})
                meta["gold_target"] = gold
                if probs is not None and len(probs) == len(gold):
                    meta.setdefault("teachers", {})["d3"] = probs
                    row["target"] = [
                        gold_weight * g + (1 - gold_weight) * p
                        for g, p in zip(gold, probs)
                    ]
                    counts["mixed"] += 1
                else:
                    counts["gold_only"] += 1
                row["meta"] = meta
                row["images"] = [
                    absolute(ref, source.parent) for ref in row.get("images") or []
                ]
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts["rows"] += 1
        tmp.replace(target)
    manifest = {
        "rule": f"target = {gold_weight:g} * gold + {1 - gold_weight:g} * d3 (gold where d3 is missing)",
        "probs": str(probs_path) if probs_path else None,
        "probs_sha256": checkpoint.file_sha256(probs_path) if probs_path else None,
        "rows_files_sha256": {name: checkpoint.file_sha256(name) for name in rows},
        "counts": counts,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    checkpoint.write_json(out / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mix")
    m.add_argument("--rows", nargs="+", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--probs")
    m.add_argument("--gold-weight", type=float, default=0.5)
    args = parser.parse_args()
    probs = Path(args.probs) if args.probs and Path(args.probs).exists() else None
    manifest = mix(args.rows, Path(args.out), probs, args.gold_weight)
    print(json.dumps(manifest["counts"] | {"probs": manifest["probs"]}))


if __name__ == "__main__":
    main()
