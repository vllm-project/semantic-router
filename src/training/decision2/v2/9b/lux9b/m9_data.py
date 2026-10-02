"""9B M9 TRAIN builds: the released 9B mixture x60 plus the release-safe IB1-r3 and IB2 blocks.

Stage 2 (amendment 2): ``train.jsonl`` = every x60 line byte for byte, then every kept IB1 line, then every kept IB2
line (file order), and ``teacher.jsonl`` = x60's own-Lux targets byte for byte (IB rows carry no teacher target, so
they train on gold only under ``--teacher-partial``). ``--exclude-family`` drops IB families (the transfer-only
ablation drops the in-distribution ones). The build refuses (no silent drop) any IB row whose id, lineage group or
canonical input hash also occurs in x60, or whose id or input hash occurs earlier in the build (an IB lineage group
holds several rows by design, so groups are checked against x60 only). IB token counts come from the releases'
``*.tokens.jsonl``. ``--ib3 F`` / ``--ib4 F`` (9B M10) append further blocks after IB2 under the same rules.

Stage 3 (amendment 3), ``--match-tokens N``: the same IB blocks at matched tokens. x60 is first cut to N minus the
kept IB native tokens in whole groups, stratified by pool x source x task type x language as the x60 recipe was
(``lux9b.m3_data.recipe_budget``, M6's KH cut; native tokens and pools from the x60 ids file ``--x60-ids``). The
kept x60 lines and their own-Lux target lines stay byte for byte in file order; every kept IB row needs a token
count. Duplicate checks run against the whole of x60, as in stage 2. ``--cut-language en`` (9B M10 KSW) cuts only
the all-English groups; every group with another language is kept whole.

usage: m9_data.py --x60-dir D --ib1 F --ib2 F [--ib3 F] [--ib4 F] [--exclude-family NAME ...]
                  [--match-tokens N --x60-ids F --keep-seed S [--keep-tolerance T] [--cut-language L]] --output DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
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


def x60_cut(
    rows: list[dict[str, Any]],
    ids_path: Path,
    budget: int,
    seed: str,
    tolerance: float,
    cut_language: str | None = None,
) -> tuple[set[str], dict[str, Any]]:
    """Ids of the x60 rows kept by the stratified whole-group cut to ``budget`` native tokens.

    With ``cut_language`` only groups whose every row has that language are cut; every other group is kept whole
    and its tokens come off the budget first."""
    try:
        from lux9b.m3_data import recipe_budget
    except ModuleNotFoundError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from lux9b.m3_data import recipe_budget

    entries: dict[str, dict[str, Any]] = {}
    with ids_path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                entry = json.loads(line)
                entries[entry["id"]] = entry
    native, pool_of = {}, {}
    for row in rows:
        entry = entries.get(row["id"])
        if entry is None or entry["source"] != row["source"]:
            raise ValueError(
                f"{row['id']}: not in the x60 ids file with the same source"
            )
        native[row["id"]] = int(entry["native"])
        pool_of[row["id"]] = entry["pool"]
    fixed: list[dict[str, Any]] = []
    if cut_language is not None:
        mixed = {r["group_id"] for r in rows if r["language"] != cut_language}
        fixed = [r for r in rows if r["group_id"] in mixed]
        rows = [r for r in rows if r["group_id"] not in mixed]
    fixed_tokens = sum(native[r["id"]] for r in fixed)
    kept, stats = recipe_budget(
        rows, native, pool_of, budget - fixed_tokens, seed, tolerance
    )
    kept = fixed + kept
    kept_ids = {r["id"] for r in kept}
    if cut_language is not None:
        stats |= {
            "cut_language": cut_language,
            "fixed_rows": len(fixed),
            "fixed_groups": len({r["group_id"] for r in fixed}),
            "fixed_native_tokens": fixed_tokens,
            "native_tokens": stats["native_tokens"] + fixed_tokens,
        }
    by_type: Counter[str] = Counter()
    for r in kept:
        by_type[r["task_type"]] += native[r["id"]]
    stats |= {
        "ids_file": str(ids_path),
        "ids_sha256": sha256(ids_path),
        "x60_native_tokens": sum(native.values()),
        "kept_native_tokens_by_type": dict(sorted(by_type.items())),
    }
    return kept_ids, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x60-dir", type=Path, required=True)
    parser.add_argument("--ib1", type=Path, required=True)
    parser.add_argument("--ib2", type=Path, required=True)
    parser.add_argument("--ib3", type=Path)
    parser.add_argument("--ib4", type=Path)
    parser.add_argument("--exclude-family", action="append", default=[])
    parser.add_argument("--match-tokens", type=int)
    parser.add_argument("--x60-ids", type=Path)
    parser.add_argument("--keep-seed")
    parser.add_argument("--keep-tolerance", type=float, default=0.01)
    parser.add_argument("--cut-language")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    match = args.match_tokens is not None
    if match and not (args.x60_ids and args.keep_seed):
        parser.error("--match-tokens needs --x60-ids and --keep-seed")
    if args.cut_language and not match:
        parser.error("--cut-language goes with --match-tokens")
    args.output.mkdir(parents=True, exist_ok=False)
    x60_train, x60_teacher = (
        args.x60_dir / "train.jsonl",
        args.x60_dir / "teacher.jsonl",
    )
    seen: dict[str, set[str]] = {"id": set(), "group_id": set(), "input_sha256": set()}
    x60_rows = 0
    order: list[str] = []
    strata_rows: list[dict[str, Any]] = []
    with x60_train.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            x60_rows += 1
            for key in seen:
                seen[key].add(row[key])
            if match:
                order.append(row["id"])
                strata_rows.append(
                    {
                        k: row[k]
                        for k in ("id", "group_id", "source", "task_type", "language")
                    }
                )
    kept: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    ib_tokens = 0
    ib_by_type: Counter[str] = Counter()
    sources: dict[str, Any] = {}
    ib_lines: list[bytes] = []
    blocks = (
        [("ib1", args.ib1), ("ib2", args.ib2)]
        + ([("ib3", args.ib3)] if args.ib3 else [])
        + ([("ib4", args.ib4)] if args.ib4 else [])
    )
    for name, path in blocks:
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
                seen["id"].add(row["id"])
                seen["input_sha256"].add(row["input_sha256"])
                if match and row["id"] not in counts:
                    raise ValueError(f"{row['id']}: no native token count")
                if not raw.endswith(b"\n"):
                    raw += b"\n"
                ib_lines.append(raw)
                kept[fam] += 1
                ib_tokens += counts.get(row["id"], 0)
                ib_by_type[row.get("task_type", "?")] += counts.get(row["id"], 0)
    out_train = args.output / "train.jsonl"
    keep_stats = None
    if match:
        keep_ids, keep_stats = x60_cut(
            strata_rows,
            args.x60_ids,
            args.match_tokens - ib_tokens,
            args.keep_seed,
            args.keep_tolerance,
            args.cut_language,
        )
        with out_train.open("xb") as out, x60_train.open("rb") as stream:
            n = 0
            for raw, rid in zip(stream, order, strict=True):
                if rid in keep_ids:
                    out.write(raw)
                    n += 1
            out.writelines(ib_lines)
        teacher_ids = set()
        with (args.output / "teacher.jsonl").open("xb") as out, x60_teacher.open(
            "rb"
        ) as stream:
            for raw in stream:
                rid = json.loads(raw)["id"]
                if rid in teacher_ids or rid not in seen["id"]:
                    raise ValueError(f"{rid}: teacher row repeated or not an x60 row")
                teacher_ids.add(rid)
                if rid in keep_ids:
                    out.write(raw)
        if not keep_ids <= teacher_ids:
            raise ValueError("a kept x60 row has no own-Lux target")
        x60_kept = n
    else:
        with out_train.open("xb") as out:
            with x60_train.open("rb") as stream:
                shutil.copyfileobj(stream, out)
            out.writelines(ib_lines)
        shutil.copyfile(x60_teacher, args.output / "teacher.jsonl")
        x60_kept = x60_rows
    x60_manifest = json.loads((args.x60_dir / "manifest.json").read_text())
    manifest = {
        "schema": "lux9b-m9-stage3-train/1" if match else "lux9b-m9-stage2-train/1",
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
        "rows": x60_kept + sum(kept.values()),
        "train_sha256": sha256(out_train),
        "teacher_sha256": sha256(args.output / "teacher.jsonl"),
    }
    if match:
        x60_tokens = keep_stats["native_tokens"]
        manifest |= {
            "match_tokens": args.match_tokens,
            "x60_keep": keep_stats,
            "x60_rows_kept": x60_kept,
            "teacher_rows": x60_kept,
            "ib_native_tokens_by_type": dict(sorted(ib_by_type.items())),
            "train_native_tokens": x60_tokens + ib_tokens,
            "ib_token_share": round(ib_tokens / (x60_tokens + ib_tokens), 4),
        }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
