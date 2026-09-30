"""Milestone 7 CPU data builds for the 9B track: the K-seed continuation files and MLX-DEV-9B.

* ``topup``: the two continuation TRAIN files on the K recipe (x60).
  - **P**: x60 replay of ``replay_tokens`` native tokens (whole groups, stratified by pool x
    source x task type x language as the x60 recipe was, ``recipe_budget``) plus the PN1-r2
    block repeated ``pn1_repeat`` times (copy k > 1 has its ids suffixed ``~r<k>``). Replay rows
    keep their x60 own-Lux targets; block rows have none (``--teacher-partial``: gold only).
  - **C** (matched-token control): x60 replay only, budgeted to P's native tokens with the same
    seed, so P's replay is contained in C's (checked); own-Lux targets on every row.
* ``mlxdev``: MLX-DEV-9B, the decoder's MLX-DEV panel minus every group that shares a group id,
  row id, input hash or a normalized state line of >= 20 characters with x60 (the K seeds'
  TRAIN, which contains both continuation files) or the PN1-r2 block, with its index.

    python3 -m lux9b.m7_data topup --spec SPEC --root NAME=PATH ... --tokenizer /model --output-dir OUT
    python3 -m lux9b.m7_data mlxdev --spec SPEC --root NAME=PATH ... --output-dir OUT

Every input is hash-verified; outputs go to a new directory with ``manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from lux9b.m3_data import (
    c1_keys,
    check_teacher,
    denied_hits,
    read_jsonl,
    recipe_budget,
    resolve,
    token_lengths,
    verified,
    write_lines,
)
from lux9b.m6_data import load_x60
from training.model.data import check_partition_isolation, file_sha256, load_partition
from v2.common import eval_only

SCHEMA = "decision2-9b-m7-data/1"


def pn1_block(rows: list[dict[str, Any]], repeat: int) -> list[dict[str, Any]]:
    """The PN1-r2 rows ``repeat`` times; copy k > 1 gets ids suffixed ``~r<k>``."""
    if repeat < 1:
        raise ValueError("pn1_repeat must be >= 1")
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids) or any("~r" in i for i in ids):
        raise ValueError("PN1-r2 ids must be unique and unsuffixed")
    block = []
    for k in range(1, repeat + 1):
        for row in rows:
            block.append(row if k == 1 else dict(row, id=f"{row['id']}~r{k}"))
    return block


def closest_budget(rows, native, pool_of, target: int, seed: str):
    """The stratified replay whose realized native tokens are closest to ``target``.

    The stratified selection overshoots a small budget (every stratum takes whole groups
    until its share is reached), so the requested budget is bisected below the target.
    """
    lo, hi, best = target // 2, target, None
    while lo <= hi:
        mid = (lo + hi) // 2
        kept, stats = recipe_budget(rows, native, pool_of, mid, seed, 1.0)
        diff = stats["native_tokens"] - target
        if best is None or abs(diff) < abs(best[1]["native_tokens"] - target):
            best = (kept, stats)
        if diff > 0:
            hi = mid - 1
        elif diff < 0:
            lo = mid + 1
        else:
            break
    kept, stats = best
    return kept, {**stats, "target_tokens": target}


def state_segments(row: dict[str, Any]) -> set[bytes]:
    """Hashes of the row's normalized state lines (>= MIN_SEGMENT characters); instructions
    and option descriptions are templates shared by every row of a source, so they are not
    screened."""
    from v2.dec.m5_block import MIN_SEGMENT, normalize

    out = set()
    for part in row["state"].split("\n"):
        part = normalize(part)
        if len(part) >= MIN_SEGMENT:
            out.add(hashlib.sha256(part.encode("utf-8")).digest()[:16])
    return out


def type_shares(rows, tokens: dict[str, int]) -> dict[str, float]:
    by_type: Counter = Counter()
    for r in rows:
        by_type[r["task_type"]] += tokens[r["id"]]
    total = sum(by_type.values())
    return {t: round(n / total, 4) for t, n in sorted(by_type.items())}


def source_guard(spec, roots, rows) -> dict[str, Any]:
    registry_path = resolve(spec["denied_sources"]["c1_registry"], roots)
    keys = c1_keys(json.loads(registry_path.read_text()))
    sources = Counter(r["source"] for r in rows)
    extra = spec["denied_sources"].get("extra", [])
    denied = {s: h for s in sources if (h := denied_hits(s, keys, extra))}
    if denied:
        raise ValueError(f"denied sources in a continuation TRAIN: {denied}")
    return {
        "c1_registry_sha256": file_sha256(registry_path),
        "c1_keys": len(keys),
        "extra": extra,
        "denied": 0,
        "sources": dict(sorted(sources.items())),
    }


def topup(spec, roots, tokenizer: Path, workers: int, out: Path) -> dict[str, Any]:
    inputs: dict[str, str] = {}
    rows, own, ids = load_x60(spec, roots, inputs)
    native = {r["id"]: ids[r["id"]]["native"] for r in rows}
    pool_of = {r["id"]: ids[r["id"]]["pool"] for r in rows}
    pn1 = load_partition(verified(spec["pn1"]["train"], roots, inputs), "train")
    x60_ids = {r["id"] for r in rows}
    x60_inputs = {r["input_sha256"] for r in rows}
    x60_groups = {r["group_id"] for r in rows}
    if any(
        r["id"] in x60_ids
        or r["input_sha256"] in x60_inputs
        or r["group_id"] in x60_groups
        for r in pn1
    ):
        raise ValueError("a PN1-r2 row shares an id, input or group with x60")
    lengths = dict(zip((r["id"] for r in pn1), token_lengths(pn1, tokenizer, workers)))
    if max(lengths.values()) > spec["max_length"]:
        raise ValueError("a PN1-r2 row exceeds max_length")
    block = pn1_block(pn1, spec["pn1"]["repeat"])
    for r in block:
        lengths[r["id"]] = lengths[r["id"].split("~r")[0]]
    block_tokens = sum(lengths[r["id"]] for r in block)
    seed = spec["seed"]
    replay_p, stats_p = recipe_budget(
        rows,
        native,
        pool_of,
        spec["replay_tokens"],
        f"{seed}:replay",
        spec["replay_tolerance"],
    )
    target_c = stats_p["native_tokens"] + block_tokens
    replay_c, stats_c = closest_budget(
        rows, native, pool_of, target_c, f"{seed}:replay"
    )
    groups_p = {r["group_id"] for r in replay_p}
    groups_c = {r["group_id"] for r in replay_c}
    if not groups_p <= groups_c:
        raise ValueError("P's replay is not contained in C's replay")
    tokens = {**native, **lengths}
    arms = {
        "P": sorted(replay_p + block, key=lambda r: r["id"]),
        "C": sorted(replay_c, key=lambda r: r["id"]),
    }
    partitions_iso = {}
    for entry in spec["isolation"]:
        partitions_iso[entry["role"]] = load_partition(
            verified(entry, roots, inputs), entry.get("split", entry["role"])
        )
    out.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "step": "topup",
        "name": spec["name"],
        "inputs_sha256": inputs,
        "seed": seed,
        "pn1": {
            "rows": len(pn1),
            "repeat": spec["pn1"]["repeat"],
            "block_rows": len(block),
            "native_tokens_once": sum(lengths[r["id"]] for r in pn1),
            "block_native_tokens": block_tokens,
            "max_tokens": max(lengths.values()),
            "rows_by_language": dict(
                sorted(Counter(r["language"] for r in pn1).items())
            ),
            "rows_by_family": dict(sorted(Counter(r["family"] for r in pn1).items())),
        },
        "replay": {"P": stats_p, "C": stats_c, "C_target_tokens": target_c},
        "arms": {},
    }
    for name, train in arms.items():
        eval_only.check_rows(train)
        if len({r["id"] for r in train}) != len(train):
            raise ValueError(f"duplicate id in TRAIN {name}")
        check_partition_isolation({"train": train, **partitions_iso})
        guard = source_guard(spec, roots, train)
        (out / name).mkdir()
        train_sha = write_lines(out / name / "train.jsonl", train)
        load_partition(out / name / "train.jsonl", "train")
        taught = [r for r in train if r["id"] in own]
        for r in taught:
            check_teacher(r, own[r["id"]])
        teacher_sha = write_lines(
            out / name / "teacher.jsonl", [own[r["id"]] for r in taught]
        )
        total = sum(tokens[r["id"]] for r in train)
        manifest["arms"][name] = {
            "train_rows": len(train),
            "train_native_tokens": total,
            "train_sha256": train_sha,
            "teacher_rows": len(taught),
            "teacher_sha256": teacher_sha,
            "gold_only_rows": len(train) - len(taught),
            "token_share_by_type": type_shares(train, tokens),
            "rows_by_type": dict(
                sorted(Counter(r["task_type"] for r in train).items())
            ),
            "replay_rows_by_pool": dict(
                sorted(Counter(pool_of[r["id"]] for r in taught).items())
            ),
            "source_guard": guard,
        }
    p_tok = manifest["arms"]["P"]["train_native_tokens"]
    c_tok = manifest["arms"]["C"]["train_native_tokens"]
    manifest["token_match"] = {
        "P": p_tok,
        "C": c_tok,
        "relative_gap": round(abs(p_tok - c_tok) / p_tok, 6),
    }
    if manifest["token_match"]["relative_gap"] > spec["tolerance"]:
        raise ValueError(
            f"P and C differ by more than the tolerance: {manifest['token_match']}"
        )
    manifest["isolation"] = sorted(partitions_iso)
    return manifest


def mlxdev(spec, roots, out: Path) -> dict[str, Any]:
    segments = state_segments
    inputs: dict[str, str] = {}
    rows, _, _ = load_x60(spec, roots, inputs)
    pn1 = load_partition(verified(spec["pn1"]["train"], roots, inputs), "train")
    panel_path = verified(spec["mlxdev"]["panel"], roots, inputs)
    panel = load_partition(panel_path, "select")
    index = read_jsonl(verified(spec["mlxdev"]["index"], roots, inputs))
    by_id = {r["id"]: r for r in panel}
    if [e["id"] for e in index] != [r["id"] for r in panel]:
        raise ValueError("MLX-DEV index and panel rows differ")
    seen = {"id": set(), "input_sha256": set(), "group_id": set()}
    seen_segments: set[bytes] = set()
    for row in rows + pn1:
        for field in seen:
            seen[field].add(row[field])
        seen_segments |= segments(row)
    dropped: dict[str, set[str]] = {k: set() for k in (*seen, "segment")}
    for row in panel:
        for field in seen:
            if row[field] in seen[field]:
                dropped[field].add(row["group_id"])
        if segments(row) & seen_segments:
            dropped["segment"].add(row["group_id"])
    drop = set().union(*dropped.values())
    kept_index = [e for e in index if e["group_id"] not in drop]
    kept = [by_id[e["id"]] for e in kept_index]
    cells = {e["cell"] for e in index}
    empty = sorted(cells - {e["cell"] for e in kept_index})
    if empty:
        raise ValueError(f"MLX-DEV-9B would leave cells empty: {empty}")
    out.mkdir(parents=True)
    panel_sha = write_lines(out / "panel.jsonl", kept)
    load_partition(out / "panel.jsonl", "select")
    index_sha = write_lines(out / "panel.jsonl.index.jsonl", kept_index)
    per_cell = {}
    for cell in sorted(cells):
        entries = [e for e in kept_index if e["cell"] == cell]
        per_cell[cell] = {
            "rows": len(entries),
            "rows_before": sum(e["cell"] == cell for e in index),
            "groups": len({e["group_id"] for e in entries}),
            "rows_by_language": dict(
                sorted(Counter(e["language"] for e in entries).items())
            ),
            "gold_by_key": dict(
                sorted(Counter(e["gold_key"] for e in entries).items())
            ),
        }
    return {
        "schema": SCHEMA,
        "step": "mlxdev",
        "name": spec["name"],
        "inputs_sha256": inputs,
        "rows_before": len(panel),
        "groups_before": len({e["group_id"] for e in index}),
        "rows": len(kept),
        "groups": len({e["group_id"] for e in kept_index}),
        "dropped_groups": {k: len(v) for k, v in dropped.items()},
        "dropped_groups_total": len(drop),
        "screened_rows": {"x60": len(rows), "pn1_r2": len(pn1)},
        "cells": per_cell,
        "panel_sha256": panel_sha,
        "index_sha256": index_sha,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=("topup", "mlxdev"))
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--root", action="append", required=True, help="name=<path>")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path)
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    args = ap.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"{args.output_dir} exists")
    roots = {}
    for item in args.root:
        name, _, path = item.partition("=")
        roots[name] = Path(path)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    eval_only.guard(args, {k: v for k, v in spec.items() if k != "note"})
    if args.step == "topup":
        if not args.tokenizer:
            ap.error("topup needs --tokenizer")
        manifest = topup(spec, roots, args.tokenizer, args.workers, args.output_dir)
    else:
        manifest = mlxdev(spec, roots, args.output_dir)
    manifest["spec_sha256"] = file_sha256(args.spec)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        k: manifest[k]
        for k in ("step", "token_match", "rows", "groups", "panel_sha256")
        if k in manifest
    }
    if "arms" in manifest:
        summary["arms"] = {
            n: {
                k: a[k]
                for k in (
                    "train_rows",
                    "train_native_tokens",
                    "train_sha256",
                    "teacher_sha256",
                )
            }
            for n, a in manifest["arms"].items()
        }
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
