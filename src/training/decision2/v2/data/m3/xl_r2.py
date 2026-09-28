"""XL release recipe revision r2 (M3b amendment 2 §4, amendment 3 §2–§3).

    python3 -m v2.data.m3.xl_r2 rows --pools xl-pools.json --r1-dir R1 \\
        --r1-manifest-sha256 HEX --out-dir RESCREEN [--workers 16]

``rows`` resolves every row of the union of the six r1 recipes by id from its pool's rows
files and writes ``rows/<pool>.jsonl`` with the pool file's line bytes unchanged, so the
rescreen reads each pool's own rendering; ``index.jsonl`` maps ids to pools and groups
(no text). There is one candidate file per pool because the overlap scan counts
boilerplate over the distinct candidate groups of one scan, as in every arm's own audit.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Any

from training.model.data import file_sha256
from v2.data.m2.common import read_jsonl

R1_RECIPES = (
    "mx-xl-full",
    "mx-xl-short",
    "cx-xl-a7v1-full",
    "cx-xl-a7v1-short",
    "cx-xl-v2v1-full",
    "cx-xl-v2v1-short",
)


def pool_file(pool: str) -> str:
    return pool.replace(":", "-")


def _write_new(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)


def _json(value: Any) -> bytes:
    return (json.dumps(value, indent=1, sort_keys=True) + "\n").encode()


def r1_recipes(
    r1_dir: Path, manifest_sha256: str
) -> tuple[dict[str, Any], dict[str, list[dict]]]:
    """The r1 manifest (checked against its expected SHA-256) and its six id lists
    (each checked against the manifest)."""
    raw = (r1_dir / "mx-xl.manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest_sha256:
        raise ValueError("r1 manifest SHA-256 differs from the expected value")
    manifest = json.loads(raw)
    recipes = {}
    for name in R1_RECIPES:
        path = r1_dir / f"{name}.ids.jsonl"
        if file_sha256(path) != manifest["recipes"][name]["sha256"]:
            raise ValueError(f"{name}: ids file differs from the r1 manifest")
        recipes[name] = list(read_jsonl(path))
    return manifest, recipes


def union_pools(recipes: dict[str, list[dict]]) -> dict[str, dict[str, int]]:
    """pool -> id -> native tokens over every recipe; an id keeps one pool."""
    pool_of: dict[str, str] = {}
    union: dict[str, dict[str, int]] = collections.defaultdict(dict)
    for rows in recipes.values():
        for row in rows:
            if pool_of.setdefault(row["id"], row["pool"]) != row["pool"]:
                raise ValueError(f"{row['id']}: listed under two pools")
            union[row["pool"]][row["id"]] = row["native"]
    return dict(union)


def _resolve(task: tuple[str, list[str], set[str], Path]) -> tuple[Any, ...]:
    pool, paths, wanted, out = task
    groups: dict[str, str] = {}
    sources = []
    digest = hashlib.sha256()
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        for path in paths:
            whole = hashlib.sha256()
            with open(path, "rb") as source:
                for raw in source:
                    whole.update(raw)
                    if not raw.strip():
                        continue
                    row = json.loads(raw)
                    if row["id"] not in wanted:
                        continue
                    if row["id"] in groups:
                        raise ValueError(f"{pool}: duplicate id {row['id']}")
                    line = raw if raw.endswith(b"\n") else raw + b"\n"
                    stream.write(line)
                    digest.update(line)
                    groups[row["id"]] = row["group_id"]
            sources.append({"path": str(path), "sha256": whole.hexdigest()})
    if len(groups) != len(wanted):
        raise ValueError(
            f"{pool}: {len(wanted) - len(groups)} recipe ids are not in the pool files"
        )
    return pool, groups, digest.hexdigest(), sources


def resolve_rows(
    specs: dict[str, dict],
    recipes: dict[str, list[dict]],
    out_dir: Path,
    workers: int = 1,
) -> dict[str, Any]:
    union = union_pools(recipes)
    unknown = sorted(set(union) - set(specs))
    if unknown:
        raise ValueError(f"recipe pools missing from the pools spec: {unknown}")
    rows_dir = out_dir / "rows"
    rows_dir.mkdir(parents=True, mode=0o700)
    tasks = [
        (
            pool,
            specs[pool]["rows"],
            set(union[pool]),
            rows_dir / f"{pool_file(pool)}.jsonl",
        )
        for pool in specs
        if pool in union
    ]
    if workers > 1:
        with multiprocessing.get_context("fork").Pool(min(workers, len(tasks))) as ctx:
            results = ctx.map(_resolve, tasks, chunksize=1)
    else:
        results = [_resolve(task) for task in tasks]
    index, pools = [], {}
    pools_of_group: dict[str, set[str]] = collections.defaultdict(set)
    for pool, groups, digest, sources in results:
        index.extend({"group": g, "id": i, "pool": pool} for i, g in groups.items())
        for group in groups.values():
            pools_of_group[group].add(pool)
        pools[pool] = {
            "rows": len(groups),
            "groups": len(set(groups.values())),
            "native_tokens": sum(union[pool].values()),
            "candidates": f"rows/{pool_file(pool)}.jsonl",
            "candidates_sha256": digest,
            "sources": sources,
        }
    index.sort(key=lambda r: (r["pool"], r["id"]))
    data = b"".join(json.dumps(r, sort_keys=True).encode() + b"\n" for r in index)
    _write_new(out_dir / "index.jsonl", data)
    return {
        "schema": "decision2-mx-xl-r2-rescreen-rows/1",
        "recipes_rows": {name: len(rows) for name, rows in recipes.items()},
        "union": {
            "rows": len(index),
            "native_tokens": sum(sum(ids.values()) for ids in union.values()),
            "pool_groups": sum(p["groups"] for p in pools.values()),
            "group_ids": len(pools_of_group),
            "group_ids_in_several_pools": sum(
                1 for found in pools_of_group.values() if len(found) > 1
            ),
        },
        "index_sha256": hashlib.sha256(data).hexdigest(),
        "pools": pools,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("rows")
    r.add_argument("--pools", type=Path, required=True)
    r.add_argument("--r1-dir", type=Path, required=True)
    r.add_argument("--r1-manifest-sha256", required=True)
    r.add_argument("--out-dir", type=Path, required=True)
    r.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == "rows":
        manifest, recipes = r1_recipes(args.r1_dir, args.r1_manifest_sha256)
        args.out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        receipt = resolve_rows(
            json.loads(args.pools.read_text()), recipes, args.out_dir, args.workers
        )
        receipt["pools_sha256"] = file_sha256(args.pools)
        receipt["r1_manifest_sha256"] = args.r1_manifest_sha256
        receipt["r1_ids_sha256"] = {
            name: manifest["recipes"][name]["sha256"] for name in R1_RECIPES
        }
        _write_new(args.out_dir / "rows.receipt.json", _json(receipt))
        print(json.dumps(receipt["union"], sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
