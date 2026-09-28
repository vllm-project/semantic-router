"""Build one v2 human-source arm (H1, H3, H5, H6, E11) from pinned sources.

    python3 -m v2.data.m2.build --arm h3 --sources /data/dev2/private/sources \\
        --v1-rows v1-rows.json --out-dir OUT [--workers 32]

Rules: data-arms-v2-prereg-2026-09-28.md and its amendments. Families of the
arm run in parallel processes; each family's rows are capped by whole groups in
seed-hash order, then the arm is sliced into TRAIN / AHO / SHO and written.
``--v1-rows`` names a JSON list of v1 row files whose source items must not be
reused (read through ``v1_local_ids``). Refuses to overwrite an existing arm.
"""

from __future__ import annotations

import argparse
import collections
import importlib
import json
import multiprocessing
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import training.model.data as contract
from training.model.data import file_sha256
from v2.data import textnorm
from v2.data.m2 import common, constructions, spec, text

SCHEMA = "decision2-m2-build/v1"
RULES = "data-arms-v2-prereg-2026-09-28.md (+ amendments)"
SOURCE_MODULES = ("src_qa", "src_multihop", "src_short", "src_score")
ROOT = Path(__file__).resolve().parents[3]
V1_ROWS_KEY = "v1_rows"
_FAMILIES: dict[str, spec.FamilySpec] = {}
_DIRS: dict[str, Path] = {}


def families(modules: Sequence[str] = SOURCE_MODULES) -> list[spec.FamilySpec]:
    found: list[spec.FamilySpec] = []
    for name in modules:
        found.extend(importlib.import_module(f"v2.data.m2.{name}").FAMILIES)
    names = [item.family for item in found]
    repeated = sorted(n for n, c in collections.Counter(names).items() if c > 1)
    if repeated:
        raise ValueError(f"duplicate family names {repeated}")
    return found


def v1_local_ids(dirs: Mapping[str, Path], source: str) -> set[str]:
    """``source_local_id`` values of v1 rows of ``source`` (empty without --v1-rows)."""
    manifest = dirs.get(V1_ROWS_KEY)
    if manifest is None:
        return set()
    ids: set[str] = set()
    for path in json.loads(Path(manifest).read_text(encoding="utf-8")):
        for row in common.read_jsonl(Path(path)):
            if row.get("source") == source:
                ids.add(row["audit_metadata"]["source_local_id"])
    return ids


def module_hashes(modules: Sequence[str]) -> dict[str, str]:
    paths = [
        Path(__file__),
        Path(common.__file__),
        Path(constructions.__file__),
        Path(spec.__file__),
        Path(text.__file__),
        Path(textnorm.__file__),
        Path(contract.__file__),
        *(Path(importlib.import_module(f"v2.data.m2.{m}").__file__) for m in modules),
    ]
    return {
        path.resolve().relative_to(ROOT).as_posix(): file_sha256(path)
        for path in sorted(set(paths), key=lambda p: p.resolve().as_posix())
    }


def _run_family(name: str) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    item = _FAMILIES[name]
    rows, report = item.build(_DIRS)
    for row in rows:
        if row["family"] != item.family or row["source"] != item.source:
            raise ValueError(
                f"{name}: row {row['id']} has family/source {row['family']}/{row['source']}"
            )
    capped = common.cap_groups(rows, item.cap_rows, item.seed)
    return (
        name,
        capped,
        {
            **report,
            "source": item.source,
            "cap_rows": item.cap_rows,
            "cap_seed": item.seed,
            "candidates": len(rows),
            "after_cap": len(capped),
            "groups": len({row["group_id"] for row in capped}),
            "shortfall": max(0, item.cap_rows - len(capped)),
            "label_histogram": dict(
                sorted(collections.Counter(str(row["label"]) for row in capped).items())
            ),
        },
    )


def build(
    arm: str,
    dirs: Mapping[str, Path],
    *,
    workers: int,
    modules: Sequence[str] = SOURCE_MODULES,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = [item for item in families(modules) if item.arm == arm]
    if not selected:
        raise ValueError(f"no families registered for arm {arm}")
    _FAMILIES.clear()
    _FAMILIES.update({item.family: item for item in selected})
    _DIRS.clear()
    _DIRS.update(dirs)
    names = [item.family for item in selected]
    if workers > 1 and len(names) > 1:
        with multiprocessing.get_context("fork").Pool(min(workers, len(names))) as pool:
            results = pool.map(_run_family, names)
    else:
        results = [_run_family(name) for name in names]
    rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    for name, capped, report in sorted(results, key=lambda item: item[0]):
        rows.extend(capped)
        reports[name] = report
    return rows, {
        "schema": SCHEMA,
        "arm": arm,
        "rules": RULES,
        "modules": module_hashes(modules),
        "families": reports,
        "slices": "AHO sha256(group_id)%10==0; SHO sha256('sho-v2:'+group_id)%50==0 among the rest",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", required=True, choices=spec.ARMS)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--v1-rows", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--modules", nargs="+", default=list(SOURCE_MODULES))
    args = parser.parse_args(argv)
    existing = [
        name
        for name in (f"{args.arm}.{part}.jsonl" for part in common.SLICES)
        if (args.out_dir / name).exists()
    ]
    if existing or (args.out_dir / f"{args.arm}.build.json").exists():
        parser.error(f"refusing to overwrite {args.arm} in {args.out_dir}")
    dirs: dict[str, Path] = spec.resolve(args.sources)
    if args.v1_rows is not None:
        dirs[V1_ROWS_KEY] = args.v1_rows
    rows, report = build(args.arm, dirs, workers=args.workers, modules=args.modules)
    if args.v1_rows is not None:
        report["v1_rows_manifest_sha256"] = file_sha256(args.v1_rows)
    manifest = common.write_arm(rows, args.out_dir, args.arm, report)
    print(
        json.dumps(
            {
                part: {k: manifest[part][k] for k in ("rows", "groups", "sha256")}
                for part in common.SLICES
            },
            indent=1,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
