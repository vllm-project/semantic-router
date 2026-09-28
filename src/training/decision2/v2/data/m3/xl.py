"""XL release recipes over data v1/v2, A7 and A0s-strict (M3b, prereg m3b §2).

    python3 -m v2.data.m3.xl --pools pools.json --out-dir OUT \\
        [--targets lux1=a.jsonl,b.jsonl --targets autojev27=c.jsonl,...]

``pools.json`` maps pool names to ``{"rows": [...], "tokens": [...], "kind": "human" |
"generated" | "anchor"}`` in dedup priority order (first pool wins an ``input_sha256``).
Rows of the excluded shortcut families are dropped everywhere. Each recipe takes all anchor
rows, then whole groups of each pool — groups holding a row with ``--prefer`` teacher targets
(and, for the short variant and the controls, a row of XL-full) first, each part in
``sha256("mx-xl:<variant>:<pool>:" + group)`` order — up to the
pool's target tokens, under a cap of 8% of the recipe budget per human source and per
generated (source, family); the short variant only takes groups whose rows are all within
1,024 native tokens. Controls at the recipe's own token total: ``cx-xl-a7v1`` (anchor + A7 + v1
arms) and ``cx-xl-v2v1`` (anchor + v2 + v1 arms). Outputs ``<recipe>.ids.jsonl`` and a
manifest with tokens by type, language, Score level count, pool and source, the English share,
and target coverage (``<recipe>.<teacher>.missing.jsonl`` lists rows without targets).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from v2.data.m2.common import read_jsonl, sha
from v2.data.m3.strict import EXCLUDED

SOURCE_CAP = 0.08
ENGLISH_CAP = 0.60
SHORT_MAX_NATIVE = 1024
FULL_BUDGET = 150_000_000
TARGETS = {
    "H1": 9.0,
    "A7m": 2.2,
    "A7h": 0.3,
    "A7i": 7.0,
    "V1:A1": 1.5,
    "V1:A5": 0.5,
    "A7g": 30.0,
    "A7o": 10.0,
    "A7p": 3.7,
    "G2": 8.0,
    "G4h": 1.6,
    "V1:A2": 2.3,
    "V1:A4v2h": 0.8,
    "H3": 11.0,
    "H5": 17.5,
    "E11": 4.5,
    "V1:A3": 2.0,
    "A7r": 1.1,
    "H6": 12.0,
    "G6": 6.0,
    "V1:A6g": 2.4,
    "V1:A6h": 1.4,
    "A7q": 10.0,
    "A7k": 0.6,
    "A7s": 2.2,
}
A7_POOLS = {"A7m", "A7h", "A7i", "A7g", "A7o", "A7p", "A7r", "A7q", "A7k", "A7s"}
V2_POOLS = {"H1", "H3", "H5", "H6", "E11", "G2", "G6", "G4h"}


def load(specs: dict[str, dict]) -> tuple[dict[str, dict[str, list[dict]]], dict]:
    seen: dict[str, str] = {}
    pools: dict[str, dict[str, list[dict]]] = {}
    dropped = collections.Counter()
    for name, spec in specs.items():
        tokens = {}
        for path in spec["tokens"]:
            for rec in read_jsonl(Path(path)):
                tokens[rec["id"]] = rec["native"]
        groups: dict[str, list[dict]] = collections.defaultdict(list)
        for path in spec["rows"]:
            for row in read_jsonl(Path(path)):
                if row["family"] in EXCLUDED:
                    dropped[f"{name}|excluded_family"] += 1
                    continue
                if row["input_sha256"] in seen:
                    dropped[f"{name}|duplicate_of:{seen[row['input_sha256']]}"] += 1
                    continue
                seen[row["input_sha256"]] = name
                cap = (
                    row["source"]
                    if spec["kind"] != "generated"
                    else f"{row['source']}|{row['family']}"
                )
                groups[row["group_id"]].append(
                    {
                        "id": row["id"],
                        "group": row["group_id"],
                        "pool": name,
                        "source": row["source"],
                        "cap": cap,
                        "task_type": row["task_type"],
                        "language": row["language"],
                        "levels": (
                            len(row["options"]) if row["task_type"] == "score" else 0
                        ),
                        "native": tokens[row["id"]],
                    }
                )
        pools[name] = groups
    return pools, dict(sorted(dropped.items()))


def take(groups, key, target, used, cap, short):
    chosen, total = [], 0
    for group in sorted(groups, key=key):
        members = groups[group]
        if short and any(m["native"] > SHORT_MAX_NATIVE for m in members):
            continue
        size = sum(m["native"] for m in members)
        if any(used[m["cap"]] + size > cap for m in members[:1]):
            continue
        if total + size > target:
            if total >= 0.98 * target:
                break
            continue
        chosen.extend(members)
        total += size
        used[members[0]["cap"]] += size
    return chosen


def summary(rows: list[dict]) -> dict[str, Any]:
    def by(field):
        c = collections.Counter()
        for r in rows:
            c[str(r[field])] += r["native"]
        return dict(sorted(c.items()))

    total = sum(r["native"] for r in rows) or 1
    types = by("task_type")
    return {
        "rows": len(rows),
        "native_tokens": total,
        "type_share": {k: round(v / total, 4) for k, v in types.items()},
        "english_share": round(by("language").get("en", 0) / total, 4),
        "by_task_type": types,
        "rows_by_task_type": dict(collections.Counter(r["task_type"] for r in rows)),
        "by_language": by("language"),
        "by_score_levels": {k: v for k, v in by("levels").items() if k != "0"},
        "by_pool": by("pool"),
        "by_source": by("source"),
    }


def write(out: Path, name: str, rows: list[dict]) -> str:
    data = "".join(
        json.dumps(
            {
                k: r[k]
                for k in ("id", "pool", "source", "task_type", "language", "native")
            },
            sort_keys=True,
        )
        + "\n"
        for r in sorted(rows, key=lambda r: r["id"])
    ).encode()
    fd = os.open(out / f"{name}.ids.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
    return hashlib.sha256(data).hexdigest()


def build(
    pools, variant: str, include: set[str] | None, budget: int, prefer: set[str]
) -> list[dict]:
    """Groups holding a preferred id (rows that already have teacher targets, or rows of the
    full recipe) come first in each pool, each part in its own hash order."""
    short = variant == "short"
    anchor = [
        m
        for g in pools["A0s-strict"].values()
        for m in g
        if not short or m["native"] <= SHORT_MAX_NATIVE
    ]
    used = collections.Counter()
    for m in anchor:
        used[m["cap"]] += m["native"]
    rows = list(anchor)
    names = [p for p in TARGETS if include is None or p in include]
    scale = 1.0
    if include is not None:
        wanted = sum(TARGETS[p] for p in names) * 1e6
        scale = (budget - sum(m["native"] for m in anchor)) / wanted
    for pool in names:
        rows += take(
            pools[pool],
            lambda g, v=variant, p=pool, groups=pools[pool]: (
                not any(m["id"] in prefer for m in groups[g]),
                sha(f"mx-xl:{v}:{p}:{g}"),
            ),
            TARGETS[pool] * 1e6 * scale,
            used,
            SOURCE_CAP * budget,
            short,
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pools", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--targets", action="append", default=[])
    parser.add_argument(
        "--prefer", default="lux1", help="teacher whose covered rows go first"
    )
    args = parser.parse_args(argv)
    specs = json.loads(args.pools.read_text())
    pools, dropped = load(specs)
    teachers = {}
    for spec in args.targets:
        name, _, paths = spec.partition("=")
        teachers[name] = {
            json.loads(line)["id"]
            for p in paths.split(",")
            for line in open(p, encoding="utf-8")
        }
    args.out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest: dict[str, Any] = {
        "schema": "decision2-mx-xl/1",
        "pools_sha256": hashlib.sha256(args.pools.read_bytes()).hexdigest(),
        "targets_million_tokens": TARGETS,
        "source_cap": SOURCE_CAP,
        "english_cap": ENGLISH_CAP,
        "dropped": dropped,
        "recipes": {},
    }
    v1 = {p for p in TARGETS if p.startswith("V1:")}
    full_ids: set[str] = set()
    for variant in ("full", "short"):
        name = f"mx-xl-{variant}"
        prefer = set(teachers.get(args.prefer, set()))
        if variant == "short":
            prefer |= full_ids
        rows = build(pools, variant, None, FULL_BUDGET, prefer)
        if variant == "full":
            full_ids = {r["id"] for r in rows}
            prefer |= full_ids
        total = sum(r["native"] for r in rows)
        recipes = {name: rows}
        for control, include in (
            ("cx-xl-a7v1", A7_POOLS | v1),
            ("cx-xl-v2v1", V2_POOLS | v1),
        ):
            recipes[f"{control}-{variant}"] = build(
                pools, variant, include, total, prefer
            )
        for rname, rrows in recipes.items():
            s = summary(rrows)
            s["sha256"] = write(args.out_dir, rname, rrows)
            s["english_cap_ok"] = s["english_share"] <= ENGLISH_CAP
            s["coverage"] = {}
            for teacher, ids in teachers.items():
                missing = sorted(r["id"] for r in rrows if r["id"] not in ids)
                by_pool = collections.Counter(
                    r["pool"] for r in rrows if r["id"] not in ids
                )
                s["coverage"][teacher] = {
                    "rows_with_targets": len(rrows) - len(missing),
                    "rows_without": len(missing),
                    "without_by_pool": dict(sorted(by_pool.items())),
                }
                path = args.out_dir / f"{rname}.{teacher}.missing.jsonl"
                with path.open("x", encoding="utf-8") as stream:
                    stream.writelines(json.dumps({"id": i}) + "\n" for i in missing)
            manifest["recipes"][rname] = s
    with (args.out_dir / "mx-xl.manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=1, sort_keys=True)
    for rname, s in manifest["recipes"].items():
        print(rname, s["rows"], s["native_tokens"], s["type_share"], s["english_share"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
