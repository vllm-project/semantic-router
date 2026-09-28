"""XL release recipe revision r2 (M3b amendment 2 §4, amendment 3 §2–§3).

    python3 -m v2.data.m3.xl_r2 rows --pools xl-pools.json --r1-dir R1 \\
        --r1-manifest-sha256 HEX --out-dir RESCREEN [--workers 16]
    python3 -m v2.data.m3.xl_r2 rescreen --rescreen-dir RESCREEN --r1-dir R1 \\
        --r1-manifest-sha256 HEX --inventory-sha256 HEX --private P.json --public Q.json
    python3 -m v2.data.m3.xl_r2 build --pools xl-pools.json --gap H7=ROWS,TOKENS \\
        --gap H8=ROWS,TOKENS --r1-dir R1 --r1-manifest-sha256 HEX --rescreen P.json \\
        [--disclose-role PATTERN ...] --out-dir OUT [--targets lux1=a,b --extra-targets lux1=c,d --targets autojev27=e \\
        --pending lux1=NAME=ROWS ...]
    python3 -m v2.data.m3.xl_r2 check --pools xl-pools.json --out-dir OUT --r1-dir R1 \\
        --r1-manifest-sha256 HEX --rescreen P.json --rows-dir RESCREEN/rows \\
        --gap H7=ROWS,TOKENS --gap H8=ROWS,TOKENS --report OUT/check.json

``rows`` resolves every row of the union of the six r1 recipes by id from its pool's rows
files and writes ``rows/<pool>.jsonl`` with the pool file's line bytes unchanged, so the
rescreen reads each pool's own rendering; ``index.jsonl`` maps ids to pools and groups
(no text). There is one candidate file per pool because the overlap scan counts
boilerplate over the distinct candidate groups of one scan, as in every arm's own audit.

``rescreen`` collects the per-pool ``v2.data.overlap`` receipts against PI-v4's
quarantining roles and the receipt of the same scan over the whole union. A group id with
a hit in either pass is flagged in every pool (v2 group ids name the upstream item, not
the arm).

``build`` excludes the flagged group ids with a hit on a role that matches none of the
``--disclose-role`` patterns (amendment 4: ``v1_aho_*``, ``a7_aho_*`` and SELECT/CAL are
disclosed, every evaluation role excludes; no pattern = every flagged group, r2-strict).
``mx-xl-<variant>-r2`` = (r1 ``mx-xl-<variant>`` minus excluded groups) + H7 / H8
whole groups in ``sha256("mx-xl-r2:<variant>:<pool>:" + group)`` order up to the gap
targets. A group is skipped when it would put a human source above 8%, or English above
60%, of the r2 total, which counts the base plus every candidate group; the short variant
only takes groups whose rows are all <= 1,024 native tokens. Controls:
``cx-xl-r2-nogap-<variant>`` = r2 minus H7 / H8; ``cx-xl-r2-{a7v1,v2v1}-<variant>`` = the
r1 controls minus flagged groups. ``--targets`` are the files r1 counted (its coverage is
reproduced first), ``--extra-targets`` later published targets, ``--pending`` prompt row
files of waves not yet published (projected coverage only). ``check`` re-derives the
set rules, input-hash uniqueness, caps, English and long-evidence shares from the
written files.
"""

from __future__ import annotations

import argparse
import collections
import fnmatch
import hashlib
import json
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Any

from training.model.data import file_sha256
from v2.data import overlap
from v2.data.m2.common import read_jsonl, sha
from v2.data.m3 import xl

R1_RECIPES = (
    "mx-xl-full",
    "mx-xl-short",
    "cx-xl-a7v1-full",
    "cx-xl-a7v1-short",
    "cx-xl-v2v1-full",
    "cx-xl-v2v1-short",
)
VARIANTS = ("full", "short")
CONTROLS = ("a7v1", "v2v1")
GAP_TARGETS = {"H7": 18.8, "H8": 12.6}
LONG_MIN_NATIVE = 2000
TEACHERS = ("lux1", "autojev27")


def r2_names(variant: str) -> dict[str, str]:
    """r2 recipe name -> its r1 counterpart."""
    names = {
        f"mx-xl-{variant}-r2": f"mx-xl-{variant}",
        f"cx-xl-r2-nogap-{variant}": f"mx-xl-{variant}",
    }
    for control in CONTROLS:
        names[f"cx-xl-r2-{control}-{variant}"] = f"cx-xl-{control}-{variant}"
    return names


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


def _scan_receipt(
    path: Path, inventory_sha256: str, candidates_sha256: list[str]
) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt["protected"]["inventory_sha256"] != inventory_sha256:
        raise ValueError(f"{path.name}: scanned against another protected inventory")
    if sorted(f["sha256"] for f in receipt["candidate_files"]) != sorted(
        candidates_sha256
    ):
        raise ValueError(f"{path.name}: candidate files differ from the rows receipt")
    return receipt


def rescreen(
    rescreen_dir: Path, recipes: dict[str, list[dict]], inventory_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """(private, public) rescreen receipts from the per-pool and single-union scans."""
    rows_raw = (rescreen_dir / "rows.receipt.json").read_bytes()
    rows_receipt = json.loads(rows_raw)
    index_path = rescreen_dir / "index.jsonl"
    if file_sha256(index_path) != rows_receipt["index_sha256"]:
        raise ValueError("index.jsonl differs from the rows receipt")
    index = list(read_jsonl(index_path))
    meta = {r["id"]: r for rows in recipes.values() for r in rows}
    pool_rows: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    pools_of_group: dict[str, set[str]] = collections.defaultdict(set)
    for row in index:
        pool_rows[(row["pool"], row["group"])].append(row["id"])
        pools_of_group[row["group"]].add(row["pool"])
    pools = rows_receipt["pools"]
    scan_dir, union_dir = rescreen_dir / "scan", rescreen_dir / "scan-union"
    hits: dict[tuple[str, str], dict[str, set[str]]] = {}
    passes: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    scans: dict[str, Any] = {"per_pool": {}, "union": {}}

    def note(key: tuple[str, str], record: dict[str, Any], scan: str) -> None:
        entry = hits.setdefault(key, {"roles": set(), "methods": set()})
        entry["roles"].update(record["roles"])
        entry["methods"].update(record["methods"])
        passes[key].add(scan)

    for pool, spec in pools.items():
        name = pool_file(pool)
        receipt = _scan_receipt(
            scan_dir / f"{name}.private.json",
            inventory_sha256,
            [spec["candidates_sha256"]],
        )
        flagged = overlap.quarantine_groups(receipt)
        for group in flagged:
            note((pool, group), receipt["groups"][group], "per_pool")
        public = json.loads((scan_dir / f"{name}.public.json").read_text())
        scans["per_pool"][pool] = {
            "public_receipt_sha256": file_sha256(scan_dir / f"{name}.public.json"),
            "candidates_sha256": spec["candidates_sha256"],
            "candidate_rows": public["candidates"]["rows"],
            "candidate_groups": public["candidates"]["groups"],
            "flagged_groups": len(flagged),
            "boilerplate": public["flagged"]["boilerplate"],
        }
    union_receipt = _scan_receipt(
        union_dir / "union.private.json",
        inventory_sha256,
        [pools[p]["candidates_sha256"] for p in pools],
    )
    union_flagged = overlap.quarantine_groups(union_receipt)
    for group in union_flagged:
        for pool in pools_of_group[group]:
            note((pool, group), union_receipt["groups"][group], "union")
    union_public = json.loads((union_dir / "union.public.json").read_text())
    scans["union"] = {
        "public_receipt_sha256": file_sha256(union_dir / "union.public.json"),
        "candidate_rows": union_public["candidates"]["rows"],
        "candidate_groups": union_public["candidates"]["groups"],
        "flagged_groups": len(union_flagged),
        "boilerplate": union_public["flagged"]["boilerplate"],
    }
    flagged_ids = {group for _, group in hits}
    removed = {
        key: ids for key, ids in pool_rows.items() if key[1] in flagged_ids
    }  # every pool's rows of a flagged group id

    def tally(keys: Any) -> dict[str, int]:
        ids = [i for key in keys for i in removed.get(key, pool_rows.get(key, []))]
        return {
            "groups": len(set(keys)),
            "rows": len(ids),
            "native_tokens": sum(meta[i]["native"] for i in ids),
        }

    by_pool = {}
    for pool in pools:
        keys = [k for k in removed if k[0] == pool]
        entry = tally(keys)
        for scan in ("per_pool", "union"):
            entry[f"hit_groups_{scan}_scan"] = sum(
                1 for k in keys if scan in passes.get(k, ())
            )
        entry["groups_by_group_id_elsewhere"] = sum(1 for k in keys if k not in hits)
        entry["share_of_pool_rows"] = round(entry["rows"] / pools[pool]["rows"], 4)
        by_pool[pool] = entry
    by_source: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    for ids in removed.values():
        for i in ids:
            by_source[meta[i]["source"]]["rows"] += 1
            by_source[meta[i]["source"]]["native_tokens"] += meta[i]["native"]
    by_role: dict[str, list] = collections.defaultdict(list)
    by_method: dict[str, list] = collections.defaultdict(list)
    combos: collections.Counter[str] = collections.Counter()
    role_pool: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    for key, entry in hits.items():
        for role in entry["roles"]:
            by_role[role].append(key)
            role_pool[role][key[0]] += 1
        for method in entry["methods"]:
            by_method[method].append(key)
        combos["+".join(m for m in overlap.METHODS if m in entry["methods"])] += 1
    only = {
        scan: [k for k, found in passes.items() if found == {scan}]
        for scan in ("per_pool", "union")
    }
    by_recipe = {}
    removed_ids = {i for ids in removed.values() for i in ids}
    for name, rows in recipes.items():
        gone = [r for r in rows if r["id"] in removed_ids]
        pool_counts: dict[str, collections.Counter] = collections.defaultdict(
            collections.Counter
        )
        for r in gone:
            pool_counts[r["pool"]]["rows"] += 1
            pool_counts[r["pool"]]["native_tokens"] += r["native"]
        total = sum(r["native"] for r in rows) or 1
        by_recipe[name] = {
            "rows": len(gone),
            "native_tokens": sum(r["native"] for r in gone),
            "rows_share": round(len(gone) / (len(rows) or 1), 4),
            "tokens_share": round(sum(r["native"] for r in gone) / total, 4),
            "by_pool": {p: dict(c) for p, c in sorted(pool_counts.items())},
        }
    public = {
        "schema": "decision2-mx-xl-r2-rescreen/1",
        "inventory_sha256": inventory_sha256,
        "rows_receipt_sha256": hashlib.sha256(rows_raw).hexdigest(),
        "r1_ids_sha256": rows_receipt.get("r1_ids_sha256", {}),
        "union": rows_receipt["union"],
        "scans": scans,
        "flagged": {
            **tally(list(removed)),
            "group_ids": len(flagged_ids),
            "hit_pool_groups": len(hits),
            "only_per_pool_scan": tally(only["per_pool"]),
            "only_union_scan": tally(only["union"]),
            "by_pool": dict(sorted(by_pool.items())),
            "by_source": {s: dict(c) for s, c in sorted(by_source.items())},
            "by_role": {r: tally(k) for r, k in sorted(by_role.items())},
            "by_role_pool": {r: dict(sorted(c.items())) for r, c in role_pool.items()},
            "by_method": {m: tally(k) for m, k in sorted(by_method.items())},
            "by_method_combination": dict(sorted(combos.items())),
        },
        "by_recipe": by_recipe,
    }
    private = {
        "schema": "decision2-mx-xl-r2-rescreen-private/1",
        "inventory_sha256": inventory_sha256,
        "flagged_group_ids": sorted(flagged_ids),
        "hits": {
            pool: {
                group: {
                    "roles": sorted(entry["roles"]),
                    "methods": sorted(entry["methods"]),
                    "passes": sorted(passes[(pool, group)]),
                }
                for (p, group), entry in sorted(hits.items())
                if p == pool
            }
            for pool in sorted({p for p, _ in hits})
        },
        "by_group_id_elsewhere": sorted(
            [pool, group] for pool, group in removed if (pool, group) not in hits
        ),
    }
    return private, public


def members_by_id(pools: dict[str, dict[str, list[dict]]]) -> dict[str, dict]:
    return {m["id"]: m for groups in pools.values() for g in groups.values() for m in g}


def r1_members(
    recipes: dict[str, list[dict]], members: dict[str, dict]
) -> dict[str, list[dict]]:
    """The r1 id lists as loaded members, each line checked against the load."""
    out = {}
    for name, rows in recipes.items():
        chosen = []
        for row in rows:
            member = members.get(row["id"])
            if member is None or any(
                member[k] != row[k]
                for k in ("pool", "source", "task_type", "language", "native")
            ):
                raise ValueError(f"{name}: {row['id']} differs from the pool load")
            chosen.append(member)
        out[name] = chosen
    return out


def gap_candidates(
    groups: dict[str, list[dict]], variant: str, pool: str, target: float
) -> list[str]:
    """Whole groups in r2 hash order up to the target (the xl.take rule, no caps)."""
    picked, total = [], 0
    for group in sorted(groups, key=lambda g: sha(f"mx-xl-r2:{variant}:{pool}:{g}")):
        members = groups[group]
        if variant == "short" and any(
            m["native"] > xl.SHORT_MAX_NATIVE for m in members
        ):
            continue
        size = sum(m["native"] for m in members)
        if total + size > target:
            if total >= 0.98 * target:
                break
            continue
        picked.append(group)
        total += size
    return picked


def select_gap(
    base: list[dict], gap: dict[str, dict[str, list[dict]]], variant: str
) -> tuple[list[dict], dict[str, Any]]:
    """H7 / H8 rows added to ``base`` under the r2 source and English caps."""
    candidates = {
        pool: gap_candidates(groups, variant, pool, GAP_TARGETS[pool] * 1e6)
        for pool, groups in gap.items()
    }
    sizes = {
        pool: sum(m["native"] for g in picked for m in gap[pool][g])
        for pool, picked in candidates.items()
    }
    total = sum(m["native"] for m in base) + sum(sizes.values())
    source_cap, english_cap = xl.SOURCE_CAP * total, xl.ENGLISH_CAP * total
    used: collections.Counter[str] = collections.Counter()
    for m in base:
        used[m["cap"]] += m["native"]
    english = sum(m["native"] for m in base if m["language"] == "en")
    rows, report = [], {}
    for pool, picked in candidates.items():
        taken, skipped = 0, collections.Counter()
        for group in picked:
            members = gap[pool][group]
            size: collections.Counter[str] = collections.Counter()
            for m in members:
                size[m["cap"]] += m["native"]
            en = sum(m["native"] for m in members if m["language"] == "en")
            over = sorted(k for k, n in size.items() if used[k] + n > source_cap)
            if over:
                skipped[f"source_cap:{over[0]}"] += 1
                continue
            if english + en > english_cap:
                skipped["english_cap"] += 1
                continue
            rows.extend(members)
            used.update(size)
            english += en
            taken += 1
        report[pool] = {
            "target_native_tokens": int(GAP_TARGETS[pool] * 1e6),
            "groups_available": len(gap[pool]),
            "candidate_groups": len(picked),
            "candidate_native_tokens": sizes[pool],
            "groups_taken": taken,
            "groups_skipped": dict(sorted(skipped.items())),
        }
    return rows, {
        "total_with_candidates": total,
        "source_cap_tokens": int(source_cap),
        "english_cap_tokens": int(english_cap),
        "english_tokens": english,
        "max_human_source_tokens_with_gap": max(
            (used[k] for k in {m["cap"] for m in rows}), default=0
        ),
        "pools": report,
    }


def build_recipes(
    r1: dict[str, list[dict]],
    flagged: set[str],
    gap: dict[str, dict[str, list[dict]]],
) -> tuple[dict[str, list[dict]], dict[str, Any]]:
    clash = sorted(g for groups in gap.values() for g in groups if g in flagged)
    if clash:
        raise ValueError(f"{len(clash)} gap groups carry flagged group ids")
    recipes, selection = {}, {}
    for variant in VARIANTS:
        base = [m for m in r1[f"mx-xl-{variant}"] if m["group"] not in flagged]
        added, selection[variant] = select_gap(base, gap, variant)
        recipes[f"mx-xl-{variant}-r2"] = base + added
        recipes[f"cx-xl-r2-nogap-{variant}"] = list(base)
        for control in CONTROLS:
            recipes[f"cx-xl-r2-{control}-{variant}"] = [
                m for m in r1[f"cx-xl-{control}-{variant}"] if m["group"] not in flagged
            ]
    return recipes, selection


def long_evidence(rows: list[dict], kinds: dict[str, str]) -> dict[str, Any]:
    tokens: collections.Counter[str] = collections.Counter()
    long: collections.Counter[str] = collections.Counter()
    for r in rows:
        kind = kinds[r["pool"]]
        tokens[kind] += r["native"]
        if r["native"] >= LONG_MIN_NATIVE:
            long[kind] += r["native"]
    total = sum(tokens.values()) or 1
    kept = tokens["anchor"] + tokens["human"]
    kept_long = long["anchor"] + long["human"]
    return {
        "long_tokens": sum(long.values()),
        "share": round(sum(long.values()) / total, 4),
        "share_without_generated": round(kept_long / (kept or 1), 4),
        "share_human_pools_only": round(long["human"] / (tokens["human"] or 1), 4),
        "non_generated_long_of_total": round(kept_long / total, 4),
        "tokens_by_kind": dict(sorted(tokens.items())),
        "long_tokens_by_kind": dict(sorted(long.items())),
    }


def describe(rows: list[dict], kinds: dict[str, str]) -> dict[str, Any]:
    s = xl.summary(rows)
    s["languages"] = len(s["by_language"])
    s["english_cap_ok"] = s["english_share"] <= xl.ENGLISH_CAP
    s["long_evidence"] = long_evidence(rows, kinds)
    return s


def shift(new: dict[str, Any], old: dict[str, Any], name: str) -> dict[str, Any]:
    points = lambda a, b: round(100 * (a - b), 2)  # noqa: E731
    types = sorted(set(new["type_share"]) | set(old["type_share"]))
    return {
        "reference": name,
        "rows": new["rows"] - old["rows"],
        "native_tokens": new["native_tokens"] - old["native_tokens"],
        "type_share_points": {
            t: points(new["type_share"].get(t, 0), old["type_share"].get(t, 0))
            for t in types
        },
        "english_share_points": points(new["english_share"], old["english_share"]),
        "long_share_points": points(
            new["long_evidence"]["share"], old["long_evidence"]["share"]
        ),
    }


def exclusion(
    receipt: dict[str, Any], disclose: list[str]
) -> tuple[set[str], set[str], dict[str, Any]]:
    """Split the rescreen's flagged group ids by role (amendment 4).

    A group id is excluded when any of its hits, in any pool, has a role matching none
    of the ``disclose`` patterns; the other flagged group ids stay and are disclosed.
    """
    hits = receipt["hits"]

    def disclosed_role(role: str) -> bool:
        return any(fnmatch.fnmatchcase(role, p) for p in disclose)

    keys = [(p, g) for p, groups in hits.items() for g in groups]
    if {g for _, g in keys} != set(receipt["flagged_group_ids"]):
        raise ValueError("rescreen hits and flagged_group_ids disagree")
    roles = sorted({r for p, g in keys for r in hits[p][g]["roles"]})
    excluding = [r for r in roles if not disclosed_role(r)]
    out = {g for p, g in keys if set(hits[p][g]["roles"]) & set(excluding)}
    kept = set(receipt["flagged_group_ids"]) - out
    shown = [(p, g) for p, g in keys if g in kept]
    by_role: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    for p, g in shown:
        for role in hits[p][g]["roles"]:
            by_role[role][p] += 1
    rule = {
        "rule": "exclude group ids with a hit on a role outside disclosed_roles",
        "disclosed_roles": list(disclose),
        "hit_roles_excluding": excluding,
        "hit_roles_disclosed": [r for r in roles if disclosed_role(r)],
        "excluded_group_ids": len(out),
        "disclosed_group_ids": len(kept),
        "disclosed_hit_pool_groups": len(shown),
        "disclosed_by_role_pool": {
            r: dict(sorted(c.items())) for r, c in sorted(by_role.items())
        },
        "disclosed_by_method": {
            m: sum(1 for p, g in shown if m in hits[p][g]["methods"])
            for m in overlap.METHODS
        },
        "disclosed_with_E_or_L": sum(
            1 for p, g in shown if {"E", "L"} & set(hits[p][g]["methods"])
        ),
    }
    return out, kept, rule


def excluded(old: list[dict], flagged: set[str]) -> dict[str, Any]:
    gone = [m for m in old if m["group"] in flagged]
    by_pool: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    for m in gone:
        by_pool[m["pool"]]["rows"] += 1
        by_pool[m["pool"]]["native_tokens"] += m["native"]
    for pool in by_pool:
        by_pool[pool]["groups"] = len({m["group"] for m in gone if m["pool"] == pool})
    return {
        "rows": len(gone),
        "groups": len({(m["pool"], m["group"]) for m in gone}),
        "native_tokens": sum(m["native"] for m in gone),
        "by_pool": {p: dict(c) for p, c in sorted(by_pool.items())},
    }


def coverage(
    rows: list[dict], ids: set[str], pending: dict[str, set[str]], gap: set[str]
) -> tuple[dict[str, Any], list[str]]:
    missing = sorted(r["id"] for r in rows if r["id"] not in ids)
    lost = set(missing)
    out: dict[str, Any] = {
        "rows_with_targets": len(rows) - len(missing),
        "rows_without": len(missing),
        "without_gap_rows": len(lost & gap),
        "without_by_pool": dict(
            sorted(
                collections.Counter(r["pool"] for r in rows if r["id"] in lost).items()
            )
        ),
    }
    if pending:
        left = lost - set().union(*pending.values())
        out["pending"] = {name: len(lost & wave) for name, wave in pending.items()}
        out["projected_rows_without"] = len(left)
        out["projected_without_by_pool"] = dict(
            sorted(
                collections.Counter(r["pool"] for r in rows if r["id"] in left).items()
            )
        )
    return out, missing


def _files(spec: str) -> tuple[str, list[Path]]:
    name, _, paths = spec.partition("=")
    return name, [Path(p) for p in paths.split(",") if p]


def _ids(paths: list[Path]) -> tuple[set[str], list[dict[str, Any]]]:
    ids: set[str] = set()
    files = []
    for path in paths:
        found = {row["id"] for row in read_jsonl(path)}
        ids |= found
        files.append(
            {"file": path.name, "sha256": file_sha256(path), "ids": len(found)}
        )
    return ids, files


def _gap_specs(values: list[str]) -> dict[str, dict]:
    specs = {}
    for value in values:
        pool, _, paths = value.partition("=")
        rows, tokens = paths.split(",")
        specs[pool] = {"rows": [rows], "tokens": [tokens], "kind": "human"}
    if set(specs) != set(GAP_TARGETS):
        raise ValueError(f"--gap must name exactly {sorted(GAP_TARGETS)}")
    return specs


def build(args: argparse.Namespace) -> dict[str, Any]:
    r1_manifest, r1_ids = r1_recipes(args.r1_dir, args.r1_manifest_sha256)
    specs = json.loads(args.pools.read_text())
    gap_specs = _gap_specs(args.gap)
    if set(gap_specs) & set(specs):
        raise ValueError("gap pools already in the pools spec")
    kinds = {name: spec["kind"] for name, spec in {**specs, **gap_specs}.items()}
    pools, dropped = xl.load({**specs, **gap_specs})
    r1_dropped = {k: v for k, v in dropped.items() if k.split("|")[0] not in gap_specs}
    if r1_dropped != r1_manifest["dropped"]:
        raise ValueError("the pool load drops other rows than r1 did")
    members = members_by_id(pools)
    r1 = r1_members(r1_ids, members)
    rescreen_raw = args.rescreen.read_bytes()
    receipt = json.loads(rescreen_raw)
    flagged, disclosed, rule = exclusion(receipt, args.disclose_role)
    gap = {pool: pools[pool] for pool in GAP_TARGETS}
    gap_ids = {m["id"] for groups in gap.values() for g in groups.values() for m in g}
    recipes, selection = build_recipes(r1, flagged, gap)
    teachers: dict[str, set[str]] = {}
    files: dict[str, Any] = {}
    for spec in args.targets:
        name, paths = _files(spec)
        teachers[name], files[name] = _ids(paths)
    r1_coverage = {}
    for name, rows in r1.items():
        for teacher, ids in teachers.items():
            with_targets = sum(1 for m in rows if m["id"] in ids)
            expected = r1_manifest["recipes"][name]["coverage"].get(teacher)
            if expected is not None and with_targets != expected["rows_with_targets"]:
                raise ValueError(f"{name}: {teacher} coverage differs from r1")
            r1_coverage.setdefault(name, {})[teacher] = with_targets
    extra_files: dict[str, Any] = {}
    for spec in args.extra_targets:
        name, paths = _files(spec)
        ids, extra_files[name] = _ids(paths)
        teachers[name] = teachers.get(name, set()) | ids
    pending: dict[str, dict[str, set[str]]] = collections.defaultdict(dict)
    pending_files: dict[str, Any] = collections.defaultdict(dict)
    for spec in args.pending:
        teacher, _, rest = spec.partition("=")
        wave, _, path = rest.partition("=")
        ids, info = _ids([Path(path)])
        pending[teacher][wave] = ids
        pending_files[teacher][wave] = info[0]
    args.out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    r1_summary = {name: describe(rows, kinds) for name, rows in r1.items()}
    r1_union = list({m["id"]: m for rows in r1.values() for m in rows}.values())
    for name, s in r1_summary.items():
        old = r1_manifest["recipes"][name]
        if (s["rows"], s["native_tokens"], s["by_pool"]) != (
            old["rows"],
            old["native_tokens"],
            old["by_pool"],
        ):
            raise ValueError(f"{name}: r1 summary differs from its manifest")
    manifest: dict[str, Any] = {
        "schema": "decision2-mx-xl/2",
        "revision_of": {
            "schema": r1_manifest["schema"],
            "manifest_sha256": args.r1_manifest_sha256,
            "hf_revision": args.r1_revision,
            "ids_sha256": {n: r1_manifest["recipes"][n]["sha256"] for n in R1_RECIPES},
            "long_evidence": {n: s["long_evidence"] for n, s in r1_summary.items()},
            "coverage_reproduced": r1_coverage,
        },
        "code": {"commit": args.code_commit, "tree": args.code_tree},
        "pools_sha256": file_sha256(args.pools),
        "gap_arms": {
            pool: {
                "rows_sha256": file_sha256(spec["rows"][0]),
                "tokens_sha256": file_sha256(spec["tokens"][0]),
                "hf_revision": args.gap_revision,
                "kind": "human",
                "rows": sum(len(g) for g in gap[pool].values()),
                "groups": len(gap[pool]),
                "native_tokens": sum(
                    m["native"] for g in gap[pool].values() for m in g
                ),
            }
            for pool, spec in gap_specs.items()
        },
        "targets_million_tokens": GAP_TARGETS,
        "source_cap": xl.SOURCE_CAP,
        "english_cap": xl.ENGLISH_CAP,
        "short_max_native": xl.SHORT_MAX_NATIVE,
        "long_min_native": LONG_MIN_NATIVE,
        "dropped": dropped,
        "rescreen": {
            "private_receipt_sha256": hashlib.sha256(rescreen_raw).hexdigest(),
            "public_receipt_sha256": receipt.get("public_receipt_sha256"),
            "inventory_sha256": receipt["inventory_sha256"],
            "flagged_group_ids": len(receipt["flagged_group_ids"]),
            **rule,
            "r1_union_excluded": excluded(r1_union, flagged),
            "r1_union_disclosed": excluded(r1_union, disclosed),
        },
        "selection": selection,
        "teacher_files": {
            "counted_by_r1": files,
            "published_later": extra_files,
            "pending": pending_files,
        },
        "recipes": {},
    }
    for variant in VARIANTS:
        for name, old_name in r2_names(variant).items():
            rows = recipes[name]
            s = describe(rows, kinds)
            s["sha256"] = xl.write(args.out_dir, name, rows)
            s["r1_counterpart"] = old_name
            s["flagged_excluded"] = excluded(r1[old_name], flagged)
            s["flagged_disclosed"] = excluded(rows, disclosed)
            s["gap"] = {
                pool: {
                    "rows": sum(1 for m in rows if m["pool"] == pool),
                    "groups": len({m["group"] for m in rows if m["pool"] == pool}),
                    "native_tokens": sum(
                        m["native"] for m in rows if m["pool"] == pool
                    ),
                }
                for pool in GAP_TARGETS
            }
            s["shift_vs_r1"] = shift(s, r1_summary[old_name], old_name)
            s["coverage"] = {}
            for teacher in TEACHERS:
                if teacher not in teachers:
                    continue
                s["coverage"][teacher], missing = coverage(
                    rows, teachers[teacher], pending.get(teacher, {}), gap_ids
                )
                path = args.out_dir / f"{name}.{teacher}.missing.jsonl"
                with path.open("x", encoding="utf-8") as stream:
                    stream.writelines(json.dumps({"id": i}) + "\n" for i in missing)
            manifest["recipes"][name] = s
    data = json.dumps(manifest, indent=1, sort_keys=True).encode()
    _write_new(args.out_dir / "mx-xl-r2.manifest.json", data)
    return manifest


def check(args: argparse.Namespace) -> dict[str, Any]:
    """Re-derive every r2 rule from the written files; returns the check report."""
    manifest_path = args.out_dir / "mx-xl-r2.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    _, r1_ids = r1_recipes(args.r1_dir, args.r1_manifest_sha256)
    receipt = json.loads(args.rescreen.read_text())
    flagged, disclosed, rule = exclusion(
        receipt, manifest["rescreen"]["disclosed_roles"]
    )
    gap_specs = _gap_specs(args.gap)
    kinds = {
        name: spec["kind"]
        for name, spec in {**json.loads(args.pools.read_text()), **gap_specs}.items()
    }
    info: dict[str, dict[str, str]] = {}
    fields = ("group_id", "input_sha256", "family")
    for path in sorted(args.rows_dir.glob("*.jsonl")):
        for row in read_jsonl(path):
            info[row["id"]] = {k: row[k] for k in fields}
    seen = {v["input_sha256"] for v in info.values()}
    gap_rows: dict[str, dict] = {}
    gap_tokens: dict[str, int] = {}
    gap_dropped: collections.Counter[str] = collections.Counter()
    for pool in GAP_TARGETS:
        spec = gap_specs[pool]
        for row in read_jsonl(Path(spec["rows"][0])):
            info[row["id"]] = {k: row[k] for k in fields}
            if row["family"] in xl.EXCLUDED or row["input_sha256"] in seen:
                gap_dropped[f"{pool}|{row['family']}"] += 1
                continue
            seen.add(row["input_sha256"])
            gap_rows[row["id"]] = dict(row, pool=pool)
        for row in read_jsonl(Path(spec["tokens"][0])):
            gap_tokens[row["id"]] = row["native"]

    def cap_key(row: dict) -> str:
        if kinds[row["pool"]] != "generated":
            return row["source"]
        return f"{row['source']}|{info[row['id']]['family']}"

    gap_group_rows: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for i, row in gap_rows.items():
        gap_group_rows[(row["pool"], row["group_id"])].add(i)
    results: dict[str, Any] = {}

    def ok(name: str, passed: bool, **detail: Any) -> None:
        results[name] = {"pass": bool(passed), **detail}

    ok(
        "exclusion_rule_matches_manifest",
        all(manifest["rescreen"][k] == v for k, v in rule.items()),
        excluded_group_ids=len(flagged),
        disclosed_group_ids=len(disclosed),
    )
    load_dropped = sum(
        n for k, n in manifest["dropped"].items() if k.split("|")[0] in GAP_TARGETS
    )
    ok(
        "gap_rows_dropped_as_in_the_build",
        sum(gap_dropped.values()) == load_dropped,
        by_pool_family=dict(sorted(gap_dropped.items())),
    )
    recipes = {}
    for name, s in manifest["recipes"].items():
        path = args.out_dir / f"{name}.ids.jsonl"
        ok(f"{name}:sha256", file_sha256(path) == s["sha256"])
        recipes[name] = list(read_jsonl(path))
    for variant in VARIANTS:
        base = {
            r["id"]
            for r in r1_ids[f"mx-xl-{variant}"]
            if info[r["id"]]["group_id"] not in flagged
        }
        main = f"mx-xl-{variant}-r2"
        r2 = [r["id"] for r in recipes[main]]
        extra = set(r2) - base
        ok(f"{main}:contains_r1_minus_flagged", base <= set(r2))
        ok(f"{main}:additions_are_gap_rows", extra <= set(gap_rows), rows=len(extra))
        ok(
            f"cx-xl-r2-nogap-{variant}:equals_r1_minus_flagged",
            {r["id"] for r in recipes[f"cx-xl-r2-nogap-{variant}"]} == base,
        )
        for control in CONTROLS:
            name = f"cx-xl-r2-{control}-{variant}"
            want = {
                r["id"]
                for r in r1_ids[f"cx-xl-{control}-{variant}"]
                if info[r["id"]]["group_id"] not in flagged
            }
            got = {r["id"] for r in recipes[name]}
            ok(f"{name}:equals_r1_control_minus_flagged", got == want)
        groups = {(gap_rows[i]["pool"], gap_rows[i]["group_id"]) for i in extra}
        ok(
            f"{main}:gap_whole_groups",
            all(gap_group_rows[g] <= extra for g in groups),
            groups=len(groups),
        )
        if variant == "short":
            ok(
                f"{main}:gap_rows_within_short_limit",
                all(gap_tokens[i] <= xl.SHORT_MAX_NATIVE for i in extra),
            )
        sel = manifest["selection"][variant]
        total = sel["total_with_candidates"]
        tokens: collections.Counter[str] = collections.Counter()
        for r in recipes[main]:
            tokens[cap_key(r)] += r["native"]
        gap_keys = {cap_key(r) for r in recipes[main] if r["id"] in extra}
        worst = max((tokens[k] for k in gap_keys), default=0)
        ok(
            f"{main}:human_source_cap",
            worst <= xl.SOURCE_CAP * total,
            max_gap_source_share_of_total=round(worst / total, 4),
        )
        english = sum(r["native"] for r in recipes[main] if r["language"] == "en")
        size = sum(r["native"] for r in recipes[main])
        ok(
            f"{main}:english_cap",
            english <= xl.ENGLISH_CAP * total and english <= xl.ENGLISH_CAP * size,
            english_share=round(english / size, 4),
        )
    for name, rows in recipes.items():
        ids = [r["id"] for r in rows]
        hashes = [info[i]["input_sha256"] for i in ids]
        ok(f"{name}:unique_ids", len(ids) == len(set(ids)))
        ok(f"{name}:unique_input_sha256", len(hashes) == len(set(hashes)))
        ok(
            f"{name}:no_flagged_group",
            not any(info[i]["group_id"] in flagged for i in ids),
        )
        ok(
            f"{name}:gap_tokens_match_token_files",
            all(
                r["native"] == gap_tokens[r["id"]] for r in rows if r["id"] in gap_rows
            ),
        )
        size = sum(r["native"] for r in rows) or 1
        long = sum(r["native"] for r in rows if r["native"] >= LONG_MIN_NATIVE)
        ok(
            f"{name}:long_share_matches_manifest",
            round(long / size, 4)
            == manifest["recipes"][name]["long_evidence"]["share"],
            long_share=round(long / size, 4),
        )
    return {
        "schema": "decision2-mx-xl-r2-check/1",
        "manifest_sha256": file_sha256(manifest_path),
        "pass": all(r["pass"] for r in results.values()),
        "checks": results,
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
    s = sub.add_parser("rescreen")
    s.add_argument("--rescreen-dir", type=Path, required=True)
    s.add_argument("--r1-dir", type=Path, required=True)
    s.add_argument("--r1-manifest-sha256", required=True)
    s.add_argument("--inventory-sha256", required=True)
    s.add_argument("--private", type=Path, required=True)
    s.add_argument("--public", type=Path, required=True)
    b = sub.add_parser("build")
    b.add_argument("--pools", type=Path, required=True)
    b.add_argument("--gap", action="append", required=True)
    b.add_argument("--r1-dir", type=Path, required=True)
    b.add_argument("--r1-manifest-sha256", required=True)
    b.add_argument("--r1-revision", default="")
    b.add_argument("--gap-revision", default="")
    b.add_argument("--rescreen", type=Path, required=True)
    b.add_argument("--disclose-role", action="append", default=[])
    b.add_argument("--out-dir", type=Path, required=True)
    b.add_argument("--targets", action="append", default=[])
    b.add_argument("--extra-targets", action="append", default=[])
    b.add_argument("--pending", action="append", default=[])
    b.add_argument("--code-commit", default="")
    b.add_argument("--code-tree", default="")
    c = sub.add_parser("check")
    c.add_argument("--pools", type=Path, required=True)
    c.add_argument("--out-dir", type=Path, required=True)
    c.add_argument("--r1-dir", type=Path, required=True)
    c.add_argument("--r1-manifest-sha256", required=True)
    c.add_argument("--rescreen", type=Path, required=True)
    c.add_argument("--rows-dir", type=Path, required=True)
    c.add_argument("--gap", action="append", required=True)
    c.add_argument("--report", type=Path, required=True)
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
    elif args.command == "rescreen":
        _, recipes = r1_recipes(args.r1_dir, args.r1_manifest_sha256)
        private, public = rescreen(args.rescreen_dir, recipes, args.inventory_sha256)
        data = _json(public)
        _write_new(args.public, data)
        private["public_receipt_sha256"] = hashlib.sha256(data).hexdigest()
        _write_new(args.private, _json(private))
        flagged = public["flagged"]
        print(
            json.dumps(
                {k: flagged[k] for k in ("group_ids", "rows", "native_tokens")},
                sort_keys=True,
            )
        )
    elif args.command == "build":
        manifest = build(args)
        for name, s in manifest["recipes"].items():
            print(
                name,
                s["rows"],
                s["native_tokens"],
                s["type_share"],
                s["english_share"],
                s["long_evidence"]["share"],
            )
    else:
        report = check(args)
        _write_new(args.report, _json(report))
        failed = sorted(k for k, v in report["checks"].items() if not v["pass"])
        print(json.dumps({"pass": report["pass"], "failed": failed}))
        return 0 if report["pass"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
