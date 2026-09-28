"""D10 mixture recipes and matched-token controls (prereg section 8, amendment 3).

    python3 -m v2.data.m2.mixtures --pools pools.json --out-dir OUT

``pools.json`` maps pool names to ``{"rows": [jsonl...], "tokens": [jsonl...]}``
(token files from ``v2.data.m2.row_tokens``). Every recipe = all of A0s plus
whole groups of each component pool taken in ``sha256("mx-v2:<variant>:" +
group_id)`` order until the component's token share of the v2 portion is
reached; a source may not exceed 8% of recipe tokens. The order does not depend
on the budget, so S ⊂ M ⊂ L. Controls at the same native-token budget:
``cx-a0s`` (A0s plus a whole-group resample of A0s) and ``cx-v1`` (A0s plus v1
arms). Outputs one JSONL of ``{id, pool, source, native}`` per recipe plus a
summary manifest with token totals by pool, task type, language and source.
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

BUDGETS = {"S": 8_000_000, "M": 20_000_000, "L": 40_000_000}
SOURCE_CAP = 0.08
ENGLISH_CAP = 0.60
SHARES = {
    "full": {
        "H6": 0.14,
        "G6": 0.10,
        "V1S": 0.06,
        "H3": 0.12,
        "H5": 0.28,
        "H1": 0.08,
        "E11": 0.07,
        "G2": 0.12,
        "G4h": 0.03,
    },
    "short": {
        "H6": 0.15,
        "G6": 0.11,
        "V1S": 0.06,
        "H5": 0.33,
        "H1": 0.10,
        "E11": 0.08,
        "G2": 0.13,
        "G4h": 0.04,
    },
}
SHORT_MAX_NATIVE = 1024
RECIPES = [("full", "S"), ("full", "M"), ("full", "L"), ("short", "S"), ("short", "M")]


def load_pool(spec: dict[str, list[str]]) -> dict[str, list[dict[str, Any]]]:
    tokens: dict[str, dict[str, int]] = {}
    for path in spec["tokens"]:
        for rec in read_jsonl(Path(path)):
            tokens[rec["id"]] = rec
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for path in spec["rows"]:
        for row in read_jsonl(Path(path)):
            t = tokens[row["id"]]
            groups[row["group_id"]].append(
                {
                    "id": row["id"],
                    "group": row["group_id"],
                    "source": row["source"],
                    "task_type": row["task_type"],
                    "language": row["language"],
                    "native": t["native"],
                    "kai": t["kai"],
                }
            )
    return groups


def take(groups, order_key, target, used_by_source, source_cap, max_native=None):
    chosen, total = [], 0
    for group in sorted(groups, key=order_key):
        members = groups[group]
        if max_native is not None and any(m["native"] > max_native for m in members):
            continue
        size = sum(m["native"] for m in members)
        src = members[0]["source"]
        if used_by_source[src] + size > source_cap:
            continue
        if total + size > target:
            if total >= 0.98 * target:
                break
            continue
        chosen.extend(members)
        total += size
        used_by_source[src] += size
    return chosen, total


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def by(field):
        c = collections.Counter()
        for r in rows:
            c[r[field]] += r["native"]
        return dict(sorted(c.items()))

    total = sum(r["native"] for r in rows)
    en = by("language").get("en", 0)
    return {
        "rows": len(rows),
        "native_tokens": total,
        "english_share": round(en / total, 4),
        "by_pool": by("pool"),
        "by_task_type": by("task_type"),
        "by_language": by("language"),
        "by_source": by("source"),
    }


def write(out: Path, name: str, rows: list[dict[str, Any]]) -> str:
    data = "".join(
        json.dumps(
            {
                "id": r["id"],
                "pool": r["pool"],
                "source": r["source"],
                "native": r["native"],
            },
            sort_keys=True,
        )
        + "\n"
        for r in sorted(rows, key=lambda r: r["id"])
    ).encode()
    fd = os.open(out / f"{name}.ids.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as s:
        s.write(data)
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pools", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    specs = json.loads(args.pools.read_text())
    pools = {name: load_pool(spec) for name, spec in specs.items()}
    args.out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest: dict[str, Any] = {
        "schema": "decision2-mx-v2/1",
        "pools_sha256": hashlib.sha256(args.pools.read_bytes()).hexdigest(),
        "shares": SHARES,
        "budgets": BUDGETS,
        "source_cap": SOURCE_CAP,
        "english_cap": ENGLISH_CAP,
        "recipes": {},
    }
    a0s = [dict(m, pool="A0s") for g in pools["A0s"].values() for m in g]
    a0s_tokens = sum(m["native"] for m in a0s)
    for variant, level in RECIPES:
        budget = BUDGETS[level]
        name = f"mx-v2-{variant}-{level}"
        base = [m for m in a0s if variant == "full" or m["native"] <= SHORT_MAX_NATIVE]
        v2_budget = budget - sum(m["native"] for m in base)
        used = collections.Counter()
        for m in base:
            used[m["source"]] += m["native"]
        rows = list(base)
        for pool, share in SHARES[variant].items():
            chosen, _ = take(
                pools[pool],
                lambda g, v=variant: sha(f"mx-v2:{v}:{g}"),
                share * v2_budget,
                used,
                SOURCE_CAP * budget,
                SHORT_MAX_NATIVE if variant == "short" else None,
            )
            rows.extend(dict(m, pool=pool) for m in chosen)
        s = summary(rows)
        s["sha256"] = write(args.out_dir, name, rows)
        s["english_cap_ok"] = s["english_share"] <= ENGLISH_CAP
        manifest["recipes"][name] = s
        total = s["native_tokens"]
        for control, pool_names in (("cx-a0s", ["A0s"]), ("cx-v1", ["V1ALL"])):
            cname = f"{control}-{variant}-{level}"
            used_c = collections.Counter()
            crows = list(base)
            pool = pools[pool_names[0]]
            if control == "cx-a0s":
                remaining, k = total - sum(m["native"] for m in base), 0
                while remaining > 0.02 * total:
                    k += 1
                    extra, got = take(
                        pool,
                        lambda g, c=cname, k=k: sha(f"cx:{c}:{k}:{g}"),
                        remaining,
                        collections.Counter(),
                        float("inf"),
                        SHORT_MAX_NATIVE if variant == "short" else None,
                    )
                    if not extra:
                        break
                    crows.extend(
                        dict(m, id=f"{m['id']}#r{k}", pool="A0s-resample")
                        for m in extra
                    )
                    remaining -= got
            else:
                extra, _ = take(
                    pool,
                    lambda g, c=cname: sha(f"cx:{c}:{g}"),
                    total - sum(m["native"] for m in base),
                    used_c,
                    float("inf"),
                    SHORT_MAX_NATIVE if variant == "short" else None,
                )
                crows.extend(dict(m, pool="V1ALL") for m in extra)
            cs = summary(crows)
            cs["sha256"] = write(args.out_dir, cname, crows)
            manifest["recipes"][cname] = cs
    manifest["a0s_native_tokens"] = a0s_tokens
    (args.out_dir / "mixtures.manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    )
    for name, s in manifest["recipes"].items():
        print(
            name,
            s["rows"],
            s["native_tokens"],
            s["english_share"],
            s.get("english_cap_ok"),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
