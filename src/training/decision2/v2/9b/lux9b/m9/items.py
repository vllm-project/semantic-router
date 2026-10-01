"""9B M9 successor items 1-7 for one formal candidate (prereg "Formal and successor"; amendment 3 item 6).

    python3 items.py subset --train F --pool F [--pool F ...] --output OUT
    python3 items.py verdict --name NAME --formal-dir D --exposure F [--exposure F ...] --subset F --output OUT

``subset``: every line of the TRAIN file (the builds copy x60 and IB lines byte for byte) must occur in one of the
pool files (x60 TRAIN, IB1-r3 TRAIN, IB2 TRAIN); counts only, no ids or text.

``verdict`` reads D/NAME.gates/successor.json (written by formal.sh; numbers only) and applies the preregistered
items against the released T = 1 run I (DEV2.0-9B-T1):
  1. v3 paired ci95.low > 0 vs I;
  2. axis_ci95.H.delta.high >= 0 vs I;
  3. every `gates types` verdict OK;
  4. card-eligible mlx-diag Choice + Noul (lux9b.mlx_paired) ci95.high >= 0 vs formal-m4/K-a13-16k-mlx;
  5. vs adopted Lux1 16K ci95.low > 0, H.delta.high >= 0 vs Lux1 and vs Nimble v2, types OK;
  6. every exposure receipt lists 0 groups and the TRAIN is a subset of its pools (``subset``);
  7. `gates public231` vs I not REGRESSION.
Item 8 (C1 post-key) is the custodian's and is not read here. Host python3, stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA = "lux9b-m9-items/1"


def line_hashes(path: Path) -> list[bytes]:
    out = []
    with path.open("rb") as stream:
        for line in stream:
            line = line.rstrip(b"\r\n")
            if line.strip():
                out.append(hashlib.sha256(line).digest())
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 23), b""):
            h.update(block)
    return h.hexdigest()


def subset(args: argparse.Namespace) -> int:
    pool: set[bytes] = set()
    pools = []
    for path in args.pool:
        hashes = line_hashes(path)
        pool.update(hashes)
        pools.append({"path": str(path), "sha256": sha256(path), "rows": len(hashes)})
    train = line_hashes(args.train)
    missing = sum(h not in pool for h in train)
    value = {
        "schema": SCHEMA + ":subset",
        "train": {
            "path": str(args.train),
            "sha256": sha256(args.train),
            "rows": len(train),
        },
        "pools": pools,
        "rows_in_pools": len(train) - missing,
        "rows_missing": missing,
        "subset": missing == 0,
    }
    args.output.write_text(json.dumps(value, indent=1) + "\n")
    print(
        json.dumps({k: value[k] for k in ("rows_in_pools", "rows_missing", "subset")})
    )
    return 0 if missing == 0 else 1


def load(path: Path) -> tuple[Any, str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def at_least(value: Any, bound: float, strict: bool) -> bool:
    if not isinstance(value, (int, float)):
        return False
    return value > bound if strict else value >= bound


def verdict(args: argparse.Namespace) -> int:
    summary_path = args.formal_dir / f"{args.name}.gates" / "successor.json"
    s, s_sha = load(summary_path)
    inputs = {str(summary_path): s_sha}
    types = s.get("types") or {}
    types_ok = set(types) == {"choice", "noul", "score"} and all(
        v == "OK" for v in types.values()
    )
    t1, lux, nimble = (
        s.get("vs_T1") or {},
        s.get("vs_Lux1") or {},
        s.get("vs_Nimble2") or {},
    )
    mlx = s.get("mlx_vs_T1") or {}
    pub = s.get("public231_vs_T1") or {}
    receipts = []
    for path in args.exposure:
        d, sha = load(path)
        inputs[str(path)] = sha
        groups = d.get("groups")
        receipts.append(
            {
                "path": str(path),
                "label": d.get("label"),
                "groups": len(groups) if isinstance(groups, list) else groups,
            }
        )
    sub, sub_sha = load(args.subset)
    inputs[str(args.subset)] = sub_sha
    items = {
        "1_v3_vs_T1": {
            "pass": at_least(t1.get("lb"), 0.0, True),
            "point": t1.get("point"),
            "ci95": [t1.get("lb"), t1.get("ub")],
        },
        "2_H_vs_T1": {
            "pass": at_least(t1.get("H_delta_high"), 0.0, False),
            "H_delta": t1.get("H_delta"),
            "ci95": [t1.get("H_delta_low"), t1.get("H_delta_high")],
        },
        "3_types": {"pass": types_ok, "types": types},
        "4_mlx_card_eligible": {
            "pass": at_least(mlx.get("ci_high"), 0.0, False),
            "delta": mlx.get("delta"),
            "ci95": [mlx.get("ci_low"), mlx.get("ci_high")],
        },
        "5_tier": {
            "pass": lux.get("right") == "Lux1-16K"
            and at_least(lux.get("lb"), 0.0, True)
            and at_least(lux.get("H_delta_high"), 0.0, False)
            and at_least(nimble.get("H_delta_high"), 0.0, False)
            and types_ok,
            "vs_Lux1_16K": {
                "point": lux.get("point"),
                "ci95": [lux.get("lb"), lux.get("ub")],
                "H_delta_high": lux.get("H_delta_high"),
            },
            "vs_Nimble2": {
                "point": nimble.get("point"),
                "ci95": [nimble.get("lb"), nimble.get("ub")],
                "H_delta_high": nimble.get("H_delta_high"),
            },
        },
        "6_exposure": {
            "pass": bool(receipts)
            and all(r["groups"] == 0 for r in receipts)
            and sub.get("subset") is True,
            "receipts": receipts,
            "train_subset_of_pools": sub.get("subset"),
            "train_rows": (sub.get("train") or {}).get("rows"),
        },
        "7_public231_vs_T1": {
            "pass": pub.get("verdict") not in (None, "REGRESSION"),
            **pub,
        },
    }
    value = {
        "schema": SCHEMA,
        "name": args.name,
        "bar": "DEV2.0-9B released T = 1 run (release/dev2-8b-t1-derived; formal-path parity exact)",
        "v3": s.get("v3"),
        "public231": s.get("public231"),
        "items": items,
        "items_1_7_pass": all(v["pass"] for v in items.values()),
        "failed": [k for k, v in items.items() if not v["pass"]],
        "inputs_sha256": inputs,
    }
    args.output.write_text(json.dumps(value, indent=1) + "\n")
    print(
        json.dumps(
            {
                "name": args.name,
                "v3": (s.get("v3") or {}).get("score"),
                "items_1_7_pass": value["items_1_7_pass"],
                "failed": value["failed"],
            }
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("subset")
    a.add_argument("--train", type=Path, required=True)
    a.add_argument("--pool", type=Path, action="append", required=True)
    a.add_argument("--output", type=Path, required=True)
    b = sub.add_parser("verdict")
    b.add_argument("--name", required=True)
    b.add_argument("--formal-dir", type=Path, required=True)
    b.add_argument("--exposure", type=Path, action="append", required=True)
    b.add_argument("--subset", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    return subset(args) if args.command == "subset" else verdict(args)


if __name__ == "__main__":
    sys.exit(main())
