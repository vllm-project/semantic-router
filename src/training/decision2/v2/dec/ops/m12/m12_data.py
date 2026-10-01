"""Decoder M12 TRAIN files (prereg dec-m12-prereg-2026-10-01.md, "Data"): additive breadth.

Every arm keeps the tier's released TRAIN file whole (every row, byte for byte, in file order) and adds IB1-r3 + IB2
rows after it. Tokens are counted with the trainer's head-readout encoding (`training.model.decision_model.encode`,
the tier's base tokenizer, no truncation); T is the base file's total. An arm spec is NAME=POOL:AMOUNT with POOL
`all` (every IB family) or `transfer` (without the in-distribution families) and AMOUNT either a share (`0.25`: IB
tokens = round(share * T), whole groups, stratified by family in proportion to the family's tokens in the pool, as
M11's m11_s2data.sample_groups) or whole-pool copies (`x3`: the pool three times; copy k > 1 of a row keeps its line
except the id, which gets the suffix `~c<k>` so the partition stays unique, as build_template_s's resample suffix).
A share larger than the pool is an error (the prereg uses copies there).

usage: m12_data.py --tier T --base B --base-sha S --ib1 F --ib1-sha S --ib2 F --ib2-sha S --tokenizer DIR \
         --indist w2c,isarc,hover,gsm2 --arm NAME=POOL:AMOUNT [...] --output DIR [--seed 20261001] [--workers N]
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "m11"))
import m11_s2data as s2  # noqa: E402

SCHEMA = "dec-m12-data/1"
MAX_LENGTH = 8192
COPY_SUFFIX = "~c"


def parse_arm(spec: str) -> tuple[str, str, float | None, int | None]:
    name, rest = spec.split("=", 1)
    pool, amount = rest.split(":", 1)
    if pool not in ("all", "transfer"):
        raise ValueError(f"{spec}: pool must be all or transfer")
    if amount.startswith("x"):
        copies = int(amount[1:])
        if copies < 1:
            raise ValueError(f"{spec}: copies must be >= 1")
        return name, pool, None, copies
    share = float(amount)
    if not 0 < share < 1:
        raise ValueError(f"{spec}: share must be in (0, 1)")
    return name, pool, share, None


def copy_line(line: bytes, k: int) -> bytes:
    if k == 1:
        return line
    row = json.loads(line)
    row["id"] = f"{row['id']}{COPY_SUFFIX}{k}"
    return json.dumps(row, ensure_ascii=False).encode() + b"\n"


def build_arm(
    base: list[dict[str, Any]],
    pool_rows: list[dict[str, Any]],
    total: int,
    share: float | None,
    copies: int | None,
    seed: int,
) -> tuple[list[tuple[dict[str, Any], int]], dict[str, Any]]:
    """Rows of the arm as (row, copy index): every base row, then the IB rows."""
    pool_tokens = sum(r["tokens"] for r in pool_rows)
    if copies is not None:
        ib = [(r, k) for k in range(1, copies + 1) for r in pool_rows]
        info = {"copies": copies, "fraction": 1.0}
        kept = set(range(len(pool_rows)))
    else:
        target = round(share * total)
        if target > pool_tokens:
            raise ValueError(
                f"share {share} needs {target} IB tokens; pool has {pool_tokens}"
            )
        kept, info = s2.sample_groups(pool_rows, target, seed)
        info = {**info, "target": target}
        ib = [(pool_rows[i], 1) for i in sorted(kept)]
    rows = [(r, 1) for r in base] + ib
    ib_sum = s2.summary(pool_rows, kept)
    ib_tokens = ib_sum["tokens"] * (copies or 1)
    return rows, {
        **info,
        "ib_unique": ib_sum,
        "ib_tokens": ib_tokens,
        "ib_share_of_T": ib_tokens / total,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tier", required=True)
    for name in ("base", "ib1", "ib2"):
        p.add_argument(f"--{name}", type=Path, required=True)
        p.add_argument(f"--{name}-sha", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--indist", required=True)
    p.add_argument("--arm", action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261001)
    p.add_argument("--workers", type=int, default=32)
    a = p.parse_args(argv)
    specs = [parse_arm(s) for s in a.arm]
    indist = set(a.indist.split(","))
    files = {"base": a.base, "ib1": a.ib1, "ib2": a.ib2}
    for name, path in files.items():
        if s2.sha256(path) != getattr(a, f"{name}_sha"):
            raise ValueError(f"{path}: sha256 differs from --{name}-sha")
    blocks: dict[str, list[dict[str, Any]]] = {}
    with Pool(a.workers, initializer=s2._init, initargs=(a.tokenizer,)) as pool:
        for name, path in files.items():
            with path.open("rb") as stream:
                lengths = pool.map(s2._length, list(stream), chunksize=64)
            blocks[name] = s2.load(path, name, lengths)
    longest = max(r["tokens"] for rows in blocks.values() for r in rows)
    if longest > MAX_LENGTH:
        raise ValueError(f"a row has {longest} tokens > {MAX_LENGTH}")
    ids = [r["id"] for rows in blocks.values() for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("repeated row id across the inputs")
    if any(COPY_SUFFIX in i for i in ids):
        raise ValueError(f"an input id already contains {COPY_SUFFIX!r}")
    base = blocks["base"]
    ib = blocks["ib1"] + blocks["ib2"]
    present = {r["family"] for r in ib}
    if not indist <= present:
        raise ValueError(
            f"in-distribution families not in IB: {sorted(indist - present)}"
        )
    pools = {"all": ib, "transfer": [r for r in ib if r["family"] not in indist]}
    total = sum(r["tokens"] for r in base)
    a.output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "tier": a.tier,
        "inputs": {
            n: {"path": str(q), "sha256": getattr(a, f"{n}_sha")}
            for n, q in files.items()
        },
        "tokenizer": a.tokenizer,
        "token_unit": "training.model.decision_model.encode (head readout), no truncation",
        "seed": a.seed,
        "in_distribution": sorted(indist),
        "T_base_tokens": total,
        "base_rows": len(base),
        "ib_pool_tokens": {k: sum(r["tokens"] for r in v) for k, v in pools.items()},
        "longest_row": longest,
        "arms": {},
    }
    for name, pool_name, share, copies in specs:
        rows, info = build_arm(base, pools[pool_name], total, share, copies, a.seed)
        out = a.output / name
        out.mkdir()
        with (out / "train.jsonl").open("xb") as sink:
            for r, k in rows:
                sink.write(copy_line(r["line"], k))
        with (out / "train.ids.jsonl").open("x") as sink:
            for r, k in rows:
                rid = r["id"] if k == 1 else f"{r['id']}{COPY_SUFFIX}{k}"
                sink.write(
                    json.dumps({"id": rid, "block": r["block"], "tokens": r["tokens"]})
                    + "\n"
                )
        tokens = sum(r["tokens"] for r, _ in rows)
        report["arms"][name] = {
            "pool": pool_name,
            "share": share,
            "copies": copies,
            "train_sha256": s2.sha256(out / "train.jsonl"),
            "ids_sha256": s2.sha256(out / "train.ids.jsonl"),
            "rows": len(rows),
            "tokens": tokens,
            "tokens_vs_T": tokens / total,
            "ib": info,
        }
    (a.output / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(
        json.dumps(
            {
                "tier": a.tier,
                "T": total,
                "arms": {
                    k: {
                        "rows": v["rows"],
                        "tokens": v["tokens"],
                        "ib_share_of_T": round(v["ib"]["ib_share_of_T"], 4),
                    }
                    for k, v in report["arms"].items()
                },
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
