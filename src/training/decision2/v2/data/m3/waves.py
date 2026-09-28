"""RP-v2 prompt waves for the AutoJev-27B targets and the published own-Lux waves.

    python3 -m v2.data.m3.waves --pools pools-mx-v2.json --recipes DIR --out-dir DIR \\
        [--expect FILE=SHA256 ...]

With S, M, L = the non-A0s ids of (full-S, short-S), (full-M, short-M), (full-L, short-M),
writes, sorted by id:

- ``rp-v2.rows.jsonl`` / ``rp-v2.prompts.jsonl``: the Milestone 2 RP-v2 pool = L
  (`v2.data.m2.rp_pool` semantics),
- ``lux-wave1/2/3.prompts.jsonl``: S within L; M minus S; L minus S and M (the own-Lux
  waves published in Milestone 2); ``lux-wave4.prompts.jsonl``: S and M rows outside L,
  which RP-v2 missed because the recipes nest only approximately,
- ``mx-v2-all.rows.jsonl``: every non-A0s row of S, M and L,
- ``aj-m.prompts.jsonl`` (M) and ``aj-sl.prompts.jsonl`` (S and L minus M): the AutoJev
  waves before and after A0s.

Every written file is checked against ``--expect`` hashes when given, so a node-A rebuild
can be proven byte-identical to the node-B files the Lux targets were produced from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from training.model.data import canonical
from v2.data.build_a0_variants import native_prompt
from v2.data.m2.common import read_jsonl

RECIPES = {
    "S": ("mx-v2-full-S", "mx-v2-short-S"),
    "M": ("mx-v2-full-M", "mx-v2-short-M"),
    "L": ("mx-v2-full-L", "mx-v2-short-M"),
}


def recipe_ids(recipes: Path, names: tuple[str, ...]) -> set[str]:
    return {
        row["id"]
        for name in names
        for row in read_jsonl(recipes / f"{name}.ids.jsonl")
        if row["pool"] != "A0s"
    }


def prompt_line(row: dict) -> str:
    return json.dumps(native_prompt(row), ensure_ascii=False) + "\n"


def _write(path: Path, lines: list[str]) -> str:
    data = "".join(lines).encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
    return hashlib.sha256(data).hexdigest()


def build(pools: dict, recipes: Path, out_dir: Path) -> dict[str, dict]:
    small, medium, large = (recipe_ids(recipes, f) for f in RECIPES.values())
    everything = small | medium | large
    rows: dict[str, dict] = {}
    for spec in pools.values():
        for path in spec["rows"]:
            for row in read_jsonl(Path(path)):
                if row["id"] in everything:
                    rows.setdefault(row["id"], row)
    missing = everything - set(rows)
    if missing:
        raise ValueError(
            f"{len(missing)} recipe ids not found, e.g. {sorted(missing)[:3]}"
        )
    ordered = [rows[i] for i in sorted(rows)]
    pool = [r for r in ordered if r["id"] in large]
    waves = {
        "lux-wave1": small & large,
        "lux-wave2": medium - small,
        "lux-wave3": large - small - medium,
        "lux-wave4": (small | medium) - large,
        "aj-m": medium,
        "aj-sl": (small | large) - medium,
    }
    out: dict[str, dict] = {}
    for name, selected, render in (
        ("rp-v2.rows.jsonl", pool, lambda r: canonical(r) + "\n"),
        ("rp-v2.prompts.jsonl", pool, prompt_line),
        ("mx-v2-all.rows.jsonl", ordered, lambda r: canonical(r) + "\n"),
    ):
        out[name] = {
            "rows": len(selected),
            "sha256": _write(out_dir / name, [render(r) for r in selected]),
        }
    for name, ids in waves.items():
        lines = [prompt_line(r) for r in ordered if r["id"] in ids]
        out[f"{name}.prompts.jsonl"] = {
            "rows": len(lines),
            "sha256": _write(out_dir / f"{name}.prompts.jsonl", lines),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pools", type=Path, required=True)
    parser.add_argument("--recipes", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expect", action="append", default=[])
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = build(
        json.loads(args.pools.read_text(encoding="utf-8")), args.recipes, args.out_dir
    )
    mismatched = []
    for spec in args.expect:
        name, _, expected = spec.partition("=")
        out[name]["expected_sha256"] = expected
        if out[name]["sha256"] != expected:
            mismatched.append(name)
    receipt = {
        "schema": "decision2-m3a-waves/1",
        "files": out,
        "mismatched": mismatched,
    }
    _write(
        args.out_dir / "waves.receipt.json",
        [json.dumps(receipt, indent=1, sort_keys=True) + "\n"],
    )
    print(json.dumps(receipt, sort_keys=True))
    return 1 if mismatched else 0


if __name__ == "__main__":
    sys.exit(main())
