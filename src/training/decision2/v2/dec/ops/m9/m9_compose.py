"""Decoder M9 matched-token mixtures (prereg dec-m9-prereg-2026-10-01.md, "Data").

One deterministic CPU pass (node A) writes two TRAIN files that share the released 4B recipe's mixture (the base):

  H9  base + the HR2 TRAIN block (whole, once; gold only, pool label `hr2`)
  C9  base + filler of |HR2| native tokens: whole groups of the XL r2 pool that share no id, group or input hash with
      the base, chosen by `build_template_s.select_groups` (strata source x task type x language) with M7's 4B filler
      seed, the budget bisected until the chosen tokens are closest to |HR2| (ops/m7/m7_compose.pick_matched)

- Quarantine: every row of the listed groups is dropped from the base and the pool first.
- Row pools (`train.ids.jsonl`, for `compose_teacher --allow-missing-pool`): base / fill rows whose recipe pool is a
  `--gold-pools` pool (H7, H8: gold only in the released recipe) get the suffix `-gold`; HR2 rows are `hr2`.
- C9's `teacher-new.jsonl` holds the `--pool-teacher` record (own-Lux, `lux-all-59m`) of every non-gold fill row.
- An HR2 row whose id, group or input hash also occurs in the base is an error (the HR2 build checked its isolation
  against the training corpora; the composer re-checks it against this base).

usage (launch.sh --cpu): python3 v2/dec/ops/m9/m9_compose.py --base B --base-sha S --pool P --pool-sha S
    --pool-ids IDS --pool-ids-sha S --pool-teacher T --pool-teacher-sha S --hr2 H --hr2-sha S --quarantine Q
    --tokenizer TOK --output DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "m7"))

from m7_compose import (  # noqa: E402
    group_tokens,
    pick_matched,
    summary,
    teacher_new,
    verified,
    write_arm,
)
from training.model.data import file_sha256  # noqa: E402
from v2.common import eval_only  # noqa: E402
from v2.dec.build_template_s import group_rows  # noqa: E402

SCHEMA = "dec-m9-compose/1"
ARMS = ("H9", "C9")
FILL_SEED = "dec-m7-fill-4b-v1"
Row = dict[str, Any]


def compose(
    base: list[Row],
    pool: list[Row],
    hr2: list[Row],
    length_of: dict[str, int],
    *,
    quarantine: set[str],
    pool_of: dict[str, str],
    gold_pools: set[str],
    fill_seed: str = FILL_SEED,
) -> tuple[dict[str, list[tuple[Row, str]]], dict[str, Any]]:
    """The two arms as (row, pool label) lists, and the composition report."""

    def ln(row: Row) -> int:
        return length_of[row["id"]]

    dropped = {
        "base": sum(1 for r in base if r["group_id"] in quarantine),
        "pool": sum(1 for r in pool if r["group_id"] in quarantine),
    }
    base = [r for r in base if r["group_id"] not in quarantine]
    pool = [r for r in pool if r["group_id"] not in quarantine]
    base_ids = {r["id"] for r in base}
    base_groups = {r["group_id"] for r in base}
    base_inputs = {r["input_sha256"] for r in base}

    clash = {
        "id": sum(1 for r in hr2 if r["id"] in base_ids),
        "group": sum(1 for r in hr2 if r["group_id"] in base_groups),
        "input": sum(1 for r in hr2 if r["input_sha256"] in base_inputs),
    }
    if any(clash.values()):
        raise ValueError(f"HR2 rows shared with the base: {clash}")
    if len({r["id"] for r in hr2}) != len(hr2):
        raise ValueError("HR2: repeated row id")

    def label(row: Row, name: str) -> str:
        return f"{name}-gold" if pool_of.get(row["id"]) in gold_pools else name

    new = [
        r
        for r in pool
        if r["id"] not in base_ids
        and r["group_id"] not in base_groups
        and r["input_sha256"] not in base_inputs
    ]
    new_groups = group_rows(new)
    new_tok = group_tokens(new_groups, length_of)
    x = sum(ln(r) for r in hr2)
    c_pick, c_rho = pick_matched(new_groups, new_tok, x, fill_seed)
    fill = [r for g in sorted(c_pick) for r in new_groups[g]]
    base_l = [(r, label(r, "base")) for r in base]
    arms = {
        "H9": base_l + [(r, "hr2") for r in hr2],
        "C9": base_l + [(r, label(r, "fill")) for r in fill],
    }
    for arm, rows in arms.items():
        ids = [r["id"] for r, _ in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{arm}: repeated row id")
    tokens = {arm: sum(ln(r) for r, _ in rows) for arm, rows in arms.items()}
    hr2_summary = summary(hr2, ln)
    hr2_summary["families"] = dict(sorted(Counter(r["family"] for r in hr2).items()))
    hr2_summary["groups"] = len({r["group_id"] for r in hr2})
    report = {
        "tier": "4b",
        "base": summary(base, ln),
        "quarantine_rows_dropped": dropped,
        "hr2": hr2_summary,
        "filler": {
            "seed": fill_seed,
            "target": x,
            **summary(fill, ln),
            "budget": c_rho,
            "groups": len(c_pick),
            "group_ids": sorted(c_pick),
            "gold_rows": sum(1 for r in fill if pool_of.get(r["id"]) in gold_pools),
            "eligible_groups": len(new_groups),
            "eligible_tokens": sum(new_tok.values()),
        },
        "added_tokens_H9": x,
        "tokens": tokens,
        "match": {
            "C9_minus_H9": tokens["C9"] - tokens["H9"],
            "max_relative": abs(tokens["C9"] - tokens["H9"]) / tokens["H9"],
        },
    }
    return arms, report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for name in ("base", "pool", "pool-ids", "pool-teacher", "hr2"):
        p.add_argument(f"--{name}", type=Path, required=True)
        p.add_argument(f"--{name}-sha", required=True)
    p.add_argument("--gold-pools", default="H7,H8")
    p.add_argument(
        "--quarantine", type=Path, required=True, help='{"group_ids": [...]}'
    )
    p.add_argument("--fill-seed", default=FILL_SEED)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    if a.output.exists():
        raise FileExistsError(a.output)
    eval_only.guard(a)
    from v2.dec.build_mixture import token_lengths

    base = verified(a.base, a.base_sha)
    pool = verified(a.pool, a.pool_sha)
    hr2 = verified(a.hr2, a.hr2_sha)
    for path, sha in (
        (a.pool_ids, a.pool_ids_sha),
        (a.pool_teacher, a.pool_teacher_sha),
    ):
        if file_sha256(path) != sha:
            raise ValueError(f"{path}: sha256 differs from {sha}")
    quarantine = set(json.loads(a.quarantine.read_text())["group_ids"])
    pool_of = {}
    with a.pool_ids.open(encoding="utf-8") as stream:
        for line in stream:
            entry = json.loads(line)
            pool_of[entry["id"]] = entry["pool"]
    unique: dict[str, Row] = {}
    for row in base + pool + hr2:
        unique.setdefault(row["id"], row)
    rows = list(unique.values())
    length_of = dict(
        zip((r["id"] for r in rows), token_lengths(rows, a.tokenizer, a.workers))
    )
    arms, report = compose(
        base,
        pool,
        hr2,
        length_of,
        quarantine=quarantine,
        pool_of=pool_of,
        gold_pools={x for x in a.gold_pools.split(",") if x},
        fill_seed=a.fill_seed,
    )
    for arm in ARMS:
        eval_only.check_rows(r for r, _ in arms[arm])
    a.output.mkdir(parents=True)
    files: dict[str, Any] = {}
    for arm in ARMS:
        d = a.output / f"m9-4b-{arm}"
        files[arm] = write_arm(d, arms[arm])
        extra = [r for r, pool in arms[arm] if pool == "fill"]
        if extra:
            files[arm]["teacher_new"] = teacher_new(
                a.pool_teacher, {r["id"] for r in extra}, d / "teacher-new.jsonl"
            )
        files[arm]["pools"] = dict(
            sorted(Counter(pool for _, pool in arms[arm]).items())
        )
    inputs = {
        "base": a.base_sha,
        "pool": a.pool_sha,
        "pool_ids": a.pool_ids_sha,
        "pool_teacher": a.pool_teacher_sha,
        "hr2": a.hr2_sha,
        "quarantine": file_sha256(a.quarantine),
    }
    doc = {
        "schema": SCHEMA,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        "inputs_sha256": inputs,
        "tokenizer_files_sha256": {
            n: file_sha256(a.tokenizer / n)
            for n in ("tokenizer.json", "tokenizer_config.json")
            if (a.tokenizer / n).is_file()
        },
        "report": report,
        "files": files,
    }
    (a.output / "compose.json").write_text(
        json.dumps(doc, indent=1, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "tokens": report["tokens"],
                "match": report["match"],
                "added_H9": report["added_tokens_H9"],
                "filler_groups": report["filler"]["groups"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
