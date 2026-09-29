"""Decoder M7 matched-token mixtures (prereg dec-m7-prereg-2026-09-30.md, "Arms and data").

One deterministic pass per tier writes three TRAIN files that share the released recipe's mixture (the base):

  H  base + HS1 block + long-prose block (LP)
  C  base + filler of |HS1| + |LP| native tokens (the matched-token control)
  P  base + PN1 block + filler of |HS1| + |LP| - |PN1| tokens; its filler is a per-stratum prefix of C's filler

Every block takes whole groups; sampled blocks use the builder's sampler (`build_template_s.select_groups`: strata
source x task type x language, seed-keyed hash order, each stratum's share of the budget), so a smaller budget with
the same seed takes a prefix of the same order. Matched budgets are bisected on the sampler's budget until the chosen
tokens are closest to the target.

- Quarantine: every row of the listed groups is dropped from the base and the pool first.
- HS1: rows of groups containing a `--hs1-drop-substring` (a confirmed template defect) are dropped; then the
  `--hs1-whole` families whole plus half (by tokens) of the `--hs1-half` family's groups.
- LP: pool groups not in the base (no shared id, group or input hash) whose longest state is over `--lp-min-chars`
  characters, sampled to `--lp-budget` tokens.
- filler `pool`: pool groups not in the base; `replay`: base groups repeated once, the copy's id suffixed `~r2`.
- PN1: the PN1 TRAIN rows `--pn1-repeat` times; copy k >= 2 has its id suffixed `~r<k>`.

Row pools (`train.ids.jsonl`, for `compose_teacher --allow-missing-pool`): base / lp / fill rows whose recipe pool is
a `--gold-pools` pool (4B: H7, H8, gold-only as in the released recipe) get the suffix `-gold`; hs1 and pn1 rows are
gold-only. Teacher sources per arm: `teacher-new.jsonl` (lines of `--pool-teacher` for the arm's non-gold lp / fill
rows) and, for replay, `teacher-replay.jsonl` (the `--base-teacher` record of each replayed row under its new id).

usage (launch.sh --cpu): python3 v2/dec/ops/m7/m7_compose.py --tier 4b --base B --base-sha S --pool P --pool-sha S
    --pool-ids IDS --pool-teacher T --hs1 H --hs1-sha S --pn1 N --pn1-sha S --quarantine Q --filler pool
    --tokenizer TOK --output DIR   [2B: --filler replay --base-teacher T --gold-pools '']
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from training.model.data import file_sha256, load_partition  # noqa: E402
from v2.dec.build_template_s import group_rows, select_groups  # noqa: E402

SCHEMA = "dec-m7-compose/1"
ARMS = ("H", "C", "P")
Row = dict[str, Any]


def state_chars(row: Row) -> int:
    state = row["state"]
    return len(
        state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    )


def group_tokens(
    groups: dict[str, list[Row]], length: dict[str, int]
) -> dict[str, int]:
    return {g: sum(length[r["id"]] for r in rows) for g, rows in groups.items()}


def pick(
    groups: dict[str, list[Row]], gtok: dict[str, int], rho: float, seed: str
) -> list[str]:
    return select_groups(groups, gtok, int(rho), seed) if rho > 0 else []


def pick_matched(
    groups: dict[str, list[Row]], gtok: dict[str, int], target: int, seed: str
) -> tuple[list[str], int]:
    """Groups whose tokens are closest to ``target``; the chosen set grows monotonically with the budget."""
    total = sum(gtok.values())
    if target > total:
        raise ValueError(f"filler target {target} exceeds the pool's {total} tokens")
    lo, hi = 0.0, float(total)
    best: tuple[int, list[str], int] = (abs(target), [], 0)
    for _ in range(60):
        mid = (lo + hi) / 2
        chosen = pick(groups, gtok, mid, seed)
        got = sum(gtok[g] for g in chosen)
        if abs(got - target) < best[0]:
            best = (abs(got - target), chosen, int(mid))
        if got < target:
            lo = mid
        elif got > target:
            hi = mid
        else:
            break
    return best[1], best[2]


def copies(rows: list[Row], k: int) -> list[Row]:
    return [dict(r, id=f"{r['id']}~r{k}") for r in rows]


def summary(rows: list[Row], length: Callable[[Row], int]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "tokens": sum(length(r) for r in rows),
        "rows_by_type": dict(sorted(Counter(r["task_type"] for r in rows).items())),
        "rows_by_language": dict(Counter(r["language"] for r in rows).most_common(12)),
        "sources": dict(Counter(r["source"] for r in rows).most_common(20)),
    }


def compose(
    tier: str,
    base: list[Row],
    pool: list[Row],
    hs1: list[Row],
    pn1: list[Row],
    length_of: dict[str, int],
    *,
    quarantine: set[str],
    pool_of: dict[str, str],
    gold_pools: set[str],
    filler: str,
    hs1_whole: tuple[str, ...],
    hs1_half: str,
    hs1_drop: tuple[str, ...],
    lp_min_chars: int,
    lp_budget: int,
    pn1_repeat: int,
) -> tuple[dict[str, list[tuple[Row, str]]], dict[str, Any]]:
    """The three arms as (row, pool label) lists, and the composition report."""

    def ln(row: Row) -> int:
        return length_of[row["id"].split("~r", 1)[0]]

    dropped = {
        "base": sum(1 for r in base if r["group_id"] in quarantine),
        "pool": sum(1 for r in pool if r["group_id"] in quarantine),
    }
    base = [r for r in base if r["group_id"] not in quarantine]
    pool = [r for r in pool if r["group_id"] not in quarantine]
    base_ids = {r["id"] for r in base}
    base_groups = {r["group_id"] for r in base}
    base_inputs = {r["input_sha256"] for r in base}

    def gold(row: Row) -> bool:
        return pool_of.get(row["id"]) in gold_pools

    def label(row: Row, name: str) -> str:
        return f"{name}-gold" if gold(row) else name

    defect = {
        r["group_id"]
        for r in hs1
        if any(s in json.dumps(r, ensure_ascii=False) for s in hs1_drop)
    }
    kept = [r for r in hs1 if r["group_id"] not in defect]
    whole = [r for r in kept if r["family"] in hs1_whole]
    half_rows = [r for r in kept if r["family"] == hs1_half]
    half_groups = group_rows(half_rows)
    half_tok = group_tokens(half_groups, length_of)
    half_pick = set(
        pick(half_groups, half_tok, sum(half_tok.values()) / 2, "dec-m7-hs1-half-v1")
    )
    hs1_block = whole + [r for r in half_rows if r["group_id"] in half_pick]

    new = [
        r
        for r in pool
        if r["id"] not in base_ids
        and r["group_id"] not in base_groups
        and r["input_sha256"] not in base_inputs
    ]
    new_groups = group_rows(new)
    new_tok = group_tokens(new_groups, length_of)
    long_groups = {
        g: rows
        for g, rows in new_groups.items()
        if max(map(state_chars, rows)) > lp_min_chars
    }
    lp_pick = set(pick(long_groups, new_tok, lp_budget, "dec-m7-lp-v1"))
    lp_block = [r for g in sorted(lp_pick) for r in new_groups[g]]

    pn1_block = list(pn1) + [
        row for k in range(2, pn1_repeat + 1) for row in copies(pn1, k)
    ]
    x = sum(ln(r) for r in hs1_block) + sum(ln(r) for r in lp_block)
    y = sum(ln(r) for r in pn1_block)
    if filler == "pool":
        fill_groups, fill_tok, seed = new_groups, new_tok, f"dec-m7-fill-{tier}-v1"
    elif filler == "replay":
        fill_groups = group_rows(base)
        fill_tok = group_tokens(fill_groups, length_of)
        seed = f"dec-m7-replay-{tier}-v1"
    else:
        raise ValueError(f"unknown filler {filler}")
    c_pick, c_rho = pick_matched(fill_groups, fill_tok, x, seed)
    p_pick, p_rho = pick_matched(fill_groups, fill_tok, x - y, seed)
    if not set(p_pick) <= set(c_pick):
        raise ValueError("P's filler is not nested in C's")

    def fill(chosen: list[str]) -> list[Row]:
        rows = [r for g in sorted(chosen) for r in fill_groups[g]]
        return copies(rows, 2) if filler == "replay" else rows

    fill_c, fill_p = fill(c_pick), fill(p_pick)
    base_l = [(r, label(r, "base")) for r in base]
    arms = {
        "H": base_l
        + [(r, "hs1") for r in hs1_block]
        + [(r, label(r, "lp")) for r in lp_block],
        "C": base_l
        + [(r, label(r, "replay" if filler == "replay" else "fill")) for r in fill_c],
        "P": base_l
        + [(r, "pn1") for r in pn1_block]
        + [(r, label(r, "replay" if filler == "replay" else "fill")) for r in fill_p],
    }
    for arm, rows in arms.items():
        ids = [r["id"] for r, _ in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{arm}: repeated row id")
    tokens = {arm: sum(ln(r) for r, _ in rows) for arm, rows in arms.items()}
    report = {
        "tier": tier,
        "base": summary(base, ln),
        "quarantine_rows_dropped": dropped,
        "hs1": {
            **summary(hs1_block, ln),
            "defect_groups_dropped": len(defect),
            "defect_rows_dropped": sum(1 for r in hs1 if r["group_id"] in defect),
            "whole_families": {
                f: sum(1 for r in whole if r["family"] == f) for f in hs1_whole
            },
            "half_family": hs1_half,
            "half_groups": f"{len(half_pick)} of {len(half_groups)}",
            "half_tokens": f"{sum(half_tok[g] for g in half_pick)} of {sum(half_tok.values())}",
        },
        "lp": {
            **summary(lp_block, ln),
            "min_state_chars": lp_min_chars,
            "budget": lp_budget,
            "eligible_groups": len(long_groups),
            "groups": len(lp_pick),
            "gold_rows": sum(1 for r in lp_block if gold(r)),
        },
        "pn1": {
            **summary(pn1_block, ln),
            "repeat": pn1_repeat,
            "unique_rows": len(pn1),
        },
        "filler": {
            "kind": filler,
            "seed": seed,
            "C": {
                "target": x,
                **summary(fill_c, ln),
                "budget": c_rho,
                "groups": len(c_pick),
            },
            "P": {
                "target": x - y,
                **summary(fill_p, ln),
                "budget": p_rho,
                "groups": len(p_pick),
            },
            "P_nested_in_C": True,
        },
        "added_tokens_H": x,
        "tokens": tokens,
        "match": {
            "C_minus_H": tokens["C"] - tokens["H"],
            "P_minus_H": tokens["P"] - tokens["H"],
            "max_relative": max(abs(tokens[a] - tokens["H"]) for a in ARMS)
            / tokens["H"],
        },
    }
    return arms, report


def write_arm(out: Path, rows: list[tuple[Row, str]]) -> dict[str, str]:
    out.mkdir(parents=True, exist_ok=False)
    with (out / "train.jsonl").open("x", encoding="utf-8") as stream:
        for row, _ in rows:
            stream.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    with (out / "train.ids.jsonl").open("x", encoding="utf-8") as stream:
        for row, pool in rows:
            stream.write(json.dumps({"id": row["id"], "pool": pool}) + "\n")
    load_partition(out / "train.jsonl", "train")
    return {
        "train_sha256": file_sha256(out / "train.jsonl"),
        "ids_sha256": file_sha256(out / "train.ids.jsonl"),
    }


def teacher_new(src: Path, wanted: set[str], out: Path) -> dict[str, Any]:
    found = 0
    with src.open(encoding="utf-8") as stream, out.open("x", encoding="utf-8") as sink:
        for line in stream:
            if json.loads(line)["id"] in wanted:
                sink.write(line if line.endswith("\n") else line + "\n")
                found += 1
    if found != len(wanted):
        raise ValueError(
            f"{src}: {len(wanted) - found} of {len(wanted)} new rows have no target"
        )
    return {"file": out.name, "rows": found, "sha256": file_sha256(out)}


def teacher_replay(src: Path, replayed: dict[str, str], out: Path) -> dict[str, Any]:
    records = {}
    with src.open(encoding="utf-8") as stream:
        for line in stream:
            rec = json.loads(line)
            if rec["id"] in replayed:
                records[rec["id"]] = rec
    missing = set(replayed) - set(records)
    if missing:
        raise ValueError(f"{src}: {len(missing)} replayed rows have no base target")
    with out.open("x", encoding="utf-8") as sink:
        for original, new_id in sorted(replayed.items(), key=lambda kv: kv[1]):
            rec = dict(records[original], id=new_id)
            sink.write(
                json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    return {"file": out.name, "rows": len(replayed), "sha256": file_sha256(out)}


def verified(path: Path, sha: str) -> list[Row]:
    actual = file_sha256(path)
    if actual != sha:
        raise ValueError(f"{path}: sha256 {actual} != {sha}")
    return load_partition(path, "train")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--tier", choices=("4b", "2b"), required=True)
    for name in ("base", "pool", "hs1", "pn1"):
        p.add_argument(f"--{name}", type=Path, required=True)
        p.add_argument(f"--{name}-sha", required=True)
    p.add_argument(
        "--pool-ids", type=Path, required=True, help="the pool recipe's {id, pool} list"
    )
    p.add_argument("--pool-teacher", type=Path, required=True)
    p.add_argument(
        "--base-teacher", type=Path, help="replay filler: the base's teacher file"
    )
    p.add_argument("--gold-pools", default="H7,H8")
    p.add_argument(
        "--quarantine", type=Path, required=True, help='{"group_ids": [...]}'
    )
    p.add_argument("--filler", choices=("pool", "replay"), required=True)
    p.add_argument("--hs1-whole", default="hs1_policy_packet,hs1_unmet_condition")
    p.add_argument("--hs1-half", default="hs1_quote_check")
    p.add_argument("--hs1-drop-substring", action="append", default=[])
    p.add_argument("--lp-min-chars", type=int, default=3000)
    p.add_argument("--lp-budget", type=int, default=3_000_000)
    p.add_argument("--pn1-repeat", type=int, default=3)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    if a.filler == "replay" and a.base_teacher is None:
        p.error("--filler replay needs --base-teacher")
    if a.output.exists():
        raise FileExistsError(a.output)
    from v2.dec.build_mixture import token_lengths

    base = verified(a.base, a.base_sha)
    pool = verified(a.pool, a.pool_sha)
    hs1 = verified(a.hs1, a.hs1_sha)
    pn1 = verified(a.pn1, a.pn1_sha)
    quarantine = set(json.loads(a.quarantine.read_text())["group_ids"])
    pool_of = {}
    with a.pool_ids.open(encoding="utf-8") as stream:
        for line in stream:
            entry = json.loads(line)
            pool_of[entry["id"]] = entry["pool"]
    unique: dict[str, Row] = {}
    for row in base + pool + hs1 + pn1:
        unique.setdefault(row["id"], row)
    rows = list(unique.values())
    length_of = dict(
        zip((r["id"] for r in rows), token_lengths(rows, a.tokenizer, a.workers))
    )
    arms, report = compose(
        a.tier,
        base,
        pool,
        hs1,
        pn1,
        length_of,
        quarantine=quarantine,
        pool_of=pool_of,
        gold_pools={x for x in a.gold_pools.split(",") if x},
        filler=a.filler,
        hs1_whole=tuple(a.hs1_whole.split(",")),
        hs1_half=a.hs1_half,
        hs1_drop=tuple(a.hs1_drop_substring),
        lp_min_chars=a.lp_min_chars,
        lp_budget=a.lp_budget,
        pn1_repeat=a.pn1_repeat,
    )
    a.output.mkdir(parents=True)
    files: dict[str, Any] = {}
    for arm in ARMS:
        d = a.output / f"m7-{a.tier}-{arm}"
        files[arm] = write_arm(d, arms[arm])
        extra = [r for r, pool in arms[arm] if pool in ("lp", "fill")]
        if extra:
            files[arm]["teacher_new"] = teacher_new(
                a.pool_teacher, {r["id"] for r in extra}, d / "teacher-new.jsonl"
            )
        replayed = {
            r["id"].split("~r", 1)[0]: r["id"]
            for r, pool in arms[arm]
            if pool == "replay"
        }
        if replayed:
            files[arm]["teacher_replay"] = teacher_replay(
                a.base_teacher, replayed, d / "teacher-replay.jsonl"
            )
        files[arm]["pools"] = dict(
            sorted(Counter(pool for _, pool in arms[arm]).items())
        )
    inputs = {
        "base": a.base_sha,
        "pool": a.pool_sha,
        "hs1": a.hs1_sha,
        "pn1": a.pn1_sha,
        "pool_ids": file_sha256(a.pool_ids),
        "pool_teacher": file_sha256(a.pool_teacher),
        "quarantine": file_sha256(a.quarantine),
        **({"base_teacher": file_sha256(a.base_teacher)} if a.base_teacher else {}),
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
                "added_H": report["added_tokens_H"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
