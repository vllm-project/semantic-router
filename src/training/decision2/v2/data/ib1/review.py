"""IB1 blind label review in two stages (prereg ``records/ib1-prereg-2026-10-01.md`` §4).

    python3 -m v2.data.ib1.review screen-sample --train TRAIN [--drop-families F ...] --out-dir S
    python3 -m v2.data.ib1.review screen-score --key S/key.jsonl --answers A [--answers ...] \
        --out PUBLIC --private PRIVATE --drop-families-out F --drop-ids-out IDS
    python3 -m v2.data.ib1.review sample --train TRAIN --screen-key S/key.jsonl [--drop-families F ...] --out-dir R
    python3 -m v2.data.ib1.review splits --key R/key.jsonl --r1 .. --r2 .. --packets .. --out R3PACKET
    python3 -m v2.data.ib1.review score --sample R/sample.json --key R/key.jsonl --r1 .. --r2 .. [--r3 ..] \
        --out REPORT --private PRIVATE

Stage S: one fresh reviewer answers 10 TRAIN rows per family; a family with >= 3 disagreements with gold is
dropped, and every screened row the reviewer disagrees with leaves TRAIN. Stage R: a fresh sample (no screened row or
group) of max(14, ceil(210 / F)) rows per surviving family; R1 and R2 answer everything, R3 the splits. A gold
error is an item both R1 and R2 disagree with, or a split R3 disagrees with. PASS needs P1 (pooled error <= 5% and
exact 95% upper bound <= 8%), P2 (population-weighted error <= 5%) and P3 (no family whose exact 95% lower bound
exceeds 5%).
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from v2.data.dq.blind_review import (
    BOOT_REPS,
    BOOT_SALT,
    assert_blind,
    clopper_pearson,
    cohen_kappa,
    proportion,
    read_answers,
    read_jsonl,
    sha,
    stratified_bootstrap,
    weighted_error,
    write_json,
    write_jsonl,
    write_new,
)
from v2.data.hr2.review import (
    agrees,
    answer,
    breakdown,
    key_item,
    packet_item,
    split_rids,
    write_packet,
)

SCREEN_PER_FAMILY = 10
SCREEN_DROP_AT = 3
SCREEN_SALT = "ib1-screen-v1"
SCREEN_PACKET_SALT = "ib1-screen-packet-v1"
REVIEW_MIN_PER_FAMILY = 14
REVIEW_MIN_TOTAL = 210
SALT = "ib1-review-v1"
PACKET_SALT = "ib1-packet-v1"
R2_SALT = "ib1-review-r2-order-v1"
R3_SALT = "ib1-review-r3-order-v1"
PACKETS = 2
FIELDS = ("rid", "task_type", "instructions", "state", "options")
THRESHOLDS = {
    "error_max": 0.05,
    "error_upper_max": 0.08,
    "weighted_error_max": 0.05,
    "family_lower_max": 0.05,
}


def train_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    text = path.read_bytes().decode("utf-8")
    rows = [json.loads(line) for line in text.split("\n") if line]
    for row in rows:
        if row.get("split") != "train":
            raise ValueError(f"{row.get('id')}: not a TRAIN row")
    return rows, sha(text)


def draw(
    rows: Sequence[Mapping[str, Any]],
    per_family: int,
    salt: str,
    skip_ids: set[str] = frozenset(),
    skip_groups: set[str] = frozenset(),
) -> list[Mapping[str, Any]]:
    """``per_family`` rows per family, one per group, gold cells capped at ceil(per_family / options)."""
    taken: collections.Counter = collections.Counter()
    cells: collections.Counter = collections.Counter()
    used: set[str] = set(skip_groups)
    picked = []
    for row in sorted(rows, key=lambda r: sha(f"{salt}:{r['id']}")):
        cell = (row["family"], row["label"])
        cap = math.ceil(per_family / len(row["options"]))
        if (
            row["id"] in skip_ids
            or taken[row["family"]] >= per_family
            or cells[cell] >= cap
            or row["group_id"] in used
        ):
            continue
        picked.append(row)
        taken[row["family"]] += 1
        cells[cell] += 1
        used.add(row["group_id"])
    return picked


def packets(
    picked: Sequence[Mapping[str, Any]], packet_salt: str, prefix: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(picked, key=lambda r: sha(f"{packet_salt}:{r['id']}"))
    rids = {row["id"]: f"{prefix}{index + 1:03d}" for index, row in enumerate(ordered)}
    items = [packet_item(rids[row["id"]], row) for row in ordered]
    key = [key_item(rids[row["id"]], row) for row in ordered]
    secrets = (
        {row["id"] for row in ordered}
        | {row["group_id"] for row in ordered}
        | {row["source"] for row in ordered}
    )
    assert_blind(items, FIELDS, secrets)
    return items, key


def population(
    rows: Sequence[Mapping[str, Any]], families: Iterable[str]
) -> dict[str, int]:
    wanted = set(families)
    return dict(
        sorted(
            collections.Counter(
                r["family"] for r in rows if r["family"] in wanted
            ).items()
        )
    )


# --------------------------------------------------------------------------- stage S


def screen_sample(
    rows: Sequence[Mapping[str, Any]], rows_sha256: str, drop: set[str]
) -> dict[str, Any]:
    eligible = [r for r in rows if r["family"] not in drop]
    picked = draw(eligible, SCREEN_PER_FAMILY, SCREEN_SALT)
    items, key = packets(picked, SCREEN_PACKET_SALT, "s")
    size = math.ceil(len(items) / PACKETS)
    return {
        "packets": [items[i : i + size] for i in range(0, len(items), size)],
        "key": key,
        "sample": {
            "schema": "decision2.ib1.screen-sample.v1",
            "rows_sha256": rows_sha256,
            "rows": len(rows),
            "salt": SCREEN_SALT,
            "per_family": SCREEN_PER_FAMILY,
            "n": len(key),
            "families_excluded": sorted(drop),
            "sampled": dict(
                sorted(collections.Counter(k["family"] for k in key).items())
            ),
        },
    }


def screen_score(
    key: Sequence[Mapping[str, Any]], answers: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    scored = []
    for row in key:
        value = answer(answers[row["rid"]], row["keys"], row["task_type"])
        scored.append(
            {
                **row,
                "s1": value,
                "agree": agrees(value, row),
                "confidence": str(answers[row["rid"]].get("confidence", "")).lower(),
                "defect": bool(answers[row["rid"]].get("defect")),
                "note": str(answers[row["rid"]].get("note", "")),
            }
        )
    by_family: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for item in scored:
        by_family[item["family"]].append(item)
    disagreements = {
        name: sum(not i["agree"] for i in group)
        for name, group in sorted(by_family.items())
    }
    dropped = sorted(name for name, k in disagreements.items() if k >= SCREEN_DROP_AT)
    public = {
        "schema": "decision2.ib1.screen.v1",
        "n": len(scored),
        "per_family": SCREEN_PER_FAMILY,
        "drop_at": SCREEN_DROP_AT,
        "disagreements_by_family": disagreements,
        "families_dropped": dropped,
        "families_kept": sorted(set(by_family) - set(dropped)),
        "disagreements": sum(disagreements.values()),
        "defect_flags": sum(i["defect"] for i in scored),
        "confidence": dict(collections.Counter(i["confidence"] for i in scored)),
    }
    private = {
        "schema": "decision2.ib1.screen-private.v1",
        "disagreements": [i for i in scored if not i["agree"]],
        "items": scored,
    }
    return public, private


# --------------------------------------------------------------------------- stage R


def review_sample(
    rows: Sequence[Mapping[str, Any]],
    rows_sha256: str,
    screen_key: Sequence[Mapping[str, Any]],
    drop: set[str],
) -> dict[str, Any]:
    eligible = [r for r in rows if r["family"] not in drop]
    families = sorted({r["family"] for r in eligible})
    per_family = max(REVIEW_MIN_PER_FAMILY, math.ceil(REVIEW_MIN_TOTAL / len(families)))
    picked = draw(
        eligible,
        per_family,
        SALT,
        skip_ids={k["id"] for k in screen_key},
        skip_groups={k["group_id"] for k in screen_key},
    )
    items, key = packets(picked, PACKET_SALT, "q")
    size = math.ceil(len(items) / PACKETS)
    packets_r1 = [items[i : i + size] for i in range(0, len(items), size)]
    packets_r2 = [
        sorted(chunk, key=lambda item: sha(f"{R2_SALT}:{item['rid']}"))
        for chunk in packets_r1
    ]
    return {
        "packets_r1": packets_r1,
        "packets_r2": packets_r2,
        "key": key,
        "sample": {
            "schema": "decision2.ib1.review-sample.v1",
            "rows_sha256": rows_sha256,
            "rows": len(rows),
            "salt": SALT,
            "per_family": per_family,
            "n": len(key),
            "families": families,
            "families_excluded": sorted(drop),
            "population": population(eligible, families),
            "sampled": dict(
                sorted(collections.Counter(k["family"] for k in key).items())
            ),
            "cells": dict(
                sorted(
                    collections.Counter(
                        f"{k['family']}|{k['gold']}" for k in key
                    ).items()
                )
            ),
            "screened_rows_excluded": len(screen_key),
            "thresholds": THRESHOLDS,
        },
    }


def items(
    key: Sequence[Mapping[str, Any]],
    r1: Mapping[str, Mapping[str, Any]],
    r2: Mapping[str, Mapping[str, Any]],
    r3: Mapping[str, Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    out = []
    for row in key:
        rid = row["rid"]
        a = answer(r1[rid], row["keys"], row["task_type"])
        b = answer(r2[rid], row["keys"], row["task_type"])
        ok_a, ok_b = agrees(a, row), agrees(b, row)
        c = None
        if ok_a != ok_b:
            if r3 is None or rid not in r3:
                raise ValueError(f"{rid}: R1 / R2 split without an R3 answer")
            c = answer(r3[rid], row["keys"], row["task_type"])
            error = not agrees(c, row)
        else:
            error = not ok_a
        out.append(
            {
                **row,
                "r1": a,
                "r2": b,
                "r3": c,
                "error": error,
                "r1_confidence": str(r1[rid].get("confidence", "")).lower(),
                "r2_confidence": str(r2[rid].get("confidence", "")).lower(),
                "defect": [bool(r1[rid].get("defect")), bool(r2[rid].get("defect"))],
                "notes": [str(r1[rid].get("note", "")), str(r2[rid].get("note", ""))]
                + ([str(r3[rid].get("note", ""))] if c is not None else []),
            }
        )
    return out


def verdict(
    scored: Sequence[Mapping[str, Any]], population_: Mapping[str, int]
) -> dict[str, Any]:
    n, errors = len(scored), sum(i["error"] for i in scored)
    _, upper = clopper_pearson(errors, n)
    by_family: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for item in scored:
        by_family[item["family"]].append(item)
    weighted = weighted_error(by_family, population_)
    family_errors = {
        name: sum(i["error"] for i in group)
        for name, group in sorted(by_family.items())
    }
    lower = {
        name: clopper_pearson(family_errors[name], len(group))[0]
        for name, group in sorted(by_family.items())
    }
    failing = sorted(
        name for name, value in lower.items() if value > THRESHOLDS["family_lower_max"]
    )
    p1 = (
        errors / n <= THRESHOLDS["error_max"] and upper <= THRESHOLDS["error_upper_max"]
    )
    p2 = weighted <= THRESHOLDS["weighted_error_max"]
    return {
        "n": n,
        "errors": errors,
        "error": round(errors / n, 6),
        "error_cp95_upper": round(upper, 6),
        "weighted_error": round(weighted, 6),
        "family_errors": family_errors,
        "family_n": {name: len(group) for name, group in sorted(by_family.items())},
        "failing_families": failing,
        "P1": p1,
        "P2": p2,
        "P3": not failing,
        "verdict": "PASS" if p1 and p2 and not failing else "FAIL",
    }


def score(
    sample_info: Mapping[str, Any],
    key: Sequence[Mapping[str, Any]],
    r1: Mapping[str, Mapping[str, Any]],
    r2: Mapping[str, Mapping[str, Any]],
    r3: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    scored = items(key, r1, r2, r3)
    pop = sample_info["population"]
    by_family: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for item in scored:
        by_family[item["family"]].append(item)
    first = verdict(scored, pop)
    weighted_ci = stratified_bootstrap(
        by_family, lambda draw_: weighted_error(draw_, pop)
    )
    n = len(scored)
    splits = [i for i in scored if i["r3"] is not None]
    report = {
        "schema": "decision2.ib1.review.v1",
        "sample": {
            k: sample_info[k]
            for k in (
                "rows_sha256",
                "rows",
                "salt",
                "n",
                "per_family",
                "families",
                "families_excluded",
            )
        },
        "thresholds": THRESHOLDS,
        "verdict": first,
        "error_unweighted": proportion(first["errors"], n),
        "error_weighted": {
            "rate": first["weighted_error"],
            "bootstrap95": [round(weighted_ci[0], 6), round(weighted_ci[1], 6)],
            "reps": BOOT_REPS,
            "salt": BOOT_SALT,
        },
        "agreement_with_gold": {
            "r1": proportion(sum(agrees(i["r1"], i) for i in scored), n),
            "r2": proportion(sum(agrees(i["r2"], i) for i in scored), n),
            "majority": proportion(sum(not i["error"] for i in scored), n),
            "r3_on_splits": proportion(
                sum(agrees(i["r3"], i) for i in splits), len(splits)
            ),
        },
        "inter_reviewer": {
            "raw": proportion(sum(i["r1"] == i["r2"] for i in scored), n),
            "kappa": round(
                cohen_kappa([i["r1"] for i in scored], [i["r2"] for i in scored]), 6
            ),
            "splits": len(splits),
        },
        "defect_flags": {
            "any": sum(any(i["defect"]) for i in scored),
            "both": sum(all(i["defect"]) for i in scored),
        },
        "by_family": breakdown(scored, "family"),
        "by_language": breakdown(scored, "language"),
        "by_task_type": breakdown(scored, "task_type"),
        "confidence": {
            "r1": dict(collections.Counter(i["r1_confidence"] for i in scored)),
            "r2": dict(collections.Counter(i["r2_confidence"] for i in scored)),
        },
    }
    private = {
        "schema": "decision2.ib1.review-private.v1",
        "errors": [i for i in scored if i["error"]],
        "splits": splits,
        "items": scored,
    }
    return report, private


def read_drop(paths: Sequence[Path]) -> set[str]:
    out: set[str] = set()
    for path in paths:
        out |= {
            line.strip()
            for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s1 = sub.add_parser("screen-sample")
    s1.add_argument("--train", type=Path, required=True)
    s1.add_argument("--drop-families", type=Path, action="append", default=[])
    s1.add_argument("--out-dir", type=Path, required=True)
    s2 = sub.add_parser("screen-score")
    s2.add_argument("--key", type=Path, required=True)
    s2.add_argument("--answers", type=Path, action="append", required=True)
    s2.add_argument("--out", type=Path, required=True)
    s2.add_argument("--private", type=Path, required=True)
    s2.add_argument("--drop-families-out", type=Path, required=True)
    s2.add_argument("--drop-ids-out", type=Path, required=True)
    r1 = sub.add_parser("sample")
    r1.add_argument("--train", type=Path, required=True)
    r1.add_argument("--screen-key", type=Path, required=True)
    r1.add_argument("--drop-families", type=Path, action="append", default=[])
    r1.add_argument("--out-dir", type=Path, required=True)
    r2 = sub.add_parser("splits")
    r3 = sub.add_parser("score")
    for p in (r2, r3):
        p.add_argument("--key", type=Path, required=True)
        p.add_argument("--r1", type=Path, action="append", required=True)
        p.add_argument("--r2", type=Path, action="append", required=True)
    r2.add_argument("--packets", type=Path, action="append", required=True)
    r2.add_argument("--out", type=Path, required=True)
    r3.add_argument("--sample", type=Path, required=True)
    r3.add_argument("--r3", type=Path, action="append", default=[])
    r3.add_argument("--out", type=Path, required=True)
    r3.add_argument("--private", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "screen-sample":
        rows, digest = train_rows(args.train)
        built = screen_sample(rows, digest, read_drop(args.drop_families))
        args.out_dir.mkdir(mode=0o700)
        for index, packet in enumerate(built["packets"], 1):
            write_packet(args.out_dir / f"packet.s1.{index}.jsonl", packet)
        write_jsonl(args.out_dir / "key.jsonl", built["key"])
        write_json(args.out_dir / "sample.json", built["sample"])
        print(json.dumps({"n": built["sample"]["n"], **built["sample"]["sampled"]}))
        return 0
    if args.command == "screen-score":
        key = read_jsonl(args.key)
        answers = read_answers(args.answers, [row["rid"] for row in key])
        public, private = screen_score(key, answers)
        write_json(args.out, public)
        write_json(args.private, private)
        write_new(
            args.drop_families_out,
            "".join(f + "\n" for f in public["families_dropped"]),
        )
        write_new(
            args.drop_ids_out, "".join(i["id"] + "\n" for i in private["disagreements"])
        )
        print(json.dumps({k: public[k] for k in ("disagreements", "families_dropped")}))
        return 0
    if args.command == "sample":
        rows, digest = train_rows(args.train)
        built = review_sample(
            rows, digest, read_jsonl(args.screen_key), read_drop(args.drop_families)
        )
        args.out_dir.mkdir(mode=0o700)
        for order in ("r1", "r2"):
            for index, packet in enumerate(built[f"packets_{order}"], 1):
                write_packet(args.out_dir / f"packet.{order}.{index}.jsonl", packet)
        write_jsonl(args.out_dir / "key.jsonl", built["key"])
        write_json(args.out_dir / "sample.json", built["sample"])
        print(
            json.dumps(
                {"n": built["sample"]["n"], "per_family": built["sample"]["per_family"]}
            )
        )
        return 0
    key = read_jsonl(args.key)
    rids = [row["rid"] for row in key]
    ans1 = read_answers(args.r1, rids)
    ans2 = read_answers(args.r2, rids)
    if args.command == "splits":
        by_rid = {
            item["rid"]: item for path in args.packets for item in read_jsonl(path)
        }
        wanted = split_rids(key, ans1, ans2)
        packet = [
            by_rid[rid]
            for rid in sorted(wanted, key=lambda rid: sha(f"{R3_SALT}:{rid}"))
        ]
        assert_blind(packet, FIELDS, {row["id"] for row in key})
        write_packet(args.out, packet)
        print(json.dumps({"splits": len(packet)}))
        return 0
    splits = split_rids(key, ans1, ans2)
    ans3 = read_answers(args.r3, splits) if args.r3 else None
    report, private = score(json.loads(args.sample.read_text()), key, ans1, ans2, ans3)
    write_json(args.out, report)
    write_json(args.private, private)
    print(json.dumps({"verdict": report["verdict"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
