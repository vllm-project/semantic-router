"""HR2 blind label review (prereg ``records/hr2-prereg-2026-09-30.md`` §4).

    python3 -m v2.data.hr2.review sample --train TRAIN --out-dir DIR
    python3 -m v2.data.hr2.review splits --key DIR/key.jsonl --r1 A1 --r1 A2 --r2 B1 --r2 B2 \
        --out DIR3/r3.packet.jsonl
    python3 -m v2.data.hr2.review score --sample DIR/sample.json --key DIR/key.jsonl \
        --r1 A1 --r1 A2 --r2 B1 --r2 B2 [--r3 C] --out REPORT.json --private PRIVATE.json

``sample`` draws 24 TRAIN rows per family, one per group, stratified by gold, in salted-hash order,
and writes two blind packets per reviewer order (R1 order and R2 order) plus the key, which stays on
the node. An item agrees with gold on the same key (Choice / Noul) or within one level (Score). R1
and R2 split when exactly one of them agrees; ``splits`` writes those items, in a fresh order, as the
R3 packet (it needs the key, so it runs on the node). A gold error is an item both R1 and R2
disagree with, or a split R3 disagrees with.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from collections.abc import Mapping, Sequence
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
)

PER_FAMILY = 24
SALT = "hr2-review-v1"
PACKET_SALT = "hr2-packet-v1"
R2_SALT = "hr2-review-r2-order-v1"
R3_SALT = "hr2-review-r3-order-v1"
PACKETS = 2
FIELDS = ("rid", "task_type", "instructions", "state", "options")
THRESHOLDS = {
    "error_max": 0.05,
    "error_upper_max": 0.08,
    "weighted_error_max": 0.05,
    "family_errors_fail": 5,
}
NOUL_ALIASES = {"yes": "true", "no": "false"}


def gold_key(row: Mapping[str, Any]) -> str:
    return str(row["options"][row["label"]]["key"])


def cell_cap(row: Mapping[str, Any]) -> int:
    return math.ceil(PER_FAMILY / len(row["options"]))


def sample(rows: Sequence[Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], dict]:
    """``PER_FAMILY`` rows per family, one per group overall, gold cells capped evenly."""
    population = collections.Counter()
    for row in rows:
        if row.get("split") != "train":
            raise ValueError(f"{row.get('id')}: not a TRAIN row")
        population[row["family"]] += 1
    taken: collections.Counter = collections.Counter()
    cells: collections.Counter = collections.Counter()
    used: set[str] = set()
    picked = []
    for row in sorted(rows, key=lambda r: sha(f"{SALT}:{r['id']}")):
        cell = (row["family"], row["label"])
        if (
            taken[row["family"]] >= PER_FAMILY
            or cells[cell] >= cell_cap(row)
            or row["group_id"] in used
        ):
            continue
        picked.append(row)
        taken[row["family"]] += 1
        cells[cell] += 1
        used.add(row["group_id"])
    return picked, dict(sorted(population.items()))


def packet_item(rid: str, row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "rid": rid,
        "task_type": row["task_type"],
        "instructions": row["instructions"],
        "state": row["state"],
        "options": [
            {"key": str(o["key"]), "description": o["description"]}
            for o in row["options"]
        ],
    }


def key_item(rid: str, row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "rid": rid,
        "id": row["id"],
        "group_id": row["group_id"],
        "family": row["family"],
        "source": row["source"],
        "language": row["language"],
        "task_type": row["task_type"],
        "label": row["label"],
        "gold": gold_key(row),
        "keys": [str(o["key"]) for o in row["options"]],
    }


def build(rows: Sequence[Mapping[str, Any]], rows_sha256: str) -> dict[str, Any]:
    picked, population = sample(rows)
    ordered = sorted(picked, key=lambda r: sha(f"{PACKET_SALT}:{r['id']}"))
    rids = {row["id"]: f"h{index + 1:03d}" for index, row in enumerate(ordered)}
    items = [packet_item(rids[row["id"]], row) for row in ordered]
    key = [key_item(rids[row["id"]], row) for row in ordered]
    secrets = (
        {row["id"] for row in ordered}
        | {row["group_id"] for row in ordered}
        | {row["source"] for row in ordered}
    )
    assert_blind(items, FIELDS, secrets)
    size = math.ceil(len(items) / PACKETS)
    packets_r1 = [items[i : i + size] for i in range(0, len(items), size)]
    packets_r2 = [
        sorted(chunk, key=lambda item: sha(f"{R2_SALT}:{item['rid']}"))
        for chunk in packets_r1
    ]
    sampled = collections.Counter(item["family"] for item in key)
    cells = collections.Counter(f"{item['family']}|{item['gold']}" for item in key)
    return {
        "packets_r1": packets_r1,
        "packets_r2": packets_r2,
        "key": key,
        "sample": {
            "schema": "decision2.hr2.review-sample.v1",
            "rows_sha256": rows_sha256,
            "rows": len(rows),
            "salt": SALT,
            "per_family": PER_FAMILY,
            "n": len(key),
            "population": population,
            "sampled": dict(sorted(sampled.items())),
            "cells": dict(sorted(cells.items())),
            "thresholds": THRESHOLDS,
        },
    }


# --------------------------------------------------------------------------- answers


def answer(item: Mapping[str, Any], keys: Sequence[str], task_type: str) -> str:
    value = str(item.get("answer", "")).strip().lower()
    if task_type == "noul":
        value = NOUL_ALIASES.get(value, value)
    if value not in keys:
        raise ValueError(f"{item.get('rid')}: answer {value!r} is not one of {keys}")
    return value


def agrees(value: str, row: Mapping[str, Any]) -> bool:
    if row["task_type"] == "score":
        return abs(int(value) - int(row["gold"])) <= 1
    return value == row["gold"]


def split_rids(
    key: Sequence[Mapping[str, Any]],
    r1: Mapping[str, Mapping[str, Any]],
    r2: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    out = []
    for row in key:
        a = answer(r1[row["rid"]], row["keys"], row["task_type"])
        b = answer(r2[row["rid"]], row["keys"], row["task_type"])
        if agrees(a, row) != agrees(b, row):
            out.append(row["rid"])
    return sorted(out, key=lambda rid: sha(f"{R3_SALT}:{rid}"))


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
                "defect": [
                    bool(r.get("defect")) for r in (r1[rid], r2[rid]) if r is not None
                ],
                "notes": [str(r1[rid].get("note", "")), str(r2[rid].get("note", ""))]
                + ([str(r3[rid].get("note", ""))] if c is not None else []),
            }
        )
    return out


def verdict(
    scored: Sequence[Mapping[str, Any]], population: Mapping[str, int]
) -> dict[str, Any]:
    n, errors = len(scored), sum(i["error"] for i in scored)
    _, upper = clopper_pearson(errors, n)
    by_family: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for item in scored:
        by_family[item["family"]].append(item)
    weighted = weighted_error(by_family, population)
    family_errors = {
        name: sum(i["error"] for i in group)
        for name, group in sorted(by_family.items())
    }
    failing = sorted(
        name
        for name, k in family_errors.items()
        if k >= THRESHOLDS["family_errors_fail"]
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
        "failing_families": failing,
        "P1": p1,
        "P2": p2,
        "P3": not failing,
        "verdict": "PASS" if p1 and p2 and not failing else "FAIL",
    }


def breakdown(scored: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for item in scored:
        groups[str(item[field])].append(item)
    return {
        name: proportion(sum(i["error"] for i in group), len(group))
        for name, group in sorted(groups.items())
    }


def score(
    sample_info: Mapping[str, Any],
    key: Sequence[Mapping[str, Any]],
    r1: Mapping[str, Mapping[str, Any]],
    r2: Mapping[str, Mapping[str, Any]],
    r3: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    scored = items(key, r1, r2, r3)
    population = sample_info["population"]
    by_family: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for item in scored:
        by_family[item["family"]].append(item)
    first = verdict(scored, population)
    fixed = None
    if first["verdict"] == "FAIL" and first["failing_families"]:
        rest = [i for i in scored if i["family"] not in first["failing_families"]]
        fixed = verdict(rest, population) if rest else None
    weighted_ci = stratified_bootstrap(
        by_family, lambda draw: weighted_error(draw, population)
    )
    kappa = cohen_kappa([i["r1"] for i in scored], [i["r2"] for i in scored])
    n = len(scored)
    splits = [i for i in scored if i["r3"] is not None]
    distance: dict[str, collections.Counter] = {
        "r1": collections.Counter(),
        "r2": collections.Counter(),
    }
    for item in scored:
        if item["task_type"] == "score":
            for who in ("r1", "r2"):
                distance[who][str(abs(int(item[who]) - int(item["gold"])))] += 1
    report = {
        "schema": "decision2.hr2.review.v1",
        "sample": {k: sample_info[k] for k in ("rows_sha256", "rows", "salt", "n")},
        "thresholds": THRESHOLDS,
        "verdict": first,
        "fix_rule_f1": fixed,
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
            "kappa": round(kappa, 6),
            "splits": len(splits),
        },
        "score_distance": {k: dict(sorted(v.items())) for k, v in distance.items()},
        "defect_flags": {
            "any": sum(any(i["defect"]) for i in scored),
            "both": sum(all(i["defect"]) for i in scored),
        },
        "by_family": breakdown(scored, "family"),
        "by_language": breakdown(scored, "language"),
        "by_task_type": breakdown(scored, "task_type"),
        "by_gold": {
            name: breakdown([i for i in scored if i["task_type"] == kind], "gold")
            for name, kind in (
                ("choice", "choice"),
                ("noul", "noul"),
                ("score", "score"),
            )
        },
        "confidence": {
            "r1": dict(collections.Counter(i["r1_confidence"] for i in scored)),
            "r2": dict(collections.Counter(i["r2_confidence"] for i in scored)),
        },
    }
    private = {
        "schema": "decision2.hr2.review-private.v1",
        "errors": [i for i in scored if i["error"]],
        "splits": splits,
        "items": scored,
    }
    return report, private


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("sample")
    one.add_argument("--train", type=Path, required=True)
    one.add_argument("--out-dir", type=Path, required=True)
    two = sub.add_parser("splits")
    three = sub.add_parser("score")
    for p in (two, three):
        p.add_argument("--key", type=Path, required=True)
        p.add_argument("--r1", type=Path, action="append", required=True)
        p.add_argument("--r2", type=Path, action="append", required=True)
    two.add_argument("--packets", type=Path, action="append", required=True)
    two.add_argument("--out", type=Path, required=True)
    three.add_argument("--sample", type=Path, required=True)
    three.add_argument("--r3", type=Path, action="append", default=[])
    three.add_argument("--out", type=Path, required=True)
    three.add_argument("--private", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "sample":
        data = args.train.read_bytes()
        rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line]
        built = build(rows, sha(data.decode("utf-8")))
        args.out_dir.mkdir(mode=0o700)
        for order in ("r1", "r2"):
            for index, packet in enumerate(built[f"packets_{order}"], 1):
                write_jsonl(args.out_dir / f"packet.{order}.{index}.jsonl", packet)
        write_jsonl(args.out_dir / "key.jsonl", built["key"])
        write_json(args.out_dir / "sample.json", built["sample"])
        print(json.dumps({"n": built["sample"]["n"], **built["sample"]["sampled"]}))
        return 0
    key = read_jsonl(args.key)
    rids = [row["rid"] for row in key]
    r1 = read_answers(args.r1, rids)
    r2 = read_answers(args.r2, rids)
    if args.command == "splits":
        wanted = set(split_rids(key, r1, r2))
        by_rid = {
            item["rid"]: item for path in args.packets for item in read_jsonl(path)
        }
        packet = [by_rid[rid] for rid in split_rids(key, r1, r2)]
        assert_blind(packet, FIELDS, {row["id"] for row in key})
        write_jsonl(args.out, packet)
        print(json.dumps({"splits": len(wanted)}))
        return 0
    splits = split_rids(key, r1, r2)
    r3 = read_answers(args.r3, splits) if args.r3 else None
    report, private = score(json.loads(args.sample.read_text()), key, r1, r2, r3)
    write_json(args.out, report)
    write_json(args.private, private)
    print(json.dumps({"verdict": report["verdict"], "fix": report["fix_rule_f1"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
