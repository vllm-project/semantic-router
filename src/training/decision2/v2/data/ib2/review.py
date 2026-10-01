"""IB2 blind label review in two stages (prereg ``records/ib2-prereg-2026-10-01.md`` §4).

IB1's review code (``v2.data.ib1.review``) with IB2's salts and item prefixes; the scoring rules are IB1's unchanged.

    python3 -m v2.data.ib2.review screen-sample --train TRAIN [--drop-families F ...] --out-dir S
    python3 -m v2.data.ib2.review screen-score --key S/key.jsonl --answers A [--answers ...] \
        --out PUBLIC --private PRIVATE --drop-families-out F --drop-ids-out IDS
    python3 -m v2.data.ib2.review sample --train TRAIN --screen-key S/key.jsonl [--drop-families F ...] --out-dir R
    python3 -m v2.data.ib2.review splits --key R/key.jsonl --r1 .. --r2 .. --packets .. --out R3PACKET
    python3 -m v2.data.ib2.review score --sample R/sample.json --key R/key.jsonl --r1 .. --r2 .. [--r3 ..] \
        --out REPORT --private PRIVATE
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
    assert_blind,
    read_answers,
    read_jsonl,
    sha,
    write_json,
    write_jsonl,
    write_new,
)
from v2.data.hr2.review import split_rids, write_packet
from v2.data.ib1 import review as ib1

SCREEN_SALT = "ib2-screen-v1"
SCREEN_PACKET_SALT = "ib2-screen-packet-v1"
MIN_PER_FAMILY = 18
MIN_TOTAL = 216
SALT = "ib2-review-v1"
PACKET_SALT = "ib2-packet-v1"
R2_SALT = "ib2-review-r2-order-v1"
R3_SALT = "ib2-review-r3-order-v1"
PREFIX = "u"


def screen_sample(
    rows: Sequence[Mapping[str, Any]], rows_sha256: str, drop: set[str]
) -> dict[str, Any]:
    eligible = [r for r in rows if r["family"] not in drop]
    picked = ib1.draw(eligible, ib1.SCREEN_PER_FAMILY, SCREEN_SALT)
    items, key = ib1.packets(picked, SCREEN_PACKET_SALT, "s")
    size = math.ceil(len(items) / ib1.PACKETS)
    return {
        "packets": [items[i : i + size] for i in range(0, len(items), size)],
        "key": key,
        "sample": {
            "schema": "decision2.ib2.screen-sample.v1",
            "rows_sha256": rows_sha256,
            "rows": len(rows),
            "salt": SCREEN_SALT,
            "per_family": ib1.SCREEN_PER_FAMILY,
            "n": len(key),
            "families_excluded": sorted(drop),
            "sampled": dict(
                sorted(collections.Counter(k["family"] for k in key).items())
            ),
        },
    }


def review_sample(
    rows: Sequence[Mapping[str, Any]],
    rows_sha256: str,
    screen_key: Sequence[Mapping[str, Any]],
    drop: set[str],
) -> dict[str, Any]:
    eligible = [r for r in rows if r["family"] not in drop]
    families = sorted({r["family"] for r in eligible})
    per_family = max(MIN_PER_FAMILY, math.ceil(MIN_TOTAL / len(families)))
    picked = ib1.draw(
        eligible,
        per_family,
        SALT,
        skip_ids={k["id"] for k in screen_key},
        skip_groups={k["group_id"] for k in screen_key},
        present_classes=True,
    )
    items, key = ib1.packets(picked, PACKET_SALT, PREFIX)
    size = math.ceil(len(items) / ib1.PACKETS)
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
            "schema": "decision2.ib2.review-sample.v1",
            "rows_sha256": rows_sha256,
            "rows": len(rows),
            "salt": SALT,
            "per_family": per_family,
            "n": len(key),
            "families": families,
            "families_excluded": sorted(drop),
            "population": ib1.population(eligible, families),
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
            "excluded_key_rows": len(screen_key),
            "thresholds": ib1.THRESHOLDS,
        },
    }


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
    r1.add_argument("--screen-key", type=Path, action="append", required=True)
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
        rows, digest = ib1.train_rows(args.train)
        built = screen_sample(rows, digest, ib1.read_drop(args.drop_families))
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
        public, private = ib1.screen_score(key, answers)
        public["schema"], private["schema"] = (
            "decision2.ib2.screen.v1",
            "decision2.ib2.screen-private.v1",
        )
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
        rows, digest = ib1.train_rows(args.train)
        excluded = [item for path in args.screen_key for item in read_jsonl(path)]
        built = review_sample(rows, digest, excluded, ib1.read_drop(args.drop_families))
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
        assert_blind(packet, ib1.FIELDS, {row["id"] for row in key})
        write_packet(args.out, packet)
        print(json.dumps({"splits": len(packet)}))
        return 0
    splits = split_rids(key, ans1, ans2)
    ans3 = read_answers(args.r3, splits) if args.r3 else None
    report, private = ib1.score(
        json.loads(args.sample.read_text()), key, ans1, ans2, ans3
    )
    report["schema"], private["schema"] = (
        "decision2.ib2.review.v1",
        "decision2.ib2.review-private.v1",
    )
    write_json(args.out, report)
    write_json(args.private, private)
    print(json.dumps({"verdict": report["verdict"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
