"""Build and finalize IB2 (prereg ``records/ib2-prereg-2026-10-01.md`` §2–§3).

    python3 -m v2.data.ib2.build build --raw RAW --out CAND
    python3 -m v2.data.ib2.build finalize --cand CAND --out FINAL \
        [--drop-groups FILE ...] [--drop-dev-groups FILE ...] [--drop-families NAME ...] \
        [--drop-ids FILE ...] [--drop-leak-ids FILE ...]

``build`` reads the pinned publisher files under RAW, converts every family, merges groups that share a
normalized state, removes exact duplicates and assigns the group-isolated DEV slice. ``finalize`` only removes rows
from the candidates (Index-row and quarantined groups, DEV near-duplicates of TRAIN, dropped families, screen and
review findings, leak-guard rows) and re-balances by downsampling, so every final row was scanned as a candidate.
The generic helpers (hashing, writing, group merging, de-duplication, label and position balancing) are IB1's.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import io
import json
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from training.model.data import validate_row
from v2.common import eval_only
from v2.data.ib1.build import (
    counts,
    dedup,
    equal_labels,
    file_sha256,
    merge_groups,
    parquet,
    position_band,
    read_list,
    read_rows,
    write_json,
    write_rows,
)
from v2.data.ib2 import families as fam
from v2.data.sources.common import sha

DEV_SALT = "ib2-dev-v1"
BALANCE_SALT = "ib2-final-v1"

# Publisher files read by ``build`` (relative to RAW) and their SHA-256 at download time (prereg §1.2).
PINS = {
    "glaiveai_glaive-function-calling-v2/glaive-function-calling-v2.json": "e9b5d671812b5ca2fbd7b625a37d5c99a19576c37252cdc806defe256aea6dad",
    "uci_youtube_spam/youtube+spam+collection.zip": "bd6182891adb3cfc8334b82c062176dfbebc563bf0ba07e31c2645f916865a0a",
    "ibm-research_argument_quality_ranking_30k/train.csv": "55910fd3599ec54c088d4e9c55f745ff814d153f13c1a7c23a4d31f167cf16f7",
    "hover-nlp_hover/data/hover/hover_train_release_v1.1.json": "1f1cd57abd616fa00c70bdc575ce77c16fc6cf1a6cffd5ff87c208030a336bb6",
    "hover_wiki_wo_links.db": "c37ee397916ec0bffacfe8902db454a5cda88a7a188409217b2e15231fe5ee2f",
    "allenai_qasc/data/train-00000-of-00001.parquet": "b9a297b5ab55f1605c7682ffbb7042c26d7ecb9ff1e1aa5a820d4e791c8302d1",
    "allenai_ai2_arc/ARC-Easy/train-00000-of-00001.parquet": "b315db8a4be597dc7daa50a4e70d48dd7c990c32085629e6ccd8c926beaa80b5",
    "allenai_ai2_arc/ARC-Challenge/train-00000-of-00001.parquet": "e488c1587ffdcfc8443f916c53488a95cd471c5790e0746c6bfe4cecf20962cb",
    "openai_gsm8k/main/train-00000-of-00001.parquet": "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d",
    "stanfordnlp_contract-nli/contract-nli.zip": "e03fc77bbf8b53e2976a250e81d8a294bc3d5e5fb014521e477dee9340d6287b",
}
FAMILIES = (
    "fc_rel",
    "fc_sel",
    "fc_args",
    "fc_ready",
    "ytspam",
    "argq",
    "hover",
    "qasc",
    "arc",
    "gsm",
    "cnli",
)
NOUL_FAMILIES = ("fc_rel", "fc_ready", "ytspam", "gsm")
RATIO_FAMILIES = {"hover": (fam.HOVER_YES, fam.HOVER_NO)}
HYPOTHESIS_FAMILIES = ("cnli",)
TOPIC_FAMILIES = ("argq",)
ROTATED_FAMILIES = ("fc_sel", "fc_args", "qasc", "arc")
YOUTUBE_FILES = (
    "Youtube01-Psy.csv",
    "Youtube02-KatyPerry.csv",
    "Youtube03-LMFAO.csv",
    "Youtube04-Eminem.csv",
    "Youtube05-Shakira.csv",
)


def csv_rows(text: str) -> list[dict[str, str]]:
    csv.field_size_limit(1 << 30)
    return list(csv.DictReader(io.StringIO(text)))


# --------------------------------------------------------------------------- build


def convert(raw: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reports: dict[str, collections.Counter] = {
        name: collections.Counter() for name in FAMILIES
    }
    rows: list[dict[str, Any]] = []
    glaive = json.loads(
        (
            raw / "glaiveai_glaive-function-calling-v2/glaive-function-calling-v2.json"
        ).read_text("utf-8")
    )
    for family_rows in fam.glaive(glaive, reports).values():
        rows += family_rows
    del glaive
    with zipfile.ZipFile(
        raw / "uci_youtube_spam/youtube+spam+collection.zip"
    ) as archive:
        tables = {
            name.split("-", 1)[1].removesuffix(".csv"): csv_rows(
                archive.read(name).decode("utf-8")
            )
            for name in YOUTUBE_FILES
        }
    rows += fam.ytspam(tables, reports["ytspam"])
    rows += fam.argq(
        csv_rows(
            (raw / "ibm-research_argument_quality_ranking_30k/train.csv").read_text(
                "utf-8"
            )
        ),
        reports["argq"],
    )
    rows += fam.hover(
        json.loads(
            (
                raw / "hover-nlp_hover/data/hover/hover_train_release_v1.1.json"
            ).read_text("utf-8")
        ),
        fam.hover_lookup(str(raw / "hover_wiki_wo_links.db")),
        reports["hover"],
    )
    rows += fam.qasc(
        parquet(raw / "allenai_qasc/data/train-00000-of-00001.parquet"), reports["qasc"]
    )
    arc_records = []
    for subset in ("ARC-Easy", "ARC-Challenge"):
        arc_records += [
            (subset, record)
            for record in parquet(
                raw / f"allenai_ai2_arc/{subset}/train-00000-of-00001.parquet"
            )
        ]
    rows += fam.arc(arc_records, reports["arc"])
    rows += fam.gsm(
        parquet(raw / "openai_gsm8k/main/train-00000-of-00001.parquet"), reports["gsm"]
    )
    with zipfile.ZipFile(raw / "stanfordnlp_contract-nli/contract-nli.zip") as archive:
        contracts = json.loads(archive.read("contract-nli/train.json").decode("utf-8"))
    rows += fam.cnli(contracts, reports["cnli"])
    return rows, {
        "families": {name: dict(sorted(r.items())) for name, r in reports.items()}
    }


def is_dev(group_id: str) -> bool:
    return int(sha(f"{DEV_SALT}:{group_id}"), 16) % 10 == 0


def as_dev(row: Mapping[str, Any]) -> dict[str, Any]:
    return validate_row(dict(row, split="select", evaluation_role="select"), "select")


def build(args: argparse.Namespace) -> int:
    eval_only.guard(args)
    raw = args.raw
    inputs = {}
    for rel, expected in PINS.items():
        actual = file_sha256(raw / rel)
        if actual != expected:
            raise ValueError(f"{rel}: sha256 {actual} != pinned {expected}")
        inputs[rel] = actual
    rows, report = convert(raw)
    unique, report["duplicates"] = dedup(rows)
    report["group_merges"] = merge_groups(unique)
    train = [row for row in unique if not is_dev(row["group_id"])]
    dev = [as_dev(row) for row in unique if is_dev(row["group_id"])]
    eval_only.check_rows(train + dev)
    args.out.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest = {
        "schema": "decision2.ib2.build.v1",
        "prereg": "records/ib2-prereg-2026-10-01.md",
        "inputs": inputs,
        "report": report,
        "train": {
            **counts(train),
            "sha256": write_rows(args.out / "ib2.train.cand.jsonl", train),
        },
        "dev": {
            **counts(dev),
            "sha256": write_rows(args.out / "ib2.dev.cand.jsonl", dev),
        },
    }
    write_json(args.out / "build.json", manifest)
    print(json.dumps({k: manifest[k]["rows"] for k in ("train", "dev")}))
    return 0


# --------------------------------------------------------------------------- finalize


def cells(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by[row["audit_metadata"]["ib2"]["cell"]].append(row)
    return by


def rebalance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in FAMILIES:
        members = [row for row in rows if row["family"] == name]
        if not members:
            continue
        if name in NOUL_FAMILIES:
            out += equal_labels(members, BALANCE_SALT)
        elif name in RATIO_FAMILIES:
            yes, no = RATIO_FAMILIES[name]
            out += fam.ratio_labels(members, yes, no, None, BALANCE_SALT)
        elif name in HYPOTHESIS_FAMILIES:
            for hyp, cell in sorted(cells(members).items()):
                out += equal_labels(cell, BALANCE_SALT)
        elif name in TOPIC_FAMILIES:
            out += fam.topic_twins(members, BALANCE_SALT, None)
        elif name in ROTATED_FAMILIES:
            out += position_band(members, BALANCE_SALT + ":pos")
        else:
            raise ValueError(f"no balance rule for {name}")
    return out


def finalize(args: argparse.Namespace) -> int:
    eval_only.guard(args)
    train = read_rows(args.cand / "ib2.train.cand.jsonl")
    dev = read_rows(args.cand / "ib2.dev.cand.jsonl")
    drop_groups = read_list(args.drop_groups)
    drop_dev = read_list(args.drop_dev_groups)
    drop_ids = read_list(args.drop_ids)
    drop_leak = read_list(args.drop_leak_ids)
    drop_families = set(args.drop_families)
    report: dict[str, Any] = collections.defaultdict(collections.Counter)

    def keep(row: Mapping[str, Any], dev_slice: bool) -> bool:
        reason = None
        if row["family"] in drop_families:
            reason = "family"
        elif row["group_id"] in drop_groups:
            reason = "dropped_group"
        elif dev_slice and row["group_id"] in drop_dev:
            reason = "dev_near_train"
        elif row["id"] in drop_ids:
            reason = "review_or_screen"
        elif row["id"] in drop_leak:
            reason = "leak_guard"
        if reason:
            report[f"{'dev' if dev_slice else 'train'}:{reason}"][row["family"]] += 1
        return reason is None

    train = rebalance([row for row in train if keep(row, False)])
    dev = rebalance([row for row in dev if keep(row, True)])
    for row in train:
        validate_row(row, "train")
    for row in dev:
        validate_row(row, "select")
    shared = {r["group_id"] for r in train} & {r["group_id"] for r in dev}
    if shared:
        raise ValueError(f"{len(shared)} groups in both TRAIN and DEV")
    eval_only.check_rows(train + dev)
    args.out.mkdir(parents=True, exist_ok=True, mode=0o700)

    def listed(values: set[str]) -> dict[str, Any]:
        return {"count": len(values), "sha256": sha("\n".join(sorted(values)))}

    manifest = {
        "schema": "decision2.ib2.final.v1",
        "candidates": {
            "train": file_sha256(args.cand / "ib2.train.cand.jsonl"),
            "dev": file_sha256(args.cand / "ib2.dev.cand.jsonl"),
        },
        "drops": {
            "groups": listed(drop_groups),
            "dev_groups": listed(drop_dev),
            "ids": listed(drop_ids),
            "leak_ids": listed(drop_leak),
            "families": sorted(drop_families),
            "removed": {
                key: dict(sorted(value.items()))
                for key, value in sorted(report.items())
            },
        },
        "train": {
            **counts(train),
            "sha256": write_rows(args.out / "ib2.train.jsonl", train),
        },
        "dev": {**counts(dev), "sha256": write_rows(args.out / "ib2.dev.jsonl", dev)},
    }
    write_json(args.out / "final.json", manifest)
    print(json.dumps({k: manifest[k]["rows"] for k in ("train", "dev")}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("build")
    one.add_argument("--raw", required=True, type=Path)
    one.add_argument("--out", required=True, type=Path)
    two = sub.add_parser("finalize")
    two.add_argument("--cand", required=True, type=Path)
    two.add_argument("--out", required=True, type=Path)
    two.add_argument("--drop-groups", action="append", default=[], type=Path)
    two.add_argument("--drop-dev-groups", action="append", default=[], type=Path)
    two.add_argument("--drop-ids", action="append", default=[], type=Path)
    two.add_argument("--drop-leak-ids", action="append", default=[], type=Path)
    two.add_argument("--drop-families", nargs="*", default=[])
    args = parser.parse_args(argv)
    return build(args) if args.command == "build" else finalize(args)


if __name__ == "__main__":
    raise SystemExit(main())
