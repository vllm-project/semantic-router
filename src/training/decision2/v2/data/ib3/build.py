"""Build and finalize IB3 (prereg ``records/ib3-prereg-2026-10-01.md`` §2–§3).

    python3 -m v2.data.ib3.build build --raw RAW --out CAND
    python3 -m v2.data.ib3.build finalize --cand CAND --out FINAL \
        [--drop-groups FILE ...] [--drop-dev-groups FILE ...] [--drop-families NAME ...] \
        [--drop-ids FILE ...] [--drop-leak-ids FILE ...]

``build`` reads the pinned publisher files under RAW, converts every family, merges groups that share a
normalized state, removes exact duplicates and assigns the group-isolated DEV slice. ``finalize`` only removes rows
from the candidates (Index-row, Index-URL and quarantined groups, DEV near-duplicates of TRAIN, dropped families,
screen and review findings, leak-guard rows) and re-balances yes = no inside every declared cell, so every final row
was scanned as a candidate. The generic helpers (hashing, writing, group merging, de-duplication) are IB1's.
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from training.model.data import validate_row
from v2.common import eval_only
from v2.data.ib1.build import (
    counts,
    dedup,
    file_sha256,
    merge_groups,
    parquet,
    read_list,
    read_rows,
    write_json,
    write_rows,
)
from v2.data.ib3 import families as fam
from v2.data.sources.common import sha

DEV_SALT = "ib3-dev-v1"
BALANCE_SALT = "ib3-final-v1"
PINS = {
    "mendeley_wpd/dataset_B_05_2020.csv": "21093e2902e5441c86a6daf95e86e7c332046e477fdf109a579d7bd81e586d6c",
    "uci_phiusiil/phiusiil+phishing+url+dataset.zip": "0a639fd03aea6308c5b1c10c92aa23c2ce1505447a9137271865cd0badc9a59a",
    "mcgill_faithdial/data/train.json": "51eff8212d804b1954b7eefa8987265750f0412b0b18b534ed2dbe636e524512",
    "rucaibox_halueval/data/qa_data.json": "89ed139ec5e3a3169a0b30e45569ac1283846f76f27f7bb5e908ee6deed57e88",
    "amazon_esci/shopping_queries_dataset_examples.parquet": "4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a",
    "amazon_esci/shopping_queries_dataset_products.parquet": "25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265",
    "mathqa/train.json": "00e8919347d65dbba9289bf04ed998a6c48dbf451ca909eeb66a35f2419c2bf6",
    "theatticusproject_maud/MAUD_v1/MAUD_train.csv": "bac9f2d034ad487d5398ee2ac1c876679afea509ba8e7f1092112955c6180ff9",
}
# Amendment 1 removed `maud`; amendment 2 removed `fdial` and `haluqa` (G4 on run d1) and added `fdial2`.
FAMILIES = ("wpd", "phiu", "fdial2", "esci", "mqa")


def csv_rows(text: str) -> list[dict[str, str]]:
    csv.field_size_limit(1 << 30)
    rows = list(csv.DictReader(io.StringIO(text)))
    return [{k.lstrip("\ufeff"): v for k, v in row.items()} for row in rows]


def esci_products(
    path: Path, wanted: set[tuple[str, str]]
) -> dict[tuple[str, str], dict[str, Any]]:
    import pyarrow.parquet as pq

    columns = [
        "product_id",
        "product_locale",
        "product_title",
        "product_bullet_point",
        "product_brand",
        "product_color",
    ]
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=200_000):
        for item in batch.to_pylist():
            key = (str(item["product_id"]), str(item["product_locale"]))
            if key in wanted:
                out[key] = item
    return out


# --------------------------------------------------------------------------- build


def convert(raw: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reports: dict[str, collections.Counter] = {
        name: collections.Counter() for name in FAMILIES
    }
    rows: list[dict[str, Any]] = []
    rows += fam.wpd(
        csv_rows((raw / "mendeley_wpd/dataset_B_05_2020.csv").read_text("utf-8")),
        reports["wpd"],
    )
    with zipfile.ZipFile(
        raw / "uci_phiusiil/phiusiil+phishing+url+dataset.zip"
    ) as archive:
        rows += fam.phiu(
            csv_rows(archive.read("PhiUSIIL_Phishing_URL_Dataset.csv").decode("utf-8")),
            reports["phiu"],
        )
    rows += fam.fdial2(
        json.loads((raw / "mcgill_faithdial/data/train.json").read_text("utf-8")),
        reports["fdial2"],
    )
    examples = parquet(
        raw / "amazon_esci/shopping_queries_dataset_examples.parquet",
        [
            "example_id",
            "query",
            "query_id",
            "product_id",
            "product_locale",
            "esci_label",
            "split",
        ],
    )
    examples = [
        e
        for e in examples
        if e["split"] == "train" or reports["esci"].update(["drop_not_train"])
    ]
    wanted = {(str(e["product_id"]), str(e["product_locale"])) for e in examples}
    rows += fam.esci(
        examples,
        esci_products(
            raw / "amazon_esci/shopping_queries_dataset_products.parquet", wanted
        ),
        reports["esci"],
    )
    del examples
    rows += fam.mqa(
        json.loads((raw / "mathqa/train.json").read_text("utf-8")), reports["mqa"]
    )
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
        "schema": "decision2.ib3.build.v1",
        "prereg": "records/ib3-prereg-2026-10-01.md",
        "inputs": inputs,
        "report": report,
        "train": {
            **counts(train),
            "sha256": write_rows(args.out / "ib3.train.cand.jsonl", train),
        },
        "dev": {
            **counts(dev),
            "sha256": write_rows(args.out / "ib3.dev.cand.jsonl", dev),
        },
    }
    write_json(args.out / "build.json", manifest)
    print(json.dumps({k: manifest[k]["rows"] for k in ("train", "dev")}))
    return 0


# --------------------------------------------------------------------------- finalize


def rebalance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in FAMILIES:
        members = [row for row in rows if row["family"] == name]
        if members:
            out += fam.cell_balance(members, None, BALANCE_SALT)
    return out


def finalize(args: argparse.Namespace) -> int:
    eval_only.guard(args)
    train = read_rows(args.cand / "ib3.train.cand.jsonl")
    dev = read_rows(args.cand / "ib3.dev.cand.jsonl")
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

    kept_train = [row for row in train if keep(row, False)]
    kept_dev = [row for row in dev if keep(row, True)]
    train = rebalance(kept_train)
    dev = rebalance(kept_dev)
    report["train:rebalance"].update(
        collections.Counter(r["family"] for r in kept_train)
        - collections.Counter(r["family"] for r in train)
    )
    report["dev:rebalance"].update(
        collections.Counter(r["family"] for r in kept_dev)
        - collections.Counter(r["family"] for r in dev)
    )
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
        "schema": "decision2.ib3.final.v1",
        "candidates": {
            "train": file_sha256(args.cand / "ib3.train.cand.jsonl"),
            "dev": file_sha256(args.cand / "ib3.dev.cand.jsonl"),
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
            "sha256": write_rows(args.out / "ib3.train.jsonl", train),
        },
        "dev": {**counts(dev), "sha256": write_rows(args.out / "ib3.dev.jsonl", dev)},
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
