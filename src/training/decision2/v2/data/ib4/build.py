"""Build and finalize IB4 (prereg ``records/ib4-prereg-2026-10-02.md`` §2–§3).

    python3 -m v2.data.ib4.build build --raw RAW --prior FILE [--prior ...] --out CAND [--families NAME ...]
    python3 -m v2.data.ib4.build finalize --cand CAND --out FINAL \
        [--drop-groups FILE ...] [--drop-dev-groups FILE ...] [--drop-families NAME ...] \
        [--drop-ids FILE ...] [--drop-leak-ids FILE ...]

``build`` reads the pinned publisher files under RAW and the IB1-r3 / IB2 / IB3-r2 files (``--prior``, dedupe §2.0),
converts every family, merges groups that share a normalized state, removes exact duplicates and assigns the
group-isolated DEV slice. ``finalize`` only removes rows from the candidates and re-balances every declared cell.
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
from v2.data.ib1.index_guard import leaves
from v2.data.ib4 import families as fam
from v2.data.sources.common import sha
from v2.data.textnorm import normalize

DEV_SALT = "ib4-dev-v1"
BALANCE_SALT = "ib4-final-v1"
PINS = {
    "rajpurkar_squad_v2/squad_v2/train-00000-of-00001.parquet": "f6da32ffb482ff463ad056477740d1bb284b96a45db3a08bee6a225ca6abf291",
    "mendeley_sms_phishing/Dataset_5971.zip": "9bbf3188fdad81495d8e82825648b9b63b53fc86841a3d26c02629990b233cc3",
    "nvidia_When2Call/train/when2call_train_sft.jsonl": "3eb20258557513579995ff55c09fcc33fabf2cd2004dea49dc3a0ba9880e631c",
}
# Files already pinned by IB1 / IB2 (their builders' PINS); copied unchanged into RAW.
REUSED = (
    "iabufarha_iSarcasmEval/train/train.En.csv",
    "iabufarha_iSarcasmEval/train/train.Ar.csv",
    "pyRis_SEntFiN/SEntFiN.csv",
    "nvidia_When2Call/train/when2call_train_pref.jsonl",
    "glaiveai_glaive-function-calling-v2/glaive-function-calling-v2.json",
)
FAMILIES = ("sqa2", "smish", "isarc2", "sentfin3", "w2c_act", "fc_pick")
LABELS = {"sentfin3": (0, 1, 2), "w2c_act": (0, 1, 2)}
LEAF_MIN_TOKENS = 6


def csv_rows(text: str) -> list[dict[str, str]]:
    csv.field_size_limit(1 << 30)
    rows = list(csv.DictReader(io.StringIO(text)))
    return [{(k or "").lstrip("\ufeff"): v for k, v in r.items()} for r in rows]


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text("utf-8").splitlines()
        if line.strip()
    ]


# --------------------------------------------------------------------------- dedupe against IB1-3 (prereg §2.0)


def state_key(state: Any) -> str:
    return sha("\x1e".join(sorted(normalize(leaf) for leaf in leaves(state))))


class PriorSets:
    def __init__(self, paths: list[Path]) -> None:
        self.states: dict[str, set[str]] = collections.defaultdict(set)
        self.leaves: dict[str, set[str]] = collections.defaultdict(set)
        self.files = {}
        replaced = set(fam.REPLACES.values())
        for path in paths:
            n = 0
            for line in path.open(encoding="utf-8"):
                if not line.strip():
                    continue
                item = json.loads(line)
                bucket = item["family"] if item["family"] in replaced else ""
                self.states[bucket].add(state_key(item["state"]))
                self.leaves[bucket].update(
                    normalize(leaf) for leaf in leaves(item["state"])
                )
                n += 1
            self.files[str(path)] = {"sha256": file_sha256(path), "rows": n}
        self.replaced_overlap: collections.Counter = collections.Counter()

    def __call__(self, item: Mapping[str, Any]) -> bool:
        key = state_key(item["state"])
        field = fam.KEY_LEAVES.get(item["family"])
        leaf = normalize(str(item["state"][field])) if field else None
        for bucket in self.states:
            if bucket and bucket == fam.REPLACES.get(item["family"]):
                if key in self.states[bucket]:
                    self.replaced_overlap[f"{item['family']}<-{bucket}"] += 1
                continue
            if key in self.states[bucket] or (
                leaf is not None and leaf in self.leaves[bucket]
            ):
                return True
        return False

    def disclose(self, rows: list[dict[str, Any]]) -> dict[str, int]:
        out: collections.Counter = collections.Counter()
        for item in rows:
            if any(
                len(normalize(leaf).split()) >= LEAF_MIN_TOKENS
                and any(normalize(leaf) in known for known in self.leaves.values())
                for leaf in leaves(item["state"])
            ):
                out[item["family"]] += 1
        return dict(sorted(out.items()))


# --------------------------------------------------------------------------- build


def convert(
    raw: Path, families: tuple[str, ...], prior: PriorSets
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reports = {name: collections.Counter() for name in families}
    rows: list[dict[str, Any]] = []
    if "sqa2" in families:
        records = parquet(
            raw / "rajpurkar_squad_v2/squad_v2/train-00000-of-00001.parquet",
            ["id", "title", "context", "question", "answers"],
        )
        rows += fam.sqa2(records, reports["sqa2"], prior)
    if "smish" in families:
        with zipfile.ZipFile(raw / "mendeley_sms_phishing/Dataset_5971.zip") as archive:
            text = archive.read("Dataset_5971.csv").decode("utf-8", "replace")
        rows += fam.smish(csv_rows(text), reports["smish"], prior)
    if "isarc2" in families:
        files = {
            lang: csv_rows(
                (raw / f"iabufarha_iSarcasmEval/train/train.{part}.csv").read_text(
                    "utf-8"
                )
            )
            for lang, part in (("en", "En"), ("ar", "Ar"))
        }
        rows += fam.isarc2(files, reports["isarc2"], prior)
    if "sentfin3" in families:
        rows += fam.sentfin3(
            csv_rows((raw / "pyRis_SEntFiN/SEntFiN.csv").read_text("utf-8")),
            reports["sentfin3"],
            prior,
        )
    if "w2c_act" in families:
        rows += fam.w2c_act(
            jsonl(raw / "nvidia_When2Call/train/when2call_train_sft.jsonl"),
            jsonl(raw / "nvidia_When2Call/train/when2call_train_pref.jsonl"),
            reports["w2c_act"],
            prior,
        )
    if "fc_pick" in families:
        glaive = json.loads(
            (
                raw
                / "glaiveai_glaive-function-calling-v2/glaive-function-calling-v2.json"
            ).read_text("utf-8")
        )
        rows += fam.fc_pick(glaive, reports["fc_pick"], prior)
        del glaive
    return rows, {
        "families": {
            name: {
                k: (dict(v) if isinstance(v, dict) else v) for k, v in sorted(r.items())
            }
            for name, r in reports.items()
        }
    }


def is_dev(group_id: str) -> bool:
    return int(sha(f"{DEV_SALT}:{group_id}"), 16) % 10 == 0


def as_dev(item: Mapping[str, Any]) -> dict[str, Any]:
    return validate_row(dict(item, split="select", evaluation_role="select"), "select")


def build(args: argparse.Namespace) -> int:
    eval_only.guard(args)
    raw = args.raw
    families = tuple(args.families)
    unknown = set(families) - set(FAMILIES)
    if unknown:
        raise ValueError(f"unknown families {sorted(unknown)}")
    inputs = {}
    for rel, expected in PINS.items():
        actual = file_sha256(raw / rel)
        if actual != expected:
            raise ValueError(f"{rel}: sha256 {actual} != pinned {expected}")
        inputs[rel] = actual
    for rel in REUSED:
        inputs[rel] = file_sha256(raw / rel)
    prior = PriorSets(list(args.prior))
    rows, report = convert(raw, families, prior)
    unique, report["duplicates"] = dedup(rows)
    report["group_merges"] = merge_groups(unique)
    report["prior_leaf_overlap_disclosed"] = prior.disclose(unique)
    report["replaced_family_state_overlap"] = dict(
        sorted(prior.replaced_overlap.items())
    )
    train = [r for r in unique if not is_dev(r["group_id"])]
    dev = [as_dev(r) for r in unique if is_dev(r["group_id"])]
    eval_only.check_rows(train + dev)
    args.out.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest = {
        "schema": "decision2.ib4.build.v1",
        "prereg": args.prereg,
        "families": list(families),
        "inputs": inputs,
        "prior": prior.files,
        "report": report,
        "train": {
            **counts(train),
            "sha256": write_rows(args.out / "ib4.train.cand.jsonl", train),
        },
        "dev": {
            **counts(dev),
            "sha256": write_rows(args.out / "ib4.dev.cand.jsonl", dev),
        },
    }
    write_json(args.out / "build.json", manifest)
    print(json.dumps({k: manifest[k]["rows"] for k in ("train", "dev")}))
    return 0


# --------------------------------------------------------------------------- finalize


def rebalance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in FAMILIES:
        members = [r for r in rows if r["family"] == name]
        if members:
            out += fam.cell_balance(
                members, LABELS.get(name, (0, 1)), None, BALANCE_SALT
            )
    return out


def finalize(args: argparse.Namespace) -> int:
    eval_only.guard(args)
    train = read_rows(args.cand / "ib4.train.cand.jsonl")
    dev = read_rows(args.cand / "ib4.dev.cand.jsonl")
    drop_groups = read_list(args.drop_groups)
    drop_dev = read_list(args.drop_dev_groups)
    drop_ids = read_list(args.drop_ids)
    drop_leak = read_list(args.drop_leak_ids)
    drop_families = set(args.drop_families)
    report: dict[str, Any] = collections.defaultdict(collections.Counter)

    def keep(item: Mapping[str, Any], dev_slice: bool) -> bool:
        reason = None
        if item["family"] in drop_families:
            reason = "family"
        elif item["group_id"] in drop_groups:
            reason = "dropped_group"
        elif dev_slice and item["group_id"] in drop_dev:
            reason = "dev_near_train"
        elif item["id"] in drop_ids:
            reason = "review_or_screen"
        elif item["id"] in drop_leak:
            reason = "leak_guard"
        if reason:
            report[f"{'dev' if dev_slice else 'train'}:{reason}"][item["family"]] += 1
        return reason is None

    kept_train = [r for r in train if keep(r, False)]
    kept_dev = [r for r in dev if keep(r, True)]
    train = rebalance(kept_train)
    dev = rebalance(kept_dev)
    for name, kept, final in (("train", kept_train, train), ("dev", kept_dev, dev)):
        report[f"{name}:rebalance"].update(
            collections.Counter(r["family"] for r in kept)
            - collections.Counter(r["family"] for r in final)
        )
    for item in train:
        validate_row(item, "train")
    for item in dev:
        validate_row(item, "select")
    shared = {r["group_id"] for r in train} & {r["group_id"] for r in dev}
    if shared:
        raise ValueError(f"{len(shared)} groups in both TRAIN and DEV")
    eval_only.check_rows(train + dev)
    args.out.mkdir(parents=True, exist_ok=True, mode=0o700)

    def listed(values: set[str]) -> dict[str, Any]:
        return {"count": len(values), "sha256": sha("\n".join(sorted(values)))}

    manifest = {
        "schema": "decision2.ib4.final.v1",
        "candidates": {
            "train": file_sha256(args.cand / "ib4.train.cand.jsonl"),
            "dev": file_sha256(args.cand / "ib4.dev.cand.jsonl"),
        },
        "drops": {
            "groups": listed(drop_groups),
            "dev_groups": listed(drop_dev),
            "ids": listed(drop_ids),
            "leak_ids": listed(drop_leak),
            "families": sorted(drop_families),
            "removed": {k: dict(sorted(v.items())) for k, v in sorted(report.items())},
        },
        "train": {
            **counts(train),
            "sha256": write_rows(args.out / "ib4.train.jsonl", train),
        },
        "dev": {**counts(dev), "sha256": write_rows(args.out / "ib4.dev.jsonl", dev)},
    }
    write_json(args.out / "final.json", manifest)
    print(json.dumps({k: manifest[k]["rows"] for k in ("train", "dev")}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("build")
    one.add_argument("--raw", required=True, type=Path)
    one.add_argument("--prior", action="append", default=[], type=Path)
    one.add_argument("--out", required=True, type=Path)
    one.add_argument("--families", nargs="+", default=list(FAMILIES))
    one.add_argument("--prereg", default="records/ib4-prereg-2026-10-02.md")
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
