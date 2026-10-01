"""Build and finalize IB1 (prereg ``records/ib1-prereg-2026-10-01.md`` §2–§3).

    python3 -m v2.data.ib1.build build --raw RAW --out CAND
    python3 -m v2.data.ib1.build finalize --cand CAND --out FINAL \
        [--drop-groups FILE ...] [--drop-dev-groups FILE ...] [--drop-families NAME ...] \
        [--drop-ids FILE ...] [--drop-leak-ids FILE ...]

``build`` reads the pinned publisher files under RAW, converts every family, merges groups that share a
normalized state, removes exact duplicates and assigns the group-isolated DEV slice. ``finalize`` only removes rows
from the candidates (Index-row and quarantined groups, DEV near-duplicates of TRAIN, dropped families, screen and
review findings, leak-guard rows) and re-balances by downsampling, so every final row was scanned as a candidate.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import hashlib
import json
import os
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from training.model.data import canonical, validate_row
from v2.common import eval_only
from v2.data.ib1 import families as fam
from v2.data.sources.common import sha
from v2.data.textnorm import normalize

DEV_SALT = "ib1-dev-v1"
BALANCE_SALT = "ib1-final-v1"
SHARE_BOUNDS = (0.45, 0.55)
POSITION_MARGIN = 0.05

# Publisher files read by ``build`` (relative to RAW) and their SHA-256 at download time (prereg §1.2).
PINS = {
    "Salesforce_summedits/summedits.json": "21e34593330020d02b7d92f4651fdcc550f7d79a2f899392f106fd816c215a89",
    "cmalaviya_expertqa/r2_compiled_anon_fixed.jsonl": "da7edb5b774ed1081f50a665fc8cc1ad64690a4b6a67a587803f53f547be4bc6",
    "ucirvine_sms_spam/plain_text/train-00000-of-00001.parquet": "6e5518e4a49cb2de8af9c89a38b742825cdddbb55942701fc2237d4364288abd",
    "nvidia_When2Call/train/when2call_train_pref.jsonl": "d90637f108fabf1b097493c5818c3692e1d73140259d2f8e490536255e765bc4",
    "sonos_nlu-benchmark/2017-06-custom-intent-engines/AddToPlaylist/train_AddToPlaylist_full.json": "fee1b4645352e70ed8a40f1f87c3b53fc3bf75f8ebc82252a22b82c107e75b01",
    "sonos_nlu-benchmark/2017-06-custom-intent-engines/BookRestaurant/train_BookRestaurant_full.json": "7677e82cd6e9a8191f0a4502568c786278c984900b4796d3628e46f1abf73ef4",
    "sonos_nlu-benchmark/2017-06-custom-intent-engines/GetWeather/train_GetWeather_full.json": "86f414b2ad8a993134d85d16e25e5d337beae47cef993c0af2a5f440405c60f6",
    "sonos_nlu-benchmark/2017-06-custom-intent-engines/PlayMusic/train_PlayMusic_full.json": "901ddbe9781d1fdb8196e789c543f80df3d29ca6e237da64947905adcc0312e4",
    "sonos_nlu-benchmark/2017-06-custom-intent-engines/RateBook/train_RateBook_full.json": "bb43e4b74134300d5766f6b420d0357a63402987a225f6dc92c806c9c040ce04",
    "sonos_nlu-benchmark/2017-06-custom-intent-engines/SearchCreativeWork/train_SearchCreativeWork_full.json": "87b3432de48c71c205aa5238b41d9e3f2d70bdfcd0bea2427544fb9e4bd5e5bb",
    "sonos_nlu-benchmark/2017-06-custom-intent-engines/SearchScreeningEvent/train_SearchScreeningEvent_full.json": "5447e9440c02e2fb08c3ae16b9f58a4d1332b4bc77fd34691402b02aa4399765",
    "webis_args_me/args-me.jsonl": "dc49ce6a5547370034be99365e8041d91ce477dbb4e0012ae16c0a6ecdd28187",
    "iabufarha_iSarcasmEval/train/train.En.csv": "fde737b69caffbcf81ebe79858ad16ac21881573c308343cd5219c4f7a800ae7",
    "iabufarha_iSarcasmEval/train/train.Ar.csv": "7a13a6ee8082c7a2f0bfd0d0e2b8bd06549ccb1feff7c013afdd509d473672f0",
    "Qwen_ProcessBench/math.json": "60cb0d7a69cbec98d43cf50855b9b7e7294599c3c9a24bdd031abd18c3b6b136",
    "pyRis_SEntFiN/SEntFiN.csv": "570af7ba58f724f3fcc414dd5ba80da5b1a3b342134d76c6ebdf53a32c54a92f",
    "wayfair_WANDS/dataset/label.csv": "c11fe81ad62f17f56f316b0ec9630ebe8fbe1393578cb0ca4f05c17253a180ef",
    "wayfair_WANDS/dataset/product.csv": "d993926254572e6eba96c8fd87cc549a17fb91ad3748308036eee4cf92b10ac6",
    "wayfair_WANDS/dataset/query.csv": "63b61660560fecc33ec490804c7e2b81402ee3e7c31a9cbb5e03736639f68e95",
    "theatticusproject_maud/MAUD_v1/MAUD_train.csv": "bac9f2d034ad487d5398ee2ac1c876679afea509ba8e7f1092112955c6180ff9",
    "pkavumba_balanced-copa/train.csv": "650ec76d0b0b46511bcf68702f9e371cc3d9ad3074f947659d98155cc19d56ef",
    "tau_commonsense_qa/data/train-00000-of-00001.parquet": "b0449767ed986bfc2ca52b1244a46ef12f732756727f3cb0a4ab69ac8b3d282b",
    "openlifescienceai_medmcqa/data/train-00000-of-00001.parquet": "b119434ba551517a6ec0ba1f7e0b4c029165ed284a4704f262ce37c791c493c5",
    "alisawuffles_WANLI/train.jsonl": "85058cf017a911e89242dc29fa0a4ddaad3664cb923dc0a82145fdda14b694e5",
    "alisawuffles_WANLI/anonymized_annotations.jsonl": "32821a1cf24208399c1d3967b2c998b2f0227f054dfddd0b83bbde5cdf5e3588",
    "biglam_gutenberg-poetry-corpus/data/train-00000-of-00001-fa9fb9e1f16eed7e.parquet": "c7117a3c7e613162679cc1fbca4b0ac4a1f105c7f6d00cebb9ec35d6788d64d2",
}
FAMILIES = (
    "sumedit",
    "expqa",
    "sms",
    "w2c",
    "snips_sel",
    "snips_rel",
    "args",
    "isarc",
    "procb",
    "sentfin",
    "wands",
    "maud",
    "copa",
    "csqa",
    "medmcqa",
    "wanli",
    "poem",
)
NOUL_FAMILIES = ("sumedit", "expqa", "sms", "snips_rel", "procb")
CLASS_FAMILIES = ("sentfin", "wands", "wanli")
AB_FAMILIES = ("w2c", "isarc", "poem", "copa")
ROTATED_FAMILIES = ("snips_sel", "maud", "csqa", "medmcqa")
TWIN_FAMILIES = ("args",)
# Round-2 constructions (amendment 2 §A): name -> (family, audit field, value); a row matches when its family is the
# named one and `label` (gold class index) or `domain` (audit metadata) equals the value.
CONSTRUCTIONS: dict[str, tuple[str, str, Any]] = {
    "sentfin-neutral": ("sentfin", "label", fam.SENTFIN_LABELS.index("neutral")),
    "sumedit-shakespeare": ("sumedit", "domain", "shakespeare"),
    "sumedit-sales_email": ("sumedit", "domain", "sales_email"),
    "wands-partial": ("wands", "label", fam.WANDS_LABELS.index("Partial")),
}


def construction(row: Mapping[str, Any], names: Iterable[str]) -> str | None:
    """The first named construction ``row`` belongs to, if any."""
    for name in names:
        family, field, value = CONSTRUCTIONS[name]
        if row["family"] != family:
            continue
        got = row["label"] if field == "label" else row["audit_metadata"]["ib1"][field]
        if got == value:
            return name
    return None


def excluded_labels(names: Iterable[str]) -> dict[str, set[int]]:
    out: dict[str, set[int]] = collections.defaultdict(set)
    for name in names:
        family, field, value = CONSTRUCTIONS[name]
        if field == "label":
            out[family].add(value)
    return out


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path: Path, data: bytes) -> str:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
    return hashlib.sha256(data).hexdigest()


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    ordered = sorted(rows, key=lambda row: row["id"])
    return write_new(
        path, "".join(canonical(row) + "\n" for row in ordered).encode("utf-8")
    )


def write_json(path: Path, payload: Any) -> str:
    return write_new(
        path, (json.dumps(payload, indent=1, sort_keys=True) + "\n").encode("utf-8")
    )


def read_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_bytes().decode("utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def read_list(paths: Iterable[Path]) -> set[str]:
    found: set[str] = set()
    for path in paths:
        found |= {
            line.strip()
            for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()
        }
    return found


def jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def parquet(path: Path, columns: Sequence[str] | None = None) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=list(columns) if columns else None).to_pylist()


def decode(data: bytes) -> str:
    """UTF-8; a SNIPS file encodes emoji as UTF-8 surrogate pairs (CESU-8), which are re-joined."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", "surrogatepass")
        return text.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")


def snips_files(root: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(glob.glob(str(root / "*/train_*_full.json"))):
        data = json.loads(decode(Path(path).read_bytes()))
        intent = next(iter(data))
        files[intent] = data
    return files


# --------------------------------------------------------------------------- build


def convert(raw: Path, round_: int = 1) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reports: dict[str, collections.Counter] = {
        name: collections.Counter() for name in FAMILIES
    }
    rows: list[dict[str, Any]] = []
    rows += fam.sumedit(
        json.loads((raw / "Salesforce_summedits/summedits.json").read_text("utf-8")),
        reports["sumedit"],
    )
    rows += fam.expqa(
        jsonl(raw / "cmalaviya_expertqa/r2_compiled_anon_fixed.jsonl"), reports["expqa"]
    )
    rows += fam.sms(
        parquet(raw / "ucirvine_sms_spam/plain_text/train-00000-of-00001.parquet"),
        reports["sms"],
    )
    rows += fam.w2c(
        jsonl(raw / "nvidia_When2Call/train/when2call_train_pref.jsonl"), reports["w2c"]
    )
    sel, rel = fam.snips(
        snips_files(raw / "sonos_nlu-benchmark/2017-06-custom-intent-engines"),
        reports["snips_sel"],
        reports["snips_rel"],
    )
    rows += sel + rel
    with (raw / "webis_args_me/args-me.jsonl").open(encoding="utf-8") as stream:
        rows += fam.args(stream, reports["args"])
    rows += fam.isarc(
        {
            "en": fam.read_csv(raw / "iabufarha_iSarcasmEval/train/train.En.csv"),
            "ar": fam.read_csv(raw / "iabufarha_iSarcasmEval/train/train.Ar.csv"),
        },
        reports["isarc"],
    )
    rows += fam.procb(
        json.loads((raw / "Qwen_ProcessBench/math.json").read_text("utf-8")),
        reports["procb"],
    )
    rows += fam.sentfin(
        fam.read_csv(raw / "pyRis_SEntFiN/SEntFiN.csv"),
        reports["sentfin"],
        two_way=round_ >= 3,
    )
    tsv = {"delimiter": "\t", "quoting": csv.QUOTE_NONE}
    rows += fam.wands(
        fam.read_csv(raw / "wayfair_WANDS/dataset/query.csv", **tsv),
        fam.read_csv(raw / "wayfair_WANDS/dataset/product.csv", **tsv),
        fam.read_csv(raw / "wayfair_WANDS/dataset/label.csv", **tsv),
        reports["wands"],
    )
    rows += fam.maud(
        fam.read_csv(raw / "theatticusproject_maud/MAUD_v1/MAUD_train.csv"),
        reports["maud"],
    )
    rows += fam.copa(
        fam.read_csv(raw / "pkavumba_balanced-copa/train.csv"), reports["copa"]
    )
    rows += fam.csqa(
        parquet(raw / "tau_commonsense_qa/data/train-00000-of-00001.parquet"),
        reports["csqa"],
    )
    rows += fam.medmcqa(
        parquet(raw / "openlifescienceai_medmcqa/data/train-00000-of-00001.parquet"),
        reports["medmcqa"],
    )
    rows += fam.wanli(
        jsonl(raw / "alisawuffles_WANLI/train.jsonl"),
        jsonl(raw / "alisawuffles_WANLI/anonymized_annotations.jsonl"),
        reports["wanli"],
    )
    poems = parquet(
        raw
        / "biglam_gutenberg-poetry-corpus/data/train-00000-of-00001-fa9fb9e1f16eed7e.parquet",
        ["line", "gutenberg_id"],
    )
    rows += fam.poem([(p["line"], p["gutenberg_id"]) for p in poems], reports["poem"])
    return rows, {
        "families": {name: dict(sorted(r.items())) for name, r in reports.items()}
    }


def state_key(row: Mapping[str, Any]) -> str:
    return normalize(" ".join(str(value) for value in row["state"].values()))


def merge_groups(rows: list[dict[str, Any]]) -> int:
    """Union groups whose rows share a normalized state; returns the number of merges."""
    parent: dict[str, str] = {}

    def find(group: str) -> str:
        while parent.setdefault(group, group) != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    first: dict[str, str] = {}
    merges = 0
    for row in rows:
        key = state_key(row)
        if key in first:
            a, b = find(first[key]), find(row["group_id"])
            if a != b:
                parent[max(a, b)] = min(a, b)
                merges += 1
        else:
            first[key] = row["group_id"]
    for row in rows:
        row["group_id"] = find(row["group_id"])
    return merges


def dedup(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Keep one copy of identical rows; drop every copy of an input with conflicting labels."""
    report: dict[str, collections.Counter] = {
        "exact": collections.Counter(),
        "conflicting": collections.Counter(),
    }
    by_input: dict[str, set[int]] = collections.defaultdict(set)
    for row in rows:
        by_input[row["input_sha256"]].add(row["label"])
    unique, seen = [], set()
    for row in sorted(rows, key=lambda r: r["id"]):
        if len(by_input[row["input_sha256"]]) > 1:
            report["conflicting"][row["family"]] += 1
            continue
        if row["input_sha256"] in seen or row["id"] in seen:
            report["exact"][row["family"]] += 1
            continue
        seen |= {row["input_sha256"], row["id"]}
        unique.append(row)
    return unique, {key: dict(sorted(value.items())) for key, value in report.items()}


def is_dev(group_id: str) -> bool:
    return int(sha(f"{DEV_SALT}:{group_id}"), 16) % 10 == 0


def as_dev(row: Mapping[str, Any]) -> dict[str, Any]:
    return validate_row(dict(row, split="select", evaluation_role="select"), "select")


def counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def by(field: str, subset: Sequence[Mapping[str, Any]] = rows) -> dict[str, int]:
        return dict(sorted(collections.Counter(str(r[field]) for r in subset).items()))

    out: dict[str, Any] = {
        "rows": len(rows),
        "groups": len({r["group_id"] for r in rows}),
        "task_type": by("task_type"),
        "language": by("language"),
        "family": by("family"),
        "label_by_family": {},
    }
    for name in sorted({r["family"] for r in rows}):
        out["label_by_family"][name] = by(
            "label", [r for r in rows if r["family"] == name]
        )
    return out


def build(args: argparse.Namespace) -> int:
    eval_only.guard(args)
    raw = args.raw
    inputs = {}
    for rel, expected in PINS.items():
        actual = file_sha256(raw / rel)
        if actual != expected:
            raise ValueError(f"{rel}: sha256 {actual} != pinned {expected}")
        inputs[rel] = actual
    rows, report = convert(raw, args.round)
    unique, report["duplicates"] = dedup(rows)
    report["group_merges"] = merge_groups(unique)
    train = [row for row in unique if not is_dev(row["group_id"])]
    dev = [as_dev(row) for row in unique if is_dev(row["group_id"])]
    eval_only.check_rows(train + dev)
    args.out.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest = {
        "schema": "decision2.ib1.build.v1",
        "prereg": "records/ib1-prereg-2026-10-01.md",
        "round": args.round,
        "inputs": inputs,
        "report": report,
        "train": {
            **counts(train),
            "sha256": write_rows(args.out / "ib1.train.cand.jsonl", train),
        },
        "dev": {
            **counts(dev),
            "sha256": write_rows(args.out / "ib1.dev.cand.jsonl", dev),
        },
    }
    write_json(args.out / "build.json", manifest)
    print(json.dumps({k: manifest[k]["rows"] for k in ("train", "dev")}))
    return 0


# --------------------------------------------------------------------------- finalize


def ranked(rows: Iterable[dict[str, Any]], salt: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: sha(f"{salt}:{row['id']}"))


def within_bounds(
    rows: list[dict[str, Any]], flag: Callable[[dict[str, Any]], Any], salt: str
) -> list[dict[str, Any]]:
    """Downsample so the share of rows with ``flag(row) is True`` among decided rows is in bounds."""
    low, high = SHARE_BOUNDS
    ordered = ranked(rows, salt)
    yes = [r for r in ordered if flag(r) is True]
    no = [r for r in ordered if flag(r) is False]
    free = [r for r in ordered if flag(r) is None]
    while yes and no and len(yes) / (len(yes) + len(no)) > high:
        yes.pop()
    while yes and no and len(yes) / (len(yes) + len(no)) < low:
        no.pop()
    return free + yes + no


def equal_labels(
    rows: list[dict[str, Any]], salt: str, excluded: Iterable[int] = ()
) -> list[dict[str, Any]]:
    by: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in ranked(rows, salt):
        by[row["label"]].append(row)
    expected = set(range(len(rows[0]["options"]))) - set(excluded)
    if set(by) != expected:
        return []
    size = min(len(members) for members in by.values())
    return [row for label in sorted(by) for row in by[label][:size]]


def position_band(rows: list[dict[str, Any]], salt: str) -> list[dict[str, Any]]:
    """Per option count k, trim over-represented gold positions until every share is within 1/k ± 0.05."""
    strata: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        strata[len(row["options"])].append(row)
    out: list[dict[str, Any]] = []
    for k, members in sorted(strata.items()):
        by: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in ranked(members, salt):
            by[row["label"]].append(row)
        while True:
            total = sum(len(v) for v in by.values())
            top = max(range(k), key=lambda p: (len(by[p]), p))
            if total and len(by[top]) / total > 1 / k + POSITION_MARGIN:
                by[top].pop()
                continue
            break
        out += [row for p in range(k) for row in by[p]]
    return out


def twins(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by[row["group_id"]].append(row)
    out = []
    for members in by.values():
        labels = collections.Counter(r["label"] for r in members)
        if labels[0] >= 1 and labels[1] >= 1:
            out += [
                min((r for r in members if r["label"] == lab), key=lambda r: r["id"])
                for lab in (0, 1)
            ]
    return out


def rebalance(
    rows: list[dict[str, Any]], constructions: Sequence[str] = ()
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    excluded = excluded_labels(constructions)
    for name in FAMILIES:
        members = [row for row in rows if row["family"] == name]
        if not members:
            continue
        if name == "sumedit":
            for domain in fam.SUMEDIT_DOMAINS:
                cell = [
                    r for r in members if r["audit_metadata"]["ib1"]["domain"] == domain
                ]
                out += equal_labels(cell, BALANCE_SALT) if cell else []
        elif name in NOUL_FAMILIES or name in CLASS_FAMILIES:
            out += equal_labels(members, BALANCE_SALT, excluded[name])
        elif name in TWIN_FAMILIES:
            out += twins(members)
        elif name in AB_FAMILIES:
            kept = within_bounds(
                members, lambda r: r["label"] == 0, BALANCE_SALT + ":ab"
            )
            out += within_bounds(
                kept,
                lambda r: r["audit_metadata"]["ib1"].get("gold_longer"),
                BALANCE_SALT + ":len",
            )
        elif name in ROTATED_FAMILIES:
            out += position_band(members, BALANCE_SALT + ":pos")
        else:
            raise ValueError(f"no balance rule for {name}")
    return out


def finalize(args: argparse.Namespace) -> int:
    eval_only.guard(args)
    train = read_rows(args.cand / "ib1.train.cand.jsonl")
    dev = read_rows(args.cand / "ib1.dev.cand.jsonl")
    drop_groups = read_list(args.drop_groups)
    drop_dev = read_list(args.drop_dev_groups)
    drop_ids = read_list(args.drop_ids)
    drop_leak = read_list(args.drop_leak_ids)
    drop_families = set(args.drop_families)
    constructions = sorted(set(args.drop_construction))
    report: dict[str, Any] = collections.defaultdict(collections.Counter)

    def keep(row: Mapping[str, Any], dev_slice: bool) -> bool:
        reason = None
        if row["family"] in drop_families:
            reason = "family"
        elif construction(row, constructions):
            reason = "construction"
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

    train = rebalance([row for row in train if keep(row, False)], constructions)
    dev = rebalance([row for row in dev if keep(row, True)], constructions)
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
        "schema": "decision2.ib1.final.v1",
        "candidates": {
            "train": file_sha256(args.cand / "ib1.train.cand.jsonl"),
            "dev": file_sha256(args.cand / "ib1.dev.cand.jsonl"),
        },
        "drops": {
            "groups": listed(drop_groups),
            "dev_groups": listed(drop_dev),
            "ids": listed(drop_ids),
            "leak_ids": listed(drop_leak),
            "families": sorted(drop_families),
            "constructions": constructions,
            "removed": {
                key: dict(sorted(value.items()))
                for key, value in sorted(report.items())
            },
        },
        "train": {
            **counts(train),
            "sha256": write_rows(args.out / "ib1.train.jsonl", train),
        },
        "dev": {**counts(dev), "sha256": write_rows(args.out / "ib1.dev.jsonl", dev)},
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
    one.add_argument("--round", type=int, choices=(1, 2, 3), default=1)
    two = sub.add_parser("finalize")
    two.add_argument("--cand", required=True, type=Path)
    two.add_argument("--out", required=True, type=Path)
    two.add_argument("--drop-groups", action="append", default=[], type=Path)
    two.add_argument("--drop-dev-groups", action="append", default=[], type=Path)
    two.add_argument("--drop-ids", action="append", default=[], type=Path)
    two.add_argument("--drop-leak-ids", action="append", default=[], type=Path)
    two.add_argument("--drop-families", nargs="*", default=[])
    two.add_argument(
        "--drop-construction",
        action="append",
        default=[],
        choices=sorted(CONSTRUCTIONS),
    )
    args = parser.parse_args(argv)
    return build(args) if args.command == "build" else finalize(args)


if __name__ == "__main__":
    raise SystemExit(main())
