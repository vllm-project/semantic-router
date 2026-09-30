"""Build and finalize HR2 (prereg ``records/hr2-prereg-2026-09-30.md`` §2–§3).

    python3 -m v2.data.hr2.build build --raw RAW --out CAND
    python3 -m v2.data.hr2.build finalize --cand CAND --out FINAL \
        [--drop-groups FILE ...] [--drop-dev-groups FILE ...] [--drop-families NAME ...] [--drop-ids FILE ...]

``build`` reads the pinned publisher TRAIN files under RAW, converts every family, merges groups that
share a normalized state, removes exact duplicates and assigns the group-isolated DEV slice. It writes
``hr2.train.cand.jsonl``, ``hr2.dev.cand.jsonl`` and ``build.json``. ``finalize`` only removes rows
from the candidates (quarantined groups, DEV near-duplicates of TRAIN, dropped families, review
errors) and re-balances by downsampling, so every final row was scanned as a candidate.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from training.model.data import canonical, validate_row
from v2.common import eval_only
from v2.data.hr2 import families as fam
from v2.data.sources.common import sha
from v2.data.textnorm import normalize

DEV_SALT = "hr2-dev-v1"
BALANCE_SALT = "hr2-final-v1"
SHARE_BOUNDS = (0.45, 0.55)

# Publisher files read by ``build`` (relative to RAW) and their audit-time SHA-256 (prereg §1.2).
PINS = {
    "nvidia_HelpSteer3/preference/train.jsonl.gz": "32b52e1d378f8dab1e4c9ae549da49a5d6fc0875aeafe3f9139e6053beb906bb",
    "nvidia_HelpSteer3/feedback/train.jsonl.gz": "6f51cf5b7224436b0c5a4c80b280eddc566f7d2b589f5a4be0ba7a3637f0519c",
    "nvidia_HelpSteer2/train.jsonl.gz": "c0d7e91d738d42e8a08070db26c4c09a9c7631308e1f0fd380ff43d130c9f713",
    "nvidia_HelpSteer2/validation.jsonl.gz": "610eeb5289494d613c4c0f70aade2df8df0b499f3a24e76d232f74e6909d010a",
    "nvidia_HelpSteer2/preference/preference.jsonl.gz": "a5cd48600fb7a330cf0ccc8f59051e24e8f236907c379f42eff1ba18da55204b",
    "hendrycks_ethics/data/utilitarianism/train.csv": "a057f29e667b19531644f5c4c317606cacf47669d9bc6b4bd1844fd4d1780050",
    "hendrycks_ethics/data/commonsense/train.csv": "286cb4098a66b71819493f964cf2afe62cdbb6c6b41983b9c0b8fc484650c752",
    "hendrycks_ethics/data/deontology/train.csv": "d31e25825c43c16fb566cc9172dd59191756111cf74ef1b90633a716299861fe",
    "hendrycks_ethics/data/justice/train.csv": "175c8255f5ee3d99ad381cd716473367274dc24b3832d5e3357f37f1779a7d3f",
    "prm800k/phase1_train.jsonl": "e9da6a73f827ffb9a8c0dc644c541d34ed76b3d4d1e4896ff5f7b37ddf5ae34d",
    "prm800k/phase2_train.jsonl": "1110237feeb51d1bc200cb37b8f965cfdc1036eac7d506094049366fe7dc1089",
    "tals_vitaminc/train.jsonl": "7461c6fd1a13459590317c5ccdc8651dd2daf7c1ad8ae4b10ccd88d164fccd5a",
    "skt_kobest_v1/boolq/train.jsonl": "312b92a96fd5d1fc8059491fbc598a3895581723d2266b840577f4e320e6315b",
    "indonli/train.jsonl": "c18a2974a8683d283d0c8eb3d944354d3c5ce54ab6232613d932c0bd2a82abf8",
    "allegro_klej-allegro-reviews/train.csv": "bc722eafd61ccec427cc3f6af112fbbfbc999bb619f8f02a7ad5027fec1c707e",
}
FAMILIES = (
    "hs3_pref",
    "hs3_help",
    "eth_util",
    "eth_cs",
    "eth_deon",
    "eth_just",
    "prm_step",
    "vitc",
    "kob_boolq",
    "indonli",
    "allegro",
)


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
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_list(paths: Iterable[Path]) -> set[str]:
    found: set[str] = set()
    for path in paths:
        found |= {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    return found


# --------------------------------------------------------------------------- build


def convert(raw: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reports: dict[str, collections.Counter] = {
        name: collections.Counter() for name in FAMILIES
    }
    hs2 = [
        record
        for name in (
            "train.jsonl.gz",
            "validation.jsonl.gz",
            "preference/preference.jsonl.gz",
        )
        for record in fam.read_jsonl(raw / "nvidia_HelpSteer2" / name)
    ]
    screen = fam.hs2_screen(hs2)
    hs3 = raw / "nvidia_HelpSteer3"
    ethics = raw / "hendrycks_ethics/data"
    rows: list[dict[str, Any]] = []
    rows += fam.hs3_pref(
        fam.read_jsonl(hs3 / "preference/train.jsonl.gz"), screen, reports["hs3_pref"]
    )
    rows += fam.hs3_help(
        fam.read_jsonl(hs3 / "feedback/train.jsonl.gz"), screen, reports["hs3_help"]
    )
    rows += fam.eth_util(
        fam.read_csv(ethics / "utilitarianism/train.csv"), reports["eth_util"]
    )
    for family, subset in (
        ("eth_cs", "commonsense"),
        ("eth_deon", "deontology"),
        ("eth_just", "justice"),
    ):
        rows += fam.eth_noul(
            fam.read_csv(ethics / f"{subset}/train.csv"), family, reports[family]
        )
    prm = raw / "prm800k"
    rows += fam.prm_step(
        (
            r
            for name in ("phase1_train.jsonl", "phase2_train.jsonl")
            for r in fam.read_jsonl(prm / name)
        ),
        reports["prm_step"],
    )
    rows += fam.vitc(fam.read_jsonl(raw / "tals_vitaminc/train.jsonl"), reports["vitc"])
    rows += fam.kob_boolq(
        fam.read_jsonl(raw / "skt_kobest_v1/boolq/train.jsonl"), reports["kob_boolq"]
    )
    rows += fam.indonli(fam.read_jsonl(raw / "indonli/train.jsonl"), reports["indonli"])
    rows += fam.allegro(
        fam.read_csv(raw / "allegro_klej-allegro-reviews/train.csv"), reports["allegro"]
    )
    return rows, {
        "helpsteer2_screen": {
            "first_turns": len(screen[0]),
            "prefixes": len(screen[1]),
        },
        "families": {
            name: dict(sorted(report.items())) for name, report in reports.items()
        },
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
        "gold_longer_by_family": {},
    }
    for name in sorted({r["family"] for r in rows}):
        members = [r for r in rows if r["family"] == name]
        out["label_by_family"][name] = by("label", members)
        flags = [r["audit_metadata"]["hr2"].get("gold_longer", "n/a") for r in members]
        if members[0]["task_type"] == "choice":
            out["gold_longer_by_family"][name] = dict(
                collections.Counter(str(f) for f in flags)
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
    rows, report = convert(raw)
    seen: set[str] = set()
    unique = []
    for row in sorted(rows, key=lambda r: r["id"]):
        if row["input_sha256"] in seen:
            report.setdefault("exact_duplicates", collections.Counter())[
                row["family"]
            ] += 1
            continue
        seen.add(row["input_sha256"])
        unique.append(row)
    ids = [row["id"] for row in unique]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate row id")
    report["group_merges"] = merge_groups(unique)
    train = [row for row in unique if not is_dev(row["group_id"])]
    dev = [as_dev(row) for row in unique if is_dev(row["group_id"])]
    eval_only.check_rows(train + dev)
    args.out.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest = {
        "schema": "decision2.hr2.build.v1",
        "prereg": "records/hr2-prereg-2026-09-30.md",
        "inputs": inputs,
        "report": json.loads(json.dumps(report, default=dict)),
        "train": {
            **counts(train),
            "sha256": write_rows(args.out / "hr2.train.cand.jsonl", train),
        },
        "dev": {
            **counts(dev),
            "sha256": write_rows(args.out / "hr2.dev.cand.jsonl", dev),
        },
    }
    write_json(args.out / "build.json", manifest)
    print(json.dumps({k: manifest[k]["rows"] for k in ("train", "dev")}))
    return 0


# --------------------------------------------------------------------------- finalize


def ranked(rows: Iterable[dict[str, Any]], salt: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: sha(f"{salt}:{row['id']}"))


def within_bounds(rows: list[dict[str, Any]], flag, salt: str) -> list[dict[str, Any]]:
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


def rebalance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in FAMILIES:
        members = [row for row in rows if row["family"] == name]
        if not members:
            continue
        kind = members[0]["task_type"]
        if name == "prm_step":
            cells = collections.defaultdict(list)
            for row in members:
                cells[row["audit_metadata"]["hr2"]["bucket"]].append(row)
            for bucket in sorted(cells):
                out += balance_labels(cells[bucket])
        elif name == "indonli":
            out += balance_indonli(members)
        elif kind == "noul":
            out += balance_labels(members)
        elif kind == "choice":
            kept = within_bounds(
                members, lambda r: r["label"] == 0, BALANCE_SALT + ":ab"
            )
            out += within_bounds(
                kept,
                lambda r: r["audit_metadata"]["hr2"]["gold_longer"],
                BALANCE_SALT + ":len",
            )
        elif name == "hs3_help":
            out += fam.cap_share(
                members, len(members), fam.HELP_MAX_SHARE, BALANCE_SALT
            )
        else:
            levels = collections.defaultdict(list)
            for row in ranked(members, BALANCE_SALT):
                levels[row["label"]].append(row)
            size = min(
                len(levels[level]) for level in range(len(members[0]["options"]))
            )
            out += [row for level in sorted(levels) for row in levels[level][:size]]
    return out


def balance_labels(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = ranked(rows, BALANCE_SALT)
    yes = [r for r in ordered if r["label"] == 1]
    no = [r for r in ordered if r["label"] == 0]
    half = min(len(yes), len(no))
    return yes[:half] + no[:half]


def balance_indonli(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = ranked(rows, BALANCE_SALT)
    by = collections.defaultdict(list)
    for row in ordered:
        by[row["audit_metadata"]["hr2"]["upstream_label"]].append(row)
    quarter = min(len(by["c"]), len(by["n"]), len(by["e"]) // 2)
    return by["e"][: 2 * quarter] + by["c"][:quarter] + by["n"][:quarter]


def finalize(args: argparse.Namespace) -> int:
    eval_only.guard(args)
    train = read_rows(args.cand / "hr2.train.cand.jsonl")
    dev = read_rows(args.cand / "hr2.dev.cand.jsonl")
    drop_groups = read_list(args.drop_groups)
    drop_dev = read_list(args.drop_dev_groups)
    drop_ids = read_list(args.drop_ids)
    drop_families = set(args.drop_families)
    report: dict[str, Any] = collections.defaultdict(collections.Counter)

    def keep(row: Mapping[str, Any], dev_slice: bool) -> bool:
        reason = None
        if row["family"] in drop_families:
            reason = "family"
        elif row["group_id"] in drop_groups:
            reason = "quarantine_group"
        elif dev_slice and row["group_id"] in drop_dev:
            reason = "dev_near_train"
        elif row["id"] in drop_ids:
            reason = "review_error"
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
    manifest = {
        "schema": "decision2.hr2.final.v1",
        "candidates": {
            "train": file_sha256(args.cand / "hr2.train.cand.jsonl"),
            "dev": file_sha256(args.cand / "hr2.dev.cand.jsonl"),
        },
        "drops": {
            "groups": {
                "count": len(drop_groups),
                "sha256": sha("\n".join(sorted(drop_groups))),
            },
            "dev_groups": {
                "count": len(drop_dev),
                "sha256": sha("\n".join(sorted(drop_dev))),
            },
            "ids": {"count": len(drop_ids), "sha256": sha("\n".join(sorted(drop_ids)))},
            "families": sorted(drop_families),
            "removed": {
                key: dict(sorted(value.items()))
                for key, value in sorted(report.items())
            },
        },
        "train": {
            **counts(train),
            "sha256": write_rows(args.out / "hr2.train.jsonl", train),
        },
        "dev": {**counts(dev), "sha256": write_rows(args.out / "hr2.dev.jsonl", dev)},
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
    two.add_argument("--drop-families", nargs="*", default=[])
    args = parser.parse_args(argv)
    return build(args) if args.command == "build" else finalize(args)


if __name__ == "__main__":
    raise SystemExit(main())
