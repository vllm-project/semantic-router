"""HR2 audits (prereg ``records/hr2-prereg-2026-09-30.md`` §3): names, quarantine lists, balance.

    python3 -m v2.data.hr2.audit names --out RECEIPT
    python3 -m v2.data.hr2.audit quarantine --candidates T --candidates D \
        --quarantining PIV4Q.private.json --quarantining PIHR2.private.json \
        --full PIV4.private.json --quarantining-manifest MANIFEST.json \
        --self-scan SELF.private.json --out-dir DIR
    python3 -m v2.data.hr2.audit families --rows ROWS --out-dir DIR
    python3 -m v2.data.hr2.audit stats --train T --dev D [--tokens TOK ...] --out RECEIPT

``names`` (G1) searches a fixed list of repository documents (training licence registries, the M6
credits, the Decision 1.0 inventory, the C1 registry and the protected-panel records) for a fixed term
list of the seven sources and their parents. ``quarantine`` (G2) turns overlap receipts into the
group lists ``finalize`` drops: any group with a hit on a quarantining role (TRAIN and DEV), and DEV
groups that near-duplicate a TRAIN group. ``families`` writes one row file per family for the
per-family shortcut audit (G4). ``stats`` checks the G5 balance rules on final files and reports the
G8 sizes. Public receipts hold counts only.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import statistics
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from training.model.data import canonical
from v2.data import overlap

V2 = Path(__file__).resolve().parents[2]
NAME_DOCUMENTS = (
    "data/records/license-registry-v1.json",
    "data/records/license-registry-v2.json",
    "data/records/license-registry-m3b.json",
    "data/records/license-registry-m4.json",
    "data/records/license-registry-hs1.json",
    "data/a7/license-registry-a7-v1.json",
    "data/a7/license-registry-a7-v2.json",
    "release/records/dev2-0p6b-m8-release-2026-09-29/credits/m6-sources.json",
    "release/records/dev2-0p6b-m8-release-2026-09-29/credits/training-attribution.txt",
    "data/a7/records/a7-inventory-2026-09-28.md",
    "eval/sealed/c1-source-terms.json",
    "eval/sealed/c1-config.json",
    "eval/records/sealed-c1-source-registry-2026-09-28.md",
    "eval/records/htdev-prereg-2026-09-29.md",
    "eval/records/htdev-isolation-2026-09-29.md",
    "eval/records/htdev2-prereg-2026-09-30.md",
    "eval/records/mlx-diag-v1-2026-09-28.md",
    "eval/records/jevbench-value-2026-09-29.md",
)
# Source identifiers and parent corpora; a hit in any document rejects the source.
NAME_TERMS = {
    "helpsteer3_train": ["helpsteer3", "helpsteer 3", "wildchat", "sharegpt"],
    "ethics_train": ["hendrycks/ethics", "ethics dataset", "shared human values"],
    "prm800k_train": ["prm800k", "prm 800k", "competition_math", "hendrycks/math"],
    "vitaminc_real_train": ["vitaminc", "vitamin c", "vitamin-c"],
    "kobest_boolq_train": ["kobest", "kb-boolq", "skt/kobest"],
    "indonli_train": ["indonli", "ir-nlp-csui", "indosum"],
    "allegro_reviews_train": ["allegro", "klej"],
}
BOUNDS = (0.45, 0.55)
HELP_MAX_SHARE = 0.30


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_new(path: Path, text: str) -> str:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> str:
    return write_new(path, json.dumps(payload, indent=1, sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- G1 names


def names(root: Path = V2) -> dict[str, Any]:
    documents, hits = [], {}
    for rel in NAME_DOCUMENTS:
        path = root / rel
        text = path.read_text(encoding="utf-8").casefold()
        documents.append({"path": rel, "sha256": file_sha256(path)})
        for source, terms in NAME_TERMS.items():
            for term in terms:
                count = text.count(term.casefold())
                if count:
                    hits.setdefault(source, {}).setdefault(term, {})[rel] = count
    return {
        "schema": "decision2.hr2.names.v1",
        "documents": documents,
        "terms": NAME_TERMS,
        "hits": hits,
        "verdict": "FAIL" if hits else "PASS",
    }


# --------------------------------------------------------------------------- G2 quarantine


def flagged(receipt: Mapping[str, Any]) -> dict[str, set[str]]:
    """Group -> roles of every non-boilerplate E / S / L / N hit."""
    return {
        group: set(record.get("roles", []))
        for group, record in receipt["groups"].items()
        if set(record["methods"]) & set(overlap.METHODS)
    }


def quarantine(
    rows: Iterable[Mapping[str, Any]],
    quarantining: Sequence[Mapping[str, Any]],
    full: Mapping[str, Any],
    quarantining_roles: set[str],
    self_scan: Mapping[str, Any],
) -> tuple[set[str], set[str], dict[str, Any]]:
    family_of: dict[str, set[str]] = collections.defaultdict(set)
    split_rows: collections.Counter = collections.Counter()
    for row in rows:
        family_of[row["group_id"]].add(row["family"])
        split_rows[(row["group_id"], row["split"])] += 1
    drop: set[str] = set()
    for receipt in quarantining:
        drop |= set(flagged(receipt))
    report_only: collections.Counter = collections.Counter()
    for group, roles in flagged(full).items():
        if roles & quarantining_roles:
            drop.add(group)
        else:
            for role in roles:
                report_only[role] += 1
    dev_drop = {pair["a"] for pair in self_scan["pairs"]} - drop

    def by_family(groups: set[str]) -> dict[str, dict[str, int]]:
        out: dict[str, collections.Counter] = collections.defaultdict(
            collections.Counter
        )
        for group in groups:
            for family in family_of.get(group, {"(unknown)"}):
                out[family]["groups"] += 1
                out[family]["train_rows"] += split_rows[(group, "train")]
                out[family]["dev_rows"] += split_rows[(group, "select")]
        return {name: dict(sorted(c.items())) for name, c in sorted(out.items())}

    unknown = sorted(group for group in drop | dev_drop if group not in family_of)
    if unknown:
        raise ValueError(f"{len(unknown)} flagged groups are not candidate groups")
    public = {
        "schema": "decision2.hr2.quarantine.v1",
        "quarantine": {"groups": len(drop), "by_family": by_family(drop)},
        "dev_near_train": {"groups": len(dev_drop), "by_family": by_family(dev_drop)},
        "report_only_roles_hit": dict(sorted(report_only.items())),
    }
    return drop, dev_drop, public


# --------------------------------------------------------------------------- G5 / G8 stats


def share(part: int, whole: int) -> float | None:
    return round(part / whole, 4) if whole else None


def balance(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    out: dict[str, Any] = {}
    fails: list[str] = []
    by_family: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    low, high = BOUNDS
    for name, members in sorted(by_family.items()):
        kind = members[0]["task_type"]
        labels = collections.Counter(row["label"] for row in members)
        entry: dict[str, Any] = {
            "rows": len(members),
            "labels": {str(k): v for k, v in sorted(labels.items())},
        }
        hr2 = [row["audit_metadata"]["hr2"] for row in members]
        if kind == "noul":
            if name == "prm_step":
                cells = collections.Counter(
                    (h["bucket"], r["label"]) for h, r in zip(hr2, members)
                )
                buckets = sorted({bucket for bucket, _ in cells})
                entry["bucket_labels"] = {
                    b: [cells[(b, 0)], cells[(b, 1)]] for b in buckets
                }
                ok = all(cells[(b, 0)] == cells[(b, 1)] for b in buckets)
                steps = {
                    label: [
                        h["step_index"]
                        for h, r in zip(hr2, members)
                        if r["label"] == label
                    ]
                    for label in (0, 1)
                }
                entry["mean_step_index"] = {
                    str(k): round(statistics.fmean(v), 3) if v else None
                    for k, v in steps.items()
                }
            elif name == "indonli":
                upstream = collections.Counter(h["upstream_label"] for h in hr2)
                entry["upstream"] = dict(sorted(upstream.items()))
                ok = (
                    upstream["e"] == upstream["c"] + upstream["n"]
                    and upstream["c"] == upstream["n"]
                )
            else:
                ok = labels[0] == labels[1]
            lengths = {
                label: [
                    sum(len(str(v)) for v in r["state"].values())
                    for r in members
                    if r["label"] == label
                ]
                for label in (0, 1)
            }
            entry["mean_state_chars"] = {
                str(k): round(statistics.fmean(v), 1) if v else None
                for k, v in lengths.items()
            }
        elif kind == "choice":
            gold_a = share(labels[0], len(members))
            longer = [h.get("gold_longer") for h in hr2]
            decided = [flag for flag in longer if flag is not None]
            gold_longer = share(sum(decided), len(decided))
            entry.update(gold_a_share=gold_a, gold_longer_share=gold_longer)
            ok = (
                low <= gold_a <= high
                and gold_longer is not None
                and low <= gold_longer <= high
            )
        else:
            top = max(labels.values()) / len(members)
            entry["top_level_share"] = round(top, 4)
            if name == "hs3_help":
                ok = top <= HELP_MAX_SHARE
            else:
                ok = len(set(labels.values())) == 1 and len(labels) == len(
                    members[0]["options"]
                )
        entry["pass"] = ok
        if not ok:
            fails.append(name)
        out[name] = entry
    return out, fails


def sizes(
    rows: Sequence[Mapping[str, Any]], tokens: Mapping[str, Mapping[str, int]]
) -> dict[str, Any]:
    by_family: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    languages: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    groups: dict[str, set[str]] = collections.defaultdict(set)
    for row in rows:
        entry = by_family[row["family"]]
        entry["rows"] += 1
        entry["native_tokens"] += tokens.get(row["id"], {}).get("native", 0)
        languages[row["family"]][row["language"]] += 1
        groups[row["family"]].add(row["group_id"])
    return {
        "rows": len(rows),
        "groups": len({row["group_id"] for row in rows}),
        "task_type": dict(
            sorted(collections.Counter(r["task_type"] for r in rows).items())
        ),
        "language": dict(
            sorted(collections.Counter(r["language"] for r in rows).items())
        ),
        "native_tokens": sum(tokens.get(r["id"], {}).get("native", 0) for r in rows),
        "tokens_missing": sum(r["id"] not in tokens for r in rows) if tokens else None,
        "by_family": {
            name: {
                **dict(entry),
                "groups": len(groups[name]),
                "languages": dict(sorted(languages[name].items())),
            }
            for name, entry in sorted(by_family.items())
        },
    }


def stats(
    train: Sequence[Mapping[str, Any]],
    dev: Sequence[Mapping[str, Any]],
    tokens: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    out: dict[str, Any] = {"schema": "decision2.hr2.stats.v1"}
    fails = []
    for name, rows in (("train", train), ("dev", dev)):
        balanced, failed = balance(rows)
        out[name] = {"sizes": sizes(rows, tokens), "balance": balanced}
        fails += [f"{name}:{family}" for family in failed]
    train_groups = {r["group_id"] for r in train}
    out["shared_groups"] = len(train_groups & {r["group_id"] for r in dev})
    out["balance_failures"] = fails
    out["verdict"] = "PASS" if not fails and not out["shared_groups"] else "FAIL"
    return out


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("names")
    one.add_argument("--out", type=Path, required=True)
    two = sub.add_parser("quarantine")
    two.add_argument("--candidates", type=Path, action="append", required=True)
    two.add_argument("--quarantining", type=Path, action="append", required=True)
    two.add_argument("--full", type=Path, required=True)
    two.add_argument("--quarantining-manifest", type=Path, required=True)
    two.add_argument("--self-scan", type=Path, required=True)
    two.add_argument("--out-dir", type=Path, required=True)
    three = sub.add_parser("families")
    three.add_argument("--rows", type=Path, required=True)
    three.add_argument("--out-dir", type=Path, required=True)
    four = sub.add_parser("stats")
    four.add_argument("--train", type=Path, required=True)
    four.add_argument("--dev", type=Path, required=True)
    four.add_argument("--tokens", type=Path, action="append", default=[])
    four.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "names":
        receipt = names()
        write_json(args.out, receipt)
        print(json.dumps({"verdict": receipt["verdict"], "hits": receipt["hits"]}))
        return 0
    if args.command == "quarantine":
        rows = [row for path in args.candidates for row in read_jsonl(path)]
        roles = {
            entry["role"]
            for entry in json.loads(
                args.quarantining_manifest.read_text(encoding="utf-8")
            )
        }
        drop, dev_drop, public = quarantine(
            rows,
            [json.loads(p.read_text(encoding="utf-8")) for p in args.quarantining],
            json.loads(args.full.read_text(encoding="utf-8")),
            roles,
            json.loads(args.self_scan.read_text(encoding="utf-8")),
        )
        args.out_dir.mkdir(mode=0o700)
        public["drop_groups_sha256"] = write_new(
            args.out_dir / "drop-groups.txt", "".join(g + "\n" for g in sorted(drop))
        )
        public["drop_dev_groups_sha256"] = write_new(
            args.out_dir / "drop-dev-groups.txt",
            "".join(g + "\n" for g in sorted(dev_drop)),
        )
        write_json(args.out_dir / "quarantine.public.json", public)
        print(json.dumps({"quarantine": len(drop), "dev_near_train": len(dev_drop)}))
        return 0
    if args.command == "families":
        by_family: dict[str, list[str]] = collections.defaultdict(list)
        for row in read_jsonl(args.rows):
            by_family[row["family"]].append(canonical(row) + "\n")
        args.out_dir.mkdir(mode=0o700)
        for name, lines in sorted(by_family.items()):
            write_new(args.out_dir / f"{name}.jsonl", "".join(lines))
        print(
            json.dumps({name: len(lines) for name, lines in sorted(by_family.items())})
        )
        return 0
    tokens = {item["id"]: item for path in args.tokens for item in read_jsonl(path)}
    receipt = stats(read_jsonl(args.train), read_jsonl(args.dev), tokens)
    receipt["inputs"] = {
        "train": file_sha256(args.train),
        "dev": file_sha256(args.dev),
        "tokens": [file_sha256(path) for path in args.tokens],
    }
    write_json(args.out, receipt)
    print(
        json.dumps(
            {"verdict": receipt["verdict"], "fails": receipt["balance_failures"]}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
