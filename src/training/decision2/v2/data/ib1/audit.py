"""IB1 audits (prereg ``records/ib1-prereg-2026-10-01.md`` §3): G1 names and the G5 / G8 statistics.

    python3 -m v2.data.ib1.audit names --out RECEIPT
    python3 -m v2.data.ib1.audit stats --train T --dev D [--tokens TOK ...] --out RECEIPT

``names`` searches the protected-list documents (C1 registry, panel records, JevBench, mlx-diag) for a fixed term
list of the IB1 sources and their parents — a hit rejects the source — and the training registries, the M6 credits
and the Decision 1.0 inventory, where a hit is only reported (lineage overlap is allowed in IB1 and disclosed).
G2 quarantine lists and per-family row files come from ``v2.data.hr2.audit quarantine`` and ``families``.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from v2.data.hr2.audit import file_sha256, read_jsonl, sizes, write_json
from v2.data.ib1 import build

V2 = Path(__file__).resolve().parents[2]
PROTECTED_DOCUMENTS = (
    "eval/sealed/c1-source-terms.json",
    "eval/sealed/c1-config.json",
    "eval/records/sealed-c1-source-registry-2026-09-28.md",
    "eval/records/htdev-prereg-2026-09-29.md",
    "eval/records/htdev-isolation-2026-09-29.md",
    "eval/records/htdev2-prereg-2026-09-30.md",
    "eval/records/mlx-diag-v1-2026-09-28.md",
    "eval/records/jevbench-value-2026-09-29.md",
)
LINEAGE_DOCUMENTS = (
    "data/records/license-registry-v1.json",
    "data/records/license-registry-v2.json",
    "data/records/license-registry-m3b.json",
    "data/records/license-registry-m4.json",
    "data/records/license-registry-hs1.json",
    "data/records/license-registry-hr2.json",
    "data/a7/license-registry-a7-v1.json",
    "data/a7/license-registry-a7-v2.json",
    "release/records/dev2-0p6b-m8-release-2026-09-29/credits/m6-sources.json",
    "release/records/dev2-0p6b-m8-release-2026-09-29/credits/training-attribution.txt",
    "data/a7/records/a7-inventory-2026-09-28.md",
)
NAME_TERMS = {
    "summedits": ["summedits", "factualnlg"],
    "expertqa_r2": ["expertqa", "expert-qa"],
    "sms_spam_collection": ["sms spam", "sms_spam", "smsspam"],
    "when2call_train_pref": ["when2call"],
    "snips_2017_custom_intents": ["snips", "nlu-benchmark"],
    "args_me_portals": [
        "args.me",
        "args_me",
        "argsme",
        "idebate",
        "debatewise",
        "debatepedia",
    ],
    "isarcasmeval_train": ["isarcasm"],
    "processbench_math_apache": ["processbench"],
    "sentfin_v1": ["sentfin"],
    "wands": ["wands", "wayfair"],
    "maud_train_main": ["maud", "merger agreement understanding"],
    "balanced_copa_train": ["copa", "choice of plausible alternatives"],
    "commonsenseqa_train": ["commonsense_qa", "commonsenseqa", "commonsense qa"],
    "medmcqa_train": ["medmcqa"],
    "wanli_train": ["wanli"],
    "gutenberg_poetry_cc0": ["gutenberg"],
}
CLASS_MARGIN = 0.05
BOUNDS = (0.45, 0.55)


def scan_terms(
    documents: Sequence[str], root: Path
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    seen, hits = [], {}
    for rel in documents:
        path = root / rel
        text = path.read_text(encoding="utf-8").casefold()
        seen.append({"path": rel, "sha256": file_sha256(path)})
        for source, terms in NAME_TERMS.items():
            for term in terms:
                count = text.count(term.casefold())
                if count:
                    hits.setdefault(source, {}).setdefault(term, {})[rel] = count
    return seen, hits


def names(root: Path = V2) -> dict[str, Any]:
    protected_docs, protected = scan_terms(PROTECTED_DOCUMENTS, root)
    lineage_docs, lineage = scan_terms(LINEAGE_DOCUMENTS, root)
    return {
        "schema": "decision2.ib1.names.v1",
        "terms": NAME_TERMS,
        "protected_documents": protected_docs,
        "protected_hits": protected,
        "lineage_documents": lineage_docs,
        "lineage_hits": lineage,
        "verdict": "FAIL" if protected else "PASS",
    }


# --------------------------------------------------------------------------- G5 / G8


def share(part: int, whole: int) -> float | None:
    return round(part / whole, 4) if whole else None


def balance(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    out: dict[str, Any] = {}
    fails: list[str] = []
    by_family: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    for name, members in sorted(by_family.items()):
        labels = collections.Counter(row["label"] for row in members)
        k = len(members[0]["options"])
        entry: dict[str, Any] = {
            "rows": len(members),
            "labels": {str(key): value for key, value in sorted(labels.items())},
        }
        meta = [row["audit_metadata"]["ib1"] for row in members]
        if name in build.NOUL_FAMILIES:
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
            if name == "sumedit":
                cells = collections.Counter(
                    (m["domain"], r["label"]) for m, r in zip(meta, members)
                )
                ok = ok and all(cells[(d, 0)] == cells[(d, 1)] for d, _ in cells)
        elif name in build.TWIN_FAMILIES:
            groups = collections.Counter(row["group_id"] for row in members)
            ok = labels[0] == labels[1] and set(groups.values()) == {2}
        elif name in build.AB_FAMILIES:
            gold_a = share(labels[0], len(members))
            decided = [
                m.get("gold_longer") for m in meta if m.get("gold_longer") is not None
            ]
            gold_longer = share(sum(decided), len(decided))
            entry.update(gold_a_share=gold_a, gold_longer_share=gold_longer)
            low, high = BOUNDS
            ok = low <= gold_a <= high and (
                gold_longer is None or low <= gold_longer <= high
            )
        else:
            strata: dict[int, collections.Counter] = collections.defaultdict(
                collections.Counter
            )
            for row in members:
                strata[len(row["options"])][row["label"]] += 1
            entry["shares_by_option_count"] = {
                str(size): [share(c[p], sum(c.values())) for p in range(size)]
                for size, c in sorted(strata.items())
            }
            ok = True
            for size, c in strata.items():
                total = sum(c.values())
                small = total < 20 * size
                entry.setdefault("small_strata_reported", [])
                if small:
                    entry["small_strata_reported"].append(size)
                    continue
                ok = ok and all(
                    abs(c[p] / total - 1 / size) <= CLASS_MARGIN + 1e-9
                    for p in range(size)
                )
        entry["pass"] = ok
        if not ok:
            fails.append(name)
        out[name] = entry
    return out, fails


def stats(
    train: Sequence[Mapping[str, Any]],
    dev: Sequence[Mapping[str, Any]],
    tokens: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    out: dict[str, Any] = {"schema": "decision2.ib1.stats.v1"}
    fails = []
    for name, rows in (("train", train), ("dev", dev)):
        balanced, failed = balance(rows)
        out[name] = {"sizes": sizes(rows, tokens), "balance": balanced}
        fails += [f"{name}:{family}" for family in failed]
    out["shared_groups"] = len(
        {r["group_id"] for r in train} & {r["group_id"] for r in dev}
    )
    out["balance_failures"] = fails
    out["verdict"] = "PASS" if not fails and not out["shared_groups"] else "FAIL"
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("names")
    one.add_argument("--out", type=Path, required=True)
    two = sub.add_parser("stats")
    two.add_argument("--train", type=Path, required=True)
    two.add_argument("--dev", type=Path, required=True)
    two.add_argument("--tokens", type=Path, action="append", default=[])
    two.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "names":
        receipt = names()
        write_json(args.out, receipt)
        print(
            json.dumps(
                {
                    "verdict": receipt["verdict"],
                    "protected_hits": receipt["protected_hits"],
                }
            )
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
