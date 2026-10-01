"""IB3 audits (prereg ``records/ib3-prereg-2026-10-01.md`` §3): G1 names, G4 audit rows, G5 / G8 statistics.

    python3 -m v2.data.ib3.audit names --out RECEIPT
    python3 -m v2.data.ib3.audit g4rows --rows TRAIN --out-dir DIR
    python3 -m v2.data.ib3.audit stats --train T --dev D [--tokens TOK ...] --out RECEIPT

``names`` is IB2's line scan with IB3's terms and the prereg's known lineage hits. ``g4rows`` writes one file per
family for ``v2.data.shortcut``, with the family's declared hypothesis field renamed to ``claim`` (the audit copy only)
and, for the URL families, a receipt of the shape-cell label baseline. ``stats`` checks G5 (yes = no overall and in
every declared cell) and reports the G8 sizes.
"""

from __future__ import annotations

import argparse
import collections
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from training.model.data import INPUT_FIELDS, digest
from v2.data.hr2.audit import file_sha256, read_jsonl, sizes, write_json
from v2.data.ib1.audit import PROTECTED_DOCUMENTS
from v2.data.ib2 import audit as ib2_audit
from v2.data.ib3.families import HYPOTHESIS_FIELDS
from v2.data.shortcut import fold_of

LINEAGE_DOCUMENTS = ib2_audit.LINEAGE_DOCUMENTS + (
    "data/records/license-registry-ib2.json",
)
NAME_TERMS = {
    "mendeley_web_page_phishing": ["web page phishing", "hannousse", "c2gw7fy2j4"],
    "uci_phiusiil": ["phiusiil"],
    "faithdial_train": ["faithdial", "wizard of wikipedia", "wizard_of_wikipedia"],
    "halueval_qa": ["halueval", "hotpotqa"],
    "amazon_esci_train": ["esci", "shopping queries", "shopping_queries"],
    "mathqa_train": ["mathqa", "math_qa", "aqua-rat", "aqua_rat"],
    "maud_train_main": ["maud", "merger agreement understanding"],
}
KNOWN_LINEAGE = frozenset(
    {
        ("eval/records/htdev-isolation-2026-09-29.md", 38, "hotpotqa"),
        ("eval/records/jevbench-value-2026-09-29.md", 252, "hotpotqa"),
        ("eval/records/jevbench-value-2026-09-29.md", 344, "hotpotqa"),
    }
)
URL_FAMILIES = ("wpd", "phiu")


def names() -> dict[str, Any]:
    ib2_audit.NAME_TERMS = NAME_TERMS
    protected_docs, protected = ib2_audit.scan_lines(PROTECTED_DOCUMENTS, ib2_audit.V2)
    lineage_docs, lineage = ib2_audit.scan_lines(LINEAGE_DOCUMENTS, ib2_audit.V2)
    for hit in protected:
        hit["class"] = (
            "lineage (prereg)"
            if (hit["path"], hit["line"], hit["term"]) in KNOWN_LINEAGE
            else "protected"
        )
    rejected = sorted({h["source"] for h in protected if h["class"] == "protected"})
    return {
        "schema": "decision2.ib3.names.v1",
        "terms": NAME_TERMS,
        "protected_documents": protected_docs,
        "protected_hits": protected,
        "rejected_sources": rejected,
        "lineage_documents": lineage_docs,
        "lineage_hits": dict(
            sorted(
                collections.Counter(
                    f"{h['source']}|{h['path']}" for h in lineage
                ).items()
            )
        ),
        "verdict": "FAIL" if rejected else "PASS",
    }


# --------------------------------------------------------------------------- G4 audit rows


def audit_copy(row: Mapping[str, Any]) -> dict[str, Any]:
    field = HYPOTHESIS_FIELDS.get(row["family"])
    if not field:
        return dict(row)
    state = {
        ("claim" if key == field else key): value for key, value in row["state"].items()
    }
    copy = {**row, "state": state}
    copy["input_sha256"] = digest({name: copy[name] for name in INPUT_FIELDS})
    return copy


def shape_baseline(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Group-disjoint 5-fold label baseline that predicts each row's cell majority (ties -> yes)."""
    correct = 0
    for fold in range(5):
        train = collections.defaultdict(collections.Counter)
        for row in rows:
            if fold_of(row["group_id"]) != fold:
                train[row["audit_metadata"]["ib3"]["cell"]][row["label"]] += 1
        for row in rows:
            if fold_of(row["group_id"]) == fold:
                seen = train[row["audit_metadata"]["ib3"]["cell"]]
                guess = 1 if seen[1] >= seen[0] else 0
                correct += guess == row["label"]
    return {
        "rows": len(rows),
        "cells": len({r["audit_metadata"]["ib3"]["cell"] for r in rows}),
        "cell_baseline_accuracy": round(correct / len(rows), 4) if rows else None,
        "yes_share": (
            round(sum(r["label"] for r in rows) / len(rows), 4) if rows else None
        ),
    }


def g4rows(path: Path, out: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    out.mkdir(mode=0o700)
    by: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by[row["family"]].append(audit_copy(row))
    counts = {}
    for family, members in sorted(by.items()):
        with (out / f"{family}.jsonl").open("x", encoding="utf-8") as stream:
            for row in members:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        counts[family] = len(members)
    baseline = {f: shape_baseline(by[f]) for f in URL_FAMILIES if f in by}
    receipt = {
        "schema": "decision2.ib3.g4rows.v1",
        "input": file_sha256(path),
        "rows": counts,
        "hypothesis_fields": HYPOTHESIS_FIELDS,
        "url_shape_baseline": baseline,
    }
    write_json(out.parent / "g4rows.json", receipt)
    return receipt


# --------------------------------------------------------------------------- G5 / G8


def balance(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    out: dict[str, Any] = {}
    fails: list[str] = []
    by_family: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    for name, members in sorted(by_family.items()):
        labels = collections.Counter(row["label"] for row in members)
        cells = collections.Counter(
            (row["audit_metadata"]["ib3"]["cell"], row["label"]) for row in members
        )
        names = {cell for cell, _ in cells}
        unbalanced = sum(cells[(c, 0)] != cells[(c, 1)] for c in names)
        ok = labels[0] == labels[1] and unbalanced == 0
        out[name] = {
            "rows": len(members),
            "labels": {str(k): v for k, v in sorted(labels.items())},
            "cells": len(names),
            "unbalanced_cells": unbalanced,
            "pass": ok,
        }
        if not ok:
            fails.append(name)
    return out, fails


def stats(
    train: Sequence[Mapping[str, Any]],
    dev: Sequence[Mapping[str, Any]],
    tokens: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    out: dict[str, Any] = {"schema": "decision2.ib3.stats.v1"}
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
    two = sub.add_parser("g4rows")
    two.add_argument("--rows", type=Path, required=True)
    two.add_argument("--out-dir", type=Path, required=True)
    three = sub.add_parser("stats")
    three.add_argument("--train", type=Path, required=True)
    three.add_argument("--dev", type=Path, required=True)
    three.add_argument("--tokens", type=Path, action="append", default=[])
    three.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "names":
        receipt = names()
        write_json(args.out, receipt)
        print(json.dumps({k: receipt[k] for k in ("verdict", "rejected_sources")}))
        return 0
    if args.command == "g4rows":
        receipt = g4rows(args.rows, args.out_dir)
        print(
            json.dumps(
                {
                    "rows": receipt["rows"],
                    "url_shape_baseline": receipt["url_shape_baseline"],
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
