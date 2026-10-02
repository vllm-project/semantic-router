"""IB4 audits (prereg ``records/ib4-prereg-2026-10-02.md`` §3): G1 names, G4 audit rows, G5 / G8 statistics.

    python3 -m v2.data.ib4.audit names --out RECEIPT
    python3 -m v2.data.ib4.audit g4rows --rows TRAIN --out-dir DIR
    python3 -m v2.data.ib4.audit stats --train T --dev D [--tokens TOK ...] --out RECEIPT

``names`` is IB2's line scan with IB4's terms and the prereg's known lineage hits. ``g4rows`` writes one file per
family for ``v2.data.shortcut`` with the family's declared hypothesis field renamed to ``claim`` (audit copy only).
``stats`` checks G5 (equal rows per label overall and inside every declared cell) and reports the G8 sizes.
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
from v2.data.ib4.build import LABELS
from v2.data.ib4.families import HYPOTHESIS_FIELDS

LINEAGE_DOCUMENTS = ib2_audit.LINEAGE_DOCUMENTS + (
    "data/records/license-registry-ib2.json",
    "data/records/license-registry-ib3.json",
)
NAME_TERMS = {
    "squad_v2_train": ["squad"],
    "mendeley_sms_phishing": ["sms phishing", "smishing", "f45bkkt8pr"],
    "isarcasmeval_train": ["isarcasm"],
    "sentfin_v1": ["sentfin"],
    "when2call_train": ["when2call"],
    "glaive_function_calling_v2": ["glaive"],
}
# Lines that name SQuAD as an existing training (project-used) source, not as a panel or C1 source.
KNOWN_LINEAGE = frozenset(
    {
        ("eval/records/sealed-c1-source-registry-2026-09-28.md", 115, "squad"),
        ("eval/records/htdev-isolation-2026-09-29.md", 25, "squad"),
        ("eval/records/htdev-isolation-2026-09-29.md", 29, "squad"),
    }
)


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
        "schema": "decision2.ib4.names.v1",
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


def audit_copy(row: Mapping[str, Any]) -> dict[str, Any]:
    field = HYPOTHESIS_FIELDS.get(row["family"])
    if not field:
        return dict(row)
    state = {("claim" if k == field else k): v for k, v in row["state"].items()}
    copy = {**row, "state": state}
    copy["input_sha256"] = digest({name: copy[name] for name in INPUT_FIELDS})
    return copy


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
    receipt = {
        "schema": "decision2.ib4.g4rows.v1",
        "input": file_sha256(path),
        "rows": counts,
        "hypothesis_fields": HYPOTHESIS_FIELDS,
    }
    write_json(out.parent / "g4rows.json", receipt)
    return receipt


def balance(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    out: dict[str, Any] = {}
    fails: list[str] = []
    by_family: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    for name, members in sorted(by_family.items()):
        wanted = LABELS.get(name, (0, 1))
        labels = collections.Counter(row["label"] for row in members)
        cells = collections.Counter(
            (row["audit_metadata"]["ib4"]["cell"], row["label"]) for row in members
        )
        names_ = {cell for cell, _ in cells}
        unbalanced = sum(
            len({cells[(c, label)] for label in wanted}) > 1 for c in names_
        )
        ok = len({labels[label] for label in wanted}) == 1 and unbalanced == 0
        out[name] = {
            "rows": len(members),
            "labels": {str(k): v for k, v in sorted(labels.items())},
            "cells": len(names_),
            "unbalanced_cells": unbalanced,
            "pass": ok,
        }
        if not ok:
            fails.append(name)
    return out, fails


def stats(train, dev, tokens) -> dict[str, Any]:
    out: dict[str, Any] = {"schema": "decision2.ib4.stats.v1"}
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
        print(json.dumps(g4rows(args.rows, args.out_dir)["rows"]))
        return 0
    tokens = {item["id"]: item for path in args.tokens for item in read_jsonl(path)}
    receipt = stats(read_jsonl(args.train), read_jsonl(args.dev), tokens)
    receipt["inputs"] = {
        "train": file_sha256(args.train),
        "dev": file_sha256(args.dev),
        "tokens": [file_sha256(p) for p in args.tokens],
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
