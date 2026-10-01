"""IB2 audits (prereg ``records/ib2-prereg-2026-10-01.md`` §3): G1 names and the G5 / G8 statistics.

    python3 -m v2.data.ib2.audit names --out RECEIPT
    python3 -m v2.data.ib2.audit stats --train T --dev D [--tokens TOK ...] --out RECEIPT

``names`` searches IB1's protected-list documents line by line for the IB2 source terms. A hit rejects the source
unless it is one of the lineage hits the prereg fixes (lines that name the source as an existing training source the
panels were isolated from); the training registries, the M6 credits and the Decision 1.0 inventory are searched too,
where a hit is only reported. ``stats`` checks G5 per family and reports the G8 sizes.
"""

from __future__ import annotations

import argparse
import collections
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from v2.data.hr2.audit import file_sha256, read_jsonl, sizes, write_json
from v2.data.ib1.audit import LINEAGE_DOCUMENTS as IB1_LINEAGE
from v2.data.ib1.audit import PROTECTED_DOCUMENTS
from v2.data.ib2 import build

V2 = Path(__file__).resolve().parents[2]
LINEAGE_DOCUMENTS = IB1_LINEAGE + ("data/records/license-registry-ib1.json",)
NAME_TERMS = {
    "glaive_fc_v2": ["glaive"],
    "uci_youtube_spam": ["youtube spam", "youtube_spam", "tubespam"],
    "ibm_argq_30k": ["argq-30k", "argq-rank", "argument_quality_ranking", "30kargs"],
    "hover_train": ["hover"],
    "qasc_train": ["qasc"],
    "arc_train": ["ai2_arc", "arc-easy", "arc-challenge", "ai2 reasoning challenge"],
    "gsm8k_train": ["gsm8k"],
    "contractnli_train": ["contractnli", "contract-nli", "contract_nli"],
}
# Protected-document hits the prereg (§3 G1) classifies as lineage: (document, line, term).
KNOWN_LINEAGE = frozenset(
    {
        ("eval/records/htdev-prereg-2026-09-29.md", 97, "argq-30k"),
        ("eval/records/htdev-isolation-2026-09-29.md", 165, "argq-30k"),
        ("eval/records/htdev-isolation-2026-09-29.md", 25, "hover"),
        ("eval/records/htdev-isolation-2026-09-29.md", 38, "hover"),
    }
)
BAND = 0.05
HOVER_BAND = (0.58, 0.62)


def scan_lines(
    documents: Sequence[str], root: Path
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    seen, hits = [], []
    for rel in documents:
        path = root / rel
        seen.append({"path": rel, "sha256": file_sha256(path)})
        for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
            low = line.casefold()
            for source, terms in NAME_TERMS.items():
                for term in terms:
                    if term.casefold() in low:
                        hits.append(
                            {
                                "source": source,
                                "term": term,
                                "path": rel,
                                "line": number,
                            }
                        )
    return seen, hits


def names(root: Path = V2) -> dict[str, Any]:
    protected_docs, protected = scan_lines(PROTECTED_DOCUMENTS, root)
    lineage_docs, lineage = scan_lines(LINEAGE_DOCUMENTS, root)
    for hit in protected:
        hit["class"] = (
            "lineage (prereg)"
            if (hit["path"], hit["line"], hit["term"]) in KNOWN_LINEAGE
            else "protected"
        )
    rejected = sorted({h["source"] for h in protected if h["class"] == "protected"})
    return {
        "schema": "decision2.ib2.names.v1",
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
        entry: dict[str, Any] = {
            "rows": len(members),
            "labels": {str(key): value for key, value in sorted(labels.items())},
        }
        cell = [row["audit_metadata"]["ib2"]["cell"] for row in members]
        if name in build.NOUL_FAMILIES:
            ok = labels[0] == labels[1]
        elif name in build.RATIO_FAMILIES:
            yes_share = share(labels[1], len(members))
            entry["yes_share"] = yes_share
            ok = HOVER_BAND[0] <= (yes_share or 0) <= HOVER_BAND[1]
        elif name in build.HYPOTHESIS_FAMILIES or name in build.TOPIC_FAMILIES:
            by = collections.Counter(zip(cell, (r["label"] for r in members)))
            groups = sorted(set(cell))
            entry["cells"] = len(groups)
            ok = all(by[(c, 0)] == by[(c, 1)] for c in groups)
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
            entry["small_strata_reported"] = []
            ok = True
            for size, c in sorted(strata.items()):
                total = sum(c.values())
                if total < 20 * size:
                    entry["small_strata_reported"].append(size)
                    continue
                ok = ok and all(
                    abs(c[p] / total - 1 / size) <= BAND + 1e-9 for p in range(size)
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
    out: dict[str, Any] = {"schema": "decision2.ib2.stats.v1"}
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
        print(json.dumps({k: receipt[k] for k in ("verdict", "rejected_sources")}))
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
