"""Combine the HT-DEV isolation checks into ADMISSION.json (prereg §3 items 2, 3, 5(ii), 6, 7).

    python3 -m v2.eval.htdev_iso.admission --iso <iso-dir> --code-commit SHA \
        --terms htdev-source-terms.json --flags flags.json --names-review names-review.json \
        --families embed-families.json [--embed-private <embed.private.json>] \
        --output <ADMISSION.json> --summary <summary.json>

Names (item 2): a hit counts as dataset presence when it is a row-provenance hit or a
text/code hit outside the eval-track code, tests and the aggregator root, unless
names-review.json clears it (by key and path regex, with a reason). Lexical (item 3):
per source, the distinct training rows matched at containment >= 0.8 or by an exact
span of >= 8 tokens; >= 5 such rows whose provenance fields carry one of the source's
terms reject the source; >= 5 without matching provenance are reported for review.
Embedding (item 5(ii)): >= 5 source rows at cosine >= 0.93 against one training family
reject the source. flagged_rows are lexical REVIEW/OVERLAP rows and embedding rows at
>= 0.93. The summary holds counts only.
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from v2.eval.sealed.independence import PROVENANCE, strings
from v2.eval.sealed.schema import normalized

STRONG_CONTAINMENT = 0.8
STRONG_TOKENS = 8
MIN_ROWS = 5


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify(path: str) -> str:
    if "/c1-corpora/aggregators/" in path:
        return "aggregator"
    if path.startswith("/data/dev2/src/"):
        if "/v2/eval/" in path or "/tests/" in path or "/test_" in path:
            return "eval-or-test-code"
        return "code-other"
    if path.endswith((".jsonl", ".jsonl.gz")):
        return "row-provenance"
    return "text-other"


def provenance_of(locations: dict[str, set[int]]) -> dict[tuple[str, int], str]:
    """Provenance-field text of the given (file, row) training locations."""
    found: dict[tuple[str, int], str] = {}
    for file, rows in locations.items():
        if file.endswith((".jsonl", ".jsonl.gz")):
            opener = gzip.open if file.endswith(".gz") else open
            with opener(file, "rt", encoding="utf-8", errors="replace") as stream:
                for number, line in enumerate(stream):
                    if number in rows:
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            row = {}
                        found[(file, number)] = (
                            " ".join(
                                str(v)
                                for k in PROVENANCE
                                if k in row
                                for v in strings(row[k])
                            )
                            if isinstance(row, dict)
                            else ""
                        )
        elif file.endswith(".parquet"):
            import pyarrow.parquet as pq

            table = pq.read_table(file)
            columns = [c for c in PROVENANCE if c in table.column_names]
            for number in rows:
                found[(file, number)] = " ".join(
                    str(table.column(c)[number].as_py()) for c in columns
                )
        else:
            for number in rows:
                found[(file, number)] = file
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iso", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--terms", type=Path, required=True)
    parser.add_argument("--flags", type=Path, required=True)
    parser.add_argument("--names-review", type=Path, required=True)
    parser.add_argument("--families", type=Path, required=True)
    parser.add_argument("--embed-private", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    iso, work = args.iso, args.iso / "work"
    terms = json.loads(args.terms.read_text(encoding="utf-8"))
    flags = json.loads(args.flags.read_text(encoding="utf-8"))
    reviewed = json.loads(args.names_review.read_text(encoding="utf-8"))
    review, confirmed = reviewed["clear"], reviewed.get("confirm", [])
    families = {
        n: re.compile(p, re.IGNORECASE)
        for n, p in json.loads(args.families.read_text())["patterns"].items()
    }
    sources = json.loads((iso / "SOURCES-MANIFEST.json").read_text())["sources"]
    names = json.loads((work / "names.json").read_text())
    receipt = json.loads((work / "overlap-receipt.json").read_text())

    result: dict[str, dict[str, Any]] = {}
    for key, entry in sources.items():
        result[key] = {
            "admitted": True,
            "reasons": [],
            "counts": {"rows": entry.get("rows_with_text", 0)},
            "flags": {
                "sni": key in flags["sni"]["tasks"],
                "sni_tasks": len(flags["sni"]["tasks"].get(key, [])),
                "aggregators": flags["aggregators"]["hits"].get(key, []),
            },
            "flagged_rows": [],
        }
        if entry.get("status") != "ok":
            result[key]["admitted"] = False
            result[key]["reasons"].append(f"not checked: {entry.get('status')}")

    for key, paths in names["hits"].items():
        cell = result[key]
        categories: Counter = Counter()
        presence = 0
        for path in paths:
            category = classify(path)
            cleared = any(
                r["key"] == key and re.search(r["path_regex"], path) for r in review
            )
            categories[category + (":cleared" if cleared else "")] += 1
            if (
                category in ("row-provenance", "code-other", "text-other")
                and not cleared
            ):
                presence += 1
        cell["counts"]["names_hit_files"] = dict(sorted(categories.items()))
        if presence:
            cell["admitted"] = False
            cell["reasons"].append(f"names: {presence} uncleared hit files")
    for item in confirmed:
        cell = result[item["key"]]
        cell["admitted"] = False
        cell["reasons"].append(f"names (confirmed by context): {item['reason']}")

    strong: dict[str, dict[tuple[str, int], int]] = defaultdict(dict)
    flagged: dict[str, dict[str, str]] = defaultdict(dict)
    verdicts: dict[str, Counter] = defaultdict(Counter)
    with (work / "overlap-hits.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            hit = json.loads(line)
            key = hit["source"]
            verdicts[key][hit["verdict"]] += 1
            if hit["verdict"] != "CLEAN":
                flagged[key][hit["id"]] = "lexical-" + hit["verdict"]
            for label, entry in hit["labels"].items():
                if entry["containment"] >= STRONG_CONTAINMENT and entry.get("file"):
                    strong[key][(entry["file"], entry["row"])] = 1
                if (entry.get("exact_tokens") or 0) >= STRONG_TOKENS:
                    strong[key][(entry["exact_file"], entry["exact_row"])] = 1
    locations: dict[str, set[int]] = defaultdict(set)
    for key, rows in strong.items():
        for file, row in rows:
            locations[file].add(row)
    provenance = provenance_of(locations)
    for key, cell in result.items():
        rows = strong.get(key, {})
        matched = sum(
            any(
                t.casefold() in provenance.get(loc, "").casefold()
                for t in terms.get(key, [])
            )
            for loc in rows
        )
        cell["counts"].update(
            {
                "lexical": dict(verdicts.get(key, {})),
                "lexical_by_label": receipt["by_source"].get(key, {}).get("labels", {}),
                "strong_training_rows": len(rows),
                "strong_training_rows_matching_provenance": matched,
            }
        )
        if matched >= MIN_ROWS:
            cell["admitted"] = False
            cell["reasons"].append(
                f"lexical: {matched} training rows with matching provenance"
            )
        elif len(rows) >= MIN_ROWS:
            cell["reasons"].append(
                f"lexical review: {len(rows)} strong training rows, provenance not the source"
            )

    if args.embed_private:
        embed = json.loads(args.embed_private.read_text())
        public = embed["public"]["by_file"]
        hits_by: dict[str, Counter] = defaultdict(Counter)
        where: dict[str, set[int]] = defaultdict(set)
        for record in embed["quarantined"]:
            path, _, number = record["protected_id"].rpartition("#")
            where[path].add(int(number))
        prov = provenance_of(where)
        for record in embed["quarantined"]:
            key = record["file"][: -len(".jsonl")]
            path, _, number = record["protected_id"].rpartition("#")
            text = prov.get((path, int(number)), "")
            family = next((n for n, p in families.items() if p.search(text)), "other")
            hits_by[key][family] += 1
            flagged[key].setdefault(record["row_id"], "embed>=0.93")
        for file, stats in public.items():
            key = file[: -len(".jsonl")]
            cell = result[key]
            cell["counts"]["embed"] = {
                "rows": stats.get("groups", 0),
                "ge_0_93": stats.get("quarantined_groups", 0),
                "band_0_85_0_93": stats.get("review_band_groups", 0),
                "ge_0_93_by_family": dict(hits_by.get(key, {})),
            }
            worst = max(hits_by.get(key, {"-": 0}).values())
            if worst >= MIN_ROWS:
                cell["admitted"] = False
                cell["reasons"].append(
                    f"embedding: {worst} rows >= 0.93 against one family"
                )
        for key, cell in result.items():
            if "embed" not in cell["counts"] and cell["admitted"]:
                cell["counts"]["embed"] = "not scanned (backup; amendment 1 item 5)"

    wanted = {i for ids in flagged.values() for i in ids}
    leaves: dict[str, tuple[str, int, list[str]]] = {}
    with (work / "protected-all-splits.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            identity = f"{row['task']}|{row['source_item_id']}"
            if identity in wanted:
                leaves[identity] = (
                    row["task"].split("/", 1)[1],
                    int(row["source_item_id"]),
                    [
                        hashlib.sha256(normalized(t).encode()).hexdigest()
                        for t in row["overlap_texts"]
                    ],
                )
    for key, ids in flagged.items():
        cell = result[key]
        cell["flagged_rows"] = [
            {
                "file": leaves[i][0],
                "row": leaves[i][1],
                "why": why,
                "leaf_sha256": leaves[i][2],
            }
            for i, why in sorted(ids.items())
            if i in leaves
        ]
        cell["counts"]["flagged_rows"] = len(cell["flagged_rows"])
    for key, cell in result.items():
        if cell["admitted"]:
            scanned = isinstance(cell["counts"].get("embed"), dict)
            cell["reasons"].insert(
                0,
                (
                    "isolated: names, lexical and embedding checks passed"
                    if scanned
                    else "isolated: names and lexical checks passed (embedding not run)"
                ),
            )
        cell["reasons"].append("c1: no JevArena-C1 registry dataset (reverse check)")

    admission = {
        "schema": "htdev-admission/1",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "code_commit": args.code_commit,
        "training_manifest_sha256": sha_file(iso / "TRAINING-MANIFEST.json"),
        "training_corpora_sha256": sha_file(iso / "training-corpora.json"),
        "sources_manifest_sha256": sha_file(iso / "SOURCES-MANIFEST.json"),
        "protected_rows_sha256": receipt["protected"]["sha256"],
        "overlap_receipt_sha256": sha_file(work / "overlap-receipt.json"),
        "names_sha256": sha_file(work / "names.json"),
        "embed_private_sha256": (
            sha_file(args.embed_private) if args.embed_private else None
        ),
        "sources": result,
    }
    data = json.dumps(admission, indent=1, sort_keys=True).encode()
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
    summary = {
        "admission_sha256": hashlib.sha256(data).hexdigest(),
        **{k: v for k, v in admission.items() if k != "sources"},
        "sources": {
            k: {
                "admitted": v["admitted"],
                "reasons": v["reasons"],
                "counts": v["counts"],
                "flags": v["flags"],
            }
            for k, v in result.items()
        },
    }
    args.summary.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                k: [v["admitted"], v["counts"].get("flagged_rows", 0)]
                for k, v in result.items()
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
