"""Verdict of the event-3 confirmatory overlap scan against the event-2 scan (ids only, never text).

    python3 -m v2.eval.sealed.scanverdict compare --hits NEW --baseline OLD --receipt RECEIPT \
        --manifest MANIFEST --protected-sha SHA --output SCAN-VERDICT.json
    python3 -m v2.eval.sealed.scanverdict check --verdict SCAN-VERDICT.json --manifest-sha SHA \
        --protected-sha SHA
    python3 -m v2.eval.sealed.scanverdict extract --hits HITS --receipt RECEIPT --baseline OLD \
        --node NAME --output EXTRACT.json
    python3 -m v2.eval.sealed.scanverdict judge --extract EXTRACT.json [--extract ...] \
        --baseline OLD --retired RETIRED.json --retired-sha SHA --protected-sha SHA \
        [--coverage NODE=COVERAGE-RECEIPT.json ...] --output SCAN-VERDICT.json
    python3 -m v2.eval.sealed.scanverdict check-v2 --verdict SCAN-VERDICT.json --verdict-sha SHA \
        --retired-sha SHA --protected-sha SHA [--schema SCHEMA]
    python3 -m v2.eval.sealed.scanverdict judge-classes --extract EXTRACT.json [--extract ...] \
        --grouped NODE=HITS ... --perfile NODE=HITS ... --perfile-manifest NODE=MANIFEST ... \
        --classes CLASSES.json --retired RETIRED.json --retired-sha SHA --protected-sha SHA \
        --output SCAN-VERDICT.json

``compare`` reads two ``overlap scan --hits`` files (one row per protected candidate) and fails when
any candidate is OVERLAP, when a candidate is non-CLEAN now but was CLEAN at event 2, when a recurring
non-CLEAN candidate has a higher containment than at event 2, or when the two scans cover different
candidates. It exits 1 on FAIL. ``check`` is the event-3 interlock: it exits 0 only for a PASS verdict
of the pinned manifest and protected rows.

``extract``, ``judge`` and ``check-v2`` do the same for an item set with retired rows (v1.2) whose
scan runs on several nodes, each scanning its own files against the same protected rows. ``extract``
keeps one node's non-CLEAN hits and its rows of every id non-CLEAN in the baseline, bound to its
receipt and the baseline. ``judge`` merges the nodes per id (worst verdict, highest containment on
any node, the nodes that flagged it) and applies the ``compare`` rules to the
ids outside the retired list's ``protected_rows``; it also fails on extracts of other protected
rows or of another baseline, on candidate counts that differ between the nodes or from the
baseline, on repeated node names and on a retired list other than the pinned one. It exits 1 on FAIL. ``check-v2`` is the matching
interlock: it exits 0 only for the pinned PASS verdict of the pinned retired list and protected rows.

``judge-classes`` applies the class-aware rule (policy P2) to the same merged extracts, using the
analysis scans of ``hitgroups`` (one label per corpus group, and one label per file for the rows with
a hit under the training roots). Among the flagged rows outside the retired list it fails on any
non-CLEAN hit in a class-(a) file (the training files of a released or candidate model, by SHA-256 or
path) and on any near-exact hit in class (b) (every other group or file under the training roots).
Classes (c) raw sources, (d) evaluation-only pools and (e) other are counted, never failing. It also
fails when a grouped scan misses a flagged row, when the per-file scan misses a row with a hit under
the training roots, or when a pinned class-(a) file is absent from every per-file manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "dev2-c1-event3-scan-verdict/1"
SCHEMA_V2 = "dev2-c1-scan-verdict/2"
SCHEMA_CLASSES = "dev2-c1-class-verdict/1"
EXTRACT_SCHEMA = "dev2-c1-scan-extract/1"
RETIRED_SCHEMA = "dev2-c1-retired/1"
CLASSES_SCHEMA = "dev2-c1-hit-classes/1"
CLASSES = ("a", "b", "c", "d", "e")
RANK = ("CLEAN", "REVIEW", "OVERLAP")


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_hits(path: Path) -> dict[str, dict[str, Any]]:
    hits = {}
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                hits[row["id"]] = row
    return hits


def read_json(path: Path) -> tuple[Any, str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def write_new(path: Path, value: Any) -> str:
    data = (json.dumps(value, indent=1, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
    return hashlib.sha256(data).hexdigest()


def node_file(spec: str) -> tuple[str, Path]:
    node, separator, path = spec.partition("=")
    if not separator or not node or not path:
        raise argparse.ArgumentTypeError(f"expected NODE=PATH, got {spec!r}")
    return node, Path(path)


def judge(
    new: dict[str, dict[str, Any]], old: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    counts = Counter(h["verdict"] for h in new.values())
    flagged = sorted(k for k, h in new.items() if h["verdict"] != "CLEAN")
    before = {k for k, h in old.items() if h["verdict"] != "CLEAN"}
    problems = []
    if set(new) != set(old):
        problems.append("the scans cover different protected candidates")
    overlap = [k for k in flagged if new[k]["verdict"] == "OVERLAP"]
    if overlap:
        problems.append(f"{len(overlap)} OVERLAP")
    fresh = [k for k in flagged if k not in before]
    if fresh:
        problems.append(f"{len(fresh)} non-CLEAN ids that were CLEAN at event 2")
    higher = [
        k
        for k in flagged
        if k in before and new[k]["containment"] > old[k]["containment"]
    ]
    if higher:
        problems.append(f"{len(higher)} recurring ids with higher containment")
    return {
        "candidates": len(new),
        "counts": dict(sorted(counts.items())),
        "non_clean_ids": flagged,
        "event2_non_clean": len(before),
        "overlap_ids": overlap,
        "new_non_clean_ids": fresh,
        "higher_containment_ids": higher,
        "problems": problems,
        "verdict": "FAIL" if problems else "PASS",
    }


def compare(args: argparse.Namespace) -> int:
    result = judge(read_hits(args.hits), read_hits(args.baseline))
    result = {
        "schema": SCHEMA,
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest_sha256": sha_file(args.manifest),
        "protected_sha256": args.protected_sha,
        "receipt_sha256": sha_file(args.receipt),
        "hits_sha256": sha_file(args.hits),
        "baseline_hits_sha256": sha_file(args.baseline),
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(
        json.dumps(
            {k: result[k] for k in ("verdict", "counts", "problems", "hits_sha256")}
        )
    )
    return 0 if result["verdict"] == "PASS" else 1


def check(args: argparse.Namespace) -> int:
    try:
        value = json.loads(args.verdict.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"no readable scan verdict: {exc}", file=sys.stderr)
        return 1
    problems = []
    if value.get("schema") != SCHEMA:
        problems.append("not a scan verdict")
    if value.get("verdict") != "PASS":
        problems.append(f"verdict {value.get('verdict')!r}")
    if value.get("manifest_sha256") != args.manifest_sha:
        problems.append("scan used another training manifest")
    if value.get("protected_sha256") != args.protected_sha:
        problems.append("scan used other protected rows")
    if problems:
        print("scan verdict refused: " + "; ".join(problems), file=sys.stderr)
        return 1
    print(f"scan verdict PASS (hits {value['hits_sha256'][:12]}, {value['utc']})")
    return 0


def extract(args: argparse.Namespace) -> int:
    receipt, receipt_sha = read_json(args.receipt)
    data = args.hits.read_bytes()
    hits_sha = hashlib.sha256(data).hexdigest()
    if hits_sha != receipt.get("hits_sha256"):
        print("extract refused: the hits are not the receipt's hits", file=sys.stderr)
        return 1
    hits = [json.loads(line) for line in data.splitlines() if line.strip()]
    before = {k for k, h in read_hits(args.baseline).items() if h["verdict"] != "CLEAN"}
    value = {
        "schema": EXTRACT_SCHEMA,
        "node": args.node,
        "hits_sha256": hits_sha,
        "receipt_sha256": receipt_sha,
        "manifest_sha256": receipt["manifest_sha256"],
        "protected_sha256": receipt["protected"]["sha256"],
        "baseline_hits_sha256": sha_file(args.baseline),
        "candidates": len(hits),
        "counts": dict(sorted(Counter(h["verdict"] for h in hits).items())),
        "non_clean": sorted(
            (h for h in hits if h["verdict"] != "CLEAN"), key=lambda h: h["id"]
        ),
        "baseline_rows": sorted(
            (h for h in hits if h["id"] in before), key=lambda h: h["id"]
        ),
    }
    digest = write_new(args.output, value)
    print(
        json.dumps(
            {"node": args.node, "counts": value["counts"], "extract_sha256": digest}
        )
    )
    return 0


def merge(extracts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per id non-CLEAN on any node: the worst verdict, the highest containment on any
    node (a node where it is CLEAN still counts through its baseline rows), the nodes.
    """
    merged: dict[str, dict[str, Any]] = {}
    for part in extracts:
        for hit in part["non_clean"]:
            entry = merged.setdefault(
                hit["id"], {"verdict": "CLEAN", "containment": 0.0, "nodes": []}
            )
            entry["verdict"] = max(entry["verdict"], hit["verdict"], key=RANK.index)
            entry["containment"] = max(entry["containment"], hit["containment"])
            entry["nodes"] = sorted({*entry["nodes"], part["node"]})
    for part in extracts:
        for hit in part.get("baseline_rows", []):
            if hit["id"] in merged:
                entry = merged[hit["id"]]
                entry["containment"] = max(entry["containment"], hit["containment"])
    return dict(sorted(merged.items()))


def judge_merged(
    merged: dict[str, dict[str, Any]], old: dict[str, dict[str, Any]], retired: set[str]
) -> dict[str, Any]:
    before = {k for k, h in old.items() if h["verdict"] != "CLEAN"}
    kept = sorted(k for k in merged if k not in retired)
    overlap = [k for k in kept if merged[k]["verdict"] == "OVERLAP"]
    fresh = [k for k in kept if k not in before]
    higher = [
        k
        for k in kept
        if k in before and merged[k]["containment"] > old[k]["containment"]
    ]
    problems = []
    if overlap:
        problems.append(f"{len(overlap)} OVERLAP outside the retired rows")
    if fresh:
        problems.append(
            f"{len(fresh)} non-CLEAN ids that were CLEAN or absent in the baseline"
        )
    if higher:
        problems.append(f"{len(higher)} recurring ids with higher containment")
    return {
        "retired_non_clean": len(merged) - len(kept),
        "non_clean_ids": [{"id": k, **merged[k]} for k in sorted(merged)],
        "overlap_ids": overlap,
        "new_non_clean_ids": fresh,
        "higher_containment_ids": higher,
        "problems": problems,
    }


def judge_v2(args: argparse.Namespace) -> int:
    problems = []
    retired, retired_sha = read_json(args.retired)
    if retired_sha != args.retired_sha:
        problems.append("the retired list differs from --retired-sha")
    if not isinstance(retired, dict) or retired.get("schema") != RETIRED_SCHEMA:
        problems.append(f"the retired list is not {RETIRED_SCHEMA}")
        retired = {}
    parts = []
    for path in args.extract:
        part, sha = read_json(path)
        if part.get("schema") != EXTRACT_SCHEMA:
            problems.append(f"{path.name} is not a scan extract")
        else:
            parts.append((part, sha))
    names = Counter(part["node"] for part, _ in parts)
    repeated = sorted(name for name, count in names.items() if count > 1)
    if repeated:
        problems.append(f"repeated node names: {', '.join(repeated)}")
    baseline_sha = sha_file(args.baseline)
    for part, _ in parts:
        if part["protected_sha256"] != args.protected_sha:
            problems.append(f"node {part['node']} scanned other protected rows")
        if part.get("baseline_hits_sha256") != baseline_sha:
            problems.append(
                f"node {part['node']} was extracted against another baseline"
            )
    old = read_hits(args.baseline)
    sizes = {part["candidates"] for part, _ in parts}
    if len(sizes) > 1:
        problems.append("the nodes scanned different numbers of candidates")
    if sizes != {len(old)}:
        problems.append(
            "the scans and the baseline cover different numbers of candidates"
        )
    coverage = dict(args.coverage)
    if len(coverage) < len(args.coverage) or not set(coverage) <= set(names):
        problems.append("coverage receipts must name distinct scanned nodes")
    merged = merge([part for part, _ in parts])
    result = judge_merged(merged, old, set(retired.get("protected_rows", [])))
    candidates = next(iter(sizes)) if len(sizes) == 1 else None
    counts = Counter(entry["verdict"] for entry in merged.values())
    if candidates is not None:
        counts["CLEAN"] = candidates - len(merged)
    problems += result.pop("problems")
    value = {
        "schema": SCHEMA_V2,
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "item_set": retired.get("version"),
        "retired_sha256": retired_sha,
        "protected_sha256": args.protected_sha,
        "baseline_hits_sha256": sha_file(args.baseline),
        "nodes": [
            {
                "node": part["node"],
                "manifest_sha256": part["manifest_sha256"],
                "hits_sha256": part["hits_sha256"],
                "receipt_sha256": part["receipt_sha256"],
                "extract_sha256": sha,
                "coverage_receipt_sha256": (
                    sha_file(coverage[part["node"]])
                    if part["node"] in coverage
                    else None
                ),
            }
            for part, sha in sorted(parts, key=lambda pair: pair[0]["node"])
        ],
        "candidates": candidates,
        "counts": dict(sorted(counts.items())),
        **result,
        "problems": problems,
        "verdict": "FAIL" if problems else "PASS",
    }
    digest = write_new(args.output, value)
    summary = ("verdict", "item_set", "counts", "retired_non_clean", "problems")
    print(json.dumps({**{k: value[k] for k in summary}, "verdict_sha256": digest}))
    return 0 if value["verdict"] == "PASS" else 1


def check_v2(args: argparse.Namespace) -> int:
    try:
        value, digest = read_json(args.verdict)
    except (OSError, ValueError) as exc:
        print(f"no readable scan verdict: {exc}", file=sys.stderr)
        return 1
    problems = []
    if digest != args.verdict_sha:
        problems.append("not the pinned scan verdict")
    if value.get("schema") != args.schema:
        problems.append(f"not a {args.schema} scan verdict")
    if value.get("verdict") != "PASS":
        problems.append(f"verdict {value.get('verdict')!r}")
    if value.get("retired_sha256") != args.retired_sha:
        problems.append("scan judged another retired list")
    if value.get("protected_sha256") != args.protected_sha:
        problems.append("scan used other protected rows")
    if problems:
        print("scan verdict refused: " + "; ".join(problems), file=sys.stderr)
        return 1
    hits = " ".join(node["hits_sha256"][:12] for node in value["nodes"])
    rule = f", rule {value['rule']}" if value.get("rule") else ""
    print(
        f"scan verdict PASS (item set {value['item_set']}{rule}, {len(value['nodes'])} nodes,"
        f" hits {hits}, {value['utc']})"
    )
    return 0


def compile_classes(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "a_sha": {s: entry["model"] for entry in value["a"] for s in entry["sha256"]},
        "a_paths": [
            (re.compile(p), entry["model"])
            for entry in value["a"]
            for p in entry["paths"]
        ],
        "roots": re.compile(value["training_roots"]),
        "d": re.compile(value["d"]),
        "c": re.compile(value["c"]),
        "near": value["near_exact"],
        "disclose": [(e["name"], re.compile(e["pattern"])) for e in value["disclose"]],
    }


def group_class(group: str, classes: dict[str, Any]) -> str:
    """Class of a corpus group; class (a) is decided per file, inside the training roots."""
    if classes["roots"].match(group):
        return "b"
    if classes["d"].search(group):
        return "d"
    if classes["c"].match(group):
        return "c"
    return "e"


def near_exact(cell: dict[str, Any], near: dict[str, Any]) -> bool:
    return (
        cell.get("containment", 0.0) >= near["containment"]
        or (cell.get("exact_tokens") or 0) >= near["exact_tokens"]
    )


def non_clean(row: dict[str, Any] | None) -> list[tuple[str, dict[str, Any]]]:
    labels = (row or {}).get("labels") or {}
    return [(k, c) for k, c in sorted(labels.items()) if c.get("verdict") != "CLEAN"]


def judge_classes(args: argparse.Namespace) -> int:
    problems = []
    retired, retired_sha = read_json(args.retired)
    if retired_sha != args.retired_sha:
        problems.append("the retired list differs from --retired-sha")
    if not isinstance(retired, dict) or retired.get("schema") != RETIRED_SCHEMA:
        problems.append(f"the retired list is not {RETIRED_SCHEMA}")
        retired = {}
    spec, classes_sha = read_json(args.classes)
    if spec.get("schema") != CLASSES_SCHEMA:
        raise SystemExit(f"{args.classes} is not a {CLASSES_SCHEMA} class map")
    classes = compile_classes(spec)
    parts = []
    for path in args.extract:
        part, sha = read_json(path)
        if part.get("schema") != EXTRACT_SCHEMA:
            problems.append(f"{path.name} is not a scan extract")
        else:
            parts.append((part, sha))
    names = Counter(part["node"] for part, _ in parts)
    repeated = sorted(name for name, count in names.items() if count > 1)
    if repeated:
        problems.append(f"repeated node names: {', '.join(repeated)}")
    for part, _ in parts:
        if part["protected_sha256"] != args.protected_sha:
            problems.append(f"node {part['node']} scanned other protected rows")
    inputs = {
        "grouped": dict(args.grouped),
        "perfile": dict(args.perfile),
        "perfile_manifest": dict(args.perfile_manifest),
    }
    for flag, given in (
        ("--grouped", args.grouped),
        ("--perfile", args.perfile),
        ("--perfile-manifest", args.perfile_manifest),
    ):
        if len(given) != len(names) or set(dict(given)) != set(names):
            problems.append(f"{flag} must name each extract's node once")
    merged = merge([part for part, _ in parts])
    retired_rows = set(retired.get("protected_rows", []))
    kept = {k for k in merged if k not in retired_rows}

    grouped = {n: read_hits(p) for n, p in inputs["grouped"].items()}
    perfile = {n: read_hits(p) for n, p in inputs["perfile"].items()}
    files: dict[tuple[str, str], str] = {}
    for n, path in inputs["perfile_manifest"].items():
        for label, entry in read_json(path)[0]["labels"].items():
            files[(n, label)] = entry["files"][0]["sha256"]
    for n, rows in grouped.items():
        if set(merged) - set(rows):
            problems.append(
                f"the grouped scan of {n} misses {len(set(merged) - set(rows))} flagged rows"
            )
    rooted = {
        k
        for rows in grouped.values()
        for k, row in rows.items()
        if k in merged and any(classes["roots"].match(g) for g, _ in non_clean(row))
    }
    for n, rows in perfile.items():
        if rooted - set(rows):
            problems.append(
                f"the per-file scan of {n} misses {len(rooted - set(rows))} rows"
                " with a hit under the training roots"
            )
    a_sets = []
    for entry in spec["a"]:
        found = sorted(
            [n, label.split("::", 1)[1], sha]
            for (n, label), sha in files.items()
            if sha in entry["sha256"]
            or any(
                p.match(label.split("::", 1)[1])
                for p in map(re.compile, entry["paths"])
            )
        )
        missing = sorted(set(entry["sha256"]) - {sha for *_, sha in found})
        if missing:
            problems.append(
                f"{len(missing)} pinned class-(a) files of {entry['model']} are in no per-file manifest"
            )
        a_sets.append({"model": entry["model"], "files": found})

    def file_class(n: str, label: str) -> str:
        path = label.split("::", 1)[1]
        if files.get((n, label)) in classes["a_sha"] or any(
            p.match(path) for p, _ in classes["a_paths"]
        ):
            return "a"
        return group_class(label.split("::", 1)[0], classes)

    by_class: dict[str, set[str]] = {c: set() for c in CLASSES}
    a_hits, b_near = set(), set()
    disclosed = {name: {} for name, _ in classes["disclose"]}
    for k in sorted(merged):
        for n in sorted(grouped):
            for g, cell in non_clean(grouped[n].get(k)):
                cls = group_class(g, classes)
                by_class[cls].add(k)
                if cls == "b" and near_exact(cell, classes["near"]):
                    b_near.add(k)
                for name, pattern in classes["disclose"]:
                    if pattern.match(g):
                        old = disclosed[name].get(k)
                        new = (
                            RANK.index(cell["verdict"]),
                            cell.get("containment", 0.0),
                            cell.get("exact_tokens") or 0,
                        )
                        disclosed[name][k] = max(old or new, new)
            for label, cell in non_clean(perfile.get(n, {}).get(k)):
                cls = file_class(n, label)
                by_class[cls].add(k)
                if cls == "a":
                    a_hits.add(k)
                elif cls == "b" and near_exact(cell, classes["near"]):
                    b_near.add(k)
    a_kept, b_kept = sorted(a_hits & kept), sorted(b_near & kept)
    if a_kept:
        problems.append(
            f"{len(a_kept)} rows still in the item set hit class-(a) training files"
        )
    if b_kept:
        problems.append(
            f"{len(b_kept)} rows still in the item set have a near-exact class-(b) hit"
        )

    def summary(rows: set[str]) -> dict[str, Any]:
        return {
            "rows": len(rows),
            "by_class": {c: len(by_class[c] & rows) for c in CLASSES},
            "by_verdict": dict(
                sorted(Counter(merged[k]["verdict"] for k in rows).items())
            ),
        }

    near = classes["near"]
    value = {
        "schema": SCHEMA_CLASSES,
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rule": spec["rule"],
        "item_set": retired.get("version"),
        "retired_sha256": retired_sha,
        "protected_sha256": args.protected_sha,
        "classes_sha256": classes_sha,
        "nodes": [
            {
                "node": part["node"],
                "manifest_sha256": part["manifest_sha256"],
                "hits_sha256": part["hits_sha256"],
                "receipt_sha256": part["receipt_sha256"],
                "extract_sha256": sha,
                **{
                    f"{key}_sha256": (
                        sha_file(given[part["node"]]) if part["node"] in given else None
                    )
                    for key, given in inputs.items()
                },
            }
            for part, sha in sorted(parts, key=lambda pair: pair[0]["node"])
        ],
        "flagged": summary(set(merged)),
        "retired": summary(set(merged) - kept),
        "kept": summary(kept),
        "class_a_sets": a_sets,
        "class_a_ids": a_kept,
        "class_b_near_exact_ids": b_kept,
        "disclosed": [
            {
                "name": name,
                "kept_rows": sum(k in kept for k in rows),
                "kept_by_verdict": dict(
                    sorted(
                        Counter(
                            RANK[v[0]] for k, v in rows.items() if k in kept
                        ).items()
                    )
                ),
                "kept_max_containment": max(
                    (v[1] for k, v in rows.items() if k in kept), default=0.0
                ),
                "kept_max_exact_tokens": max(
                    (v[2] for k, v in rows.items() if k in kept), default=0
                ),
                "kept_near_exact": sum(
                    k in kept
                    and (v[1] >= near["containment"] or v[2] >= near["exact_tokens"])
                    for k, v in rows.items()
                ),
                "retired_rows": sum(k not in kept for k in rows),
            }
            for name, rows in disclosed.items()
        ],
        "problems": problems,
        "verdict": "FAIL" if problems else "PASS",
    }
    digest = write_new(args.output, value)
    print(
        json.dumps(
            {
                "verdict": value["verdict"],
                "item_set": value["item_set"],
                "kept": value["kept"],
                "problems": problems,
                "verdict_sha256": digest,
            }
        )
    )
    return 0 if value["verdict"] == "PASS" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("compare")
    for flag in ("--hits", "--baseline", "--receipt", "--manifest", "--output"):
        a.add_argument(flag, type=Path, required=True)
    a.add_argument("--protected-sha", required=True)
    b = sub.add_parser("check")
    b.add_argument("--verdict", type=Path, required=True)
    b.add_argument("--manifest-sha", required=True)
    b.add_argument("--protected-sha", required=True)
    c = sub.add_parser("extract")
    for flag in ("--hits", "--receipt", "--baseline", "--output"):
        c.add_argument(flag, type=Path, required=True)
    c.add_argument("--node", required=True)
    d = sub.add_parser("judge")
    d.add_argument("--extract", type=Path, action="append", required=True)
    for flag in ("--baseline", "--retired", "--output"):
        d.add_argument(flag, type=Path, required=True)
    d.add_argument("--retired-sha", required=True)
    d.add_argument("--protected-sha", required=True)
    d.add_argument(
        "--coverage", type=node_file, action="append", default=[], metavar="NODE=PATH"
    )
    e = sub.add_parser("check-v2")
    e.add_argument("--verdict", type=Path, required=True)
    for flag in ("--verdict-sha", "--retired-sha", "--protected-sha"):
        e.add_argument(flag, required=True)
    e.add_argument("--schema", default=SCHEMA_V2, choices=(SCHEMA_V2, SCHEMA_CLASSES))
    f = sub.add_parser("judge-classes")
    f.add_argument("--extract", type=Path, action="append", required=True)
    for flag in ("--grouped", "--perfile", "--perfile-manifest"):
        f.add_argument(
            flag, type=node_file, action="append", required=True, metavar="NODE=PATH"
        )
    for flag in ("--classes", "--retired", "--output"):
        f.add_argument(flag, type=Path, required=True)
    f.add_argument("--retired-sha", required=True)
    f.add_argument("--protected-sha", required=True)
    args = parser.parse_args(argv)
    return {
        "compare": compare,
        "check": check,
        "extract": extract,
        "judge": judge_v2,
        "check-v2": check_v2,
        "judge-classes": judge_classes,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
