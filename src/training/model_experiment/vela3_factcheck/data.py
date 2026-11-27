"""Check the FactCheck training split against held-out groups and text."""

import argparse
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

NEGATIVE = "NO_FACT_CHECK_NEEDED"
POSITIVE = "FACT_CHECK_NEEDED"

# Count/add/rewrite supplied material, or obtain missing content at a linked URL.
REVIEWED_LABELS = {
    "fact_check-train-001047": NEGATIVE,
    "fact_check-train-001744": NEGATIVE,
    "fact_check-train-002046": NEGATIVE,
    "fact_check-train-001888": POSITIVE,
    "fact_check-train-004319": POSITIVE,
    "fact_check-train-004889": POSITIVE,
}


def read_rows(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def shingles(text):
    text = re.sub(r"\s+", "", text)
    return {text[i : i + 5] for i in range(len(text) - 4)}


def overlaps(training, held_out):
    exact, groups, index = (defaultdict(set) for _ in range(3))
    fingerprints = {}
    for row in held_out:
        key = row["id"]
        normalized = normalize(row["text"])
        exact[normalized].add(key)
        for field in ("group", "group_sha256"):
            if row.get(field):
                groups[row[field]].add(key)
        fingerprint = shingles(normalized)
        if len(fingerprint) >= 20:
            fingerprints[key] = fingerprint
            for shingle in fingerprint:
                index[shingle].add(key)

    matches = set()
    for row in training:
        normalized = normalize(row["text"])
        matching = set(exact.get(normalized, ()))
        for field in ("group", "group_sha256"):
            matching.update(groups.get(row.get(field), ()))
        fingerprint = shingles(normalized)
        if len(fingerprint) >= 20:
            candidates = set().union(*(index.get(s, ()) for s in fingerprint))
            for key in candidates - matching:
                other = fingerprints[key]
                if min(len(other), len(fingerprint)) < 0.85 * max(
                    len(other), len(fingerprint)
                ):
                    continue
                common = len(fingerprint & other)
                if common >= 0.85 * (len(fingerprint) + len(other) - common):
                    matching.add(key)
        matches.update((row["id"], key) for key in matching)
    return matches


def suite_rows(suite, split):
    return [
        row
        for path in sorted(suite.glob(f"*/fact_check/{split}.jsonl"))
        for row in read_rows(path)
    ]


def audit(suite, training, augmentation, probes=()):
    seen = [r for r in augmentation if r.get("kind") != "pair"]
    for pair in (r for r in augmentation if r.get("kind") == "pair"):
        seen.extend(
            {"id": pair["id"], "text": pair[field]}
            for field in ("neg_text", "pos_text")
        )
    held_out = [
        row
        for path in sorted(suite.glob("*/fact_check/*.jsonl"))
        if path.stem != "train"
        for row in read_rows(path)
    ]
    held_out.extend(probes)
    return {
        "train_rows": len(training),
        "held_out_rows": len(held_out),
        "train_overlaps": sorted(overlaps(training, held_out)),
        "augmentation_overlaps": sorted(overlaps(seen, held_out)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--train", type=Path, help="override the suite training rows")
    parser.add_argument("--augmentation", type=Path, required=True)
    parser.add_argument("--probes", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--clean-train", type=Path, help="write training rows without held-out matches"
    )
    args = parser.parse_args()
    training = read_rows(args.train) if args.train else suite_rows(args.suite, "train")
    augmentation = read_rows(args.augmentation)
    probes = [row for path in args.probes for row in read_rows(path)]
    report = audit(args.suite, training, augmentation, probes)
    report["augmentation_sha256"] = hashlib.sha256(
        args.augmentation.read_bytes()
    ).hexdigest()
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for field in ("train_overlaps", "augmentation_overlaps"):
        print(f"{field}: {len(report[field])}")
    if args.clean_train is not None and not report["augmentation_overlaps"]:
        excluded = {train_id for train_id, _ in report["train_overlaps"]}
        with args.clean_train.open("x", encoding="utf-8") as output:
            for row in training:
                if row["id"] not in excluded:
                    if args.train is None and row["id"] in REVIEWED_LABELS:
                        row = {**row, "label": REVIEWED_LABELS[row["id"]]}
                    print(json.dumps(row, ensure_ascii=False), file=output)
        return 0
    return int(bool(report["train_overlaps"] or report["augmentation_overlaps"]))


if __name__ == "__main__":
    raise SystemExit(main())
