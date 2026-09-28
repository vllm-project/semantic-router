"""Fail-closed aggregate-only FEVEROUS publisher TRAIN metadata audit v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from training.data.audit_feverous_score_ranges import LABELS, evidence_profile

EXPECTED_BYTES = 175_493_294
EXPECTED_MD5 = "d8d4634760dad714b4cc30e43d25e589"
SOURCE_KIND = re.compile(r"_(?:sentence|header_cell|cell|table_caption|item)_")


def all_referenced_pages(row: dict) -> frozenset[str]:
    pages: set[str] = set()
    for evidence_set in row["evidence"]:
        for element in evidence_set["content"]:
            kind = SOURCE_KIND.search(element)
            if kind is None:
                raise ValueError("Unknown evidence element type")
            pages.add(element[: kind.start()])
    return frozenset(pages)


def normalized_claim(claim: str) -> str:
    return " ".join(claim.casefold().split())


def audit(
    path: Path, expected_bytes: int = EXPECTED_BYTES, expected_md5: str = EXPECTED_MD5
) -> dict:
    md5 = hashlib.md5(usedforsecurity=False)
    sha = hashlib.sha256()
    length = 0
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            length += len(block)
            md5.update(block)
            sha.update(block)
    if length != expected_bytes or md5.hexdigest() != expected_md5:
        raise ValueError("Publisher TRAIN file identity differs")

    labels: Counter[str] = Counter()
    text_labels: Counter[str] = Counter()
    quarantined_empty = 0
    seen_ids: set[int] = set()
    claim_counts: Counter[str] = Counter()
    text_pages: dict[str, set[str]] = defaultdict(set)
    candidates: list[tuple[int, str, frozenset[str], str]] = []
    first_page_record: dict[str, int] = {}
    parent: dict[int, int] = {}

    def root(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    with path.open("rb") as stream:
        for raw in stream:
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError("Publisher record is not an object")
            label = row.get("label")
            if label == "":
                quarantined_empty += 1
                continue
            if label not in LABELS:
                raise ValueError("Unexpected nonempty publisher label")
            source_id = row.get("id")
            if not isinstance(source_id, int) or source_id in seen_ids:
                raise ValueError("Missing or duplicate source ID")
            seen_ids.add(source_id)
            claim = row.get("claim")
            if not isinstance(claim, str) or not claim.strip():
                raise ValueError("Missing claim")
            if not isinstance(row.get("evidence"), list):
                raise ValueError("Missing evidence list")
            kinds, sentence_pages = evidence_profile(row)
            pages = all_referenced_pages(row)
            if not kinds and row["evidence"]:
                raise ValueError("Evidence set has no recognized elements")
            labels[label] += 1
            normalized = normalized_claim(claim)
            claim_counts[normalized] += 1
            if sentence_pages is None or not pages:
                continue
            text_labels[label] += 1
            text_pages[label].update(sentence_pages)
            candidates.append((source_id, label, pages, normalized))
            parent[source_id] = source_id
            for page in pages:
                prior = first_page_record.setdefault(page, source_id)
                left, right = root(source_id), root(prior)
                parent[left] = right

    component_count = len({root(source_id) for source_id in parent})
    chosen: list[int] = []
    chosen_labels: Counter[str] = Counter()
    occupied_pages: set[str] = set()
    occupied_claims: set[str] = set()
    for source_id, label, pages, claim in sorted(
        candidates,
        key=lambda record: hashlib.sha256(str(record[0]).encode()).hexdigest(),
    ):
        if (
            chosen_labels[label] >= 64
            or pages & occupied_pages
            or claim in occupied_claims
        ):
            continue
        chosen.append(source_id)
        chosen_labels[label] += 1
        occupied_pages.update(pages)
        occupied_claims.add(claim)
    metadata_floor = all(chosen_labels[label] == 64 for label in LABELS)
    roster_sha = hashlib.sha256(",".join(map(str, chosen)).encode()).hexdigest()
    return {
        "protocol": "decision2-eos08-feverous-full-train-metadata-v2",
        "publisher_train_sha256": sha.hexdigest(),
        "publisher_train_bytes": length,
        "total_source_rows": len(seen_ids) + quarantined_empty,
        "empty_label_quarantine": quarantined_empty,
        "labeled_rows": dict(sorted(labels.items())),
        "text_only_rows": dict(sorted(text_labels.items())),
        "text_only_distinct_referenced_pages": {
            label: len(text_pages[label]) for label in LABELS
        },
        "duplicate_normalized_claims": sum(
            count - 1 for count in claim_counts.values()
        ),
        "page_linked_text_only_components": component_count,
        "page_disjoint_candidate_rows": dict(sorted(chosen_labels.items())),
        "page_disjoint_candidate_roster_sha256": roster_sha,
        "metadata_floor_pass": metadata_floor,
        "full_page_native_length_checked": False,
        "protected_overlap_cleared": False,
        "train_admission": "HOLD",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publisher-train", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.publisher_train), sort_keys=True))


if __name__ == "__main__":
    main()
