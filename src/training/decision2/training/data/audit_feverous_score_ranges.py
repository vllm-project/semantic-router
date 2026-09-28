"""Aggregate-only FEVEROUS TRAIN byte-range screen for an Eos Score hypothesis.

Only publisher TRAIN byte ranges are accepted. Raw claims, evidence, article
names, row IDs and row-level labels are never emitted by this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

TOTAL_BYTES = 175_493_294
RANGE_BYTES = 1_048_576
OFFSETS = (0, 43_873_323, 87_746_647, 131_619_970)
LABELS = ("REFUTES", "NOT ENOUGH INFO", "SUPPORTS")
CONTENT_RANGE = re.compile(
    r"^content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)\s*$", re.I | re.M
)


def check_range(header: bytes, body: bytes, start: int) -> str:
    """Fail closed unless one exact publisher-size HTTP byte range was served."""
    statuses = re.findall(rb"HTTP/\S+\s+(\d{3})", header)
    if not statuses or statuses[-1] != b"206":
        raise ValueError("Expected HTTP 206 Partial Content")
    match = CONTENT_RANGE.search(header.decode("latin1"))
    if match is None:
        raise ValueError("Missing Content-Range")
    observed = tuple(map(int, match.groups()))
    expected = (start, start + RANGE_BYTES - 1, TOTAL_BYTES)
    if observed != expected or len(body) != RANGE_BYTES:
        raise ValueError("Byte-range identity or size differs")
    return hashlib.sha256(body).hexdigest()


def complete_rows(body: bytes, start: int) -> list[dict]:
    """Decode only complete JSONL lines; drop both boundary fragments."""
    parts = body.split(b"\n")
    if start != 0:
        parts = parts[1:]
    if not body.endswith(b"\n"):
        parts = parts[:-1]
    rows = []
    for part in parts:
        if not part:
            continue
        row = json.loads(part)
        if not isinstance(row, dict):
            raise ValueError("Source JSONL row is not an object")
        rows.append(row)
    return rows


def evidence_profile(row: dict) -> tuple[set[str], set[str] | None]:
    """Return element kinds and one fully contextualized sentence-only page set."""
    sets = row.get("evidence")
    if not isinstance(sets, list):
        raise ValueError("Unexpected evidence schema")
    all_kinds: set[str] = set()
    text_pages = None
    for evidence_set in sets:
        if not isinstance(evidence_set, dict):
            raise ValueError("Unexpected evidence-set schema")
        content = evidence_set.get("content")
        context = evidence_set.get("context")
        if not isinstance(content, list) or not isinstance(context, dict):
            raise ValueError("Evidence content or context is absent")
        kinds: set[str] = set()
        pages: set[str] = set()
        for element in content:
            if not isinstance(element, str):
                raise ValueError("Nontext evidence element ID")
            found = re.search(
                r"_(sentence|header_cell|cell|table_caption|item)_", element
            )
            if found is None:
                raise ValueError("Unknown evidence element type")
            kinds.add(found.group(1))
            pages.add(element[: found.start()])
        all_kinds.update(kinds)
        if (
            content
            and kinds == {"sentence"}
            and all(element in context for element in content)
        ):
            text_pages = pages if text_pages is None else text_pages
    return all_kinds, text_pages


def aggregate(
    chunks: list[tuple[bytes, bytes]], offsets: tuple[int, ...] = OFFSETS
) -> dict:
    if len(chunks) != len(offsets):
        raise ValueError("Expected one chunk for each frozen range")
    label_counts: Counter[str] = Counter()
    text_counts: Counter[str] = Counter()
    evidence_kinds: Counter[str] = Counter()
    text_pages: dict[str, set[str]] = defaultdict(set)
    seen_ids: set[str] = set()
    digests: list[str] = []
    invalid_label_rows = 0
    for (header, body), offset in zip(chunks, offsets, strict=True):
        digests.append(check_range(header, body, offset))
        for row in complete_rows(body, offset):
            label = row.get("label")
            if label not in LABELS:
                invalid_label_rows += 1
                continue
            source_id = row.get("id")
            if not isinstance(source_id, int) or str(source_id) in seen_ids:
                raise ValueError("Missing or duplicate source ID")
            seen_ids.add(str(source_id))
            if not isinstance(row.get("claim"), str) or not row["claim"].strip():
                raise ValueError("Missing claim text")
            kinds, pages = evidence_profile(row)
            label_counts[label] += 1
            evidence_kinds.update(kinds)
            if pages:
                text_counts[label] += 1
                text_pages[label].update(pages)
    floor = (
        invalid_label_rows == 0
        and len(text_pages["REFUTES"]) >= 16
        and len(text_pages["SUPPORTS"]) >= 16
        and len(text_pages["NOT ENOUGH INFO"]) >= 8
        and text_counts["NOT ENOUGH INFO"] >= 8
    )
    return {
        "protocol": "decision2-eos08-feverous-range-screen-v1",
        "range_start_offsets": list(offsets),
        "range_sha256": digests,
        "complete_rows": sum(label_counts.values()) + invalid_label_rows,
        "invalid_label_rows": invalid_label_rows,
        "source_label_counts": {label: label_counts[label] for label in LABELS},
        "text_only_counts": {label: text_counts[label] for label in LABELS},
        "text_only_distinct_referenced_pages": {
            label: len(text_pages[label]) for label in LABELS
        },
        "evidence_kind_record_occurrences": dict(sorted(evidence_kinds.items())),
        "small_screen_floor_pass": floor,
        "stop_reason": (
            "publisher TRAIN has an unlabeled row" if invalid_label_rows else None
        ),
        "train_admission": "HOLD",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--range-dir", type=Path, required=True)
    args = parser.parse_args()
    chunks = [
        (
            (args.range_dir / f"range{i}.headers").read_bytes(),
            (args.range_dir / f"range{i}.bin").read_bytes(),
        )
        for i in range(len(OFFSETS))
    ]
    print(json.dumps(aggregate(chunks), sort_keys=True))


if __name__ == "__main__":
    main()
