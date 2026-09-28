"""Reissue the ShARC blind packet with hidden-ID-based balanced state order.

V3's index-parity order revealed the targets to anyone reading its builder.
The V3 private packet is invalid for blind review and remains only an audit
receipt. V4 retains the exact V3 source selection but changes the blind order.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any

from .audit_sharc_policy_source import _visible, read_train
from .audit_sharc_policy_source_v2 import ARCHIVE_SHA256, normalized
from .build_sharc_blind_packet import (
    TRAIN_SHA256,
    _write_private,
    rank,
    read_exclusions,
    select_pairs,
    sha256,
)


def _visible_state(row: dict[str, Any]) -> dict[str, Any]:
    # Whitelist only the publisher's documented visible history fields, so a
    # future archive revision cannot smuggle a target through an extra key.
    return {
        "scenario": row["scenario"],
        "history": [
            {
                "follow_up_question": item["follow_up_question"],
                "follow_up_answer": item["follow_up_answer"],
            }
            for item in row["history"]
        ],
    }


def make_packet(
    selected: list[tuple[dict[str, Any], dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(selected) % 2:
        raise ValueError("Blind order must be balanced across an even packet")
    hidden_order = sorted(
        range(len(selected)),
        key=lambda index: rank(
            "v4-state-order",
            selected[index][0]["tree_id"],
            selected[index][0]["utterance_id"],
            selected[index][1]["utterance_id"],
        ),
    )
    yes_first = set(hidden_order[: len(selected) // 2])
    blind_pairs: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for index, (yes, no) in enumerate(selected):
        # Publisher IDs are hidden from the reviewer. Public P01/P02 order
        # provides no label key, while the full packet remains reproducible.
        first, second = (yes, no) if index in yes_first else (no, yes)
        blind_id = f"P{index + 1:02d}"
        blind_pairs.append(
            {
                "blind_id": blind_id,
                "rule_snippet": first["snippet"],
                "question": first["question"],
                "states": [
                    {"state_id": "A", **_visible_state(first)},
                    {"state_id": "B", **_visible_state(second)},
                ],
            }
        )
        mapping.append(
            {
                "blind_id": blind_id,
                "source_url": first["source_url"],
                "tree_id": first["tree_id"],
                "exact_snippet_sha256": sha256(first["snippet"].encode("utf-8")),
                "normalized_snippet_sha256": sha256(
                    normalized(first["snippet"]).encode("utf-8")
                ),
                "utterance_ids": {
                    "A": first["utterance_id"],
                    "B": second["utterance_id"],
                },
            }
        )
    return (
        {"schema": "decision2-sharc-blind-review/4", "pairs": blind_pairs},
        {"schema": "decision2-sharc-blind-mapping/4", "pairs": mapping},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    rows, identity = read_train(args.archive)
    if identity["archive_sha256"] != ARCHIVE_SHA256:
        raise ValueError("Publisher archive SHA-256 changed")
    if identity["train_member_sha256"] != TRAIN_SHA256:
        raise ValueError("Publisher TRAIN member SHA-256 changed")
    excluded, negative_counts = read_exclusions(args.archive)
    selected, aggregate = select_pairs(rows, excluded)
    packet, mapping = make_packet(selected)
    args.out_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    packet_hash = _write_private(args.out_dir / "blind_review.json", packet)
    mapping_hash = _write_private(args.out_dir / "private_mapping.json", mapping)
    lengths = [len(_visible(row)) for pair in selected for row in pair]
    receipt = {
        "schema": "decision2-sharc-blind-receipt/4",
        "archive_sha256": identity["archive_sha256"],
        "train_member_sha256": identity["train_member_sha256"],
        "negative_list_counts": negative_counts,
        "blind_packet_sha256": packet_hash,
        "private_mapping_sha256": mapping_hash,
        "visible_chars_min": min(lengths),
        "visible_chars_median": statistics.median(lengths),
        "visible_chars_max": max(lengths),
        "training_admitted": False,
        "gpu_hours": 0,
        **aggregate,
    }
    receipt_hash = _write_private(args.out_dir / "aggregate_receipt.json", receipt)
    print(
        json.dumps(
            {
                "blind_packet_sha256": packet_hash,
                "private_mapping_sha256": mapping_hash,
                "aggregate_receipt_sha256": receipt_hash,
                "selected_pairs": aggregate["selected_pairs"],
                "eligible_source_urls": aggregate["eligible_source_urls"],
                "training_admitted": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
