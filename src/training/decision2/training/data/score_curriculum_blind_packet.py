"""Seal a reproducible gold-free editorial packet from Score TRAIN only."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any

from training.data import build_pilot as pilot
from training.data.build_score_curriculum import FAMILIES
from training.model.data import load_partition

REVIEW_SEED = "decision20-score-curriculum-blind-review-v1"
GROUPS_PER_FAMILY = 12
PACKET_FIELDS = (
    "id",
    "group_id",
    "family",
    "language",
    "state",
    "instructions",
    "options",
)


def select(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    selected_groups = []
    for family in FAMILIES:
        family_groups = [
            group
            for group, variants in groups.items()
            if variants[0]["family"] == f"score_{family}"
        ]
        family_groups.sort(
            key=lambda group: (
                hashlib.sha256(f"{REVIEW_SEED}\0{group}".encode()).hexdigest(),
                group,
            )
        )
        if len(family_groups) < GROUPS_PER_FAMILY:
            raise ValueError(f"Too few {family} groups for blind review")
        selected_groups.extend(family_groups[:GROUPS_PER_FAMILY])
    selected = [
        {field: row[field] for field in PACKET_FIELDS}
        for group in selected_groups
        for row in groups[group]
    ]
    if len(selected) != GROUPS_PER_FAMILY * len(FAMILIES) * 3:
        raise ValueError("Blind packet split a three-level source group")
    selected.sort(
        key=lambda row: (
            hashlib.sha256(f"{REVIEW_SEED}\0order\0{row['id']}".encode()).hexdigest(),
            row["id"],
        )
    )
    return selected, sorted(selected_groups)


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    actual = pilot.sha_file(args.train)
    if actual != args.expected_train_sha256:
        raise ValueError("TRAIN bytes differ from reviewed candidate")
    rows = load_partition(args.train, "train")
    packet, groups = select(rows)
    if any(set(row) != set(PACKET_FIELDS) for row in packet):
        raise AssertionError("Gold or metadata leaked into blind packet")
    payload = pilot.jsonl_bytes(packet)
    args.output_dir.mkdir(parents=True, mode=0o700)
    pilot._atomic_write(args.output_dir / "packet.jsonl", payload)
    manifest = {
        "schema_version": "decision20-score-curriculum-blind-packet/1",
        "train_sha256": actual,
        "packet_builder_sha256": pilot.sha_file(Path(__file__)),
        "review_seed_sha256": hashlib.sha256(REVIEW_SEED.encode()).hexdigest(),
        "packet_sha256": pilot.sha_bytes(payload),
        "packet_rows": len(packet),
        "selected_groups": groups,
        "selected_group_count": len(groups),
        "status": "gold_free_sealed_before_editorial_review",
    }
    pilot._atomic_write(
        args.output_dir / "manifest.json",
        (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode(),
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args)
    print(
        json.dumps(
            {
                "packet_sha256": manifest["packet_sha256"],
                "rows": manifest["packet_rows"],
                "groups": manifest["selected_group_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
