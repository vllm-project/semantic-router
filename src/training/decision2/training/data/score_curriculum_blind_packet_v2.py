"""Make an opaque-ID, gold-free editorial packet from frozen Score TRAIN rows.

The first packet exposed the answer through the source row ID suffix. Retain
that packet for audit; this version requires a private 32-byte alias salt and
stores the review-to-source join outside the review packet directory.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import stat
from pathlib import Path
from typing import Any

from training.data import build_pilot as pilot
from training.data import score_curriculum_blind_packet as first_packet
from training.model.data import load_partition

VERSION = "decision20-score-curriculum-blind-packet/2"
PUBLIC_FIELDS = (
    "review_id",
    "group_id",
    "family",
    "language",
    "state",
    "instructions",
    "options",
)


def select(
    rows: list[dict[str, Any]], salt: bytes
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[str]]:
    if len(salt) != 32:
        raise ValueError("A private 32-byte alias salt is required")
    selected, groups = first_packet.select(rows)
    public = []
    private_map = []
    for source in selected:
        digest = hmac.new(salt, source["id"].encode(), hashlib.sha256).hexdigest()
        alias = f"sbr-{digest[:20]}"
        public.append(
            {
                "review_id": alias,
                **{name: source[name] for name in PUBLIC_FIELDS if name != "review_id"},
            }
        )
        private_map.append(
            {
                "review_id": alias,
                "source_id": source["id"],
                "group_id": source["group_id"],
            }
        )
    if len({row["review_id"] for row in public}) != len(public):
        raise ValueError("Review alias collision")
    public.sort(key=lambda row: row["review_id"])
    private_map.sort(key=lambda row: row["review_id"])
    return public, private_map, groups


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    map_output = args.map_output.resolve()
    if (
        output_dir.exists()
        or map_output.exists()
        or map_output.is_relative_to(output_dir)
    ):
        raise FileExistsError("Review packet and separate private map must be new")
    train_sha = pilot.sha_file(args.train)
    if train_sha != args.expected_train_sha256:
        raise ValueError("TRAIN bytes differ from frozen candidate")
    salt = args.private_salt.read_bytes()
    rows = load_partition(args.train, "train")
    public, private_map, groups = select(rows, salt)
    if any(set(row) != set(PUBLIC_FIELDS) for row in public):
        raise AssertionError("Review packet includes unintended fields")
    packet_bytes = pilot.jsonl_bytes(public)
    mapping_bytes = pilot.jsonl_bytes(private_map)
    map_output.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if stat.S_IMODE(map_output.parent.stat().st_mode) & 0o077:
        raise PermissionError("Private map directory must be owner-only")
    output_dir.mkdir(parents=True, mode=0o700)
    pilot._atomic_write(output_dir / "packet.jsonl", packet_bytes)
    pilot._atomic_write(map_output, mapping_bytes)
    manifest = {
        "schema_version": VERSION,
        "train_sha256": train_sha,
        "packet_builder_sha256": pilot.sha_file(Path(__file__)),
        "private_salt_sha256": pilot.sha_bytes(salt),
        "packet_sha256": pilot.sha_bytes(packet_bytes),
        "private_map_sha256": pilot.sha_bytes(mapping_bytes),
        "packet_rows": len(public),
        "selected_group_count": len(groups),
        "selected_groups": groups,
        "status": "gold_free_sealed_before_editorial_review",
    }
    pilot._atomic_write(
        output_dir / "manifest.json",
        (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode(),
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--private-salt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--map-output", type=Path, required=True)
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
