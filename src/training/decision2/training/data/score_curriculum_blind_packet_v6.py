"""Seal a new Score v6 gold-free packet with opaque row and group aliases.

The private source join and salt stay outside the reviewer directory. Reviewers
receive only packet.jsonl and its source-hash manifest, never TRAIN labels.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

from training.data import build_pilot as pilot
from training.data.build_score_curriculum_v6 import FAMILIES, SOURCE
from training.model.data import load_partition

VERSION = "decision20-score-curriculum-blind-packet/6"
REVIEW_SEED = "decision20-score-curriculum-blind-v6-20260927"
GROUPS_PER_FAMILY = 12
GROUPS_PER_LANGUAGE = {"en": 9, "zh": 3}
PUBLIC_FIELDS = (
    "review_id",
    "group_id",
    "family",
    "language",
    "state",
    "instructions",
    "options",
)


def _alias(salt: bytes, domain: bytes, source_id: str, prefix: str) -> str:
    digest = hmac.new(
        salt, domain + b"\0" + source_id.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return f"{prefix}-{digest[:20]}"


def select(
    rows: list[dict[str, Any]], salt: bytes
) -> tuple[list[dict[str, Any]], list[dict[str, str]], int]:
    if len(salt) != 32:
        raise ValueError("A new private 32-byte alias salt is required")
    if sum(GROUPS_PER_LANGUAGE.values()) != GROUPS_PER_FAMILY:
        raise AssertionError("Reviewer group strata do not sum to the sample size")
    if any(
        row.get("source") != SOURCE
        or not row.get("render_template", "").endswith("_v6")
        for row in rows
    ):
        raise ValueError("Only the frozen Score v6 TRAIN source may be sampled")
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    selected_groups = []
    for family in FAMILIES:
        for language, required in GROUPS_PER_LANGUAGE.items():
            candidates = [
                group
                for group, variants in groups.items()
                if variants[0]["family"] == f"score_{family}"
                and variants[0]["language"] == language
            ]
            candidates.sort(
                key=lambda group: (
                    hashlib.sha256(f"{REVIEW_SEED}\0{group}".encode()).hexdigest(),
                    group,
                )
            )
            if len(candidates) < required:
                raise ValueError(f"Too few {family}/{language} groups for blind review")
            selected_groups.extend(candidates[:required])
    public = []
    private_map = []
    for group in selected_groups:
        variants = groups[group]
        if (
            len(variants) != 3
            or {row["label"] for row in variants} != {0, 1, 2}
            or len({row["family"] for row in variants}) != 1
        ):
            raise ValueError("A blind source group is incomplete")
        group_alias = _alias(salt, b"group", group, "sbg6")
        for row in variants:
            row_alias = _alias(salt, b"row", row["id"], "sbr6")
            public.append(
                {
                    "review_id": row_alias,
                    "group_id": group_alias,
                    **{
                        name: row[name]
                        for name in PUBLIC_FIELDS
                        if name not in ("review_id", "group_id")
                    },
                }
            )
            private_map.append(
                {
                    "review_id": row_alias,
                    "group_id": group_alias,
                    "source_id": row["id"],
                    "source_group_id": group,
                }
            )
    if (
        len(public) != GROUPS_PER_FAMILY * len(FAMILIES) * 3
        or len({row["review_id"] for row in public}) != len(public)
        or len({row["group_id"] for row in public}) != len(selected_groups)
    ):
        raise ValueError("Blind packet cardinality or aliases differ")
    public.sort(
        key=lambda row: (
            hmac.new(
                salt,
                b"order\0" + row["review_id"].encode(),
                hashlib.sha256,
            ).hexdigest(),
            row["review_id"],
        )
    )
    private_map.sort(key=lambda row: row["review_id"])
    return public, private_map, len(selected_groups)


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    map_output = args.map_output.resolve()
    if (
        output_dir.exists()
        or map_output.exists()
        or map_output.is_relative_to(output_dir)
    ):
        raise FileExistsError("Review packet and separate private map must be new")
    actual_train = pilot.sha_file(args.train)
    if actual_train != args.expected_train_sha256:
        raise ValueError("TRAIN bytes differ from frozen v6 candidate")
    if args.private_salt.stat().st_mode & 0o077:
        raise PermissionError("Private salt is accessible to other users")
    salt = args.private_salt.read_bytes()
    train = load_partition(args.train, "train")
    public, private_map, selected_count = select(train, salt)
    if any(set(row) != set(PUBLIC_FIELDS) for row in public):
        raise AssertionError("Gold or source IDs leaked into public packet")
    packet_bytes = pilot.jsonl_bytes(public)
    private_bytes = pilot.jsonl_bytes(private_map)
    output_dir.mkdir(parents=True, mode=0o700)
    map_output.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if map_output.parent.stat().st_mode & 0o077:
        raise PermissionError("Private map directory is accessible to other users")
    pilot._atomic_write(output_dir / "packet.jsonl", packet_bytes)
    pilot._atomic_write(map_output, private_bytes)
    manifest = {
        "schema_version": VERSION,
        "train_sha256": actual_train,
        "packet_builder_sha256": pilot.sha_file(Path(__file__)),
        "review_seed_sha256": hashlib.sha256(REVIEW_SEED.encode()).hexdigest(),
        "private_salt_sha256": pilot.sha_bytes(salt),
        "packet_sha256": pilot.sha_bytes(packet_bytes),
        "private_map_sha256": pilot.sha_bytes(private_bytes),
        "packet_rows": len(public),
        "selected_group_count": selected_count,
        "groups_per_family": GROUPS_PER_FAMILY,
        "groups_per_language_per_family": GROUPS_PER_LANGUAGE,
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
    manifest = build(parser.parse_args())
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
