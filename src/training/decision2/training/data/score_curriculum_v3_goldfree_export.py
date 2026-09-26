"""Export the frozen Score v3 TRAIN prompts for overlap checks without gold."""

from __future__ import annotations

import argparse
import collections
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any

EXPECTED_TRAIN_SHA256 = (
    "c4ea3294247022fdf2359e6ed74c0abec0bc6de0295fad06af6271c37a9d0fb7"
)
PUBLIC_FIELDS = (
    "review_id",
    "group_id",
    "family",
    "language",
    "state",
    "instructions",
    "options",
)


def _alias(salt: bytes, domain: bytes, value: str, prefix: str) -> str:
    digest = hmac.new(salt, domain + b"\0" + value.encode(), hashlib.sha256).hexdigest()
    return f"{prefix}-{digest[:24]}"


def project(rows: list[dict[str, Any]], salt: bytes) -> list[dict[str, Any]]:
    if len(salt) != 32:
        raise ValueError("A fresh private 32-byte salt is required")
    if len(rows) != 960:
        raise ValueError("Expected 960 frozen v3 TRAIN rows")
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        if set(
            (
                "id",
                "group_id",
                "family",
                "language",
                "state",
                "instructions",
                "options",
                "label",
            )
        ) - set(row):
            raise ValueError("TRAIN schema is incomplete")
        groups[row["group_id"]].append(row)
    if len(groups) != 320:
        raise ValueError("Expected 320 complete v3 TRAIN groups")
    if any(
        len(variants) != 3 or {row["label"] for row in variants} != {0, 1, 2}
        for variants in groups.values()
    ):
        raise ValueError("A v3 TRAIN group is incomplete")
    public = []
    for row in rows:
        public.append(
            {
                "review_id": _alias(salt, b"row", row["id"], "sg3r"),
                "group_id": _alias(salt, b"group", row["group_id"], "sg3g"),
                "family": row["family"],
                "language": row["language"],
                "state": row["state"],
                "instructions": row["instructions"],
                "options": row["options"],
            }
        )
    if any(set(row) != set(PUBLIC_FIELDS) for row in public):
        raise AssertionError("Gold or source field leaked into projection")
    if len({row["review_id"] for row in public}) != len(public):
        raise AssertionError("Opaque row alias collision")
    if len({row["group_id"] for row in public}) != len(groups):
        raise AssertionError("Opaque group alias collision")
    public.sort(
        key=lambda row: hmac.new(
            salt, b"order\0" + row["review_id"].encode(), hashlib.sha256
        ).hexdigest()
    )
    return public


def export(train: Path, salt_file: Path, output_dir: Path) -> dict[str, Any]:
    source = train.read_bytes()
    source_sha = hashlib.sha256(source).hexdigest()
    if source_sha != EXPECTED_TRAIN_SHA256:
        raise ValueError("Frozen v3 TRAIN bytes differ")
    if salt_file.stat().st_mode & 0o077:
        raise PermissionError("Private salt permissions are too broad")
    salt = salt_file.read_bytes()
    rows = [json.loads(line) for line in source.splitlines() if line.strip()]
    public = project(rows, salt)
    if output_dir.exists():
        raise FileExistsError("Gold-free export path must be fresh")
    output_dir.mkdir(mode=0o700, parents=True)
    body = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()
        for row in public
    )
    packet_path = output_dir / "prompts.jsonl"
    descriptor = os.open(packet_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(body)
    manifest = {
        "schema_version": "decision20-score-v3-train-goldfree-overlap/1",
        "source_train_sha256": source_sha,
        "source_projection_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "prompts_sha256": hashlib.sha256(body).hexdigest(),
        "rows": len(public),
        "groups": len({row["group_id"] for row in public}),
        "fields": PUBLIC_FIELDS,
        "gold_included": False,
        "use": "overlap_detection_only",
    }
    manifest_path = output_dir / "manifest.json"
    descriptor = os.open(manifest_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--private-salt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = export(args.train, args.private_salt, args.output_dir)
    print(
        json.dumps(
            {
                "rows": result["rows"],
                "groups": result["groups"],
                "prompts_sha256": result["prompts_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
