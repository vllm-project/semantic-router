"""Produce a private gold-blind English audit packet for every MASSIVE TRAIN group."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

from training.data import build_massive_multilingual as massive


def review_id(source_id: str) -> str:
    return hashlib.sha256(
        f"massive-expanded-review-v1\0{source_id}".encode()
    ).hexdigest()[:20]


def build(candidate: Path, expected_manifest_sha: str, output: Path) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    manifest_path = candidate / "manifest.json"
    if massive.sha(manifest_path) != expected_manifest_sha:
        raise ValueError("Candidate manifest digest changed")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest["training_approved"]
        or manifest["selected_source_groups"]["train"] != 600
    ):
        raise ValueError("Expected the unapproved 600-group MASSIVE candidate")
    input_path = candidate / "train.private.jsonl"
    if massive.sha(input_path) != manifest["outputs"]["train.private.jsonl"]["sha256"]:
        raise ValueError("Candidate TRAIN digest changed")
    groups = defaultdict(list)
    with input_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            groups[row["group_id"]].append(row)
    if len(groups) != 600 or any(len(rows) != 7 for rows in groups.values()):
        raise ValueError("Candidate parallel groups changed")
    packet, key = [], []
    for group_id, rows in groups.items():
        english = [
            row for row in rows if row["audit_metadata"]["source_locale"] == "en-US"
        ]
        if len(english) != 1:
            raise ValueError("Expected exactly one English row per source group")
        row = english[0]
        source_id = row["audit_metadata"]["source_id"]
        identifier = review_id(source_id)
        packet.append(
            {
                "review_id": identifier,
                "utterance": row["state"],
                "instruction": row["instructions"],
                "options": row["options"],
            }
        )
        key.append(
            {
                "review_id": identifier,
                "source_id": source_id,
                "source_intent": row["audit_metadata"]["intent"],
                "gold_option_key": row["options"][row["label"]]["key"],
            }
        )
    packet.sort(key=lambda row: row["review_id"])
    key.sort(key=lambda row: row["review_id"])
    if len({row["review_id"] for row in packet}) != 600:
        raise ValueError("Review IDs collided")
    output.mkdir(mode=0o700, parents=True)
    massive.write_jsonl(output / "english-all600.blind.private.jsonl", packet)
    massive.write_jsonl(output / "english-all600.key.private.jsonl", key)
    receipt = {
        "schema_version": "decision2-massive-expanded-blind-review/1",
        "candidate_manifest_sha256": expected_manifest_sha,
        "candidate_train_sha256": manifest["outputs"]["train.private.jsonl"]["sha256"],
        "blind_review_rows": len(packet),
        "gold_key_separate": True,
        "training_approved": False,
        "files": {
            name: massive.sha(output / name)
            for name in (
                "english-all600.blind.private.jsonl",
                "english-all600.key.private.jsonl",
            )
        },
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    os.chmod(output / "english-all600.blind.private.jsonl", 0o600)
    os.chmod(output / "english-all600.key.private.jsonl", 0o600)
    os.chmod(output / "receipt.json", 0o600)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--manifest-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.candidate, args.manifest_sha, args.output)
    print(
        json.dumps(
            {
                "blind_review_rows": result["blind_review_rows"],
                "training_approved": result["training_approved"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
