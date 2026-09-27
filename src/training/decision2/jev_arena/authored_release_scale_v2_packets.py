"""Seal gold-free private v2 editorial packets after the mechanical gates.

Two independently salted original packets support two separate reviewers.
A third packet holds paired source substitutions for a different reviewer.
The join, salts, keys and raw packets never leave task-private remote space.
Packet creation is a request for editorial review, not release acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jev_arena.authored_release_scale_v1 import file_sha, write_private

VERSION = "jevarena-authored-release-scale-v2-blind-packets/1"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _blind_rows(
    source: list[dict[str, Any]], salt: bytes
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    packet = []
    joins = {}
    for row in source:
        opaque = hashlib.sha256(salt + b"\0" + row["id"].encode()).hexdigest()[:24]
        if opaque in joins:
            raise ValueError("Blind review ID collision")
        joins[opaque] = row["id"]
        packet.append(
            {"review_id": opaque, "state": row["state"], "questions": row["questions"]}
        )
    secrets.SystemRandom().shuffle(packet)
    return packet, joins


def _write_packet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            # Choice criteria order is part of the native input contract.
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in rows
        )
    )
    path.chmod(0o600)


def _validate_written_packet(
    path: Path, source: list[dict[str, Any]], joins: dict[str, str]
) -> None:
    by_id = {row["id"]: row for row in source}
    written = _rows(path)
    if len(by_id) != len(source) or len(written) != len(source):
        raise ValueError("Review packet/source count or identity changed")
    if {row["review_id"] for row in written} != set(joins):
        raise ValueError("Review packet opaque IDs changed")
    for row in written:
        original = by_id[joins[row["review_id"]]]
        for field in ("state", "questions"):
            if json.dumps(row[field], ensure_ascii=False) != json.dumps(
                original[field], ensure_ascii=False
            ):
                raise ValueError("Review packet changed native input or option order")


def seal(
    prepared: Path, preflight: Path, witness_audit: Path, output: Path
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("Private review packets cannot be overwritten")
    receipt = json.loads((prepared / "receipt.private.json").read_text())
    for name, expected in receipt["components_sha256"].items():
        if file_sha(prepared / f"{name}.private.jsonl") != expected:
            raise ValueError("Candidate snapshot changed after preparation")
    overlap = json.loads(preflight.read_text())
    witness = json.loads(witness_audit.read_text())
    if (
        overlap["status"] != "MECHANICAL_PASS_EDITORIAL_PENDING"
        or overlap["hold_reasons"]
        or overlap["candidate_receipt_sha256"]
        != file_sha(prepared / "receipt.private.json")
        or witness["status"] != "DOMAIN_SCREEN_PASS"
        or witness["casebook_sha256"] != receipt["input_casebook_sha256"]
    ):
        raise ValueError("Blind packet gates are incomplete or mismatched")
    originals = _rows(prepared / "originals.private.jsonl")
    variants = _rows(prepared / "variants.private.jsonl")
    if (
        not len(originals)
        == len(variants)
        == receipt["independent_original_candidates"]
    ):
        raise ValueError("Blind packet cardinality mismatch")
    output.mkdir(mode=0o700, parents=True)
    joins: dict[str, Any] = {}
    packet_hashes = {}
    for name, source in (
        ("original-review-a", originals),
        ("original-review-b", originals),
        ("paired-review", variants),
    ):
        salt = secrets.token_bytes(32)
        packet, lookup = _blind_rows(source, salt)
        path = output / f"{name}.private.jsonl"
        _write_packet(path, packet)
        _validate_written_packet(path, source, lookup)
        packet_hashes[name] = file_sha(path)
        joins[name] = {"salt_hex": salt.hex(), "opaque_to_source_id": lookup}
    write_private(output / "joins.private.json", joins)
    packet_receipt = {
        "version": VERSION,
        "status": "BLIND_PACKET_SEALED_REVIEW_PENDING",
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_source_commit": receipt["source_commit"],
        "sealer_sha256": file_sha(Path(__file__)),
        "candidate_receipt_sha256": file_sha(prepared / "receipt.private.json"),
        "preflight_sha256": file_sha(preflight),
        "witness_audit_sha256": file_sha(witness_audit),
        "packet_sha256": packet_hashes,
        "joins_sha256": file_sha(output / "joins.private.json"),
        "originals_per_packet": len(originals),
        "paired_views_not_independent": len(variants),
        "reviewer_assignments": 0,
        "human_reviews_completed": 0,
        "model_inference": False,
        "release_qualified": False,
    }
    write_private(output / "receipt.private.json", packet_receipt)
    return {
        key: packet_receipt[key]
        for key in (
            "status",
            "originals_per_packet",
            "paired_views_not_independent",
            "reviewer_assignments",
            "human_reviews_completed",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--witness-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(seal(args.prepared, args.preflight, args.witness_audit, args.output))
    )


if __name__ == "__main__":
    main()
