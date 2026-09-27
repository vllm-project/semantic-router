"""Prepare and seal human editorial responses to gold-free authored packets.

This tool validates a reviewer's direct typed answer and row-level quality
judgment. It never reads oracle answers, source joins, proofs, or model output.
An adjudicator must independently verify reviewer identities and open the
separately held key only after all required reviews have been sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VERSION = "jevarena-authored-human-review/1"
ROLES = ("original-review-a", "original-review-b", "paired-review")
FIELDS = frozenset(
    {
        "review_id",
        "native_answer",
        "source_a_evidence",
        "source_b_evidence",
        "both_sources_necessary",
        "ambiguity",
        "document_realism",
        "shortcut_risk",
        "rights_concern",
        "all_paragraphs_checked",
        "paragraph_notes",
        "notes",
    }
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _rows(payload: bytes) -> list[dict[str, Any]]:
    result = [json.loads(line) for line in payload.decode().splitlines() if line]
    if not result or any(not isinstance(row, dict) for row in result):
        raise ValueError("Review file must contain nonempty JSON object rows")
    return result


def _private(path: Path) -> None:
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ValueError("Review inputs must be private to their owner")


def _write_private_bytes(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)


def _packet(
    packet: Path, receipt: Path, role: str
) -> tuple[list[dict[str, Any]], bytes, bytes, dict[str, Any]]:
    if role not in ROLES or packet.name != f"{role}.private.jsonl":
        raise ValueError("Unknown or mismatched blind packet role")
    packet_bytes = packet.read_bytes()
    receipt_bytes = receipt.read_bytes()
    meta = json.loads(receipt_bytes)
    if (
        meta.get("status") != "BLIND_PACKET_SEALED_REVIEW_PENDING"
        or meta.get("packet_sha256", {}).get(role) != _sha(packet_bytes)
        or meta.get("reviewer_assignments") != 0
        or meta.get("human_reviews_completed") != 0
    ):
        raise ValueError("Blind packet seal is missing or changed")
    rows = _rows(packet_bytes)
    ids = [row.get("review_id") for row in rows]
    if len(set(ids)) != len(rows) or any(not isinstance(value, str) for value in ids):
        raise ValueError("Blind packet review IDs are invalid")
    for row in rows:
        if set(row) != {"review_id", "state", "questions"}:
            raise ValueError("Blind packet contains unexpected fields")
        questions = row["questions"]
        if not isinstance(questions, dict) or len(questions) != 1:
            raise ValueError("Authored review expects one native typed question")
    return rows, packet_bytes, receipt_bytes, meta


def template(packet: Path, receipt: Path, role: str, output: Path) -> None:
    if output.exists():
        raise FileExistsError("Review template cannot overwrite an existing file")
    _private(packet)
    _private(receipt)
    rows, _, _, _ = _packet(packet, receipt, role)
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    forms = []
    for row in rows:
        review = {
            "review_id": row["review_id"],
            "native_answer": None,
            "source_a_evidence": "",
            "source_b_evidence": "",
            "both_sources_necessary": None,
            "ambiguity": None,
            "document_realism": None,
            "shortcut_risk": None,
            "rights_concern": None,
            "all_paragraphs_checked": None,
            "paragraph_notes": "",
            "notes": "",
        }
        forms.append(json.dumps(review, ensure_ascii=False) + "\n")
    _write_private_bytes(output, "".join(forms).encode())


def _valid_native(answer: Any, question: dict[str, Any]) -> bool:
    kind, criteria = question.get("type"), question.get("criteria")
    if kind == "choice":
        return (
            isinstance(answer, str)
            and isinstance(criteria, dict)
            and answer in criteria
        )
    if kind == "noul":
        return isinstance(answer, bool)
    if kind == "score":
        return (
            isinstance(answer, int)
            and not isinstance(answer, bool)
            and isinstance(criteria, list)
            and 0 <= answer < len(criteria)
        )
    return False


def _validate_answers(
    packet: list[dict[str, Any]], answers: list[dict[str, Any]]
) -> dict[str, int]:
    by_id = {row["review_id"]: next(iter(row["questions"].values())) for row in packet}
    seen: set[str] = set()
    flagged = 0
    for row in answers:
        if set(row) != FIELDS:
            raise ValueError("Review row has missing or unexpected fields")
        review_id = row["review_id"]
        if (
            not isinstance(review_id, str)
            or review_id not in by_id
            or review_id in seen
        ):
            raise ValueError("Review ID missing from packet or repeated")
        seen.add(review_id)
        if not _valid_native(row["native_answer"], by_id[review_id]):
            raise ValueError("Review answer is invalid for the native question")
        if any(
            not isinstance(row[field], str) or not row[field].strip()
            for field in ("source_a_evidence", "source_b_evidence", "paragraph_notes")
        ):
            raise ValueError("Both source citations and paragraph notes are required")
        if not isinstance(row["notes"], str):
            raise ValueError("Review notes must be text")
        if any(
            not isinstance(row[field], bool)
            for field in (
                "both_sources_necessary",
                "rights_concern",
                "all_paragraphs_checked",
            )
        ):
            raise ValueError("Review quality booleans must be explicit")
        if row["ambiguity"] not in ("none", "minor", "material"):
            raise ValueError("Review ambiguity must be explicit")
        if row["document_realism"] not in ("plausible", "concern"):
            raise ValueError("Review realism must be explicit")
        if row["shortcut_risk"] not in ("none", "concern"):
            raise ValueError("Review shortcut risk must be explicit")
        flagged += int(
            not row["both_sources_necessary"]
            or row["ambiguity"] != "none"
            or row["document_realism"] == "concern"
            or row["shortcut_risk"] == "concern"
            or row["rights_concern"]
            or not row["all_paragraphs_checked"]
        )
    if seen != set(by_id):
        raise ValueError("Every blind packet row must receive one review")
    return {"reviewed_rows": len(seen), "quality_flagged_rows": flagged}


def seal(
    packet: Path,
    packet_receipt: Path,
    role: str,
    answers: Path,
    reviewer_id_file: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("Sealed human review cannot be overwritten")
    for path in (packet, packet_receipt, answers, reviewer_id_file):
        _private(path)
    rows, packet_bytes, receipt_bytes, receipt = _packet(packet, packet_receipt, role)
    reviewer = reviewer_id_file.read_text().strip()
    if not reviewer or "\n" in reviewer:
        raise ValueError("Private reviewer identity is missing")
    answer_bytes = answers.read_bytes()
    summary = _validate_answers(rows, _rows(answer_bytes))
    sealed_at = datetime.now(timezone.utc)
    if sealed_at <= datetime.fromisoformat(receipt["sealed_at_utc"]):
        raise ValueError("Review cannot predate the blind packet")
    output.mkdir(mode=0o700, parents=True)
    review_copy = output / "review.private.jsonl"
    _write_private_bytes(review_copy, answer_bytes)
    meta = {
        "version": VERSION,
        "status": "HUMAN_REVIEW_SEALED_ADJUDICATION_PENDING",
        "role": role,
        "sealed_at_utc": sealed_at.isoformat(),
        "packet_receipt_sha256": _sha(receipt_bytes),
        "packet_sha256": _sha(packet_bytes),
        "review_sha256": _sha(answer_bytes),
        "reviewer_identity_sha256": hashlib.sha256(reviewer.encode()).hexdigest(),
        **summary,
        "key_opened": False,
        "release_qualified": False,
    }
    _write_private_bytes(
        output / "receipt.private.json",
        (
            json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode(),
    )
    return {key: meta[key] for key in ("status", "role", *summary)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("template", "seal"):
        command = sub.add_parser(name)
        command.add_argument("--packet", type=Path, required=True)
        command.add_argument("--packet-receipt", type=Path, required=True)
        command.add_argument("--role", choices=ROLES, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "seal":
            command.add_argument("--answers", type=Path, required=True)
            command.add_argument("--reviewer-id-file", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "template":
        template(args.packet, args.packet_receipt, args.role, args.output)
    else:
        print(
            json.dumps(
                seal(
                    args.packet,
                    args.packet_receipt,
                    args.role,
                    args.answers,
                    args.reviewer_id_file,
                    args.output,
                )
            )
        )


if __name__ == "__main__":
    main()
