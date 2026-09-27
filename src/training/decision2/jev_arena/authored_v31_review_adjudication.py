"""Audit submitted authored reviews and seal a private rejection ledger.

This consumes completed review seals with declared human/AI provenance. It
cannot certify personhood, AI run isolation or qualify a release pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jev_arena.authored_release_scale_v1 import file_sha, native_row, render_source
from jev_arena.authored_release_scale_v2_review import (
    ROLES,
    _validate_answers,
)

VERSION = "jevarena-authored-v31-review-adjudication/1"
ATTESTATION_VERSION = "jevarena-authored-v31-review-attestation/1"
POLICY_VERSION = "jevarena-authored-v31-review-policy/1"
VETO_VERSION = "jevarena-authored-v31-adjudicator-vetoes/1"
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _private(path: Path) -> bytes:
    if not path.is_file() or stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ValueError("Every adjudication input must be a private regular file")
    return path.read_bytes()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = _private(path)
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value, payload


def _jsonl(
    path: Path, *, allow_empty: bool = False
) -> tuple[list[dict[str, Any]], bytes]:
    payload = _private(path)
    rows = [json.loads(line) for line in payload.decode().splitlines() if line]
    if (not rows and not allow_empty) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Expected JSON object rows")
    return rows, payload


def _utc(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("UTC timestamp is missing")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Timestamp must have a UTC offset")
    return parsed.astimezone(timezone.utc)


def _identity(path: Path) -> str:
    value = _private(path).decode().strip()
    if not value or "\n" in value:
        raise ValueError("Private identity must be one nonempty line")
    return _sha(value.encode())


def _write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)


def _sealed_review(
    role: str,
    directory: Path,
    packet: Path,
    packet_receipt_sha: str,
    packet_sealed: datetime,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], str]:
    receipt, receipt_bytes = _json(directory / "receipt.private.json")
    rows, review_bytes = _jsonl(directory / "review.private.jsonl")
    packet_rows, packet_bytes = _jsonl(packet)
    if (
        receipt.get("status") != "HUMAN_REVIEW_SEALED_ADJUDICATION_PENDING"
        or receipt.get("role") != role
        or receipt.get("packet_receipt_sha256") != packet_receipt_sha
        or receipt.get("packet_sha256") != _sha(packet_bytes)
        or receipt.get("review_sha256") != _sha(review_bytes)
        or receipt.get("key_opened") is not False
        or receipt.get("release_qualified") is not False
        or _utc(receipt.get("sealed_at_utc")) <= packet_sealed
    ):
        raise ValueError(f"Invalid or changed sealed review: {role}")
    reviewer_id = receipt.get("reviewer_identity_sha256")
    if not isinstance(reviewer_id, str) or not HEX_SHA256.fullmatch(reviewer_id):
        raise ValueError("Reviewer identity digest is missing")
    packet_ids = [row.get("review_id") for row in packet_rows]
    if (
        len(set(packet_ids)) != len(packet_rows)
        or any(set(row) != {"review_id", "state", "questions"} for row in packet_rows)
        or any(not isinstance(item, str) for item in packet_ids)
    ):
        raise ValueError("Blind packet rows are invalid")
    counts = _validate_answers(packet_rows, rows)
    if (
        receipt.get("reviewed_rows") != counts["reviewed_rows"]
        or receipt.get("quality_flagged_rows") != counts["quality_flagged_rows"]
    ):
        raise ValueError("Sealed review summary changed")
    return receipt, {row["review_id"]: row for row in rows}, _sha(receipt_bytes)


def _quote_in_source(quote: Any, source: str) -> bool:
    if not isinstance(quote, str):
        return False
    text = " ".join(quote.split())
    body = " ".join(source.split("\n", 1)[-1].split())
    return len(text) >= 12 and text in body


def _quality_reasons(row: dict[str, Any], sources: tuple[str, str]) -> list[str]:
    reasons = []
    if not _quote_in_source(row["source_a_evidence"], sources[0]):
        reasons.append("source_a_citation")
    if not _quote_in_source(row["source_b_evidence"], sources[1]):
        reasons.append("source_b_citation")
    if row["both_sources_necessary"] is not True:
        reasons.append("source_necessity")
    if row["ambiguity"] != "none":
        reasons.append("ambiguity")
    if row["document_realism"] != "plausible":
        reasons.append("document_realism")
    if row["shortcut_risk"] != "none":
        reasons.append("shortcut_risk")
    if row["rights_concern"] is not False:
        reasons.append("rights_concern")
    if row["all_paragraphs_checked"] is not True:
        reasons.append("paragraph_coverage")
    return reasons


def _review_provenance(attestation: dict[str, Any]) -> dict[str, dict[str, str]]:
    provenance = attestation.get("reviewer_provenance")
    if not isinstance(provenance, dict) or set(provenance) != set(ROLES):
        raise ValueError("Reviewer provenance must cover all roles")
    ai_runs: set[str] = set()
    for role in ROLES:
        item = provenance[role]
        if not isinstance(item, dict) or item.get("kind") not in {"human", "ai"}:
            raise ValueError("Reviewer kind must be human or ai")
        if item["kind"] == "human":
            if (
                set(item) != {"kind", "verification"}
                or item["verification"] != "coordinator_attested"
            ):
                raise ValueError("Human reviewer provenance is incomplete")
        else:
            if set(item) != {"kind", "model_id", "run_id"} or any(
                not isinstance(item[field], str) or not item[field].strip()
                for field in ("model_id", "run_id")
            ):
                raise ValueError("AI reviewer model/run provenance is incomplete")
            if item["run_id"] in ai_runs:
                raise ValueError("AI reviewer runs must be distinct")
            ai_runs.add(item["run_id"])
    return provenance


def _review_policy(
    policy: dict[str, Any], provenance: dict[str, dict[str, str]]
) -> tuple[bool, dict[str, int]]:
    if policy.get("version") != POLICY_VERSION or set(policy) != {
        "version",
        "frozen_at_utc",
        "min_human_original_reviewers",
        "min_human_paired_reviewers",
    }:
        raise ValueError("Unknown or incomplete review policy")
    original_floor, paired_floor = (
        policy["min_human_original_reviewers"],
        policy["min_human_paired_reviewers"],
    )
    if (
        type(original_floor) is not int
        or not 0 <= original_floor <= 2
        or type(paired_floor) is not int
        or not 0 <= paired_floor <= 1
    ):
        raise ValueError("Invalid human reviewer floor")
    counts = Counter(item["kind"] for item in provenance.values())
    original_humans = sum(provenance[role]["kind"] == "human" for role in ROLES[:2])
    paired_humans = int(provenance["paired-review"]["kind"] == "human")
    return original_humans >= original_floor and paired_humans >= paired_floor, {
        "human": counts["human"],
        "ai": counts["ai"],
        "original_human": original_humans,
        "paired_human": paired_humans,
    }


def _source_texts(
    case: dict[str, Any], variant: dict[str, Any] | None
) -> tuple[str, str]:
    left, right = (source["data"] for source in case["sources"])
    if variant is not None:
        if variant["side"] == "left":
            left = variant["data"]
        elif variant["side"] == "right":
            right = variant["data"]
        else:
            raise ValueError("Unknown paired substitution side")
    return (
        render_source(case["sources"][0], left),
        render_source(case["sources"][1], right),
    )


def _check_packet_join(
    role: str,
    packet_rows: list[dict[str, Any]],
    joins: dict[str, Any],
    source_rows: dict[str, dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    variants: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    lookup = joins.get(role, {}).get("opaque_to_source_id")
    if not isinstance(lookup, dict) or set(lookup) != {
        row["review_id"] for row in packet_rows
    }:
        raise ValueError("Private packet join is incomplete")
    if len(set(lookup.values())) != len(lookup) or set(lookup.values()) != set(
        source_rows
    ):
        raise ValueError("Private packet join does not cover source rows")
    result = {}
    for row in packet_rows:
        source_id = lookup[row["review_id"]]
        expected = source_rows[source_id]
        if json.dumps(row["state"], ensure_ascii=False) != json.dumps(
            expected["state"], ensure_ascii=False
        ) or json.dumps(row["questions"], ensure_ascii=False) != json.dumps(
            expected["questions"], ensure_ascii=False
        ):
            raise ValueError("Blind packet changed native input")
        slug = source_id.removesuffix(":pair")
        case = cases[slug]
        variant = variants[slug] if role == "paired-review" else None
        sources = _source_texts(case, variant)
        generated = native_row(
            case,
            (
                variant["data"]
                if variant and variant["side"] == "left"
                else case["sources"][0]["data"]
            ),
            (
                variant["data"]
                if variant and variant["side"] == "right"
                else case["sources"][1]["data"]
            ),
            source_id,
        )
        if json.dumps(expected, ensure_ascii=False) != json.dumps(
            generated, ensure_ascii=False
        ):
            raise ValueError("Prepared native row differs from frozen source facts")
        result[source_id] = {
            "review_id": row["review_id"],
            "sources": sources,
            "question": next(iter(row["questions"].values())),
        }
    return result


def adjudicate(
    prepared: Path,
    packets: Path,
    preflight: Path,
    witness_audit: Path,
    review_dirs: dict[str, Path],
    author_id_file: Path,
    adjudicator_id_file: Path,
    coordinator_id_file: Path,
    coordinator_attestation: Path,
    review_policy: Path,
    adjudicator_vetoes: Path,
    output: Path,
) -> dict[str, Any]:
    """Verify sealed review inputs, then open authored keys and record exclusions."""
    if output.exists():
        raise FileExistsError("Adjudication output must be append-only")
    if set(review_dirs) != set(ROLES):
        raise ValueError("All three review roles are required")
    candidate, candidate_bytes = _json(prepared / "receipt.private.json")
    if (
        candidate.get("status")
        != "PRIVATE_V2_CANDIDATE_PREFLIGHT_PENDING_NO_BLIND_PACKET"
    ):
        raise ValueError("Unknown private candidate version/status")
    components = candidate.get("components_sha256")
    if not isinstance(components, dict) or set(components) != {
        "sources",
        "substitutions",
        "originals",
        "variants",
        "answers",
        "proofs",
    }:
        raise ValueError("Candidate component manifest is incomplete")
    # Hash protected answers without parsing them until every blind-review gate passes.
    for name, digest in components.items():
        if file_sha(prepared / f"{name}.private.jsonl") != digest:
            raise ValueError("Frozen candidate component changed")
    preflight_report, preflight_bytes = _json(preflight)
    witness, witness_bytes = _json(witness_audit)
    packet_receipt, packet_receipt_bytes = _json(packets / "receipt.private.json")
    if (
        preflight_report.get("status") != "MECHANICAL_PASS_EDITORIAL_PENDING"
        or preflight_report.get("hold_reasons") != []
        or preflight_report.get("candidate_receipt_sha256") != _sha(candidate_bytes)
        or witness.get("status") != "DOMAIN_SCREEN_PASS"
        or witness.get("casebook_sha256") != candidate.get("input_casebook_sha256")
        or packet_receipt.get("status") != "BLIND_PACKET_SEALED_REVIEW_PENDING"
        or packet_receipt.get("candidate_receipt_sha256") != _sha(candidate_bytes)
        or packet_receipt.get("preflight_sha256") != _sha(preflight_bytes)
        or packet_receipt.get("witness_audit_sha256") != _sha(witness_bytes)
        or packet_receipt.get("release_qualified") is not False
    ):
        raise ValueError("Candidate/preflight/packet chain is inconsistent")
    packet_sealed = _utc(packet_receipt.get("sealed_at_utc"))
    if packet_sealed <= _utc(candidate.get("prepared_at_utc")):
        raise ValueError("Blind packet predates the private candidate")
    if packet_sealed > datetime.now(timezone.utc):
        raise ValueError("Blind packet seal is in the future")
    packet_hashes = packet_receipt.get("packet_sha256")
    if not isinstance(packet_hashes, dict) or set(packet_hashes) != set(ROLES):
        raise ValueError("Blind packet manifest must contain three roles")
    joins_path = packets / "joins.private.json"
    if file_sha(joins_path) != packet_receipt.get("joins_sha256"):
        raise ValueError("Private join changed")
    review_receipts: dict[str, dict[str, Any]] = {}
    review_rows: dict[str, dict[str, dict[str, Any]]] = {}
    review_receipt_hashes = {}
    packet_rows = {}
    for role in ROLES:
        packet_path = packets / f"{role}.private.jsonl"
        packet, packet_bytes = _jsonl(packet_path)
        if _sha(packet_bytes) != packet_hashes[role]:
            raise ValueError("Blind packet changed")
        packet_rows[role] = packet
        meta, rows, meta_sha = _sealed_review(
            role,
            review_dirs[role],
            packet_path,
            _sha(packet_receipt_bytes),
            packet_sealed,
        )
        review_receipts[role] = meta
        review_rows[role] = rows
        review_receipt_hashes[role] = meta_sha
    identity_digests = [
        review_receipts[role]["reviewer_identity_sha256"] for role in ROLES
    ]
    author_digest = _identity(author_id_file)
    adjudicator_digest = _identity(adjudicator_id_file)
    coordinator_digest = _identity(coordinator_id_file)
    if (
        len(
            {
                *identity_digests,
                author_digest,
                adjudicator_digest,
                coordinator_digest,
            }
        )
        != 6
    ):
        raise ValueError("Author, reviewers, adjudicator and coordinator must differ")
    attestation, attestation_bytes = _json(coordinator_attestation)
    policy, policy_bytes = _json(review_policy)
    latest_review = max(_utc(review_receipts[role]["sealed_at_utc"]) for role in ROLES)
    if (
        attestation.get("version") != ATTESTATION_VERSION
        or attestation.get("candidate_receipt_sha256") != _sha(candidate_bytes)
        or attestation.get("packet_receipt_sha256") != _sha(packet_receipt_bytes)
        or attestation.get("review_receipt_sha256") != review_receipt_hashes
        or attestation.get("reviewer_identity_sha256")
        != dict(zip(ROLES, identity_digests))
        or attestation.get("author_identity_sha256") != author_digest
        or attestation.get("adjudicator_identity_sha256") != adjudicator_digest
        or attestation.get("coordinator_identity_sha256") != coordinator_digest
        or attestation.get("identity_provenance_recorded") is not True
        or attestation.get("independence_checked") is not True
        or attestation.get("key_closed_until_reviews_sealed") is not True
        or _utc(attestation.get("attested_at_utc")) <= latest_review
    ):
        raise ValueError("Review coordinator provenance attestation is incomplete")
    provenance = _review_provenance(attestation)
    policy_pass, kind_counts = _review_policy(policy, provenance)
    if _utc(policy["frozen_at_utc"]) > packet_sealed:
        raise ValueError("Review policy must predate blind packet sealing")
    if _utc(attestation["attested_at_utc"]) > datetime.now(timezone.utc):
        raise ValueError("Coordinator attestation is in the future")
    veto_report, veto_bytes = _json(adjudicator_vetoes)
    if (
        veto_report.get("version") != VETO_VERSION
        or veto_report.get("adjudicator_identity_sha256") != adjudicator_digest
        or veto_report.get("coordinator_attestation_sha256") != _sha(attestation_bytes)
        or not isinstance(veto_report.get("vetoes"), list)
        or _utc(veto_report.get("submitted_at_utc"))
        <= _utc(attestation["attested_at_utc"])
        or _utc(veto_report.get("submitted_at_utc")) > datetime.now(timezone.utc)
    ):
        raise ValueError("Adjudicator veto receipt is incomplete or out of order")
    veto_by_id = {}
    for row in veto_report["vetoes"]:
        if (
            set(row) != {"original_id", "reason"}
            or not isinstance(row["original_id"], str)
            or not isinstance(row["reason"], str)
            or not row["reason"].strip()
            or row["original_id"] in veto_by_id
        ):
            raise ValueError("Adjudicator veto ledger is malformed or repeated")
        veto_by_id[row["original_id"]] = row["reason"]

    # All blind-review and chronology checks precede this first authored-key read.
    answers, _ = _jsonl(prepared / "answers.private.jsonl")
    originals, _ = _jsonl(prepared / "originals.private.jsonl")
    variants_rows, _ = _jsonl(prepared / "variants.private.jsonl")
    source_cases, _ = _jsonl(prepared / "sources.private.jsonl")
    substitutions, _ = _jsonl(prepared / "substitutions.private.jsonl")
    joins, _ = _json(joins_path)
    if any(not isinstance(row.get("id"), str) for row in originals + variants_rows):
        raise ValueError("Prepared source IDs are invalid")
    original_by_id = {row["id"]: row for row in originals}
    variant_by_id = {row["id"]: row for row in variants_rows}
    case_by_id = {row["slug"]: row for row in source_cases}
    substitution_by_id = {row["slug"]: row for row in substitutions}
    answer_by_id = {row["slug"]: row for row in answers}
    expected_ids = set(original_by_id)
    if (
        len(expected_ids) != len(originals)
        or set(case_by_id) != expected_ids
        or set(substitution_by_id) != expected_ids
        or set(answer_by_id) != expected_ids
        or len(case_by_id) != len(source_cases)
        or len(substitution_by_id) != len(substitutions)
        or len(answer_by_id) != len(answers)
        or set(variant_by_id) != {f"{slug}:pair" for slug in expected_ids}
        or len(variant_by_id) != len(variants_rows)
        or set(veto_by_id) - expected_ids
    ):
        raise ValueError("Candidate IDs or adjudicator veto IDs are inconsistent")
    by_role = {
        role: _check_packet_join(
            role,
            packet_rows[role],
            joins,
            variant_by_id if role == "paired-review" else original_by_id,
            case_by_id,
            substitution_by_id,
        )
        for role in ROLES
    }
    if packet_receipt.get("originals_per_packet") != len(
        expected_ids
    ) or packet_receipt.get("paired_views_not_independent") != len(variant_by_id):
        raise ValueError("Packet count disagrees with prepared candidate")
    ledger = []
    reason_counts: Counter[str] = Counter()
    for slug in sorted(expected_ids):
        original_answer = answer_by_id[slug]["original"]
        paired_answer = answer_by_id[slug]["variant"]
        original_question = next(iter(original_by_id[slug]["questions"].values()))
        paired_question = next(
            iter(variant_by_id[f"{slug}:pair"]["questions"].values())
        )
        from jev_arena.authored_release_scale_v2_review import _valid_native

        if not _valid_native(original_answer, original_question) or not _valid_native(
            paired_answer, paired_question
        ):
            raise ValueError("Authored oracle answer is not native-valid")
        if original_answer == paired_answer:
            raise ValueError("Paired substitution does not change the answer")
        found = {}
        reasons = set()
        for role, source_id in (
            ("original-review-a", slug),
            ("original-review-b", slug),
            ("paired-review", f"{slug}:pair"),
        ):
            joined = by_role[role][source_id]
            review = review_rows[role][joined["review_id"]]
            found[role] = review["native_answer"]
            reasons.update(
                f"{role}:{reason}"
                for reason in _quality_reasons(review, joined["sources"])
            )
        if found["original-review-a"] != found["original-review-b"]:
            reasons.add("original_review_disagreement")
        if (
            found["original-review-a"] != original_answer
            or found["original-review-b"] != original_answer
        ):
            reasons.add("original_oracle_disagreement")
        if found["paired-review"] != paired_answer:
            reasons.add("paired_oracle_disagreement")
        if slug in veto_by_id:
            reasons.add("adjudicator_veto")
        reason_counts.update(reasons)
        ledger.append(
            {
                "original_id": slug,
                "status": (
                    "REJECT" if reasons else "REVIEW_DATA_PASS_PENDING_POOL_SIGNOFF"
                ),
                "reasons": sorted(reasons),
                "veto_reason": veto_by_id.get(slug),
            }
        )
    passed = sum(row["status"] != "REJECT" for row in ledger)
    status = (
        "NO_ELIGIBLE_ORIGINALS"
        if not passed
        else (
            "REVIEW_POLICY_HOLD"
            if not policy_pass
            else "REVIEW_DATA_AUDITED_POOL_SIGNOFF_PENDING"
        )
    )
    receipt = {
        "version": VERSION,
        "status": status,
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_receipt_sha256": _sha(candidate_bytes),
        "packet_receipt_sha256": _sha(packet_receipt_bytes),
        "review_receipt_sha256": review_receipt_hashes,
        "coordinator_attestation_sha256": _sha(attestation_bytes),
        "review_policy_sha256": _sha(policy_bytes),
        "adjudicator_vetoes_sha256": _sha(veto_bytes),
        "originals_reviewed": len(ledger),
        "originals_surviving_review_data_checks": passed,
        "originals_rejected": len(ledger) - passed,
        "rejection_reasons": dict(sorted(reason_counts.items())),
        "review_policy_pass": policy_pass,
        "reviewer_kind_counts": kind_counts,
        "reviewer_provenance": provenance,
        "pool_signoff_pending": True,
        "release_qualified": False,
        "final_gold_accessed": False,
        "model_inference": False,
    }
    output.mkdir(mode=0o700, parents=True)
    _write_new(
        output / "rejection-ledger.private.jsonl",
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in ledger
        ).encode(),
    )
    receipt["rejection_ledger_sha256"] = file_sha(
        output / "rejection-ledger.private.jsonl"
    )
    _write_new(
        output / "receipt.private.json",
        (
            json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode(),
    )
    return {
        key: receipt[key]
        for key in (
            "status",
            "originals_reviewed",
            "originals_surviving_review_data_checks",
            "originals_rejected",
            "rejection_reasons",
            "review_policy_pass",
            "reviewer_kind_counts",
            "release_qualified",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "prepared",
        "packets",
        "preflight",
        "witness-audit",
        "review-a",
        "review-b",
        "paired-review",
        "author-id-file",
        "adjudicator-id-file",
        "coordinator-id-file",
        "coordinator-attestation",
        "review-policy",
        "adjudicator-vetoes",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = adjudicate(
        args.prepared,
        args.packets,
        args.preflight,
        args.witness_audit,
        {
            "original-review-a": args.review_a,
            "original-review-b": args.review_b,
            "paired-review": args.paired_review,
        },
        args.author_id_file,
        args.adjudicator_id_file,
        args.coordinator_id_file,
        args.coordinator_attestation,
        args.review_policy,
        args.adjudicator_vetoes,
        args.output,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
