"""Synthetic fixtures test review-data auditing, never human approval."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from jev_arena.authored_release_scale_v1 import file_sha, native_row, write_private
from jev_arena.authored_release_scale_v2_review import ROLES, seal, template
from jev_arena.authored_v31_review_adjudication import (
    ATTESTATION_VERSION,
    POLICY_VERSION,
    VETO_VERSION,
    adjudicate,
)


def _private(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)
    path.chmod(0o600)


def _jsonl(path: Path, rows: list[dict]) -> None:
    _private(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _case() -> dict:
    return {
        "slug": "synthetic-one",
        "operation": "quorum_veto",
        "scene": "A committee must decide whether the release is valid.",
        "contract": "Two listed signers and no veto are required.",
        "question": "Is the release valid?",
        "criteria": {"true": "valid", "false": "invalid"},
        "sources": [
            {
                "title": "Signer record",
                "form": "signed_memo",
                "document": "The signer ledger confirms {signers}.",
                "data": {"signers": ["A", "B"]},
            },
            {
                "title": "Veto roster",
                "form": "signed_memo",
                "document": "The approved roster is {roster}. The veto list is {veto}.",
                "data": {"roster": ["A", "B"], "veto": []},
            },
        ],
    }


def _receipt(paths: dict[str, Path], prepared: Path, packets: Path) -> dict:
    review_hashes = {
        role: file_sha(paths[role] / "receipt.private.json") for role in ROLES
    }
    names = {
        "original-review-a": "reviewer-a",
        "original-review-b": "reviewer-b",
        "paired-review": "reviewer-p",
    }
    return {
        "version": ATTESTATION_VERSION,
        "candidate_receipt_sha256": file_sha(prepared / "receipt.private.json"),
        "packet_receipt_sha256": file_sha(packets / "receipt.private.json"),
        "review_receipt_sha256": review_hashes,
        "reviewer_identity_sha256": {role: _sha(names[role]) for role in ROLES},
        "reviewer_provenance": {
            "original-review-a": {
                "kind": "human",
                "verification": "coordinator_attested",
            },
            "original-review-b": {
                "kind": "ai",
                "model_id": "synthetic-test-model",
                "run_id": "run-b",
            },
            "paired-review": {
                "kind": "ai",
                "model_id": "synthetic-test-model",
                "run_id": "run-p",
            },
        },
        "author_identity_sha256": _sha("author"),
        "adjudicator_identity_sha256": _sha("adjudicator"),
        "coordinator_identity_sha256": _sha("coordinator"),
        "identity_provenance_recorded": True,
        "independence_checked": True,
        "key_closed_until_reviews_sealed": True,
        "attested_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _fixture(tmp_path: Path) -> dict:
    prepared, packets = tmp_path / "prepared", tmp_path / "packets"
    prepared.mkdir(mode=0o700, parents=True)
    packets.mkdir(mode=0o700)
    case = _case()
    variant = {
        "slug": case["slug"],
        "side": "right",
        "data": {"roster": ["A", "B"], "veto": ["A"]},
    }
    original = native_row(
        case, case["sources"][0]["data"], case["sources"][1]["data"], case["slug"]
    )
    paired = native_row(
        case, case["sources"][0]["data"], variant["data"], case["slug"] + ":pair"
    )
    components = {
        "sources": [case],
        "substitutions": [variant],
        "originals": [original],
        "variants": [paired],
        "answers": [
            {"slug": case["slug"], "type": "noul", "original": True, "variant": False}
        ],
        "proofs": [{"slug": case["slug"], "proof": "fixture only"}],
    }
    hashes = {}
    for name, rows in components.items():
        path = prepared / f"{name}.private.jsonl"
        _jsonl(path, rows)
        hashes[name] = file_sha(path)
    write_private(
        prepared / "receipt.private.json",
        {
            "status": "PRIVATE_V2_CANDIDATE_PREFLIGHT_PENDING_NO_BLIND_PACKET",
            "prepared_at_utc": (
                datetime.now(timezone.utc) - timedelta(minutes=3)
            ).isoformat(),
            "components_sha256": hashes,
            "input_casebook_sha256": "casebook-fixture-hash",
        },
    )
    preflight = tmp_path / "preflight.private.json"
    write_private(
        preflight,
        {
            "status": "MECHANICAL_PASS_EDITORIAL_PENDING",
            "hold_reasons": [],
            "candidate_receipt_sha256": file_sha(prepared / "receipt.private.json"),
        },
    )
    witness = tmp_path / "witness.private.json"
    write_private(
        witness,
        {"status": "DOMAIN_SCREEN_PASS", "casebook_sha256": "casebook-fixture-hash"},
    )
    mappings = {
        "original-review-a": {"opaque-a": case["slug"]},
        "original-review-b": {"opaque-b": case["slug"]},
        "paired-review": {"opaque-p": case["slug"] + ":pair"},
    }
    packet_hashes = {}
    for role, source in (
        ("original-review-a", original),
        ("original-review-b", original),
        ("paired-review", paired),
    ):
        opaque = next(iter(mappings[role]))
        packet = packets / f"{role}.private.jsonl"
        _jsonl(
            packet,
            [
                {
                    "review_id": opaque,
                    "state": source["state"],
                    "questions": source["questions"],
                }
            ],
        )
        packet_hashes[role] = file_sha(packet)
    write_private(
        packets / "joins.private.json",
        {
            role: {"salt_hex": "test-only", "opaque_to_source_id": lookup}
            for role, lookup in mappings.items()
        },
    )
    write_private(
        packets / "receipt.private.json",
        {
            "status": "BLIND_PACKET_SEALED_REVIEW_PENDING",
            "sealed_at_utc": (
                datetime.now(timezone.utc) - timedelta(minutes=2)
            ).isoformat(),
            "candidate_receipt_sha256": file_sha(prepared / "receipt.private.json"),
            "preflight_sha256": file_sha(preflight),
            "witness_audit_sha256": file_sha(witness),
            "packet_sha256": packet_hashes,
            "joins_sha256": file_sha(packets / "joins.private.json"),
            "originals_per_packet": 1,
            "paired_views_not_independent": 1,
            "reviewer_assignments": 0,
            "human_reviews_completed": 0,
            "release_qualified": False,
        },
    )
    review_dirs = {}
    for role, answer in (
        ("original-review-a", True),
        ("original-review-b", True),
        ("paired-review", False),
    ):
        packet = packets / f"{role}.private.jsonl"
        form = tmp_path / f"{role}-form.private.jsonl"
        template(packet, packets / "receipt.private.json", role, form)
        row = json.loads(form.read_text())
        row.update(
            native_answer=answer,
            source_a_evidence='The signer ledger confirms ["A", "B"].',
            source_b_evidence=(
                'The veto list is ["A"].'
                if role == "paired-review"
                else 'The approved roster is ["A", "B"].'
            ),
            both_sources_necessary=True,
            ambiguity="none",
            document_realism="plausible",
            shortcut_risk="none",
            rights_concern=False,
            all_paragraphs_checked=True,
            paragraph_notes="The two presented sources were checked.",
        )
        _jsonl(form, [row])
        reviewer_id = tmp_path / f"{role}-identity.private.txt"
        short = (
            "reviewer-a"
            if role == "original-review-a"
            else "reviewer-b" if role == "original-review-b" else "reviewer-p"
        )
        _private(reviewer_id, short + "\n")
        directory = tmp_path / f"{role}-sealed"
        seal(
            packet, packets / "receipt.private.json", role, form, reviewer_id, directory
        )
        review_dirs[role] = directory
    identities = {}
    for role in ("author", "adjudicator", "coordinator"):
        path = tmp_path / f"{role}.private.txt"
        _private(path, role + "\n")
        identities[role] = path
    attestation = tmp_path / "attestation.private.json"
    write_private(attestation, _receipt(review_dirs, prepared, packets))
    policy = tmp_path / "policy.private.json"
    write_private(
        policy,
        {
            "version": POLICY_VERSION,
            "frozen_at_utc": (
                datetime.now(timezone.utc) - timedelta(minutes=4)
            ).isoformat(),
            "min_human_original_reviewers": 1,
            "min_human_paired_reviewers": 0,
        },
    )
    vetoes = tmp_path / "vetoes.private.json"
    write_private(
        vetoes,
        {
            "version": VETO_VERSION,
            "adjudicator_identity_sha256": _sha("adjudicator"),
            "coordinator_attestation_sha256": file_sha(attestation),
            "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
            "vetoes": [],
        },
    )
    return {
        "prepared": prepared,
        "packets": packets,
        "preflight": preflight,
        "witness": witness,
        "reviews": review_dirs,
        "ids": identities,
        "attestation": attestation,
        "policy": policy,
        "vetoes": vetoes,
        "output": tmp_path / "adjudicated",
    }


def _run(fixture: dict) -> dict:
    return adjudicate(
        fixture["prepared"],
        fixture["packets"],
        fixture["preflight"],
        fixture["witness"],
        fixture["reviews"],
        fixture["ids"]["author"],
        fixture["ids"]["adjudicator"],
        fixture["ids"]["coordinator"],
        fixture["attestation"],
        fixture["policy"],
        fixture["vetoes"],
        fixture["output"],
    )


def _refresh_veto_binding(fixture: dict) -> None:
    vetoes = json.loads(fixture["vetoes"].read_text())
    vetoes["coordinator_attestation_sha256"] = file_sha(fixture["attestation"])
    vetoes["submitted_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_private(fixture["vetoes"], vetoes)


def test_review_data_pass_still_cannot_qualify_release(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = _run(fixture)
    assert result["originals_surviving_review_data_checks"] == 1
    assert result["review_policy_pass"] is True
    assert result["reviewer_kind_counts"]["human"] == 1
    assert result["reviewer_kind_counts"]["ai"] == 2
    assert result["release_qualified"] is False
    receipt = json.loads((fixture["output"] / "receipt.private.json").read_text())
    assert receipt["pool_signoff_pending"] is True
    assert receipt["final_gold_accessed"] is False


def test_stricter_human_policy_records_hold_without_relabeling_ai(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    write_private(
        fixture["policy"],
        {
            "version": POLICY_VERSION,
            "frozen_at_utc": (
                datetime.now(timezone.utc) - timedelta(minutes=4)
            ).isoformat(),
            "min_human_original_reviewers": 2,
            "min_human_paired_reviewers": 1,
        },
    )
    result = _run(fixture)
    assert result["status"] == "REVIEW_POLICY_HOLD"
    assert result["review_policy_pass"] is False
    assert result["reviewer_kind_counts"]["human"] == 1
    assert result["release_qualified"] is False


def test_review_disagreement_excludes_original_and_pair(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    review_path = fixture["reviews"]["original-review-b"] / "review.private.jsonl"
    row = json.loads(review_path.read_text())
    row["native_answer"] = False
    _jsonl(review_path, [row])
    receipt_path = fixture["reviews"]["original-review-b"] / "receipt.private.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["review_sha256"] = file_sha(review_path)
    write_private(receipt_path, receipt)
    write_private(
        fixture["attestation"],
        _receipt(fixture["reviews"], fixture["prepared"], fixture["packets"]),
    )
    _refresh_veto_binding(fixture)
    result = _run(fixture)
    assert result["originals_rejected"] == 1
    assert result["rejection_reasons"]["original_review_disagreement"] == 1
    ledger = json.loads(
        (fixture["output"] / "rejection-ledger.private.jsonl").read_text()
    )
    assert ledger["status"] == "REJECT"


def test_adjudicator_veto_is_one_way_rejection(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    vetoes = json.loads(fixture["vetoes"].read_text())
    vetoes["vetoes"] = [{"original_id": "synthetic-one", "reason": "source realism"}]
    write_private(fixture["vetoes"], vetoes)
    result = _run(fixture)
    assert result["originals_rejected"] == 1
    assert result["rejection_reasons"]["adjudicator_veto"] == 1


def test_citation_must_quote_the_correct_source_body(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    review_path = fixture["reviews"]["original-review-a"] / "review.private.jsonl"
    row = json.loads(review_path.read_text())
    row["source_a_evidence"] = 'The approved roster is ["A", "B"].'
    _jsonl(review_path, [row])
    receipt_path = fixture["reviews"]["original-review-a"] / "receipt.private.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["review_sha256"] = file_sha(review_path)
    write_private(receipt_path, receipt)
    write_private(
        fixture["attestation"],
        _receipt(fixture["reviews"], fixture["prepared"], fixture["packets"]),
    )
    _refresh_veto_binding(fixture)
    result = _run(fixture)
    assert result["status"] == "NO_ELIGIBLE_ORIGINALS"
    assert result["rejection_reasons"]["original-review-a:source_a_citation"] == 1


def test_policy_must_predate_packet_and_review_seals_are_immutable(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    policy = json.loads(fixture["policy"].read_text())
    policy["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_private(fixture["policy"], policy)
    with pytest.raises(ValueError, match="predate blind packet"):
        _run(fixture)
    assert not fixture["output"].exists()

    fixture = _fixture(tmp_path / "changed-review")
    review_path = fixture["reviews"]["paired-review"] / "review.private.jsonl"
    _private(review_path, review_path.read_text() + "\n")
    with pytest.raises(ValueError, match="Invalid or changed sealed review"):
        _run(fixture)
    assert not fixture["output"].exists()


def test_tamper_or_same_ai_run_fails_before_key_output(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    packet = fixture["packets"] / "original-review-a.private.jsonl"
    _private(packet, packet.read_text() + "\n")
    with pytest.raises(ValueError, match="Blind packet changed"):
        _run(fixture)
    assert not fixture["output"].exists()

    fixture = _fixture(tmp_path / "another")
    attestation = json.loads(fixture["attestation"].read_text())
    attestation["reviewer_provenance"]["paired-review"]["run_id"] = "run-b"
    write_private(fixture["attestation"], attestation)
    with pytest.raises(ValueError, match="AI reviewer runs"):
        _run(fixture)
    assert not fixture["output"].exists()
