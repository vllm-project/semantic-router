from __future__ import annotations

import json
from pathlib import Path

import pytest
from cli.evaluation.suite_qualification import (
    BenchmarkQualificationReceipt,
    NormalizationOrigin,
    UnqualifiedBenchmarkEvidence,
)
from pydantic import ValidationError


def _receipt(
    *,
    origin: NormalizationOrigin = "user_provided_import",
    parser_verified: bool = False,
) -> BenchmarkQualificationReceipt:
    return BenchmarkQualificationReceipt(
        manifest_subject_digest="sha256:" + "1" * 64,
        source_receipt_digest="sha256:" + "2" * 64,
        artifact_set_digest="sha256:" + "3" * 64,
        qualification=UnqualifiedBenchmarkEvidence(
            origin=origin,
            parser_verified=parser_verified,
        ),
    )


def test_normalized_import_receipt_is_explicitly_e0_and_non_promotable() -> None:
    receipt = _receipt()

    assert receipt.evidence_level == "E0"
    assert receipt.qualification.status == "exploratory_import"
    assert receipt.qualification.native_execution_attested is False
    assert receipt.qualification.promotion_eligible is False
    assert receipt.qualified_gate_ids == ()


def test_normalized_import_provenance_matches_cross_language_golden() -> None:
    path = (
        Path(__file__).parent
        / "fixtures"
        / "evaluation"
        / "normalized-import-provenance.json"
    )
    golden = json.loads(path.read_text(encoding="utf-8"))

    assert golden == {
        "schema_version": "evaluation-suite-qualification.v2",
        "evidence_level": "E0",
        "origins": ["registered_parser_import", "user_provided_import"],
        "native_execution_attested": False,
        "promotion_eligible": False,
        "qualified_gate_ids": [],
    }


@pytest.mark.parametrize(
    ("origin", "parser_verified"),
    (
        ("registered_parser_import", False),
        ("user_provided_import", True),
    ),
)
def test_parser_verification_claim_must_match_import_origin(
    origin: NormalizationOrigin,
    parser_verified: bool,
) -> None:
    with pytest.raises(ValidationError, match="parser verification"):
        _receipt(origin=origin, parser_verified=parser_verified)


def test_forged_native_execution_or_promotion_claim_is_rejected() -> None:
    payload = _receipt().model_dump(mode="json")
    payload["qualification"]["native_execution_attested"] = True
    payload["qualification"]["promotion_eligible"] = True
    with pytest.raises(ValidationError):
        BenchmarkQualificationReceipt.model_validate(payload)


def test_non_e0_and_old_qualified_receipt_shapes_are_rejected() -> None:
    payload = _receipt().model_dump(mode="json")
    payload["evidence_level"] = "E4"
    payload["qualification"] = {
        "schema_version": "evaluation-suite-qualification.v2",
        "adapter_id": "routerarena",
        "evidence_level": "E4",
        "qualified_gate_ids": ["G4"],
        "normalizer_id": "routerarena.normalizer",
    }
    with pytest.raises(ValidationError):
        BenchmarkQualificationReceipt.model_validate(payload)
