"""Cross-language conformance: the CLI's version gate must agree with the
Router's (canonical_version.go), which this corpus also drives from
Go's TestVersionGateMatchesSharedCrossLanguageConformance. See issue #2469."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from cli.config_schema import schema_document
from cli.config_schema.validation import (
    ConfigSchemaValidator,
    validate_config_structure,
)
from cli.models import UserConfig
from pydantic import ValidationError

_VERSION_GATE_CONFORMANCE_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "router_config_version_gate_conformance.v1.json"
)


def _document_for_case(case: dict[str, Any]) -> dict[str, Any]:
    if not case["version_present"]:
        return {}
    return {"version": case["version"]}


def _accepts_current_contract(document: dict[str, Any]) -> bool:
    """Mirror cli.parser.parse_user_config's gate order: the generated JSON
    Schema runs first and short-circuits, then Pydantic. Either can reject."""
    if validate_config_structure(document):
        return False
    try:
        UserConfig.model_validate(document, extra="allow")
    except ValidationError:
        return False
    return True


def _accepts_with_version_enum(
    document: dict[str, Any], accepted_versions: list[str]
) -> bool:
    """Structural-only check against a schema whose version enum is patched to
    a different accepted set, mirroring how the Go test swaps the package-level
    acceptedCanonicalVersions to simulate a retained contract."""
    schema = copy.deepcopy(schema_document())
    schema["properties"]["version"] = {
        "type": "string",
        "enum": ["", *accepted_versions],
    }
    validator = ConfigSchemaValidator(schema)
    return next(validator.iter_errors(document), None) is None


def test_version_gate_matches_shared_conformance_current_contract() -> None:
    corpus = json.loads(_VERSION_GATE_CONFORMANCE_FIXTURE.read_text(encoding="utf-8"))
    assert corpus["schema_version"] == "router-config-version-gate-conformance.v1"
    contract = corpus["current_contract"]
    assert contract["accepted_versions"]
    case_ids = [case["id"] for case in contract["cases"]]
    assert case_ids
    assert len(case_ids) == len(set(case_ids))

    for case in contract["cases"]:
        assert set(case) == {"id", "version_present", "version", "expected_valid"}
        document = _document_for_case(case)
        accepted = _accepts_current_contract(document)
        assert accepted is case["expected_valid"], case["id"]


def test_version_gate_matches_shared_conformance_retained_contract_simulation() -> None:
    corpus = json.loads(_VERSION_GATE_CONFORMANCE_FIXTURE.read_text(encoding="utf-8"))
    contract = corpus["retained_contract_simulation"]
    accepted_versions = contract["accepted_versions"]
    assert (
        len(accepted_versions) > 1
    ), "the simulation must retain more than one contract"
    case_ids = [case["id"] for case in contract["cases"]]
    assert case_ids
    assert len(case_ids) == len(set(case_ids))

    for case in contract["cases"]:
        document = _document_for_case(case)
        accepted = _accepts_with_version_enum(document, accepted_versions)
        assert accepted is case["expected_valid"], case["id"]
