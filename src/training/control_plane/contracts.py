"""Validate worker messages against the generated Go contract, without duplicate models."""

import json
from pathlib import Path

from jsonschema import Draft202012Validator

CONTRACT_ROOT = (
    Path(__file__).resolve().parents[2] / "semantic-router/pkg/trainingcontract"
)
SCHEMA = json.loads((CONTRACT_ROOT / "training-v1.schema.json").read_text())


def validate(message: dict, definition: str) -> None:
    """Check message structure against the versioned wire schema.

    Resource relationships and run state transitions are owned by Go management;
    classifier bundle semantics remain in the existing provenance validator.
    """
    schema = {**SCHEMA, "$ref": f"#/$defs/{definition}"}
    Draft202012Validator(schema).validate(message)
