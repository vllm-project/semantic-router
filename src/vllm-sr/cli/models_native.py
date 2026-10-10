"""Lossless CLI projections of native contracts owned by the Router schema.

The CLI forwards these objects without interpreting their execution payloads.
Validate their structure from generated Go definitions rather than maintaining
another copy of question, stage, budget and calibration field inventories.
"""

from functools import cache, partial
from typing import Annotated, Any

from pydantic import AfterValidator

from cli.config_schema import schema_document
from cli.config_schema.validation import ConfigSchemaValidator


@cache
def _definition_validator(name: str) -> ConfigSchemaValidator:
    document = schema_document()
    return ConfigSchemaValidator(
        {"$ref": f"#/$defs/{name}", "$defs": document["$defs"]}
    )


def _validate_definition(name: str, value: dict[str, Any]) -> dict[str, Any]:
    errors = sorted(
        _definition_validator(name).iter_errors(value),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        path = ".".join(str(part) for part in error.absolute_path)
        raise ValueError(f"{name}{'.' + path if path else ''}: {error.message}")
    return value


NativeQuality = Annotated[
    dict[str, Any], AfterValidator(partial(_validate_definition, "NativeQualityConfig"))
]
NativeStage = Annotated[
    dict[str, Any], AfterValidator(partial(_validate_definition, "CascadeStage"))
]
AlgorithmBudget = Annotated[
    dict[str, Any], AfterValidator(partial(_validate_definition, "AlgorithmBudget"))
]
CalibrationArtifact = Annotated[
    dict[str, Any], AfterValidator(partial(_validate_definition, "CalibrationArtifact"))
]
