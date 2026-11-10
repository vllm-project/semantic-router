"""Offline structural validation against the generated Router contract."""

from __future__ import annotations

import copy
import re
from datetime import date, datetime
from typing import Any

from jsonschema import Draft202012Validator, validators

from cli.context_bands import references_environment

from . import schema_document

_CONST = Draft202012Validator.VALIDATORS["const"]


def _const(validator, const, instance, schema):
    # The Router expands an environment reference to text before it decodes a
    # field, so a reference can stand for a text constant such as the "auto"
    # of a number-or-auto field. A number or boolean field takes no text.
    if isinstance(const, str) and isinstance(instance, str):
        if references_environment(instance):
            return
    yield from _CONST(validator, const, instance, schema)


_Validator = validators.extend(Draft202012Validator, {"const": _const})


def _normalise_scalar(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _normalise_scalar(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalise_scalar(item) for item in value]
    return value


def validate_config_structure(data: dict[str, Any]) -> list[str]:
    """Return stable JSON-Schema errors without applying Router semantics."""

    document = _normalise_scalar(copy.deepcopy(data))
    validator = _Validator(schema_document())
    errors = sorted(
        validator.iter_errors(document),
        key=lambda error: (
            tuple(str(part) for part in error.absolute_path),
            error.message,
        ),
    )
    rendered: list[str] = []
    for error in errors:
        path = ".".join(str(part) for part in error.absolute_path) or "config"
        if error.validator == "additionalProperties":
            unexpected = re.findall(r"'([^']+)'", error.message)
            if unexpected:
                rendered.extend(
                    f"{path}.{field}: {error.message}" for field in unexpected
                )
                continue
        rendered.append(f"{path}: {error.message}")
    return rendered
