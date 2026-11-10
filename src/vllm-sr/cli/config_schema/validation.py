"""Offline structural validation against the generated Router contract."""

from __future__ import annotations

import copy
import re
from datetime import date, datetime
from typing import Any

from jsonschema import Draft202012Validator, validators

from . import schema_document

_ENV_REFERENCE = re.compile(r"^\$\{[^{}]+\}$")


def _environment_reference(value: Any) -> bool:
    return isinstance(value, str) and _ENV_REFERENCE.fullmatch(value) is not None


_BASE_TYPE_CHECKER = Draft202012Validator.TYPE_CHECKER


def _number_or_environment(_checker, value: Any) -> bool:
    return _BASE_TYPE_CHECKER.is_type(value, "number") or _environment_reference(value)


def _integer_or_environment(_checker, value: Any) -> bool:
    return _BASE_TYPE_CHECKER.is_type(value, "integer") or _environment_reference(value)


def _boolean_or_environment(_checker, value: Any) -> bool:
    return _BASE_TYPE_CHECKER.is_type(value, "boolean") or _environment_reference(value)


_CONFIG_TYPE_CHECKER = _BASE_TYPE_CHECKER.redefine_many(
    {
        "number": _number_or_environment,
        "integer": _integer_or_environment,
        "boolean": _boolean_or_environment,
    }
)
ConfigSchemaValidator = validators.extend(
    Draft202012Validator,
    type_checker=_CONFIG_TYPE_CHECKER,
)


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
    validator = ConfigSchemaValidator(schema_document())
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
