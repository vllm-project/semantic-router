"""Raw value assertions: units belong to each signal, independently of matches."""

from __future__ import annotations

import math
import re
from typing import Any

SIGNAL_VALUE_KEY = re.compile(r"^[a-z][a-z0-9_]*:\S(?:.*\S)?$")
VALUE_RESPONSE_TYPES = {
    "keyword": "keywords",
    "embedding": "embeddings",
    "domain": "domains",
    "preference": "preferences",
}
VALUE_SUFFIXES = {
    "embedding": {"best", "support", "prototype_count"},
    "complexity": {
        "text_hard_score",
        "text_easy_score",
        "text_margin",
        "image_hard_score",
        "image_easy_score",
        "image_margin",
        "margin",
        "score",
    },
}


def finite_number(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def normalize_signal_values(raw: Any, label: str) -> dict[str, dict[str, float]]:
    if not isinstance(raw, dict):
        raise TypeError(f"{label} expected_signal_values must be a mapping")
    result: dict[str, dict[str, float]] = {}
    for key, bounds in raw.items():
        if not isinstance(key, str) or not SIGNAL_VALUE_KEY.fullmatch(key):
            raise ValueError(
                f"{label} expected_signal_values requires a runtime type:name key"
            )
        if not isinstance(bounds, dict) or not bounds or set(bounds) - {"gte", "lte"}:
            raise ValueError(
                f"{label} expected_signal_values.{key} requires gte and/or lte"
            )
        if any(not finite_number(value) for value in bounds.values()):
            raise ValueError(
                f"{label} expected_signal_values.{key} bounds must be finite numbers, not bool"
            )
        if "gte" in bounds and "lte" in bounds and bounds["gte"] > bounds["lte"]:
            raise ValueError(
                f"{label} expected_signal_values.{key} gte must not exceed lte"
            )
        result[key] = dict(bounds)
    return result


def signal_value_reference(
    key: str, configured: dict[str, set[str]]
) -> tuple[str, str]:
    """Resolve only a configured rule, preserving literal colons in rule names."""
    raw_type, name = key.split(":", 1)
    family = VALUE_RESPONSE_TYPES.get(raw_type, raw_type)
    names = configured.get(family, set())
    if name in names:
        return family, name
    rule, separator, suffix = name.rpartition(":")
    if (
        separator
        and rule in names
        and (raw_type == "classifier" or suffix in VALUE_SUFFIXES.get(raw_type, set()))
    ):
        return family, rule
    raise ValueError(f"expected_signal_values references unknown signal rule {key!r}")


def compare_signal_values(
    expected: dict[str, dict[str, float]],
    actual: Any,
    signal_errors: Any,
) -> dict[str, Any]:
    errors: list[str] = []
    observed: dict[str, Any] = {}
    actual = actual if isinstance(actual, dict) else {}
    signal_errors = signal_errors if isinstance(signal_errors, dict) else {}
    for key, bounds in expected.items():
        pieces = key.split(":")
        error_keys = {":".join(pieces[:i]) for i in range(1, len(pieces) + 1)}
        related_errors = [name for name in signal_errors if name in error_keys]
        if pieces[0] == "complexity":
            # Complexity publishes a final metric suffix and errors under a
            # final verdict suffix. Keep any colons inside the rule name.
            base, _, metric = key.rpartition(":")
            if base == "complexity" or metric not in VALUE_SUFFIXES["complexity"]:
                base = key
            related_errors = [
                name
                for name in signal_errors
                if isinstance(name, str)
                and (
                    name in (key, "complexity")
                    or (
                        name.rpartition(":")[0] == base
                        and name.rpartition(":")[2] in {"easy", "medium", "hard"}
                    )
                )
            ]
        if related_errors:
            errors.append(
                f"{key}: signal error ({', '.join(sorted(set(related_errors)))})"
            )
        if key not in actual:
            errors.append(f"{key}: missing signal value")
            continue
        value = actual[key]
        # Invalid values stay identifiable without producing non-JSON numbers
        # in the new diagnostic field. The original raw response is retained.
        observed[key] = value if finite_number(value) else repr(value)
        if not finite_number(value):
            errors.append(f"{key}: expected a finite numeric value, got {value!r}")
            continue
        if "gte" in bounds and value < bounds["gte"]:
            errors.append(f"{key}: {value!r} is below gte {bounds['gte']!r}")
        if "lte" in bounds and value > bounds["lte"]:
            errors.append(f"{key}: {value!r} exceeds lte {bounds['lte']!r}")
    return {"matched": not errors, "errors": errors, "observed": observed}
