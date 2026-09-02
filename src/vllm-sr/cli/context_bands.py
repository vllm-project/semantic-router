"""Token-count parsing and band checks for routing.signals.context rules.

This mirrors the Router's TokenCount and ContextRule.Bounds contract so a
configuration the CLI accepts also loads in the Router, and a configuration
the Router rejects fails early with the same message.
"""

import math

TOKEN_COUNT_MULTIPLIERS = {"K": 1_000, "M": 1_000_000}


def normalize_token_count(value: object, field: str) -> str | None:
    """Return the string form of a YAML token count, or None when omitted.

    YAML authors write ``min_tokens: 8001`` and ``max_tokens: 64K``
    interchangeably, so plain integers and floats are accepted alongside
    strings. Booleans and containers are rejected.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a token count such as 8001 or 64K")
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    raise ValueError(f"{field} must be a token count such as 8001 or 64K")


def token_count_is_set(value: str | None) -> bool:
    """Report whether a token count was configured (non-empty after trimming)."""
    return value is not None and value.strip() != ""


def parse_token_count(value: str | None) -> int:
    """Parse ``"8001"``, ``"64K"`` or ``"1.5M"`` into an integer token count.

    An omitted or empty value parses as 0, matching the Router.
    """
    text = "" if value is None else value.strip().upper()
    if text == "":
        return 0
    multiplier = 1
    if text[-1] in TOKEN_COUNT_MULTIPLIERS:
        multiplier = TOKEN_COUNT_MULTIPLIERS[text[-1]]
        text = text[:-1]
    try:
        number = float(text)
    except ValueError:
        raise ValueError(f"invalid token count format: {value}") from None
    if not math.isfinite(number):
        raise ValueError(f"invalid token count format: {value}")
    if number < 0:
        raise ValueError(f"token count must not be negative: {value}")
    return int(number * multiplier)


def validate_context_band(min_tokens: str | None, max_tokens: str | None) -> None:
    """Reject a band the Router would reject.

    A band needs at least one limit. An omitted min_tokens means 0 and an
    omitted max_tokens makes the band open-ended. Equal limits are an
    exact-match band, so only min_tokens above max_tokens is an error.
    """
    min_set = token_count_is_set(min_tokens)
    max_set = token_count_is_set(max_tokens)
    if not min_set and not max_set:
        raise ValueError("min_tokens or max_tokens must be set")
    try:
        min_value = parse_token_count(min_tokens)
    except ValueError as error:
        raise ValueError(f"min_tokens: {error}") from None
    if not max_set:
        return
    try:
        max_value = parse_token_count(max_tokens)
    except ValueError as error:
        raise ValueError(f"max_tokens: {error}") from None
    if min_value > max_value:
        raise ValueError(
            f"min_tokens ({min_tokens.strip()}) must not exceed max_tokens "
            f"({max_tokens.strip()}); use equal values for an exact match "
            "or omit max_tokens for no upper bound"
        )
