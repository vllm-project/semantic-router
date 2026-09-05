"""Token-count parsing and band checks for routing.signals.context rules.

This mirrors the Router's TokenCount and ContextRule.Bounds contract so a
configuration the CLI accepts also loads in the Router, and a configuration
the Router rejects fails early with the same message.
"""

import math
import re

TOKEN_COUNT_MULTIPLIERS = {"K": 1_000, "M": 1_000_000}

# The characters Go's strings.TrimSpace removes (unicode.IsSpace). Python's
# str.strip() also removes the U+001C..U+001F information separators, which
# the Router does not, so trimming must use exactly this set.
GO_WHITESPACE = (
    " \t\n\v\f\r\x85\xa0\u1680"
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u2028\u2029\u202f\u205f\u3000"
)

# The Router parses the number with Go's strconv.ParseFloat, whose grammar is
# not Python's float(): only ASCII digits, decimal floats with an optional
# e-exponent, hexadecimal floats with a mandatory p-exponent, and underscores
# between digits as in Go literals. These patterns and _go_underscores_ok
# reproduce that grammar so the CLI accepts and rejects the same strings.
# Non-finite specials (inf, nan) are left to fall through as invalid because
# the Router rejects them after parsing anyway.
_GO_DECIMAL_FLOAT = re.compile(
    r"^[+-]?(?=[0-9_.]*[0-9])[0-9_]*\.?[0-9_]*(?:[eE][+-]?[0-9][0-9_]*)?$", re.ASCII
)
_GO_HEX_FLOAT = re.compile(
    r"^[+-]?0[xX](?=[0-9a-fA-F_.]*[0-9a-fA-F])[0-9a-fA-F_]*\.?[0-9a-fA-F_]*"
    r"[pP][+-]?[0-9][0-9_]*$",
    re.ASCII,
)

# The Router rejects a scaled count at or above math.MaxInt, which is 2**63 - 1
# on the 64-bit platforms it ships for. Compared as a float, as the Router does,
# so "9223372036854775807" rounds up to 2**63 and is rejected on both sides.
ROUTER_MAX_TOKEN_COUNT = float(2**63 - 1)


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


def trim_space(value: str) -> str:
    """Trim leading and trailing whitespace exactly as Go's strings.TrimSpace."""
    return value.strip(GO_WHITESPACE)


def token_count_is_set(value: str | None) -> bool:
    """Report whether a token count was configured (non-empty after trimming)."""
    return value is not None and trim_space(value) != ""


def _go_underscores_ok(text: str) -> bool:
    """Port of Go's strconv underscoreOK: an underscore must sit between digits.

    A base prefix such as ``0x`` counts as a digit on its left side.
    """
    if text[:1] in ("+", "-"):
        text = text[1:]
    start = 0
    is_hex = False
    if text[:1] == "0" and text[1:2].lower() in ("b", "o", "x"):
        start = 2
        is_hex = text[1].lower() == "x"
        saw = "0"
    else:
        saw = "^"
    for char in text[start:]:
        if "0" <= char <= "9" or (is_hex and char.lower() in "abcdef"):
            saw = "0"
        elif char == "_":
            if saw != "0":
                return False
            saw = "_"
        elif saw == "_":
            return False
        else:
            saw = "!"
    return saw != "_"


def parse_go_float(text: str) -> float:
    """Parse ``text`` with the grammar of Go's ``strconv.ParseFloat``.

    Raises ValueError for anything Go rejects as a syntax error. Values Go
    reports as out of range come back as infinity, so callers reject them
    with the same finiteness check the Router applies.
    """
    if _GO_DECIMAL_FLOAT.match(text):
        is_hex = False
    elif _GO_HEX_FLOAT.match(text):
        is_hex = True
    else:
        raise ValueError(f"not a Go float literal: {text}")
    if "_" in text and not _go_underscores_ok(text):
        raise ValueError(f"misplaced underscore in Go float literal: {text}")
    digits = text.replace("_", "")
    if not is_hex:
        return float(digits)
    try:
        return float.fromhex(digits)
    except OverflowError:
        return math.inf if not digits.startswith("-") else -math.inf


def parse_token_count(value: str | None) -> int:
    """Parse ``"8001"``, ``"64K"`` or ``"1.5M"`` into an integer token count.

    An omitted or empty value parses as 0, matching the Router. The number
    is read with Go's float grammar (see parse_go_float) so that every string
    the Router loads is accepted here and every string it rejects fails here.
    """
    text = "" if value is None else trim_space(value).upper()
    if text == "":
        return 0
    multiplier = 1
    if text[-1] in TOKEN_COUNT_MULTIPLIERS:
        multiplier = TOKEN_COUNT_MULTIPLIERS[text[-1]]
        text = text[:-1]
    try:
        number = parse_go_float(text)
    except ValueError:
        raise ValueError(f"invalid token count format: {value}") from None
    if not math.isfinite(number):
        raise ValueError(f"invalid token count format: {value}")
    if number < 0:
        raise ValueError(f"token count must not be negative: {value}")
    scaled = number * multiplier
    if scaled >= ROUTER_MAX_TOKEN_COUNT:
        raise ValueError(f"token count is too large: {value}")
    return int(scaled)


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
            f"min_tokens ({trim_space(min_tokens)}) must not exceed max_tokens "
            f"({trim_space(max_tokens)}); use equal values for an exact match "
            "or omit max_tokens for no upper bound"
        )
