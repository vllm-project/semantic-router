"""Plain-scalar typing of gopkg.in/yaml.v2, as the Router applies it to config.

The Router loads config.yaml in three steps (pkg/config/loader.go):
``yaml.Unmarshal`` into ``map[string]interface{}``, ``yaml.Marshal`` of that
map, and ``yaml.Unmarshal`` of the result into the typed config. A plain
scalar is therefore typed by yaml.v2's ``resolve`` and re-emitted before a
string field such as ``TokenCount`` receives it: ``0123`` arrives as ``83``,
``0x10`` as ``16``, ``1e3`` as ``1000`` and ``yes`` as ``true``, while ``1:30``
stays ``1:30`` because yaml.v2 has no base-60 numbers.

``router_scalar_text`` reproduces that pipeline for one plain, untagged scalar
so the CLI can validate exactly the text the Router parses. Quoted scalars and
explicit tags are not typed by yaml.v2 and never pass through here.
"""

from __future__ import annotations

import math
import re
from decimal import Decimal

from cli.context_bands import parse_go_float

# resolve.go: values yaml.v2 looks up before trying numbers, mapped to the
# text they are re-emitted as. None marks null, which the typed decoder turns
# into an empty TokenCount.
_RESOLVE_MAP: dict[str, str | None] = {
    **dict.fromkeys(("y", "Y", "yes", "Yes", "YES", "true", "True", "TRUE"), "true"),
    **dict.fromkeys(("on", "On", "ON"), "true"),
    **dict.fromkeys(("n", "N", "no", "No", "NO", "false", "False", "FALSE"), "false"),
    **dict.fromkeys(("off", "Off", "OFF"), "false"),
    **dict.fromkeys(("", "~", "null", "Null", "NULL"), None),
    **dict.fromkeys((".nan", ".NaN", ".NAN"), ".nan"),
    **dict.fromkeys((".inf", ".Inf", ".INF", "+.inf", "+.Inf", "+.INF"), ".inf"),
    **dict.fromkeys(("-.inf", "-.Inf", "-.INF"), "-.inf"),
}

# resolve.go: the first byte decides which conversions are attempted.
_SIGN_OR_DIGIT = frozenset("+-0123456789")
_MAP_ONLY = frozenset("yYnNtTfFoO~")

# resolve.go yamlStyleFloat, applied after every underscore is removed.
_YAML_STYLE_FLOAT = re.compile(r"^[-+]?(\.[0-9]+|[0-9]+(\.[0-9]*)?)([eE][-+]?[0-9]+)?$")

# strconv.ParseInt with base 0: an optional sign, then a 0b/0o/0x prefix, a
# leading-zero octal, or a decimal. Underscores are already gone.
_GO_BASE0_INT = re.compile(
    r"^(?P<sign>[+-]?)(?:0[bB](?P<bin>[01]+)|0[oO](?P<oct>[0-7]+)"
    r"|0[xX](?P<hex>[0-9a-fA-F]+)|0(?P<oct0>[0-7]*)|(?P<dec>[1-9][0-9]*))$"
)

_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1
_UINT64_MAX = 2**64 - 1

# strconv.FormatFloat with the 'g' format and shortest precision switches to
# %e below this decimal exponent or from this one upward (ftoa.go).
_GO_G_EXPONENT_LOW = -4
_GO_G_EXPONENT_HIGH = 6


def router_scalar_text(scalar: str) -> str | None:
    """Return the text a Router string field receives for a plain scalar.

    ``None`` means yaml.v2 resolved the scalar to null, which the typed
    decoder turns into an empty string.
    """
    first = scalar[:1]
    if scalar == "" or first in _MAP_ONLY or first in _SIGN_OR_DIGIT or first == ".":
        if scalar in _RESOLVE_MAP:
            return _RESOLVE_MAP[scalar]
        if first == ".":
            number = _finite_go_float(scalar)
            return scalar if number is None else format_go_float(number)
        if first in _SIGN_OR_DIGIT:
            return _resolve_number(scalar)
    return scalar


def _resolve_number(scalar: str) -> str:
    """The 'D'/'S' branch of resolve.go: int, uint, float, else string.

    A timestamp-looking value (four digits and a dash) is kept as a string by
    yaml.v2 when decoding into ``interface{}``, and it is never a number, so
    it needs no special case here.
    """
    plain = scalar.replace("_", "")
    integer = _go_base0_int(plain)
    if integer is not None and (
        _INT64_MIN <= integer <= _INT64_MAX
        or (not plain.startswith(("+", "-")) and 0 <= integer <= _UINT64_MAX)
    ):
        return str(integer)
    if _YAML_STYLE_FLOAT.match(plain):
        number = _finite_go_float(plain)
        if number is not None:
            return format_go_float(number)
    return scalar


def _go_base0_int(text: str) -> int | None:
    match = _GO_BASE0_INT.match(text)
    if match is None:
        return None
    sign = -1 if match.group("sign") == "-" else 1
    if match.group("bin") is not None:
        return sign * int(match.group("bin"), 2)
    if match.group("oct") is not None:
        return sign * int(match.group("oct"), 8)
    if match.group("hex") is not None:
        return sign * int(match.group("hex"), 16)
    if match.group("oct0") is not None:
        return sign * int(match.group("oct0") or "0", 8)
    return sign * int(match.group("dec"))


def _finite_go_float(text: str) -> float | None:
    """strconv.ParseFloat succeeding: Go syntax and no overflow."""
    try:
        number = parse_go_float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def format_go_float(value: float) -> str:
    """strconv.FormatFloat(value, 'g', -1, 64), as yaml.v2 emits a float.

    The shortest digits that round-trip are laid out as %e when the decimal
    exponent is below -4 or at least 6, and as %f otherwise, so 1e6 becomes
    ``1e+06`` and 123456.0 becomes ``123456``.
    """
    if math.isinf(value):
        return ".inf" if value > 0 else "-.inf"
    if math.isnan(value):
        return ".nan"
    sign = "-" if math.copysign(1.0, value) < 0 else ""
    _, digit_tuple, exponent = Decimal(repr(abs(value))).as_tuple()
    digits = "".join(map(str, digit_tuple)).rstrip("0")
    exponent += len(digit_tuple) - len(digits)
    if not digits:
        return sign + "0"
    point = len(digits) + exponent
    decimal_exponent = point - 1
    if decimal_exponent < _GO_G_EXPONENT_LOW or decimal_exponent >= _GO_G_EXPONENT_HIGH:
        mantissa = digits[0] + ("." + digits[1:] if len(digits) > 1 else "")
        return f"{sign}{mantissa}e{'-' if decimal_exponent < 0 else '+'}{abs(decimal_exponent):02d}"
    if point <= 0:
        return f"{sign}0.{'0' * -point}{digits}"
    if point >= len(digits):
        return sign + digits + "0" * (point - len(digits))
    return f"{sign}{digits[:point]}.{digits[point:]}"
