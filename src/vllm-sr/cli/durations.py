"""Router config durations: Go syntax in, Envoy's protobuf syntax out."""

import re
from decimal import Decimal

_UNIT_SECONDS = {
    "ns": Decimal("1e-9"),
    "us": Decimal("1e-6"),
    "µs": Decimal("1e-6"),
    "ms": Decimal("1e-3"),
    "s": Decimal(1),
    "m": Decimal(60),
    "h": Decimal(3600),
}
_PART = re.compile(r"(\d+(?:\.\d*)?|\.\d+)(ns|us|µs|ms|s|m|h)")


def parse_duration(value: str) -> Decimal:
    """Parse a Go duration such as ``30s``, ``250ms`` or ``1m30s`` into seconds.

    The Router reads these fields with Go's time.ParseDuration, so the CLI
    accepts exactly that grammar (a bare ``0`` included).
    """

    text = value.strip()
    if text == "0":
        return Decimal(0)
    position, total = 0, Decimal(0)
    while position < len(text):
        match = _PART.match(text, position)
        if match is None:
            raise ValueError(
                f"invalid duration {value!r}; use a value such as 30s or 250ms"
            )
        total += Decimal(match.group(1)) * _UNIT_SECONDS[match.group(2)]
        position = match.end()
    if position == 0:
        raise ValueError(
            f"invalid duration {value!r}; use a value such as 30s or 250ms"
        )
    return total


def envoy_duration(value: str) -> str:
    """Render a Go duration in the seconds form Envoy's config requires."""

    seconds = parse_duration(value)
    rendered = format(seconds.normalize(), "f")
    return f"{rendered}s"
