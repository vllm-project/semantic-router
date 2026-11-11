"""Read the Router's x-vsr-trace and x-vsr-warnings response headers.

x-vsr-trace is an RFC 8941 dictionary of routing details, which the Router
sends only to a caller ``global.router.headers.debug`` allows; x-vsr-warnings
is an RFC 8941 list of warning codes. Strings percent-encode "%" and every
byte outside printable ASCII.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import unquote

TRACE_HEADER = "x-vsr-trace"
WARNINGS_HEADER = "x-vsr-warnings"


@dataclass
class StructuredMember:
    """One member: its value (strings and tokens unquoted, numbers as
    written), the items of an inner list, and its parameters."""

    value: str = ""
    items: list[str] = field(default_factory=list)
    params: dict[str, str] = field(default_factory=dict)


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.at = 0

    def done(self) -> bool:
        return self.at >= len(self.text)

    def consume(self, char: str) -> bool:
        if self.text[self.at : self.at + 1] != char:
            return False
        self.at += 1
        return True

    def skip_spaces(self) -> None:
        while self.text[self.at : self.at + 1] in (" ", "\t"):
            self.at += 1

    def key(self) -> str:
        start = self.at
        while not self.done() and (
            self.text[self.at].islower()
            or self.text[self.at].isdigit()
            or self.text[self.at] in "_.*-"
        ):
            self.at += 1
        return self.text[start : self.at]

    def bare_item(self) -> str:
        if not self.consume('"'):
            start = self.at
            while not self.done() and self.text[self.at] not in " ,;()":
                self.at += 1
            return self.text[start : self.at]
        raw: list[str] = []
        while not self.done():
            char = self.text[self.at]
            self.at += 1
            if char == "\\" and not self.done():
                raw.append(self.text[self.at])
                self.at += 1
            elif char == '"':
                break
            else:
                raw.append(char)
        return unquote("".join(raw))

    def params(self) -> dict[str, str]:
        params: dict[str, str] = {}
        while self.consume(";"):
            self.skip_spaces()
            name = self.key()
            params[name] = self.bare_item() if self.consume("=") else "?1"
        return params

    def member(self) -> StructuredMember:
        member = StructuredMember()
        if self.consume("("):
            self.skip_spaces()
            while not self.done() and not self.consume(")"):
                member.items.append(self.bare_item())
                self.params()
                self.skip_spaces()
        else:
            member.value = self.bare_item()
        member.params = self.params()
        return member

    def next(self) -> bool:
        self.skip_spaces()
        if not self.consume(","):
            return False
        self.skip_spaces()
        return True


def parse_trace(value: str | None) -> dict[str, StructuredMember]:
    """Parse x-vsr-trace; a malformed value parses as far as it is well formed."""

    parser = _Parser(value or "")
    members: dict[str, StructuredMember] = {}
    parser.skip_spaces()
    while not parser.done():
        name = parser.key()
        if not name:
            break
        if parser.consume("="):
            members[name] = parser.member()
        else:
            members[name] = StructuredMember(value="?1", params=parser.params())
        if not parser.next():
            break
    return members


def warning_codes(value: str | None) -> list[str]:
    """Return the codes x-vsr-warnings lists, in order."""

    parser = _Parser(value or "")
    codes: list[str] = []
    parser.skip_spaces()
    while not parser.done():
        codes.append(parser.member().value)
        if not parser.next():
            break
    return codes


def trace_value(headers, name: str) -> str:
    """Return one trace member's value, "" when the response has none."""

    member = parse_trace(headers.get(TRACE_HEADER)).get(name)
    return member.value if member else ""
