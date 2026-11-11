"""Read the Router's x-vsr-trace response header for the benchmarks.

x-vsr-trace is an RFC 8941 dictionary of routing details that the Router
sends only to a caller its global.router.headers.debug allows. Strings
percent-encode "%" and every byte outside printable ASCII.
"""

from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import unquote

TRACE_HEADER = "x-vsr-trace"


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

    def item(self) -> str:
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

    def skip_params(self) -> None:
        while self.consume(";"):
            self.skip_spaces()
            self.key()
            if self.consume("="):
                self.item()

    def value(self) -> str:
        if not self.consume("("):
            return self.item()
        items: list[str] = []
        self.skip_spaces()
        while not self.done() and not self.consume(")"):
            items.append(self.item())
            self.skip_params()
            self.skip_spaces()
        return ",".join(items)


def trace_members(value: str | None) -> dict[str, str]:
    """Each member's value: strings unquoted, an inner list's items
    comma-separated, a member without a value "?1". Parameters are left out."""

    parser = _Parser(value or "")
    members: dict[str, str] = {}
    parser.skip_spaces()
    while not parser.done():
        name = parser.key()
        if not name:
            break
        members[name] = parser.value() if parser.consume("=") else "?1"
        parser.skip_params()
        parser.skip_spaces()
        if not parser.consume(","):
            break
        parser.skip_spaces()
    return members


def warning_codes(value: str | None) -> list[str]:
    """The codes of an x-vsr-warnings list, each warning's parameters left out."""

    parser = _Parser(value or "")
    codes: list[str] = []
    parser.skip_spaces()
    while not parser.done():
        code = parser.item()
        if not code:
            break
        codes.append(code)
        parser.skip_params()
        parser.skip_spaces()
        if not parser.consume(","):
            break
        parser.skip_spaces()
    return codes


def routing_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """The response headers, lowercased, with the trace's members beside them
    and its matched signals as matched_<type>, comma-separated."""

    fields = {key.lower(): value for key, value in headers.items()}
    trace = trace_members(fields.get(TRACE_HEADER))
    signals = trace.pop("signals", "")
    fields.update(trace)
    for signal in filter(None, signals.split(",")):
        signal_type, _, name = signal.partition(":")
        key = f"matched_{signal_type}"
        fields[key] = f"{fields[key]},{name}" if fields.get(key) else name
    return fields
