#!/usr/bin/env python3
"""Detect reusable username/password pairs in public source documentation."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

from public_docs_paths import iter_public_document_paths

PAIR_LINE_WINDOW = 12
MIN_TABLE_COLUMNS = 2
MAX_CREDENTIAL_VALUE_LENGTH = 1024
MIN_SERVICE_PASSWORD_LENGTH = 15

_FIELD_ASSIGNMENT_RE = re.compile(
    r"(?i)(?<![\w-])"
    r"(?:(?:[a-z0-9]+)[._-]+)*"
    r"(?P<label>user[\s_-]*name|email(?:[\s_-]+address)?|login|password|pass)"
    r"(?![\w-])\s*[:=]"
)
_CLOSING_LABEL_MARKUP_RE = re.compile(
    r"(?i)(user[\s_-]*name|email(?:[\s_-]+address)?|login|password|pass)"
    r"[`*_~]{1,3}(?=\s*[:=])"
)
_HTML_TAG_RE = re.compile(r"<[^>]*>", re.DOTALL)
_HTML_TABLE_RE = re.compile(r"(?is)<table\b[^>]*>(.*?)</table\s*>")
_HTML_ROW_RE = re.compile(r"(?is)<tr\b[^>]*>(.*?)</tr\s*>")
_HTML_CELL_RE = re.compile(r"(?is)<t[dh]\b[^>]*>(.*?)</t[dh]\s*>")
_HTML_DEFINITION_LIST_RE = re.compile(r"(?is)<dl\b[^>]*>(.*?)</dl\s*>")
_HTML_DEFINITION_RE = re.compile(
    r"(?is)<dt\b[^>]*>(.*?)</dt\s*>\s*<dd\b[^>]*>(.*?)</dd\s*>"
)
_MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
_MARKDOWN_TABLE_SEPARATOR_RE = re.compile(r"^:?-{3,}:?$")
_MASKED_VALUE_RE = re.compile(r"^(?:\*{3,}|x{3,}|\.{3,}|-{3,})$", re.IGNORECASE)
_ENV_PLACEHOLDER_RE = re.compile(
    r"^(?:\$\{[^}]+\}|\{\{[^}]+\}\}|\$[A-Z_][A-Z0-9_]*|%[A-Z_][A-Z0-9_]*%)$"
)
_INLINE_PLACEHOLDER_RE = re.compile(
    r"(?:\$\{[^}\n]+\}|\{\{[^}\n]+\}\}|%[A-Z_][A-Z0-9_]*%)"
)
_ANGLE_OR_BRACKET_PLACEHOLDER_RE = re.compile(r"^(?:<[^>]+>|\[[^]]+])$")
_PLACEHOLDER_VALUES = frozenset(
    {
        "change-me",
        "change_me",
        "dummy",
        "example",
        "fake",
        "placeholder",
        "replace-me",
        "replace_me",
        "sample",
    }
)
_GUIDANCE_TOKENS = frozenset(
    {
        "a",
        "at",
        "avoid",
        "choose",
        "configure",
        "do",
        "enter",
        "generated",
        "identify",
        "keep",
        "must",
        "never",
        "optional",
        "provided",
        "required",
        "set",
        "should",
        "store",
        "str",
        "string",
        "the",
        "use",
    }
)


@dataclass(frozen=True)
class CredentialDisclosure:
    """Location-only finding; credential values must never enter logs."""

    path: Path
    identity_line: int
    password_line: int


@dataclass(frozen=True)
class _CredentialField:
    role: str
    line: int
    value: str
    group: str


def scan_public_documentation(repo_root: Path) -> list[CredentialDisclosure]:
    """Scan every public source document below ``repo_root`` for credential pairs."""
    findings: list[CredentialDisclosure] = []
    for path in iter_public_document_paths(repo_root):
        text = path.read_text(encoding="utf-8", errors="replace")
        for identity_line, password_line in find_credential_pairs(text):
            findings.append(
                CredentialDisclosure(
                    path=path.relative_to(repo_root),
                    identity_line=identity_line,
                    password_line=password_line,
                )
            )
    return findings


def find_credential_pairs(text: str) -> list[tuple[int, int]]:
    """Return line locations for reusable identity/password pairs in one document."""
    fields = [
        *_assignment_fields(text),
        *_markdown_table_fields(text),
        *_html_table_fields(text),
        *_html_definition_fields(text),
    ]
    fields = list(dict.fromkeys(fields))

    # Pair unique line locations within each structural group. Credential
    # values never belong in findings, and collapsing repeated fields on one
    # line prevents attacker-controlled documents from forcing an
    # identity-by-password Cartesian product.
    grouped_lines: dict[str, dict[str, set[int]]] = {}
    for field in fields:
        grouped_lines.setdefault(field.group, {"identity": set(), "password": set()})[
            field.role
        ].add(field.line)

    pairs: set[tuple[int, int]] = set()
    lines = text.splitlines()
    for roles in grouped_lines.values():
        identity_lines = sorted(roles["identity"])
        password_lines = sorted(roles["password"])
        left = 0
        right = 0
        for identity_line in identity_lines:
            minimum = identity_line - PAIR_LINE_WINDOW
            maximum = identity_line + PAIR_LINE_WINDOW
            while left < len(password_lines) and password_lines[left] < minimum:
                left += 1
            right = max(right, left)
            while right < len(password_lines) and password_lines[right] <= maximum:
                right += 1
            for password_line in password_lines[left:right]:
                if not _heading_between(lines, identity_line, password_line):
                    pairs.add((identity_line, password_line))
    return sorted(pairs)


def _assignment_fields(text: str) -> list[_CredentialField]:
    fields: list[_CredentialField] = []
    section_line = 0
    for line_number, source_line in enumerate(text.splitlines(), start=1):
        if _MARKDOWN_HEADING_RE.match(source_line):
            section_line = line_number
        visible_line = _visible_text(source_line)
        visible_line = _CLOSING_LABEL_MARKUP_RE.sub(r"\1", visible_line)
        placeholder_spans = [
            match.span() for match in _INLINE_PLACEHOLDER_RE.finditer(visible_line)
        ]
        matches = []
        span_index = 0
        for match in _FIELD_ASSIGNMENT_RE.finditer(visible_line):
            while (
                span_index < len(placeholder_spans)
                and placeholder_spans[span_index][1] <= match.start()
            ):
                span_index += 1
            if span_index < len(placeholder_spans):
                start, end = placeholder_spans[span_index]
                if start <= match.start() < end:
                    continue
            matches.append(match)
        for index, match in enumerate(matches):
            value_end = matches[index + 1].start() if index + 1 < len(matches) else None
            value = _credential_value(visible_line[match.end() : value_end])
            if not _is_usable_value(value):
                continue
            fields.append(
                _CredentialField(
                    role=_field_role(match.group("label")),
                    line=line_number,
                    value=value,
                    group=f"section:{section_line}",
                )
            )
    return fields


def _markdown_table_fields(text: str) -> list[_CredentialField]:
    lines = text.splitlines()
    fields: list[_CredentialField] = []
    index = 0
    while index < len(lines):
        if not _looks_like_markdown_table_row(lines[index]):
            index += 1
            continue
        start = index
        rows: list[tuple[int, list[str]]] = []
        while index < len(lines) and _looks_like_markdown_table_row(lines[index]):
            rows.append((index + 1, _markdown_cells(lines[index])))
            index += 1
        fields.extend(_fields_from_table_rows(rows, f"markdown-table:{start + 1}"))
    return fields


def _html_table_fields(text: str) -> list[_CredentialField]:
    fields: list[_CredentialField] = []
    for table_index, table_match in enumerate(_HTML_TABLE_RE.finditer(text), start=1):
        table_start_line = text.count("\n", 0, table_match.start()) + 1
        rows: list[tuple[int, list[str]]] = []
        table_body = table_match.group(1)
        for row_match in _HTML_ROW_RE.finditer(table_body):
            row_line = table_start_line + table_body.count("\n", 0, row_match.start())
            cells = [
                _visible_text(cell_match.group(1))
                for cell_match in _HTML_CELL_RE.finditer(row_match.group(1))
            ]
            if cells:
                rows.append((row_line, cells))
        fields.extend(_fields_from_table_rows(rows, f"html-table:{table_index}"))
    return fields


def _html_definition_fields(text: str) -> list[_CredentialField]:
    fields: list[_CredentialField] = []
    for list_index, list_match in enumerate(
        _HTML_DEFINITION_LIST_RE.finditer(text), start=1
    ):
        list_start_line = text.count("\n", 0, list_match.start()) + 1
        list_body = list_match.group(1)
        for definition in _HTML_DEFINITION_RE.finditer(list_body):
            role = _field_role(_visible_text(definition.group(1)))
            value = _credential_value(_visible_text(definition.group(2)))
            if role and _is_usable_value(value):
                fields.append(
                    _CredentialField(
                        role=role,
                        line=list_start_line
                        + list_body.count("\n", 0, definition.start()),
                        value=value,
                        group=f"html-definition-list:{list_index}",
                    )
                )
    return fields


def _fields_from_table_rows(
    rows: list[tuple[int, list[str]]], group: str
) -> list[_CredentialField]:
    fields: list[_CredentialField] = []

    # Key/value layouts: | Login | account | followed by | Pass | value |.
    for line, cells in rows:
        if len(cells) < MIN_TABLE_COLUMNS or _is_markdown_separator_row(cells):
            continue
        role = _field_role(cells[0])
        value = _credential_value(cells[1])
        if role and _is_usable_value(value):
            fields.append(_CredentialField(role, line, value, group))

    # Column layouts: | Login | Password | followed by one or more data rows.
    for header_index, (_header_line, header_cells) in enumerate(rows):
        roles = [_field_role(cell) for cell in header_cells]
        if "identity" not in roles or "password" not in roles:
            continue
        identity_index = roles.index("identity")
        password_index = roles.index("password")
        required_columns = max(identity_index, password_index) + 1
        for data_line, data_cells in rows[header_index + 1 :]:
            if len(data_cells) < required_columns or _is_markdown_separator_row(
                data_cells
            ):
                continue
            identity_value = _credential_value(data_cells[identity_index])
            password_value = _credential_value(data_cells[password_index])
            if _is_usable_value(identity_value) and _is_usable_value(password_value):
                fields.extend(
                    (
                        _CredentialField("identity", data_line, identity_value, group),
                        _CredentialField("password", data_line, password_value, group),
                    )
                )
        # One credential header is enough per table; later rows are data.
        break
    return fields


def _field_role(label: str) -> str:
    normalized = re.sub(r"[^a-z]", "", _visible_text(label).casefold())
    if normalized in {"username", "email", "emailaddress", "login"}:
        return "identity"
    if normalized in {"password", "pass"}:
        return "password"
    return ""


def _credential_value(fragment: str) -> str:
    value = html.unescape(fragment).strip()
    if not value:
        return ""

    if value[0] in {'"', "'", "`"}:
        quote = value[0]
        closing = value.find(quote, 1)
        value = value[1:closing] if closing > 0 else value[1:]
    else:
        value = re.split(r"[\s,;|]+", value, maxsplit=1)[0]
    for marker in ("***", "___", "**", "__", "~~", "*", "_"):
        if (
            value.startswith(marker)
            and value.endswith(marker)
            and len(value) > 2 * len(marker)
            and any(
                character not in marker
                for character in value[len(marker) : -len(marker)]
            )
        ):
            value = value[len(marker) : -len(marker)]
            break
    # Preserve punctuation and braces: they are valid password characters, and
    # stripping them can turn a service-accepted value into an empty mask or a
    # different placeholder token.
    return value.strip()


def _is_usable_value(value: str) -> bool:
    if not value or len(value) > MAX_CREDENTIAL_VALUE_LENGTH:
        return False
    normalized = value.casefold()
    if normalized in {
        "false",
        "masked",
        "n/a",
        "nil",
        "none",
        "null",
        "omitted",
        "redacted",
        "true",
        "unset",
    }:
        return False
    if normalized in _GUIDANCE_TOKENS:
        return False
    if _MASKED_VALUE_RE.fullmatch(value) and len(value) < MIN_SERVICE_PASSWORD_LENGTH:
        return False
    if _ENV_PLACEHOLDER_RE.fullmatch(value):
        return False
    if _ANGLE_OR_BRACKET_PLACEHOLDER_RE.fullmatch(value):
        return False
    return normalized not in _PLACEHOLDER_VALUES


def _visible_text(fragment: str) -> str:
    fragment = re.sub(r"(?is)<br\s*/?>", " ", fragment)
    fragment = _HTML_TAG_RE.sub(" ", fragment)
    fragment = html.unescape(fragment)
    return fragment.strip()


def _looks_like_markdown_table_row(line: str) -> bool:
    return "|" in line and len(_markdown_cells(line)) >= MIN_TABLE_COLUMNS


def _markdown_cells(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    cells = re.split(r"(?<!\\)\|", stripped)
    return [_visible_text(cell.replace(r"\|", "|")) for cell in cells]


def _is_markdown_separator_row(cells: list[str]) -> bool:
    return bool(cells) and all(
        _MARKDOWN_TABLE_SEPARATOR_RE.fullmatch(cell.strip()) for cell in cells
    )


def _heading_between(lines: list[str], first_line: int, second_line: int) -> bool:
    start, end = sorted((first_line, second_line))
    return any(_MARKDOWN_HEADING_RE.match(line) for line in lines[start:end])
