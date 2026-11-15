"""Set and read a few entries of a TOML file, and leave the rest as it is written.

An entry is a top-level key, held as its line (`model = "x"`), or a table with
the tables under it (a dotted name such as `model_providers.vllm-sr`), held as
their text from each header to its last line before the next one, blank lines
and comments left out. Every edit is read back with tomllib and refused when it
changed anything but the entries it set, so a file this cannot edit cleanly is
left alone.
"""

from __future__ import annotations

import re
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = None

HEADER = re.compile(r"^\s*\[\[?\s*(.+?)\s*\]\]?\s*(?:#.*)?$")


def _spans(lines: list[str]) -> list[tuple[str | None, int, int]]:
    """Each part of the file: the top (None) and each table, as its name, its
    first line, and the line after its last key."""
    starts = [(None, 0)]
    for index, line in enumerate(lines):
        found = HEADER.match(line)
        if found:
            name = re.sub(r"\s*\.\s*", ".", found.group(1)).replace('"', "")
            starts.append((name, index))
    spans = []
    for (name, start), (_, after) in zip(
        starts, [*starts[1:], (None, len(lines))], strict=True
    ):
        end = after
        while end > start and (not lines[end - 1].strip() or _comment(lines[end - 1])):
            end -= 1
        spans.append((name, start, end))
    return spans


def _comment(line: str) -> bool:
    return line.lstrip().startswith("#")


def _mine(spans, key: str):
    return [s for s in spans if s[0] == key or (s[0] or "").startswith(key + ".")]


def _key_line(lines: list[str], spans, key: str) -> int | None:
    _, start, end = spans[0]
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
    return next((i for i in range(start, end) if pattern.match(lines[i])), None)


def values(text: str, keys) -> dict[str, str | None]:
    """What each entry holds in text, None when it is absent."""
    lines = text.splitlines(keepends=True)
    spans = _spans(lines)
    found: dict[str, str | None] = {}
    for key in keys:
        if "." in key:
            parts = ["".join(lines[s:e]).rstrip("\n") for _, s, e in _mine(spans, key)]
            found[key] = "\n\n".join(parts) or None
        else:
            index = _key_line(lines, spans, key)
            found[key] = None if index is None else lines[index].rstrip("\n")
    return found


def set_values(text: str, wanted: dict[str, str | None]) -> str:
    """text with each entry set to its text, or removed for None."""
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    for key, value in wanted.items():
        new = [line + "\n" for line in value.split("\n")] if value else []
        spans = _spans(lines)
        if "." in key:
            mine = _mine(spans, key)
            for number, (_, start, end) in reversed(list(enumerate(mine))):
                # a subtable takes the blank lines that set it apart with it
                cut = start
                while number and cut > 0 and not lines[cut - 1].strip():
                    cut -= 1
                del lines[cut:end]
            at = mine[0][1] if mine else len(lines)
            if not mine and new and lines and lines[-1].strip():
                new.insert(0, "\n")
            lines[at:at] = new
            if mine and not new:
                _join_gap(lines, at)
            continue
        index = _key_line(lines, spans, key)
        if index is not None:
            lines[index : index + 1] = new
        else:
            at = spans[0][2]
            lines[at:at] = new
    result = "".join(lines)
    _check(text, result, wanted)
    return result


def _join_gap(lines: list[str], at: int) -> None:
    """Drop the blank line left where an entry was taken out from between two."""
    before = at > 0 and not lines[at - 1].strip()
    if before and (at == len(lines) or not lines[at].strip()):
        del lines[at - 1]


def _check(old: str, new: str, wanted: dict[str, str | None]) -> None:
    """Refuse an edit that changed more than the entries in wanted, or that
    does not read back as them."""
    if tomllib is None:
        raise ValueError("editing TOML needs Python 3.11 or later")
    try:
        before = tomllib.loads(old)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"not valid TOML ({exc}), so it was left as it is") from exc
    try:
        after = tomllib.loads(new)
        expected = {key: tomllib.loads(text or "") for key, text in wanted.items()}
    except tomllib.TOMLDecodeError:
        expected = None  # such as a table it adds that the file defines another way
    for key, alone in (expected or {}).items():
        path = key.split(".")
        pop_path(before, path)
        if pop_path(after, path) != pop_path(alone, path):
            break
    else:
        if expected is not None and before == after:
            return
    raise ValueError(
        "its layout could not be edited without changing other settings, so it"
        " was left as it is"
    )


def pop_path(doc: dict[str, Any], path: list[str]) -> Any:
    """Take the value at path out of nested tables (or JSON objects), and any
    table it leaves empty."""
    parents = [doc]
    for part in path[:-1]:
        child = parents[-1].get(part)
        if not isinstance(child, dict):
            return None
        parents.append(child)
    value = parents[-1].pop(path[-1], None)
    for parent, part in zip(reversed(parents[:-1]), reversed(path[:-1]), strict=True):
        if parent.get(part) == {}:
            del parent[part]
    return value
