"""Consistent terminal presentation for the vllm-sr CLI."""

from __future__ import annotations

import logging
import os
import shutil
import textwrap
from collections.abc import Iterable
from typing import Any

import click

_DEFAULT_OUTPUT_WIDTH = 120
_MIN_OUTPUT_WIDTH = 40
_MAX_OUTPUT_WIDTH = 160


def echo(message: Any = "", *, err: bool = False, nl: bool = True) -> None:
    """Write plain command output to the appropriate Click stream."""
    click.echo(message, err=err, nl=nl)


def heading(message: str, *, indent: int = 0) -> None:
    """Write a bold, accessible section heading."""
    available = max(1, output_width() - indent)
    for line in _wrap(message, available):
        click.secho(
            f"{' ' * indent}{line}",
            bold=True,
            color=_color_setting(),
        )


def brand(message: str) -> None:
    """Write the CLI brand treatment without affecting redirected output."""
    click.secho(
        message,
        fg="cyan",
        bold=True,
        color=_color_setting(),
    )


def success(message: str) -> None:
    """Write a successful human-facing result to stdout."""
    click.secho(
        _wrap_labeled(f"✓ {message}", output_width()),
        fg="green",
        color=_color_setting(),
    )


def progress(message: str) -> None:
    """Write operational progress to stderr, leaving stdout pipe-safe."""
    echo(_wrap_labeled(message, output_width()), err=True)


def warning(message: str) -> None:
    """Write a warning to stderr with one stable textual label."""
    _write_labeled("Warning", message, fg="yellow")


def error(message: str) -> None:
    """Write an error to stderr with one stable textual label."""
    _write_labeled("Error", message, fg="red")


def hint(message: str) -> None:
    """Write an actionable error hint to stderr."""
    _write_labeled("Hint", message, fg="cyan")


def fields(
    items: Iterable[tuple[str, Any]],
    *,
    indent: int = 2,
    width: int | None = None,
) -> None:
    """Write aligned, wrapped label/value fields."""
    values = [(label, str(value)) for label, value in items]
    if not values:
        return
    label_width = max(len(label) for label, _ in values)
    available = max(12, (width or output_width()) - indent - label_width - 2)
    prefix = " " * indent
    continuation = " " * (indent + label_width + 2)
    for label, value in values:
        lines = _wrap(value, available)
        echo(f"{prefix}{label:<{label_width}}  {lines[0]}")
        for line in lines[1:]:
            echo(f"{continuation}{line}")


def output_width() -> int:
    """Return a bounded width for TTY rendering and a stable pipe width."""
    stream = click.get_text_stream("stdout")
    if not stream.isatty():
        return _DEFAULT_OUTPUT_WIDTH
    columns = shutil.get_terminal_size((_DEFAULT_OUTPUT_WIDTH, 24)).columns
    return max(_MIN_OUTPUT_WIDTH, min(columns, _MAX_OUTPUT_WIDTH))


class TerminalLogHandler(logging.Handler):
    """Route CLI logs through the same Click-aware terminal contract."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            if record.levelno >= logging.ERROR:
                error(message)
            elif record.levelno >= logging.WARNING:
                warning(message)
            else:
                progress(message)
        except Exception:  # pragma: no cover - logging must never crash the CLI
            self.handleError(record)


def _write_labeled(label: str, message: str, *, fg: str) -> None:
    rendered = _ensure_label(label, str(message))
    wrapped = _wrap_labeled(rendered, output_width())
    click.secho(
        wrapped,
        fg=fg,
        err=True,
        color=_color_setting(),
    )


def _ensure_label(label: str, message: str) -> str:
    stripped = message.lstrip("\n")
    leading = message[: len(message) - len(stripped)]
    if stripped.casefold().startswith(f"{label.casefold()}:"):
        return message
    return f"{leading}{label}: {stripped}"


def _color_setting() -> bool | None:
    if "NO_COLOR" in os.environ:
        return False
    return None


def _wrap(value: str, width: int) -> list[str]:
    return textwrap.wrap(
        value,
        width=max(1, width),
        break_long_words=True,
        break_on_hyphens=False,
    ) or [""]


def _wrap_labeled(value: str, width: int) -> str:
    wrapper = textwrap.TextWrapper(
        width=max(1, width),
        subsequent_indent="  ",
        break_long_words=True,
        break_on_hyphens=False,
    )
    lines: list[str] = []
    for source_line in value.splitlines() or [""]:
        lines.extend(wrapper.wrap(source_line) or [""])
    return "\n".join(lines)
