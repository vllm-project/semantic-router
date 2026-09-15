"""Shared terminal presentation contracts."""

from __future__ import annotations

import logging

import click
from cli import terminal
from cli.terminal import echo, error, fields, heading, warning
from cli.url_display import redact_url
from cli.utils import get_logger
from click.testing import CliRunner


def _fresh_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.NOTSET)
    return get_logger(name)


def test_logger_routes_progress_and_failures_to_clean_stderr() -> None:
    logger = _fresh_logger("tests.terminal.streams")

    @click.command()
    def command() -> None:
        echo("result")
        logger.info("Preparing runtime")
        logger.warning("Check this setting")
        logger.error("Operation failed")

    result = CliRunner().invoke(command)

    assert result.exit_code == 0
    assert result.stdout == "result\n"
    assert result.stderr.splitlines() == [
        "Preparing runtime",
        "Warning: Check this setting",
        "Error: Operation failed",
    ]
    assert " - INFO - " not in result.stderr
    assert " - ERROR - " not in result.stderr


def test_logger_reuse_does_not_duplicate_messages() -> None:
    logger = _fresh_logger("tests.terminal.reuse")

    @click.command()
    def command() -> None:
        get_logger(logger.name).info("one message")

    result = CliRunner().invoke(command)

    assert result.exit_code == 0
    assert result.stderr == "one message\n"


def test_no_color_disables_ansi_even_when_click_color_is_forced(monkeypatch) -> None:
    @click.command()
    def command() -> None:
        heading("Overview")
        warning("Review this")

    monkeypatch.delenv("NO_COLOR", raising=False)
    colored = CliRunner().invoke(command, color=True)
    plain = CliRunner().invoke(command, color=True, env={"NO_COLOR": "1"})

    assert "\x1b[" in colored.output
    assert "\x1b[" not in plain.output
    assert plain.stdout == "Overview\n"
    assert plain.stderr == "Warning: Review this\n"


def test_fields_wrap_long_values_to_requested_width() -> None:
    @click.command()
    def command() -> None:
        fields(
            (("Endpoint", "https://example.test/a-very-long-segment/another-part"),),
            width=40,
        )

    result = CliRunner().invoke(command)

    assert result.exit_code == 0
    assert max(map(len, result.stdout.splitlines())) <= 40


def test_labeled_messages_do_not_repeat_existing_prefix() -> None:
    @click.command()
    def command() -> None:
        warning("Warning: already labeled")

    result = CliRunner().invoke(command)

    assert result.stderr == "Warning: already labeled\n"


def test_long_error_wraps_on_a_narrow_terminal(monkeypatch) -> None:
    monkeypatch.setattr(terminal, "output_width", lambda: 40)

    @click.command()
    def command() -> None:
        error(
            "Configuration could not be parsed because the selected file "
            "contains a long invalid value that needs operator attention."
        )

    result = CliRunner().invoke(command)

    assert result.exit_code == 0
    assert result.stderr.startswith("Error: Configuration")
    assert max(map(len, result.stderr.splitlines())) <= 40


def test_display_url_redacts_userinfo_and_credential_parameters() -> None:
    rendered = redact_url(
        "https://user:TOPSECRET@router.test/v1?apiKey=QUERYSECRET#token=FRAGMENTSECRET"
    )

    assert rendered == "https://***@router.test/v1?apiKey=***#token=***"
    assert "TOPSECRET" not in rendered
    assert "QUERYSECRET" not in rendered
    assert "FRAGMENTSECRET" not in rendered
