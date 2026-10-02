"""Ctrl-C handling of commands wrapped by exit_with_logged_error."""

from __future__ import annotations

from pathlib import Path

import pytest
import requests
from cli.commands import runtime
from cli.commands.optimize import optimize
from cli.commands.route import probe as route_probe_command
from click.testing import CliRunner


def _interrupt(*_args, **_kwargs):
    raise KeyboardInterrupt


def test_interrupted_route_probe_does_not_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(requests, "post", _interrupt)

    result = CliRunner().invoke(
        route_probe_command,
        ["--prompt", "hello", "--base-url", "http://localhost:8801"],
    )

    assert result.exit_code == 1
    assert "Aborted!" in result.stderr
    assert result.stdout == ""


def test_interrupted_recipe_learning_fails_without_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("cli.router_management_client.requests.request", _interrupt)
    output_dir = tmp_path / "out"

    result = CliRunner().invoke(
        optimize,
        [
            "recipe-learning",
            "--endpoint",
            "http://localhost:8080",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 1
    assert "Aborted!" in result.stderr
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("command", "arguments", "interrupted", "message"),
    [
        (runtime.serve, [], "_execute_serve", "Interrupted by user"),
        (runtime.logs, ["router"], "_build_backend", "Log streaming stopped"),
    ],
)
def test_serve_and_logs_still_stop_on_ctrl_c(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    command,
    arguments: list[str],
    interrupted: str,
    message: str,
) -> None:
    monkeypatch.setattr(runtime, interrupted, _interrupt)

    with caplog.at_level("INFO", logger="cli.commands.runtime"):
        result = CliRunner().invoke(command, arguments)

    assert result.exit_code == 0
    assert message in caplog.text
