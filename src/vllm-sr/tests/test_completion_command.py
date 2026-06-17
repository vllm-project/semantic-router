import importlib
import sys
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

main = importlib.import_module("cli.main").main


def test_completion_bash_outputs_script():
    runner = CliRunner()

    result = runner.invoke(main, ["completion", "bash"])

    assert result.exit_code == 0
    assert "_VLLM_SR_COMPLETE" in result.output
    assert "bash" in result.output.lower()


def test_completion_zsh_outputs_script():
    runner = CliRunner()

    result = runner.invoke(main, ["completion", "zsh"])

    assert result.exit_code == 0
    assert "_VLLM_SR_COMPLETE" in result.output
    assert "zsh" in result.output.lower()


def test_completion_fish_outputs_script():
    runner = CliRunner()

    result = runner.invoke(main, ["completion", "fish"])

    assert result.exit_code == 0
    assert "_VLLM_SR_COMPLETE" in result.output
    assert "fish" in result.output.lower()


def test_completion_help_shows_usage():
    runner = CliRunner()

    result = runner.invoke(main, ["completion", "--help"])

    assert result.exit_code == 0
    assert "bash" in result.output
    assert "zsh" in result.output
    assert "fish" in result.output
    assert "Generate shell completion script" in result.output


def test_completion_auto_detects_bash_shell():
    runner = CliRunner()

    with patch.dict("os.environ", {"SHELL": "/bin/bash"}):
        result = runner.invoke(main, ["completion"])

    assert result.exit_code == 0
    assert "_VLLM_SR_COMPLETE" in result.output


def test_completion_auto_detects_zsh_shell():
    runner = CliRunner()

    with patch.dict("os.environ", {"SHELL": "/bin/zsh"}):
        result = runner.invoke(main, ["completion"])

    assert result.exit_code == 0
    assert "_VLLM_SR_COMPLETE" in result.output


def test_completion_fails_when_shell_undetectable():
    runner = CliRunner()

    with patch.dict("os.environ", {"SHELL": ""}, clear=False):
        result = runner.invoke(main, ["completion"])

    # The error is logged via the logger (not Click echo), so it won't appear
    # in result.output.  The important contract is the non-zero exit code.
    assert result.exit_code != 0


def test_completion_listed_in_main_help():
    runner = CliRunner()

    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "completion" in result.output


def test_completion_invalid_shell_rejected():
    runner = CliRunner()

    result = runner.invoke(main, ["completion", "powershell"])

    assert result.exit_code != 0
