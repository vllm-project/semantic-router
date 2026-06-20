import importlib
import sys
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

main = importlib.import_module("cli.main").main
completion_mod = importlib.import_module("cli.commands.completion")


# ---------------------------------------------------------------------------
# Script generation tests (vllm-sr completion show <shell>)
# ---------------------------------------------------------------------------


def test_completion_show_bash_outputs_script():
    runner = CliRunner()

    result = runner.invoke(main, ["completion", "show", "bash"])

    assert result.exit_code == 0
    assert "_VLLM_SR_COMPLETE" in result.output
    assert "bash" in result.output.lower()


def test_completion_show_zsh_outputs_script():
    runner = CliRunner()

    result = runner.invoke(main, ["completion", "show", "zsh"])

    assert result.exit_code == 0
    assert "_VLLM_SR_COMPLETE" in result.output
    assert "zsh" in result.output.lower()


def test_completion_show_fish_outputs_script():
    runner = CliRunner()

    result = runner.invoke(main, ["completion", "show", "fish"])

    assert result.exit_code == 0
    assert "_VLLM_SR_COMPLETE" in result.output
    assert "fish" in result.output.lower()


def test_completion_show_auto_detects_bash_shell():
    runner = CliRunner()

    with patch.dict("os.environ", {"SHELL": "/bin/bash"}):
        result = runner.invoke(main, ["completion", "show"])

    assert result.exit_code == 0
    assert "_VLLM_SR_COMPLETE" in result.output


def test_completion_show_auto_detects_zsh_shell():
    runner = CliRunner()

    with patch.dict("os.environ", {"SHELL": "/bin/zsh"}):
        result = runner.invoke(main, ["completion", "show"])

    assert result.exit_code == 0
    assert "_VLLM_SR_COMPLETE" in result.output


def test_completion_show_fails_when_shell_undetectable():
    runner = CliRunner()

    with patch.dict("os.environ", {"SHELL": ""}, clear=False):
        result = runner.invoke(main, ["completion", "show"])

    # The error is logged via the logger (not Click echo), so it won't appear
    # in result.output.  The important contract is the non-zero exit code.
    assert result.exit_code != 0


def test_completion_show_invalid_shell_rejected():
    runner = CliRunner()

    result = runner.invoke(main, ["completion", "show", "powershell"])

    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Group-level tests
# ---------------------------------------------------------------------------


def test_completion_help_lists_subcommands():
    runner = CliRunner()

    result = runner.invoke(main, ["completion", "--help"])

    assert result.exit_code == 0
    assert "show" in result.output
    assert "install" in result.output


def test_completion_bare_shows_help():
    runner = CliRunner()

    result = runner.invoke(main, ["completion"])

    assert result.exit_code == 0
    assert "show" in result.output
    assert "install" in result.output


def test_completion_listed_in_main_help():
    runner = CliRunner()

    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "completion" in result.output


# ---------------------------------------------------------------------------
# Install subcommand tests (vllm-sr completion install <shell>)
# ---------------------------------------------------------------------------


def test_install_bash_appends_to_bashrc(tmp_path):
    bashrc = tmp_path / ".bashrc"
    bashrc.write_text("# existing content\n", encoding="utf-8")

    with patch.dict(completion_mod._RC_FILES, {"bash": str(bashrc)}):
        runner = CliRunner()
        result = runner.invoke(main, ["completion", "install", "bash"])

    assert result.exit_code == 0
    content = bashrc.read_text(encoding="utf-8")
    assert "# vllm-sr shell completion" in content
    assert 'eval "$(vllm-sr completion show bash)"' in content
    assert "# existing content" in content
    assert "✓" in result.output


def test_install_zsh_appends_to_zshrc(tmp_path):
    zshrc = tmp_path / ".zshrc"
    zshrc.write_text("# existing zsh config\n", encoding="utf-8")

    with patch.dict(completion_mod._RC_FILES, {"zsh": str(zshrc)}):
        runner = CliRunner()
        result = runner.invoke(main, ["completion", "install", "zsh"])

    assert result.exit_code == 0
    content = zshrc.read_text(encoding="utf-8")
    assert "# vllm-sr shell completion" in content
    assert 'eval "$(vllm-sr completion show zsh)"' in content
    assert "✓" in result.output


def test_install_fish_writes_completion_file(tmp_path):
    fish_dir = tmp_path / ".config" / "fish" / "completions"

    original_expanduser = Path.expanduser

    def fake_expanduser(self):
        path_str = str(self)
        if path_str.startswith("~"):
            return tmp_path / path_str[2:]
        return original_expanduser(self)

    with patch.object(Path, "expanduser", fake_expanduser):
        runner = CliRunner()
        result = runner.invoke(main, ["completion", "install", "fish"])

    assert result.exit_code == 0
    fish_file = fish_dir / "vllm-sr.fish"
    assert fish_file.exists()
    content = fish_file.read_text(encoding="utf-8")
    assert "_VLLM_SR_COMPLETE" in content
    assert "✓" in result.output


def test_install_is_idempotent(tmp_path):
    bashrc = tmp_path / ".bashrc"
    bashrc.write_text("# existing content\n", encoding="utf-8")

    with patch.dict(completion_mod._RC_FILES, {"bash": str(bashrc)}):
        runner = CliRunner()
        # Install twice
        runner.invoke(main, ["completion", "install", "bash"])
        result = runner.invoke(main, ["completion", "install", "bash"])

    assert result.exit_code == 0
    content = bashrc.read_text(encoding="utf-8")
    # The marker should appear exactly once
    assert content.count("# vllm-sr shell completion") == 1
    assert "already configured" in result.output


def test_install_creates_rc_file_if_missing(tmp_path):
    bashrc = tmp_path / ".bashrc"
    assert not bashrc.exists()

    with patch.dict(completion_mod._RC_FILES, {"bash": str(bashrc)}):
        runner = CliRunner()
        result = runner.invoke(main, ["completion", "install", "bash"])

    assert result.exit_code == 0
    assert bashrc.exists()
    content = bashrc.read_text(encoding="utf-8")
    assert "# vllm-sr shell completion" in content


def test_install_auto_detects_shell(tmp_path):
    zshrc = tmp_path / ".zshrc"
    zshrc.write_text("", encoding="utf-8")

    with (
        patch.dict("os.environ", {"SHELL": "/bin/zsh"}),
        patch.dict(completion_mod._RC_FILES, {"zsh": str(zshrc)}),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["completion", "install"])

    assert result.exit_code == 0
    content = zshrc.read_text(encoding="utf-8")
    assert "# vllm-sr shell completion" in content


def test_install_help_shows_usage():
    runner = CliRunner()

    result = runner.invoke(main, ["completion", "install", "--help"])

    assert result.exit_code == 0
    assert "Install shell completions" in result.output
    assert "bash" in result.output
    assert "zsh" in result.output
    assert "fish" in result.output
