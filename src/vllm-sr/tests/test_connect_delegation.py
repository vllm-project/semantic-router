"""vllm-sr connect --delegate: Claude Code and Codex hand work to each other,
each in its own official client and on its own login or key."""

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

main = importlib.import_module("cli.main").main
command = importlib.import_module("cli.commands.connect")
# Codex settings in config.toml are edited and read back as TOML (Python 3.11)
tomllib = pytest.importorskip("tomllib")

URL = "http://localhost:8899"
# The status click gives an option it refuses
USAGE_ERROR = 2
# The gateway as `vllm-sr connect codex --app` leaves Codex's default config
CODEX_APP = f"""\
model = "vllm-sr/auto"
model_provider = "vllm-sr"

[model_providers.vllm-sr]
name = "vLLM Semantic Router"
base_url = "{URL}/v1"
wire_api = "responses"
"""


class FakeClaude:
    """`claude plugin` and a key command as connect runs them: the plugins and
    marketplaces Claude Code has, changed by install, add, uninstall and remove."""

    def __init__(self, plugins=(), marketplaces=()):
        self.plugins = set(plugins)
        self.marketplaces = set(marketplaces)
        self.ran = []

    def __call__(self, argv, **kw):
        self.ran.append(list(argv))
        words, out = list(argv[1:]), ""
        if argv[0] == "/bin/sh":
            out = "sk-from-command\n"
        elif words[:2] == ["plugin", "list"]:
            out = json.dumps([{"id": p, "enabled": True} for p in sorted(self.plugins)])
        elif words[:3] == ["plugin", "marketplace", "list"]:
            out = json.dumps([{"name": m} for m in sorted(self.marketplaces)])
        elif words[:3] == ["plugin", "marketplace", "add"]:
            self.marketplaces.add("openai-codex")
        elif words[:3] == ["plugin", "marketplace", "remove"]:
            self.marketplaces.discard(words[3])
        elif words[:2] == ["plugin", "install"]:
            self.plugins.add(words[2])
        elif words[:2] == ["plugin", "uninstall"]:
            self.plugins.discard(words[2])
        return subprocess.CompletedProcess(argv, 0, out, "")


@pytest.fixture(autouse=True)
def router_answers(monkeypatch):
    """A Router answers at every configuration's listener, so connect starts none."""
    monkeypatch.setattr(command, "_answers", lambda _: True)


def _installs(fake):
    """The commands that changed Claude Code's plugins or marketplaces."""
    changing = (["marketplace", "add"], ["marketplace", "remove"])
    return [
        argv[1:]
        for argv in fake.ran
        if argv[1:2] == ["plugin"]
        and (argv[2:4] in changing or argv[2] in ("install", "uninstall"))
    ]


@pytest.fixture
def homes(tmp_path, monkeypatch):
    """Claude Code's and Codex's homes, both CLIs on PATH, and no gateway probe."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "pi"))
    monkeypatch.delenv("VLLM_SR_API_KEY", raising=False)
    monkeypatch.setattr(command, "_probe", lambda *args: None)
    replied = SimpleNamespace(ok=True, model="agent-model", detail="")
    monkeypatch.setattr(command.rc, "basic_reply", lambda *args: replied)
    monkeypatch.setattr(command.shutil, "which", lambda name: f"/bin/{name}")
    (tmp_path / "codex").mkdir()
    return tmp_path / "claude", tmp_path / "codex"


def _claude_args(*more):
    return [
        "connect",
        "claude-code",
        "--base-url",
        URL,
        "--no-vsr",
        "--delegate",
        "codex",
        "--key-command",
        "pass show gw",
        *more,
    ]


def test_claude_code_hands_work_to_codex_through_openais_plugin(homes, monkeypatch):
    claude_home, codex_home = homes
    config = codex_home / "config.toml"
    theirs = (
        CODEX_APP
        + '\n[shell_environment_policy]\ninherit = "core"\nexclude = ["AWS_*"]\n'
    )
    config.write_text(theirs)
    fake = FakeClaude()
    monkeypatch.setattr(command.subprocess, "run", fake)

    result = CliRunner().invoke(main, _claude_args("--dry-run"))
    assert result.exit_code == 0, result.output
    assert "claude plugin marketplace add openai/codex-plugin-cc" in result.output
    assert "claude plugin install codex@openai-codex" in result.output
    assert '+exclude = ["AWS_*", "ANTHROPIC_*"]' in result.output
    assert f"Codex's default config sends to {URL}" in result.output
    assert _installs(fake) == [] and config.read_text() == theirs
    assert not (claude_home / "settings.json").exists()

    result = CliRunner().invoke(
        main, _claude_args("--delegate-source", "/src/cc", "--yes")
    )
    assert result.exit_code == 0, result.output
    assert _installs(fake) == [
        ["plugin", "marketplace", "add", "/src/cc"],
        ["plugin", "install", "codex@openai-codex"],
    ]
    # Claude Code's own variables stay out of every command Codex runs; Codex's other settings stay
    policy = tomllib.loads(config.read_text())["shell_environment_policy"]
    assert policy == {"inherit": "core", "exclude": ["AWS_*", "ANTHROPIC_*"]}
    assert tomllib.loads(config.read_text())["model_provider"] == "vllm-sr"
    # the key is fetched by a command, never held in the settings' env
    settings = json.loads((claude_home / "settings.json").read_text())
    assert settings["apiKeyHelper"] == "pass show gw"
    assert settings["env"]["ANTHROPIC_AUTH_TOKEN"] == ""
    assert "sk-from-command" not in (claude_home / "settings.json").read_text()

    result = CliRunner().invoke(main, ["connect", "status", "--json"])
    claude = json.loads(result.output)["connections"][0]
    assert claude["delegate"] == {
        "to": "codex",
        "plugin": "codex@openai-codex",
        "installed_by_connect": True,
    }
    assert {"path": str(config), "state": "as written"} in claude["files"]
    output = " ".join(CliRunner().invoke(main, ["connect", "status"]).output.split())
    assert (
        "delegates to codex: plugin codex@openai-codex (installed by connect)" in output
    )

    fake.ran.clear()
    result = CliRunner().invoke(main, ["connect", "claude-code", "--undo"])
    assert result.exit_code == 0, result.output
    assert _installs(fake) == [
        ["plugin", "uninstall", "codex@openai-codex"],
        ["plugin", "marketplace", "remove", "openai-codex"],
    ]
    assert config.read_text() == theirs


def test_undo_leaves_a_plugin_and_marketplace_connect_did_not_add(homes, monkeypatch):
    _, codex_home = homes
    (codex_home / "config.toml").write_text(CODEX_APP)
    fake = FakeClaude(plugins={"codex@openai-codex"}, marketplaces={"openai-codex"})
    monkeypatch.setattr(command.subprocess, "run", fake)
    result = CliRunner().invoke(main, _claude_args("--yes"))
    assert result.exit_code == 0, result.output
    assert _installs(fake) == []
    result = CliRunner().invoke(main, ["connect", "claude-code", "--undo"])
    assert result.exit_code == 0, result.output
    assert _installs(fake) == [] and fake.plugins == {"codex@openai-codex"}
    assert (
        not (codex_home / "config.toml").read_text().count("shell_environment_policy")
    )


def test_delegating_to_codex_keeps_the_gateway_key_out_of_claude_codes_env(
    homes, monkeypatch
):
    claude_home, _ = homes
    monkeypatch.setenv("VLLM_SR_API_KEY", "sk-secret")
    fake = FakeClaude()
    monkeypatch.setattr(command.subprocess, "run", fake)
    args = [
        "connect",
        "claude-code",
        "--base-url",
        URL,
        "--no-vsr",
        "--delegate",
        "codex",
    ]
    result = CliRunner().invoke(main, [*args, "--yes"])
    assert result.exit_code == 1
    assert "--key-command" in result.output and "sk-secret" not in result.output
    assert not (claude_home / "settings.json").exists() and _installs(fake) == []


def test_delegating_to_a_codex_not_on_the_gateway_points_at_its_app_connection(
    homes, monkeypatch
):
    _, codex_home = homes
    (codex_home / "vllm-sr.config.toml").write_text(CODEX_APP)  # a profile only
    monkeypatch.setattr(command.subprocess, "run", FakeClaude())
    result = CliRunner().invoke(main, _claude_args("--dry-run"))
    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "vllm-sr connect codex --app" in output
    assert "runs codex with no profile" in output


@pytest.mark.parametrize("harness", ["claude-code", "codex"])
def test_delegation_needs_both_clients_and_a_written_setting(
    homes, monkeypatch, harness
):
    other = {"claude-code": "codex", "codex": "claude-code"}[harness]
    args = ["connect", harness, "--base-url", URL, "--no-vsr", "--delegate", other]
    result = CliRunner().invoke(main, [*args, "--launch"])
    assert (
        result.exit_code == USAGE_ERROR and "does not take --delegate" in result.output
    )
    monkeypatch.setattr(command.shutil, "which", lambda name: None)
    result = CliRunner().invoke(main, [*args, "--key-command", "pass show gw", "--yes"])
    assert result.exit_code == 1 and "is not on PATH" in result.output


def _codex_mcp(doc):
    return doc["mcp_servers"]["claude"]


@pytest.mark.parametrize("app", [False, True])
def test_codex_asks_claude_code_read_only_over_mcp(homes, monkeypatch, app):
    _, codex_home = homes
    monkeypatch.setattr(command.subprocess, "run", FakeClaude())
    mode = ["--app"] if app else []
    args = ["connect", "codex", "--base-url", URL, "--no-vsr", *mode]
    args += ["--key-command", "pass show gw", "--delegate", "claude-code"]
    path = codex_home / ("config.toml" if app else "vllm-sr.config.toml")
    theirs = '[mcp_servers.docs]\ncommand = "docs-mcp"\n'
    if app:
        path.write_text(theirs)

    result = CliRunner().invoke(main, [*args, "--yes"])
    assert result.exit_code == 0, result.output
    doc = tomllib.loads(path.read_text())
    assert _codex_mcp(doc) == {
        "command": "claude",
        "args": ["mcp", "serve"],
        "env_vars": ["CLAUDE_CONFIG_DIR"],
        "enabled_tools": ["Agent"],
        "tool_timeout_sec": 300,
        "tools": {"Agent": {"approval_mode": "approve"}},
    }
    # the provider's key command is still read as the provider's
    assert doc["model_providers"]["vllm-sr"]["auth"]["args"] == ["-c", "pass show gw"]
    assert not (codex_home / "skills").exists()
    assert "read-only" in result.output

    result = CliRunner().invoke(main, ["connect", "status", "--json"])
    codex = json.loads(result.output)["connections"][1]
    assert codex["delegate"] == {"to": "claude-code", "edits": False}
    assert (codex["login"], codex["key_command"]) == ("key command", "pass show gw")

    assert CliRunner().invoke(main, ["connect", "codex", "--undo"]).exit_code == 0
    assert (path.read_text() if path.exists() else None) == (theirs if app else None)


def test_codex_writing_back_the_mcp_table_leaves_it_connects(homes, monkeypatch):
    # codex mcp add writes config.toml back in its own form: 300.0 for 300 and
    # keys in another order, the same values
    _, codex_home = homes
    monkeypatch.setattr(command.subprocess, "run", FakeClaude())
    args = ["connect", "codex", "--base-url", URL, "--no-vsr", "--app"]
    args += ["--key-command", "pass show gw", "--delegate", "claude-code"]
    path = codex_home / "config.toml"
    theirs = '[mcp_servers.docs]\ncommand = "docs-mcp"\n'
    path.write_text(theirs)
    assert CliRunner().invoke(main, [*args, "--yes"]).exit_code == 0
    text = path.read_text()
    rewritten = text.replace("tool_timeout_sec = 300", "tool_timeout_sec = 300.0")
    rewritten = rewritten.replace(
        'command = "claude"\nargs = ["mcp", "serve"]',
        'args = ["mcp", "serve"]\ncommand = "claude"',
    )
    assert rewritten != text
    path.write_text(rewritten)

    output = " ".join(CliRunner().invoke(main, ["connect", "status"]).output.split())
    assert f"{path}: as written" in output
    assert CliRunner().invoke(main, ["connect", "codex", "--undo"]).exit_code == 0
    assert path.read_text() == theirs


def test_codex_hands_edits_to_claude_code_only_through_an_opt_in_skill(
    homes, monkeypatch
):
    _, codex_home = homes
    monkeypatch.setattr(command.subprocess, "run", FakeClaude())
    args = [
        "connect",
        "codex",
        "--base-url",
        URL,
        "--no-vsr",
        "--key-command",
        "pass show gw",
    ]
    result = CliRunner().invoke(main, [*args, "--delegate-edits", "--yes"])
    assert result.exit_code == USAGE_ERROR and "--delegate claude-code" in result.output

    result = CliRunner().invoke(
        main, [*args, "--delegate", "claude-code", "--delegate-edits", "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    skill = codex_home / "skills" / "vllm-sr-claude-code" / "SKILL.md"
    assert f"+++ {skill}" in result.output and not skill.exists()

    result = CliRunner().invoke(
        main, [*args, "--delegate", "claude-code", "--delegate-edits", "--yes"]
    )
    assert result.exit_code == 0, result.output
    text = skill.read_text()
    assert text.startswith("---\nname: vllm-sr-claude-code\n")
    assert "claude -p --output-format json --max-turns" in text
    # edit tools only when asked, and one command outside the sandbox, not network for all
    assert "--tools Read,Grep,Glob " in text and "acceptEdits" in text
    assert 'sandbox_permissions: "require_escalated"' in text
    assert "--bare" not in text and "network_access" not in result.output
    profile = tomllib.loads((codex_home / "vllm-sr.config.toml").read_text())
    assert "sandbox_workspace_write" not in profile and _codex_mcp(profile)

    result = CliRunner().invoke(main, ["connect", "status", "--json"])
    codex = json.loads(result.output)["connections"][1]
    assert codex["delegate"] == {"to": "claude-code", "edits": True}
    assert {"path": str(skill), "state": "as written"} in codex["files"]

    assert CliRunner().invoke(main, ["connect", "codex", "--undo"]).exit_code == 0
    assert not skill.parent.exists()
