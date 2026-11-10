import importlib
import json
import stat
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

main = importlib.import_module("cli.main").main
hc = importlib.import_module("cli.harness_connect")
rc = importlib.import_module("cli.harness_router_check")

MESSAGES_REPLY = {
    "type": "message",
    "content": [{"type": "tool_use", "id": "t", "name": "get_time", "input": {}}],
}
RESPONSES_REPLY = {
    "object": "response",
    "status": "completed",
    "output": [{"type": "function_call", "name": "get_time"}],
}


@pytest.fixture
def gateway():
    """A stub gateway that records each request and answers like the Router."""
    seen = []
    replies = {"/v1/messages": MESSAGES_REPLY, "/v1/responses": RESPONSES_REPLY}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            seen.append((self.path, body, self.headers.get("authorization")))
            reply = replies.get(self.path)
            self.send_response(200 if reply is not None else 404)
            self.send_header("content-type", "application/json")
            self.send_header("x-vsr-selected-model", "agent-model")
            self.end_headers()
            payload = reply if reply is not None else {"error": "not_found"}
            self.wfile.write(json.dumps(payload).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", seen, replies
    server.shutdown()


def _claude(path, gateway="http://gw:8899/", token="tok"):
    return hc.Change(path, "json-env", hc.claude_env(gateway, "vllm-sr/auto", token))


def _mode(path):
    return stat.S_IMODE(path.stat().st_mode)


def test_claude_settings_keep_other_keys_and_undo_keeps_later_edits(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"theme": "light", "env": {"FOO": "1", "ANTHROPIC_MODEL": "opus"}})
    )
    hc.apply_changes([_claude(path)], tmp_path)
    settings = json.loads(path.read_text())
    assert settings["theme"] == "light"
    assert settings["env"]["FOO"] == "1"
    assert settings["env"]["ANTHROPIC_BASE_URL"] == "http://gw:8899"
    assert settings["env"]["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "vllm-sr/auto"

    settings["permissions"] = {"allow": ["Bash"]}
    settings["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = "64000"  # changed after connect
    path.write_text(json.dumps(settings))
    report = hc.undo_changes(tmp_path)

    restored = json.loads(path.read_text())
    assert restored["env"] == {
        "FOO": "1",
        "ANTHROPIC_MODEL": "opus",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "64000",
    }
    assert restored["permissions"] == {"allow": ["Bash"]}
    assert any("kept CLAUDE_CODE_MAX_OUTPUT_TOKENS" in line for line in report)
    assert not (tmp_path / hc.MANIFEST).exists()


def test_second_run_keeps_the_first_runs_record(tmp_path):
    path = tmp_path / "settings.json"
    hc.apply_changes([_claude(path, "http://a")], tmp_path)
    hc.apply_changes([_claude(path, "http://b")], tmp_path)
    assert json.loads(path.read_text())["env"]["ANTHROPIC_BASE_URL"] == "http://b"
    hc.undo_changes(tmp_path)
    assert not path.exists()


def test_files_holding_keys_are_private(tmp_path):
    existing = tmp_path / "settings.json"
    existing.write_text("{}")
    existing.chmod(0o644)
    hc.apply_changes([_claude(existing)], tmp_path)
    assert _mode(existing) == 0o600
    assert _mode(tmp_path / hc.MANIFEST) == 0o600
    created = tmp_path / "new" / "settings.json"
    hc.apply_changes([_claude(created)], tmp_path / "new")
    assert _mode(created) == 0o600


def test_the_displayed_diff_shows_owned_keys_and_hides_the_key(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {"apiKeyHelper": "pass sk-other", "env": {"ANTHROPIC_AUTH_TOKEN": "sk-old"}}
        )
    )
    change = _claude(path, token="sk-new")
    old = change.shown(change.old_text(), "<$VLLM_SR_API_KEY>")
    new = change.shown(change.new_text(), "<$VLLM_SR_API_KEY>")
    assert "sk-" not in old + new
    assert '"ANTHROPIC_AUTH_TOKEN": "<hidden>"' in old
    assert '"ANTHROPIC_AUTH_TOKEN": "<$VLLM_SR_API_KEY>"' in new
    assert '"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "32000"' in new


def test_codex_profile_holds_the_provider_and_undo_restores_the_old_one(tmp_path):
    tomllib = pytest.importorskip("tomllib")
    profile = tmp_path / "vllm-sr.config.toml"
    profile.write_text('model = "gpt-5"\n')
    change = hc.codex_change(tmp_path, "vllm-sr", "http://gw/", "vllm-sr/auto", "KEY")
    hc.apply_changes([change], tmp_path)
    parsed = tomllib.loads(profile.read_text())
    assert parsed["model"] == "vllm-sr/auto" and parsed["web_search"] == "disabled"
    assert parsed["model_provider"] == "vllm-sr"
    assert parsed["model_providers"]["vllm-sr"] == {
        "name": "vLLM Semantic Router",
        "base_url": "http://gw/v1",
        "wire_api": "responses",
        "env_key": "KEY",
    }
    assert not (tmp_path / "config.toml").exists()
    hc.undo_changes(tmp_path)
    assert profile.read_text() == 'model = "gpt-5"\n'


def test_undo_removes_what_it_created_and_keeps_what_changed_since(tmp_path):
    profile = tmp_path / "p.config.toml"
    change = hc.codex_change(tmp_path, "p", "http://gw", "m", "K")
    hc.apply_changes([change], tmp_path)
    hc.undo_changes(tmp_path)
    assert not profile.exists()

    hc.apply_changes([change], tmp_path)
    # the Codex TUI appends its own tables after the profile's text
    profile.write_text(
        profile.read_text() + "\n[tui]\nscreen_reader_detection_done = true\n"
    )
    hc.undo_changes(tmp_path)
    assert profile.read_text() == "[tui]\nscreen_reader_detection_done = true\n"

    profile.unlink()
    hc.apply_changes([change], tmp_path)
    profile.write_text(profile.read_text().replace("http://gw", "http://other"))
    report = hc.undo_changes(tmp_path)
    assert "http://other" in profile.read_text()
    assert any(line.startswith("kept ") for line in report)


def test_undo_does_not_recreate_a_file_deleted_since(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{}")
    hc.apply_changes([_claude(path)], tmp_path)
    path.unlink()
    assert any("no longer exists" in line for line in hc.undo_changes(tmp_path))
    assert not path.exists()


def _router(
    lossy=(), learning=None, services=None, api_format="anthropic", plugins=None
):
    model = {"name": "m"}
    if api_format:
        model["api_format"] = api_format
    decision = {"name": "d", "modelRefs": [{"model": "m"}], "plugins": plugins or []}
    doc = {
        "providers": {"models": [model]},
        "recipes": [{"decisions": [decision]}],
        "global": {"router": {"allow_lossy_translation": list(lossy)}},
    }
    if learning is not None:
        doc["global"]["router"]["learning"] = learning
    if services is not None:
        doc["global"]["services"] = services
    return doc


def _learning(headers, enabled=True):
    return {
        "enabled": enabled,
        "protection": {
            "enabled": True,
            "scope": "session",
            "identity": {"headers": headers},
        },
    }


def test_router_check_for_claude_code():
    needs = [f.need for f in rc.check_router_config(_router(), "claude-code")]
    assert any("tools.defer_loading" in n for n in needs)
    assert any("session protection" in n for n in needs)
    complete = _router(["tools.defer_loading"], _learning(rc.CLAUDE_HEADERS))
    assert rc.check_router_config(complete, "claude-code") == []
    off = _router(["tools.defer_loading"], _learning(rc.CLAUDE_HEADERS, enabled=False))
    assert rc.check_router_config(off, "claude-code")


def test_router_check_for_codex():
    needs = [f.need for f in rc.check_router_config(_router(), "codex")]
    assert any("reasoning.signature" in n for n in needs)
    assert any("output limit on decision d" in n for n in needs)
    assert not any("Responses" in n for n in needs)  # on by default
    off = _router(services={"response_api": {"enabled": False}})
    assert any("Responses" in f.need for f in rc.check_router_config(off, "codex"))
    unknown = [f.need for f in rc.check_router_config(_router(api_format=""), "codex")]
    assert any("api_format is not set" in n for n in unknown)
    openai = rc.check_router_config(_router(api_format="openai"), "codex")
    assert not any("reasoning.signature" in f.need for f in openai)
    complete = _router(
        ["reasoning.signature"],
        _learning(rc.CODEX_HEADERS),
        plugins=[
            {"type": "request_params", "configuration": {"default_max_tokens": 32000}}
        ],
    )
    assert rc.check_router_config(complete, "codex") == []


def test_probe_sends_one_tool_call_in_each_protocol(gateway):
    url, seen, _ = gateway
    result = rc.probe_gateway(url, "claude-code", "vllm-sr/auto", "tok")
    assert result.ok and result.model == "agent-model"
    assert result.detail == "called the tool"
    path, body, auth = seen[-1]
    assert path == "/v1/messages" and body["tools"][0]["input_schema"]
    assert auth == "Bearer tok"
    assert rc.probe_gateway(url, "codex", "vllm-sr/auto", "tok").ok
    path, body, _ = seen[-1]
    assert path == "/v1/responses" and body["tools"][0]["type"] == "function"
    assert not rc.probe_gateway(url + "/nope", "codex", "m", "t").ok


def test_probe_refuses_a_200_that_is_not_a_reply(gateway):
    url, _, replies = gateway
    replies["/v1/messages"] = {}
    replies["/v1/responses"] = {"object": "response", "status": "failed"}
    assert not rc.probe_gateway(url, "claude-code", "m", "t").ok
    assert not rc.probe_gateway(url, "codex", "m", "t").ok


def test_cli_dry_run_writes_nothing_and_hides_the_key(tmp_path, gateway, monkeypatch):
    url, _, _ = gateway
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_SR_API_KEY", "sk-secret")
    monkeypatch.setattr("shutil.which", lambda _: None)
    result = CliRunner().invoke(
        main, ["connect", "claude-code", "--gateway", url, "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "sk-secret" not in result.output
    assert '"ANTHROPIC_AUTH_TOKEN": "<$VLLM_SR_API_KEY>"' in result.output
    assert not (tmp_path / "settings.json").exists()


def test_cli_writes_probes_and_undoes(tmp_path, gateway, monkeypatch):
    url, seen, _ = gateway
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    monkeypatch.delenv("VLLM_SR_API_KEY", raising=False)
    result = CliRunner().invoke(main, ["connect", "codex", "--gateway", url, "--yes"])
    assert result.exit_code == 0, result.output
    assert "System Two: ok" in result.output
    assert seen[-1][0] == "/v1/responses"
    assert (tmp_path / "vllm-sr.config.toml").exists()
    result = CliRunner().invoke(main, ["connect", "codex", "--undo"])
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "vllm-sr.config.toml").exists()


def test_cli_fails_when_the_gateway_refuses(tmp_path, gateway, monkeypatch):
    url, _, _ = gateway
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    result = CliRunner().invoke(
        main, ["connect", "codex", "--gateway", url + "/nope", "--yes"]
    )
    assert result.exit_code != 0
    assert "System Two: failed" in result.output


def test_cli_project_scope_keeps_the_key_out_of_git(tmp_path, gateway, monkeypatch):
    url, _, _ = gateway
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))  # no global ignore
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "gitconfig"))
    monkeypatch.setenv("VLLM_SR_API_KEY", "sk-secret")
    monkeypatch.setattr("shutil.which", lambda _: None)
    result = CliRunner().invoke(
        main,
        ["connect", "claude-code", "--scope", "project", "--gateway", url, "--yes"],
    )
    assert result.exit_code == 0, result.output
    assert "sk-secret" in (tmp_path / ".claude" / "settings.local.json").read_text()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert status.stdout == ""
