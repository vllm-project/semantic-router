import importlib
import json
import stat
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

main = importlib.import_module("cli.main").main
hc = importlib.import_module("cli.harness_connect")
rc = importlib.import_module("cli.harness_router_check")
chat_client = importlib.import_module("cli.chat_client")

# What the stub gateway answers at each stage: a plain reply, the streamed tool
# call, and the streamed answer once the tool result goes back.
MESSAGES_EVENTS = [
    {"type": "message_start", "message": {"type": "message", "content": []}},
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "type": "tool_use",
            "id": "t",
            "name": "get_time",
            "input": {},
        },
    },
    {"type": "message_stop"},
]
RESPONSES_EVENTS = [
    {
        "type": "response.completed",
        "response": {
            "object": "response",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "name": "get_time",
                    "call_id": "c1",
                    "arguments": "{}",
                    "status": "completed",
                }
            ],
        },
    }
]
STAGES = {
    "/v1/messages": {
        "plain": {"type": "message", "content": [{"type": "text", "text": "ok"}]},
        "tool": MESSAGES_EVENTS,
        "result": [
            {"type": "message_start", "message": {"type": "message", "content": []}},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "It is 12:00."},
            },
            {"type": "message_stop"},
        ],
    },
    "/v1/responses": {
        "plain": {"object": "response", "status": "completed", "output": []},
        "tool": RESPONSES_EVENTS,
        "result": [
            {
                "type": "response.completed",
                "response": {
                    "object": "response",
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": "It is 12:00."}
                            ],
                        }
                    ],
                },
            }
        ],
    },
}


def _stage(body: dict) -> str:
    if not body.get("stream"):
        return "plain"
    sent = json.dumps(body.get("messages") or body.get("input"))
    return (
        "result" if "tool_result" in sent or "function_call_output" in sent else "tool"
    )


# A translation loss whose reason holds a comma, as the Router writes it: an RFC 8941 list.
PLAIN_WARNINGS = (
    'protocol;action=approximated;reason="stop_sequence, absent, read as null";'
    'field="stop_sequence", truncated;count=2'
)


@pytest.fixture
def gateway():
    """A stub gateway that records each request and answers like the Router: JSON
    for a plain request, events for a streamed one."""
    seen = []
    replies = {path: dict(stages) for path, stages in STAGES.items()}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            seen.append((self.path, body, self.headers.get("authorization")))
            stage = _stage(body)
            reply = replies.get(self.path, {}).get(stage)
            self.send_response(200 if reply is not None else 404)
            kind = "application/json" if stage == "plain" else "text/event-stream"
            self.send_header("content-type", kind)
            self.send_header("x-vsr-selected-model", "agent-model")
            if stage == "plain":
                self.send_header("x-vsr-warnings", PLAIN_WARNINGS)
            self.end_headers()
            if stage == "plain":
                self.wfile.write(json.dumps(reply or {"error": "not_found"}).encode())
                return
            for event in reply or [{"type": "error"}]:
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())

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


def test_a_failed_write_leaves_the_settings_as_they_were(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    path.write_text('{"theme": "dark"}')

    def crash(*_):
        raise OSError("disk full")

    monkeypatch.setattr(hc.os, "replace", crash)
    with pytest.raises(OSError):
        hc.write_private(path, "{}", always=True)
    assert path.read_text() == '{"theme": "dark"}'
    assert [p.name for p in tmp_path.iterdir()] == ["settings.json"]


def test_writing_through_a_linked_settings_file_keeps_the_link(tmp_path):
    target = tmp_path / "dotfiles" / "settings.json"
    target.parent.mkdir()
    target.write_text("{}")
    link = tmp_path / "settings.json"
    link.symlink_to(target)
    hc.write_private(link, "{}\n", always=True)
    assert link.is_symlink() and target.read_text() == "{}\n"


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

    profile.write_text('model = "gpt-5"')  # no final line break
    hc.apply_changes([change], tmp_path)
    profile.write_text(
        profile.read_text() + "\n[tui]\nscreen_reader_detection_done = true\n"
    )
    hc.undo_changes(tmp_path)
    assert tomllib.loads(profile.read_text()) == {
        "model": "gpt-5",
        "tui": {"screen_reader_detection_done": True},
    }


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
    openai = _router(
        ["tools.defer_loading", "content.file"],
        _learning(rc.CLAUDE_HEADERS),
        api_format="openai",
    )
    (need,) = [f.need for f in rc.check_router_config(openai, "claude-code")]
    assert need.startswith("tool_result.is_error, messages[].role, or")
    unknown = _router(
        ["tools.defer_loading"], _learning(rc.CLAUDE_HEADERS), api_format=""
    )
    assert (
        "api_format is not set"
        in rc.check_router_config(unknown, "claude-code")[0].need
    )


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


def test_router_check_reads_canonical_api_formats_like_their_aliases():
    claude = _router(["tools.defer_loading"], _learning(rc.CLAUDE_HEADERS))
    for name in ("openai", "chat_completions", "responses"):
        claude["providers"]["models"][0]["api_format"] = name
        (need,) = [f.need for f in rc.check_router_config(claude, "claude-code")]
        assert need.startswith("tool_result.is_error") and "not set" not in need
    claude["providers"]["models"][0]["api_format"] = "messages"
    assert rc.check_router_config(claude, "claude-code") == []
    for name in ("anthropic", "messages"):
        needs = [
            f.need for f in rc.check_router_config(_router(api_format=name), "codex")
        ]
        assert any(
            n.startswith("reasoning.signature") and "not set" not in n for n in needs
        )


def test_probe_checks_a_plain_reply_then_a_streamed_tool_loop(gateway):
    url, seen, _ = gateway
    plain, loop = rc.probe_gateway(url, "claude-code", "vllm-sr/auto", "tok")
    assert (plain.name, plain.ok, plain.model) == ("Basic reply", True, "agent-model")
    assert plain.warnings == [
        "stop_sequence approximated: stop_sequence, absent, read as null",
        "2 more left out of the header",
    ]
    assert (loop.name, loop.ok) == ("Streamed tool loop", True)
    assert loop.detail == "called the tool and answered its result"
    (_, first, auth), (_, call, _), (path, answer, _) = seen
    assert "stream" not in first and "tools" not in first
    assert call["stream"] is True and call["tools"][0]["input_schema"]
    assert path == "/v1/messages" and auth == "Bearer tok"
    assert answer["messages"][-1]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "t",
        "content": "12:00",
    }

    seen.clear()
    assert all(r.ok for r in rc.probe_gateway(url, "codex", "vllm-sr/auto", "tok"))
    path, answer, _ = seen[-1]
    assert path == "/v1/responses" and seen[1][1]["tools"][0]["type"] == "function"
    # the call goes back as Codex sends it, then its result
    assert answer["input"][1] == {
        "type": "function_call",
        "name": "get_time",
        "call_id": "c1",
        "arguments": "{}",
    }
    assert answer["input"][2]["type"] == "function_call_output"
    failed = rc.probe_gateway(url + "/nope", "codex", "m", "t")
    assert [r.ok for r in failed] == [False]


def test_probe_refuses_a_200_that_is_not_a_reply(gateway):
    url, _, replies = gateway
    replies["/v1/messages"]["plain"] = {}
    (plain,) = rc.probe_gateway(url, "claude-code", "m", "t")
    assert not plain.ok and plain.name == "Basic reply"

    replies["/v1/messages"]["plain"] = STAGES["/v1/messages"]["plain"]
    replies["/v1/messages"]["tool"] = MESSAGES_EVENTS[:1]  # never finished
    replies["/v1/responses"]["tool"] = [{"type": "response.failed", "response": {}}]
    assert not rc.probe_gateway(url, "claude-code", "m", "t")[1].ok
    assert not rc.probe_gateway(url, "codex", "m", "t")[1].ok

    # a stream that fails where a block closes still began with a 200
    replies["/v1/messages"]["tool"] = [
        *MESSAGES_EVENTS[:2],
        {"type": "error", "error": {"type": "lossy_translation"}},
    ]
    loop = rc.probe_gateway(url, "claude-code", "m", "t")[1]
    assert not loop.ok and "lossy_translation" in loop.detail

    # and the answer to the tool result can fail on its own
    replies["/v1/messages"]["tool"] = MESSAGES_EVENTS
    replies["/v1/messages"]["result"] = [{"type": "error", "error": {"type": "x"}}]
    loop = rc.probe_gateway(url, "claude-code", "m", "t")[1]
    assert not loop.ok and "ended in an error" in loop.detail
    # or finish without the text the result was meant for
    replies["/v1/messages"]["result"] = [MESSAGES_EVENTS[0], MESSAGES_EVENTS[-1]]
    loop = rc.probe_gateway(url, "claude-code", "m", "t")[1]
    assert not loop.ok and "has no text" in loop.detail
    assert "output tokens" not in loop.detail
    # a model that thinks until the token limit names the limit, not the gateway
    replies["/v1/messages"]["result"] = [
        MESSAGES_EVENTS[0],
        {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}},
        MESSAGES_EVENTS[-1],
    ]
    loop = rc.probe_gateway(url, "claude-code", "m", "t")[1]
    assert not loop.ok and f"{rc.PROBE_TOKENS}-token output limit" in loop.detail
    # and a model that runs out before it calls the tool says so too
    replies["/v1/messages"]["tool"] = replies["/v1/messages"]["result"]
    loop = rc.probe_gateway(url, "claude-code", "m", "t")[1]
    assert loop.ok and "output limit before calling the tool" in loop.detail
    replies["/v1/messages"]["tool"] = MESSAGES_EVENTS
    # a Responses reply cut for another reason, such as a content filter, is not a token limit
    replies["/v1/responses"]["tool"] = RESPONSES_EVENTS
    incomplete = {
        "type": "response.completed",
        "response": {"object": "response", "status": "incomplete", "output": []},
    }
    for reason, says in (("max_output_tokens", True), ("content_filter", False)):
        incomplete["response"]["incomplete_details"] = {"reason": reason}
        replies["/v1/responses"]["result"] = [incomplete]
        loop = rc.probe_gateway(url, "codex", "m", "t")[1]
        assert not loop.ok and ("output limit" in loop.detail) is says


def test_probe_answers_every_tool_call_and_says_when_none_came(gateway):
    url, seen, replies = gateway
    two = {**RESPONSES_EVENTS[0]["response"]}
    two["output"] = [
        *two["output"],
        {
            "type": "function_call",
            "name": "get_time",
            "call_id": "c2",
            "arguments": "{}",
        },
    ]
    replies["/v1/responses"]["tool"] = [{"type": "response.completed", "response": two}]
    assert all(r.ok for r in rc.probe_gateway(url, "codex", "m", "t"))
    outputs = [
        i for i in seen[-1][1]["input"] if i.get("type") == "function_call_output"
    ]
    assert [o["call_id"] for o in outputs] == ["c1", "c2"]

    replies["/v1/responses"]["tool"] = [
        {
            "type": "response.completed",
            "response": {"object": "response", "status": "completed", "output": []},
        }
    ]
    loop = rc.probe_gateway(url, "codex", "m", "t")[1]
    assert loop.ok and "loop was not tested" in loop.detail


def test_cli_dry_run_writes_nothing_and_hides_the_key(tmp_path, gateway, monkeypatch):
    url, _, _ = gateway
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_SR_API_KEY", "sk-secret")
    monkeypatch.setattr("shutil.which", lambda _: None)
    result = CliRunner().invoke(
        main, ["connect", "claude-code", "--base-url", url, "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "sk-secret" not in result.output
    assert '"ANTHROPIC_AUTH_TOKEN": "<$VLLM_SR_API_KEY>"' in result.output
    assert not (tmp_path / "settings.json").exists()


def test_cli_writes_probes_and_undoes(tmp_path, gateway, monkeypatch):
    url, seen, _ = gateway
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    monkeypatch.delenv("VLLM_SR_API_KEY", raising=False)
    result = CliRunner().invoke(main, ["connect", "codex", "--base-url", url, "--yes"])
    assert result.exit_code == 0, result.output
    assert "Basic reply: ok" in result.output
    assert (
        "  note: stop_sequence approximated: stop_sequence, absent, read as null\n"
        in result.output
    )
    assert "Streamed tool loop: ok" in result.output
    assert seen[-1][0] == "/v1/responses"
    # Codex refuses to start while a provider's env_key names an unset variable,
    # so a listener without keys gets a profile without one
    assert "env_key" not in (tmp_path / "vllm-sr.config.toml").read_text()
    result = CliRunner().invoke(main, ["connect", "codex", "--undo"])
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "vllm-sr.config.toml").exists()

    monkeypatch.setenv("VLLM_SR_API_KEY", "sk-test")
    result = CliRunner().invoke(main, ["connect", "codex", "--base-url", url, "--yes"])
    assert result.exit_code == 0, result.output
    assert (
        'env_key = "VLLM_SR_API_KEY"' in (tmp_path / "vllm-sr.config.toml").read_text()
    )
    assert seen[-1][2] == "Bearer sk-test"


def test_cli_leaves_the_login_to_claude_code_for_a_caller_gateway(
    tmp_path, gateway, monkeypatch
):
    url, seen, _ = gateway
    config = tmp_path / "config.yaml"
    backend = {
        "name": "anthropic",
        "provider": "anthropic",
        "endpoint": "https://api.anthropic.com",
        "auth": {"credential": "caller"},
    }
    config.write_text(
        yaml.safe_dump(
            {
                "providers": {
                    "models": [{"name": "claude-sonnet-5-5", "backend_refs": [backend]}]
                }
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("VLLM_SR_API_KEY", "sk-secret")
    monkeypatch.setattr("shutil.which", lambda _: None)
    asked = len(seen)
    result = CliRunner().invoke(
        main,
        ["connect", "claude-code", "--config", str(config), "--base-url", url, "--yes"],
    )
    assert result.exit_code == 0, result.output
    written = (tmp_path / "claude" / "settings.json").read_text()
    # empty, so neither a placeholder nor a token the shell exports replaces the login
    assert json.loads(written)["env"]["ANTHROPIC_AUTH_TOKEN"] == ""
    assert "sk-secret" not in written
    assert "Not probed" in result.output
    assert len(seen) == asked


def test_cli_fails_when_the_gateway_refuses(tmp_path, gateway, monkeypatch):
    url, _, _ = gateway
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    result = CliRunner().invoke(
        main, ["connect", "codex", "--base-url", url + "/nope", "--yes"]
    )
    assert result.exit_code != 0
    assert "Basic reply: failed" in result.output


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
        ["connect", "claude-code", "--scope", "project", "--base-url", url, "--yes"],
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


def test_cli_refuses_a_key_in_a_settings_file_git_tracks(
    tmp_path, gateway, monkeypatch
):
    url, _, _ = gateway
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)
    settings = tmp_path / ".claude" / "settings.local.json"
    settings.parent.mkdir()
    settings.write_text("{}")
    subprocess.run(["git", "add", "-f", str(settings)], check=True)
    monkeypatch.setenv("VLLM_SR_API_KEY", "sk-secret")
    monkeypatch.setattr("shutil.which", lambda _: None)
    result = CliRunner().invoke(
        main,
        ["connect", "claude-code", "--scope", "project", "--base-url", url, "--yes"],
    )
    assert result.exit_code != 0
    assert "git tracks" in result.output
    assert settings.read_text() == "{}"


def test_cli_refuses_a_key_in_an_undo_manifest_git_tracks(
    tmp_path, gateway, monkeypatch
):
    url, _, _ = gateway
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)
    manifest = tmp_path / ".claude" / hc.MANIFEST
    manifest.parent.mkdir()
    manifest.write_text('{"files": []}')
    subprocess.run(["git", "add", "-f", str(manifest)], check=True)
    monkeypatch.setenv("VLLM_SR_API_KEY", "sk-secret")
    monkeypatch.setattr("shutil.which", lambda _: None)
    result = CliRunner().invoke(
        main,
        ["connect", "claude-code", "--scope", "project", "--base-url", url, "--yes"],
    )
    assert result.exit_code != 0
    assert f"git tracks {manifest}" in result.output
    assert manifest.read_text() == '{"files": []}'
    assert not (tmp_path / ".claude" / "settings.local.json").exists()


def test_cli_context_window_reaches_both_harnesses(tmp_path, gateway, monkeypatch):
    url, _, _ = gateway
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    monkeypatch.setattr("shutil.which", lambda _: None)
    args = ["--base-url", url, "--yes"]
    result = CliRunner().invoke(main, ["connect", "claude-code", *args])
    assert "assumes a window of its own" in result.output
    result = CliRunner().invoke(
        main, ["connect", "claude-code", *args, "--context-window", "131072"]
    )
    assert result.exit_code == 0, result.output
    assert "assumes a window of its own" not in result.output
    env = json.loads((tmp_path / "settings.json").read_text())["env"]
    # Claude Code neither warns about vllm-sr/auto nor assumes its own window
    assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "131072"
    result = CliRunner().invoke(
        main, ["connect", "codex", *args, "--context-window", "131072"]
    )
    assert result.exit_code == 0, result.output
    profile = (tmp_path / "vllm-sr.config.toml").read_text()
    assert "model_context_window = 131072\n" in profile


def test_cli_takes_the_gateway_from_the_configurations_listener(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr("shutil.which", lambda _: None)
    (tmp_path / "config.yaml").write_text(
        "version: v0.5\nlisteners:\n  - {name: http, address: 0.0.0.0, port: 8899}\n"
    )
    host, port = chat_client.resolve_listener_host_port("config.yaml")
    result = CliRunner().invoke(main, ["connect", "claude-code", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert f'"ANTHROPIC_BASE_URL": "http://{host}:{port}"' in result.output


def test_vsr_is_pointed_at_the_listeners_system_one(monkeypatch, capsys):
    connect_command = importlib.import_module("cli.commands.connect")
    doc = {
        "listeners": [{"port": 8899, "apis": {"systemone": {"models": ["vela3-fast"]}}}]
    }
    inputs = {"router": "false", "runtimeUrl": "http://127.0.0.1:8790/v1/decisions"}
    monkeypatch.setattr(connect_command, "_vsr_plugin", lambda env: {"inputs": inputs})
    connect_command._check_vsr({}, "http://localhost:8899", doc)
    shown = capsys.readouterr().out
    assert '"runtimeUrl": "http://localhost:8899/v1/decisions"' in shown
    assert '"runtimeModel": "vela3-fast"' in shown

    # System One checks the listener's key too; the shell fills it in
    connect_command._check_vsr({}, "http://localhost:8899", doc, "VLLM_SR_API_KEY")
    shown = capsys.readouterr().out
    assert (
        '"runtimeToken": "%s"}\' "$VLLM_SR_API_KEY" | claude plugin configure' in shown
    )

    inputs["runtimeUrl"] = "http://localhost:8899/v1/decisions"
    connect_command._check_vsr({}, "http://localhost:8899", doc)
    assert "System One" not in capsys.readouterr().out


def test_status_line_names_the_models_that_answered(tmp_path):
    script = tmp_path / "status.py"
    script.write_text(hc.STATUS_LINE_SOURCE)
    transcript = tmp_path / "session.jsonl"
    lines = [
        {"type": "user", "message": {"role": "user", "content": "hi"}},
        {
            "type": "assistant",
            "timestamp": "T1",
            "message": {"id": "m1", "model": "qwen36"},
        },
        {
            "type": "assistant",
            "timestamp": "T2",
            "message": {"id": "m2", "model": "qwen38-27b"},
        },
        # one reply is written as one line per content block
        {
            "type": "assistant",
            "timestamp": "T2",
            "message": {"id": "m2", "model": "qwen38-27b"},
        },
        {
            "type": "assistant",
            "timestamp": "T3",
            "message": {"id": "m3", "model": "qwen38-27b"},
        },
    ]
    transcript.write_text("\n".join(json.dumps(line) for line in lines) + "\n")

    def run(session):
        return subprocess.run(
            [sys.executable, str(script)],
            input=json.dumps(session),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    assert (
        run({"transcript_path": str(transcript)})
        == "vllm-sr ⇢ qwen38-27b (2) · qwen36 (1)"
    )
    # a subagent keeps its replies in its own transcript beside the session's
    subagent = tmp_path / "session" / "subagents" / "agent-a1.jsonl"
    subagent.parent.mkdir(parents=True)
    replies = [
        ("T4", "s1", "haiku-like"),
        ("T5", "s2", "qwen36"),
        ("T6", "s3", "qwen36"),
    ]
    subagent.write_text(
        "".join(
            json.dumps(
                {"type": "assistant", "timestamp": t, "message": {"id": i, "model": m}}
            )
            + "\n"
            for t, i, m in replies
        )
    )
    assert (
        run({"transcript_path": str(transcript)})
        == "vllm-sr ⇢ qwen36 (3) · haiku-like (1) · qwen38-27b (2)"
    )
    assert (
        run({"transcript_path": str(tmp_path / "missing.jsonl")})
        == "vllm-sr: no reply yet"
    )
    assert run({}) == "vllm-sr"


def test_cli_status_line_is_written_and_taken_back(tmp_path, gateway, monkeypatch):
    url, _, _ = gateway
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr("shutil.which", lambda _: None)
    own = {"type": "command", "command": "my-status"}
    (tmp_path / "settings.json").write_text(json.dumps({"statusLine": own}))
    result = CliRunner().invoke(
        main, ["connect", "claude-code", "--base-url", url, "--status-line", "--yes"]
    )
    assert result.exit_code == 0, result.output
    settings = json.loads((tmp_path / "settings.json").read_text())
    script = tmp_path / hc.STATUS_LINE_SCRIPT
    assert settings["statusLine"]["command"] == f'python3 "{script}"'
    assert script.read_text() == hc.STATUS_LINE_SOURCE

    result = CliRunner().invoke(main, ["connect", "claude-code", "--undo"])
    assert result.exit_code == 0, result.output
    assert json.loads((tmp_path / "settings.json").read_text()) == {"statusLine": own}
    assert not script.exists()


ROUTER = """\
version: v0.5
listeners:
  - {name: http, address: 127.0.0.1, port: 8899}
providers:
  defaults: {model: small}
  models:
    - {name: small, api_format: chat_completions, context_window_size: 32768}
    - {name: big, api_format: messages, context_window_size: 131072}
    - {name: card, catalog: qwen/qwen3.8-27b, api_format: messages}
    - {name: painter, api_format: images, context_window_size: 77}
recipes:
  - name: default
    decisions:
      - {name: hard, modelRefs: [{model: big}, {model: card}]}
      - {name: draw, modelRefs: [{model: painter}]}
"""


def test_the_entrypoint_reaches_its_recipes_models_and_the_default():
    doc = yaml.safe_load(ROUTER)
    # vllm-sr/auto is the Router's own name for the default recipe; an image model
    # never answers a harness's turn
    assert rc.entrypoint_models(doc, "vllm-sr/auto") == ["big", "card", "small"]
    assert rc.entrypoint_models(doc, "big") == ["big"]
    assert rc.entrypoint_models(doc, "elsewhere") is None
    windows = rc.context_windows(doc, ["big", "card", "small"])
    # a catalog-backed model without a window of its own takes its card's
    assert windows["big"] == 131072 and windows["card"] > 0
    # a chat entrypoint that maps the default recipe takes the name away
    doc["entrypoints"] = [{"model_names": ["team/auto"], "recipe": "default"}]
    assert rc.entrypoint_models(doc, "vllm-sr/auto") is None
    assert rc.entrypoint_models(doc, "team/auto") == ["big", "card", "small"]


def test_a_model_a_decision_iterates_over_counts_toward_the_window():
    doc = yaml.safe_load(ROUTER)
    doc["providers"]["models"].append({"name": "tiny", "context_window_size": 4096})
    hard = doc["recipes"][0]["decisions"][0]
    hard["candidateIterations"] = [{"variable": "m", "models": [{"model": "tiny"}]}]
    assert "tiny" in rc.entrypoint_models(doc, "vllm-sr/auto")


def test_cli_reads_the_context_window_from_the_configuration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    monkeypatch.setattr("shutil.which", lambda _: None)
    (tmp_path / "config.yaml").write_text(ROUTER)
    result = CliRunner().invoke(main, ["connect", "codex", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "model_context_window = 32768" in result.output
    assert (
        "Context window: 32,768 tokens, the smallest of the 3 models" in result.output
    )
    result = CliRunner().invoke(main, ["connect", "claude-code", "--dry-run"])
    assert '"CLAUDE_CODE_MAX_CONTEXT_TOKENS": "32768"' in result.output
    # the flag wins
    result = CliRunner().invoke(
        main, ["connect", "codex", "--dry-run", "--context-window", "8000"]
    )
    assert "model_context_window = 8000" in result.output
    assert "Context window:" not in result.output
    # a model without a window is named, and nothing is guessed
    (tmp_path / "config.yaml").write_text(
        ROUTER.replace(", context_window_size: 32768", "")
    )
    result = CliRunner().invoke(main, ["connect", "codex", "--dry-run"])
    assert "model_context_window" not in result.output
    assert "not set: small, which vllm-sr/auto can reach, states no" in result.output


def test_a_shell_api_key_stays_off_the_gateway_and_is_never_shown(
    tmp_path, gateway, monkeypatch
):
    url, _, _ = gateway
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-shell")
    monkeypatch.setattr("shutil.which", lambda _: None)
    (tmp_path / "settings.json").write_text(
        '{"env": {"ANTHROPIC_API_KEY": "sk-ant-saved"}}'
    )
    result = CliRunner().invoke(
        main, ["connect", "claude-code", "--base-url", url, "--yes"]
    )
    assert result.exit_code == 0, result.output
    assert "sk-ant-" not in result.output
    assert '"ANTHROPIC_API_KEY": "<hidden>"' in result.output
    assert "ANTHROPIC_API_KEY your shell exports no longer goes to the gateway" in (
        result.output
    )
    env = json.loads((tmp_path / "settings.json").read_text())["env"]
    assert env["ANTHROPIC_API_KEY"] == ""
    CliRunner().invoke(main, ["connect", "claude-code", "--undo"])
    env = json.loads((tmp_path / "settings.json").read_text())["env"]
    assert env == {"ANTHROPIC_API_KEY": "sk-ant-saved"}


def test_settings_that_win_over_what_connect_wrote_are_named(
    tmp_path, gateway, monkeypatch
):
    url, _, _ = gateway
    project = tmp_path / "project"
    (project / ".claude").mkdir(parents=True)
    (project / ".claude" / "settings.json").write_text(
        '{"env": {"ANTHROPIC_BASE_URL": "https://elsewhere", "OTHER": "1"}}'
    )
    monkeypatch.chdir(project)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    monkeypatch.setattr("shutil.which", lambda _: None)
    result = CliRunner().invoke(
        main, ["connect", "claude-code", "--base-url", url, "--yes"]
    )
    assert result.exit_code == 0, result.output
    assert "settings.json sets ANTHROPIC_BASE_URL: Claude Code started in" in (
        " ".join(result.output.split())
    )

    # Codex reads a trusted project's configuration from the repository's top down
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    (project / "sub").mkdir()
    (project / ".codex").mkdir()
    (project / ".codex" / "config.toml").write_text('model = "gpt-x"\n')
    monkeypatch.chdir(project / "sub")
    result = CliRunner().invoke(main, ["connect", "codex", "--base-url", url, "--yes"])
    assert result.exit_code == 0, result.output
    assert f"{project / '.codex' / 'config.toml'} sets model" in " ".join(
        result.output.split()
    )
    # a key of the same name inside a table is not the model
    (project / ".codex" / "config.toml").write_text('[tui]\nmodel = "x"\n')
    assert hc.codex_model_overrides(project / "sub") == []


@pytest.fixture
def harness_bin(tmp_path, monkeypatch):
    """A claude and a codex on PATH that record their arguments, the settings file
    they were given, and the key in their environment, then exit 3."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "ran.json"
    for name in ("claude", "codex"):
        script = bin_dir / name
        script.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "args = sys.argv[1:]\n"
            "given = args[args.index('--settings') + 1] if '--settings' in args else None\n"
            f"json.dump({{'args': args, 'settings': open(given).read() if given else None,"
            f" 'key': os.environ.get('VLLM_SR_API_KEY')}}, open({str(record)!r}, 'w'))\n"
            "sys.exit(3)\n"
        )
        script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    return record


def test_launch_starts_the_harness_through_the_gateway_and_writes_nothing(
    tmp_path, gateway, harness_bin, monkeypatch
):
    url, _, _ = gateway
    home = tmp_path / "home"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(home))
    monkeypatch.setenv("CODEX_HOME", str(home))
    monkeypatch.setenv("VLLM_SR_API_KEY", "sk-secret")
    launch = ["--base-url", url, "--context-window", "65536", "--launch", "--"]
    result = CliRunner().invoke(
        main, ["connect", "claude-code", *launch, "--resume", "x"]
    )
    assert result.exit_code == 3, result.output
    assert "Streamed tool loop: ok" in result.output
    ran = json.loads(harness_bin.read_text())
    assert ran["args"][-2:] == ["--resume", "x"]
    settings = json.loads(ran["settings"])["env"]
    assert settings["ANTHROPIC_BASE_URL"] == url
    assert settings["ANTHROPIC_AUTH_TOKEN"] == "sk-secret"
    assert settings["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "65536"
    # the key never reaches the command line, and its file is gone afterwards
    assert "sk-secret" not in " ".join(ran["args"]) + result.output
    assert not Path(ran["args"][ran["args"].index("--settings") + 1]).exists()

    result = CliRunner().invoke(main, ["connect", "codex", *launch, "exec", "hi"])
    assert result.exit_code == 3, result.output
    ran = json.loads(harness_bin.read_text())
    assert ran["args"][-2:] == ["exec", "hi"]
    overrides = dict(arg.split("=", 1) for arg in ran["args"][1:-2:2] if "=" in arg)
    assert overrides["model_providers.vllm-sr.base_url"] == json.dumps(url + "/v1")
    assert overrides["model_providers.vllm-sr.env_key"] == '"VLLM_SR_API_KEY"'
    assert overrides["model_context_window"] == "65536"
    assert ran["key"] == "sk-secret" and "sk-secret" not in " ".join(ran["args"])
    assert not home.exists()


def test_launch_starts_nothing_when_the_gateway_fails(
    tmp_path, gateway, harness_bin, monkeypatch
):
    url, _, _ = gateway
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "home"))
    result = CliRunner().invoke(
        main, ["connect", "codex", "--base-url", url + "/nope", "--launch"]
    )
    assert result.exit_code == 1
    assert "codex was not started" in result.output
    assert not harness_bin.exists()
    result = CliRunner().invoke(main, ["connect", "codex", "--launch", "--undo"])
    assert result.exit_code == 2 and "does not take --undo" in result.output
    result = CliRunner().invoke(main, ["connect", "codex", "--", "exec"])
    assert result.exit_code == 2 and "only with --launch" in result.output


class FakeClaudePlugins:
    """`claude plugin` as the commands --vsr runs see it: the vSR plugin's options,
    with the token kept apart as Claude Code's credential store keeps it."""

    def __init__(self, installed):
        self.installed = installed
        # what `configure --json` gives before any option is set: an unset string empty
        self.inputs = {
            "runtime": "local-cpu",
            "runtimeUrl": "",
            "runtimeModel": "",
            "router": "true",
            "guard": "true",
        }
        self.configured = set()
        self.schema = {
            "runtimeUrl": {"default": "http://127.0.0.1:8790/v1/decisions"},
            "router": {"default": True},
        }
        self.token = None
        self.ran = []
        self.unreadable = False

    def __call__(self, argv, **kw):
        self.ran.append(argv)
        out = ""
        if argv[1:3] == ["plugin", "list"]:
            out = json.dumps(
                [{"id": "vsr@vllm-sr", "enabled": True}] if self.installed else []
            )
        elif argv[1:3] == ["plugin", "install"]:
            self.installed = True
        elif argv[-1] == "--json" and self.unreadable:
            return subprocess.CompletedProcess(argv, 1, "", "error")
        elif argv[-1] == "--json":
            inputs = {**self.inputs, "runtimeToken": ""}
            configured = sorted(self.configured)
            found = {"inputs": inputs, "configured": configured, "schema": self.schema}
            out = json.dumps(found)
        elif argv[-1] == "--values-stdin":
            given = json.loads(kw["input"])
            self.token = given.pop("runtimeToken", self.token)
            self.inputs.update(given)
            self.configured |= set(given) | ({"runtimeToken"} if self.token else set())
        return subprocess.CompletedProcess(argv, 0, out, "")


@pytest.mark.parametrize("installed", [False, True])
def test_vsr_sets_up_claude_code_through_its_plugin_command_and_undoes_it(
    tmp_path, monkeypatch, installed
):
    url = "http://localhost:8899"
    command = importlib.import_module("cli.commands.connect")
    monkeypatch.setattr(command, "_probe", lambda *args: None)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_SR_API_KEY", "sk-secret")
    monkeypatch.setattr(command.shutil, "which", lambda name: f"/bin/{name}")
    claude = FakeClaudePlugins(installed)
    monkeypatch.setattr(command.subprocess, "run", claude)
    (tmp_path / "config.yaml").write_text(
        "listeners:\n  - port: 8899\n    apis:\n      systemone:\n"
        "        models: [vela3-lite, second]\n"
    )
    args = ["connect", "claude-code", "--vsr", "--base-url", url]

    result = CliRunner().invoke(main, [*args, "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "claude plugin install vsr@vllm-sr" in result.output or installed
    assert '"runtimeModel": "vela3-lite"' in result.output
    assert '"runtimeToken": "<$VLLM_SR_API_KEY>"' in result.output
    assert "sk-secret" not in result.output
    assert not any("--values-stdin" in argv for argv in claude.ran)

    result = CliRunner().invoke(main, [*args, "--yes"])
    assert result.exit_code == 0, result.output
    installs = [argv for argv in claude.ran if argv[2] in ("marketplace", "install")]
    assert installs == (
        [] if installed else command._installs("claude", command.VSR_SOURCE)
    )
    # the gateway routes, so the plugin's router steps aside; the token goes to the credential store
    assert claude.inputs == {
        "runtime": "remote",
        "runtimeUrl": url + "/v1/decisions",
        "runtimeModel": "vela3-lite",
        "router": "false",
        "guard": "true",
    }
    assert claude.token == "sk-secret"
    assert "sk-secret" not in (tmp_path / "settings.json").read_text().replace(
        json.loads((tmp_path / "settings.json").read_text())["env"][
            "ANTHROPIC_AUTH_TOKEN"
        ],
        "",
    )

    result = CliRunner().invoke(main, ["connect", "claude-code", "--undo"])
    assert result.exit_code == 0, result.output
    assert claude.inputs["runtime"] == "local-cpu" and claude.inputs["router"] == "true"
    assert claude.inputs["runtimeUrl"] == "http://127.0.0.1:8790/v1/decisions"


class FakeCodexPlugins:
    """`codex plugin` as connect runs it: what is installed, and what it ran."""

    def __init__(self, installed, unreachable=False):
        self.installed = installed
        self.unreachable = unreachable
        self.ran = []

    def __call__(self, argv, **kw):
        self.ran.append(argv)
        out = ""
        if argv[1:3] == ["plugin", "list"]:
            listed = (
                [{"pluginId": "vsr@vllm-sr", "enabled": True}] if self.installed else []
            )
            out = json.dumps({"installed": listed, "available": []})
        elif argv[1:3] == ["plugin", "add"]:
            self.installed = True
        elif argv[2:4] == ["marketplace", "add"] and self.unreachable:
            raise subprocess.CalledProcessError(1, argv)
        return subprocess.CompletedProcess(argv, 0, out, "")


def _codex_env(tmp_path, monkeypatch, codex):
    command = importlib.import_module("cli.commands.connect")
    monkeypatch.setattr(command, "_probe", lambda *args: None)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    monkeypatch.setattr(command.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(command.subprocess, "run", codex)


def test_connect_installs_the_codex_plugin_from_its_source_unless_it_is_there(
    tmp_path, monkeypatch
):
    codex = FakeCodexPlugins(installed=False)
    _codex_env(tmp_path, monkeypatch, codex)
    (tmp_path / "config.yaml").write_text(
        "listeners:\n  - port: 8899\n    apis:\n      systemone:\n        models: [vela3-lite]\n"
    )
    args = ["connect", "codex", "--base-url", "http://localhost:8899"]
    result = CliRunner().invoke(main, [*args, "--vsr-source", "/src/sr", "--yes"])
    assert result.exit_code == 0, result.output
    installs = [argv for argv in codex.ran if argv[2] in ("marketplace", "add")]
    assert installs == [
        ["codex", "plugin", "marketplace", "add", "/src/sr"],
        ["codex", "plugin", "add", "vsr@vllm-sr"],
    ]
    codex.ran.clear()
    assert CliRunner().invoke(main, [*args, "--yes"]).exit_code == 0
    assert [argv for argv in codex.ran if argv[2] in ("marketplace", "add")] == []
    # --no-vsr leaves vSR alone
    codex.ran.clear()
    (tmp_path / "vsr.json").unlink()
    assert CliRunner().invoke(main, [*args, "--no-vsr", "--yes"]).exit_code == 0
    assert not any(argv[0] == "codex" for argv in codex.ran)
    assert not (tmp_path / "vsr.json").exists()


def test_vsr_points_the_codex_plugin_at_the_listener_and_keeps_its_other_settings(
    tmp_path, monkeypatch
):
    url = "http://localhost:8899"
    _codex_env(tmp_path, monkeypatch, FakeCodexPlugins(installed=True))
    (tmp_path / "config.yaml").write_text(
        "listeners:\n  - port: 8899\n    apis:\n      systemone:\n        models: [vela3-lite]\n"
    )
    held = tmp_path / "vsr.json"
    held.write_text(
        json.dumps(
            {"guardReview": 0.8, "runtimeUrl": "http://127.0.0.1:8790/v1/decisions"}
        )
    )
    args = ["connect", "codex", "--vsr", "--base-url", url]

    result = CliRunner().invoke(main, [*args, "--yes"])
    assert result.exit_code == 0, result.output
    assert "the Codex plugin asks vela3-lite" in result.output
    assert json.loads(held.read_text()) == {
        "guardReview": 0.8,
        "runtimeUrl": url + "/v1/decisions",
        "runtimeModel": "vela3-lite",
    }
    result = CliRunner().invoke(main, ["connect", "codex", "--undo"])
    assert result.exit_code == 0, result.output
    assert json.loads(held.read_text()) == {
        "guardReview": 0.8,
        "runtimeUrl": "http://127.0.0.1:8790/v1/decisions",
    }


def test_a_codex_plugin_that_does_not_install_is_not_reported_as_set_up(
    tmp_path, monkeypatch
):
    codex = FakeCodexPlugins(installed=False, unreachable=True)
    _codex_env(tmp_path, monkeypatch, codex)
    (tmp_path / "config.yaml").write_text(
        "listeners:\n  - port: 8899\n    apis:\n      systemone:\n        models: [vela3-lite]\n"
    )
    args = ["connect", "codex", "--base-url", "http://localhost:8899"]
    result = CliRunner().invoke(main, [*args, "--vsr-source", "/gone", "--yes"])
    assert result.exit_code == 0, result.output
    assert (
        "vSR is not set up: codex plugin marketplace add /gone exited with status 1"
        in result.output
    )
    assert "the Codex plugin asks" not in result.output
    assert ["codex", "plugin", "add", "vsr@vllm-sr"] not in codex.ran


def test_a_keyed_listener_names_its_key_variable_to_the_codex_plugin(
    tmp_path, monkeypatch
):
    _codex_env(tmp_path, monkeypatch, FakeCodexPlugins(installed=True))
    monkeypatch.setenv("VLLM_SR_API_KEY", "listener-key")
    (tmp_path / "config.yaml").write_text(
        "listeners:\n  - port: 8899\n    api_keys: [listener-key]\n"
        "    apis:\n      systemone:\n        models: [vela3-lite]\n"
    )
    result = CliRunner().invoke(
        main, ["connect", "codex", "--base-url", "http://localhost:8899", "--yes"]
    )
    assert result.exit_code == 0, result.output
    held = (tmp_path / "vsr.json").read_text()
    assert json.loads(held)["runtimeTokenEnv"] == "VLLM_SR_API_KEY"
    assert "listener-key" not in held
    assert (
        "Codex and its vSR plugin read the key from $VLLM_SR_API_KEY" in result.output
    )
    # a listener without a key, later: the variable named for the first one goes
    monkeypatch.delenv("VLLM_SR_API_KEY")
    (tmp_path / "config.yaml").write_text(
        "listeners:\n  - port: 8899\n    apis:\n      systemone:\n        models: [vela3-lite]\n"
    )
    result = CliRunner().invoke(
        main, ["connect", "codex", "--base-url", "http://localhost:8899", "--yes"]
    )
    assert result.exit_code == 0, result.output
    assert "runtimeTokenEnv" not in (tmp_path / "vsr.json").read_text()


@pytest.mark.parametrize("harness", ["claude-code", "codex"])
def test_vsr_without_a_model_and_with_launch(tmp_path, monkeypatch, harness):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    args = ["connect", harness, "--vsr"]
    result = CliRunner().invoke(
        main, [*args, "--base-url", "http://localhost:8899", "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "publishes no model" in result.output


def test_without_claude_on_path_the_gateway_is_set_up_and_vsr_is_left_out(
    tmp_path, monkeypatch
):
    command = importlib.import_module("cli.commands.connect")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(command.shutil, "which", lambda _: None)
    (tmp_path / "config.yaml").write_text(
        "listeners:\n  - port: 8899\n    apis:\n      systemone:\n        models: [vela3-lite]\n"
    )
    monkeypatch.setattr(command, "_probe", lambda *args: None)
    result = CliRunner().invoke(main, ["connect", "claude-code", "--yes"])
    # the gateway settings are written, and vSR says why it is left out
    assert result.exit_code == 0, result.output
    assert "vSR is not set up: claude is not on PATH" in result.output
    assert (tmp_path / "settings.json").exists()


def test_a_credential_in_a_file_connect_owns_is_never_shown(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    monkeypatch.setattr("shutil.which", lambda _: None)
    (tmp_path / "vllm-sr.config.toml").write_text(
        'model = "vllm-sr/auto"\n'
        'experimental_bearer_token = "sk-live-123"\n'
        "[model_providers.vllm-sr]\n"
        'http_headers = { "X-Team" = "Bearer sk-live-456" }\n'
        'env_key = "VLLM_SR_API_KEY"\n'
    )
    result = CliRunner().invoke(
        main, ["connect", "codex", "--base-url", "http://localhost:8899", "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "sk-live" not in result.output
    assert 'experimental_bearer_token = "<hidden>"' in result.output
    # an env_key only names the variable, and stays readable
    assert 'env_key = "VLLM_SR_API_KEY"' in result.output


def test_a_project_that_sets_the_api_key_gets_no_promise_that_it_is_off(
    tmp_path, monkeypatch
):
    command = importlib.import_module("cli.commands.connect")
    monkeypatch.setattr(command, "_probe", lambda *args: None)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "home"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-shell")
    monkeypatch.setattr("shutil.which", lambda _: None)
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "settings.json").write_text(
        json.dumps({"env": {"ANTHROPIC_API_KEY": "sk-ant-project"}})
    )
    result = CliRunner().invoke(
        main,
        ["connect", "claude-code", "--base-url", "http://localhost:8899", "--yes"],
    )
    assert result.exit_code == 0, result.output
    # The warning wraps at the output width, and the temp path's length moves the break.
    output = " ".join(result.output.split())
    assert "no longer goes to the gateway" not in output
    assert "sets ANTHROPIC_API_KEY" in output


def test_a_dry_run_launch_writes_no_settings_file(tmp_path, monkeypatch):
    command = importlib.import_module("cli.commands.connect")
    written = []
    monkeypatch.setattr(command.hc, "write_private", lambda *args: written.append(args))
    monkeypatch.setattr(command.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        main,
        [
            "connect",
            "claude-code",
            "--base-url",
            "http://localhost:8899",
            "--launch",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "--settings" in result.output and written == []


def _vsr_claude_env(tmp_path, monkeypatch, claude):
    command = importlib.import_module("cli.commands.connect")
    monkeypatch.setattr(command, "_probe", lambda *args: None)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(command.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(command.subprocess, "run", claude)
    (tmp_path / "config.yaml").write_text(
        "listeners:\n  - port: 8899\n    apis:\n      systemone:\n"
        "        models: [vela3-lite]\n"
    )
    return ["connect", "claude-code", "--vsr", "--base-url", "http://localhost:8899"]


def test_a_stored_token_is_not_sent_to_another_listener(tmp_path, monkeypatch):
    claude = FakeClaudePlugins(installed=True)
    claude.token, claude.configured = "sk-old", {"runtimeToken"}
    monkeypatch.delenv("VLLM_SR_API_KEY", raising=False)
    args = _vsr_claude_env(tmp_path, monkeypatch, claude)
    result = CliRunner().invoke(main, [*args, "--yes"])
    assert result.exit_code == 0, result.output
    assert "holds a runtime token from before" in result.output
    assert not any(argv[-1] == "--values-stdin" for argv in claude.ran)
    assert claude.inputs["runtimeUrl"] == ""


def test_a_plugin_whose_options_cannot_be_read_is_left_alone(tmp_path, monkeypatch):
    claude = FakeClaudePlugins(installed=True)
    claude.unreadable = True
    monkeypatch.setenv("VLLM_SR_API_KEY", "sk-secret")
    args = _vsr_claude_env(tmp_path, monkeypatch, claude)
    result = CliRunner().invoke(main, [*args, "--yes"])
    assert result.exit_code == 0 and "gave no options" in result.output
    assert not any(argv[-1] == "--values-stdin" for argv in claude.ran)


def test_undo_that_cannot_read_the_plugin_undoes_nothing(tmp_path, monkeypatch):
    claude = FakeClaudePlugins(installed=True)
    monkeypatch.setenv("VLLM_SR_API_KEY", "sk-secret")
    args = _vsr_claude_env(tmp_path, monkeypatch, claude)
    assert CliRunner().invoke(main, [*args, "--yes"]).exit_code == 0
    written = (tmp_path / "settings.json").read_text()
    claude.unreadable = True
    result = CliRunner().invoke(main, ["connect", "claude-code", "--undo"])
    assert result.exit_code == 1
    assert (tmp_path / "settings.json").read_text() == written
    assert (tmp_path / ".vllm-sr-connect.json").exists()


def test_undo_restores_the_runtime_keys_of_vsr_json_and_keeps_later_edits(
    tmp_path, monkeypatch
):
    _codex_env(tmp_path, monkeypatch, FakeCodexPlugins(installed=True))
    (tmp_path / "config.yaml").write_text(
        "listeners:\n  - port: 8899\n    apis:\n      systemone:\n"
        "        models: [vela3-lite]\n"
    )
    held = tmp_path / "vsr.json"
    held.write_text(json.dumps({"guardReview": 0.8}))
    args = ["connect", "codex", "--vsr", "--base-url", "http://localhost:8899"]
    assert CliRunner().invoke(main, [*args, "--yes"]).exit_code == 0
    held.write_text(json.dumps({**json.loads(held.read_text()), "guardReview": 0.95}))
    result = CliRunner().invoke(main, ["connect", "codex", "--undo"])
    assert result.exit_code == 0, result.output
    assert json.loads(held.read_text()) == {"guardReview": 0.95}


def test_vsr_asks_the_listener_the_gateway_url_names():
    doc = {
        "listeners": [
            {"port": 8899, "apis": {"systemone": {"models": ["first"]}}},
            {"port": 9900, "apis": {"systemone": {"models": ["second"]}}},
        ]
    }
    assert rc.system_one_models(doc) == ["first"]
    assert rc.system_one_models(doc, 9900) == ["second"]
    assert rc.system_one_models(doc, 7000) == []


def test_started_in_the_home_directory_the_user_settings_are_not_their_own_override(
    tmp_path, monkeypatch
):
    command = importlib.import_module("cli.commands.connect")
    monkeypatch.setattr(command, "_probe", lambda *args: None)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / ".claude"))
    monkeypatch.setattr("shutil.which", lambda _: None)
    args = ["connect", "claude-code", "--base-url", "http://localhost:8899", "--yes"]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert "takes those over" not in result.output
