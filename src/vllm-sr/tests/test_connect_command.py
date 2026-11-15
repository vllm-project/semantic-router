import importlib
import json
import os
import signal
import stat
import subprocess
import sys
import threading
import urllib.parse
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
    for a plain request, events for a streamed one. Its replay list, as the
    Router's management API serves it, holds one record for each request."""
    seen = []
    replies = {path: dict(stages) for path, stages in STAGES.items()}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            path, _, query = self.path.partition("?")
            wanted = urllib.parse.parse_qs(query).get("model", [None])[0]
            records = [
                {"id": f"r{index}", "original_model": body.get("model")}
                for index, (_, body, _) in enumerate(seen)
                if wanted in (None, body.get("model"))
            ]
            found = path == "/api/v1/observability/replays"
            self.send_response(200 if found else 404)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"data": records} if found else {}).encode())

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


def test_codex_findings_cover_only_the_models_the_entrypoint_reaches():
    doc = _router()
    doc["providers"]["models"].append({"name": "c", "api_format": "openai"})
    doc["recipes"] = [
        {"name": "agent", "decisions": [{"name": "d", "modelRefs": [{"model": "m"}]}]},
        {"name": "chat", "decisions": [{"name": "e", "modelRefs": [{"model": "c"}]}]},
    ]
    doc["entrypoints"] = [
        {"model_names": ["vllm-sr/auto"], "recipe": "agent", "default": True},
        {"model_names": ["vllm-sr/chat"], "recipe": "chat"},
    ]

    def signature(model):
        needs = [f.need for f in rc.check_router_config(doc, "codex", model)]
        return any(n.startswith("reasoning.signature") for n in needs)

    # vllm-sr/chat reaches no anthropic model, so Codex on it needs no signature
    assert signature("vllm-sr/auto") and signature(None)
    assert not signature("vllm-sr/chat")


def test_a_preview_cuts_a_long_line_short():
    shown = hc.hide_secrets('{"base_instructions": "' + "x" * 5000 + '"}\n')
    assert shown.endswith("... (4,865 more characters)\n")
    assert len(shown) < 200


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


def _caller_config(tmp_path, provider):
    config = tmp_path / "config.yaml"
    backend = {"name": provider, "provider": provider, "auth": {"credential": "caller"}}
    model = {"name": "a-model", "backend_refs": [backend]}
    config.write_text(yaml.safe_dump({"providers": {"models": [model]}}))
    return config


def test_cli_leaves_the_login_to_codex_for_a_chatgpt_caller_gateway(
    tmp_path, gateway, harness_bin, monkeypatch
):
    url, seen, _ = gateway
    config = _caller_config(tmp_path, "openai")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    monkeypatch.setenv("VLLM_SR_API_KEY", "sk-secret")
    asked = len(seen)
    result = CliRunner().invoke(
        main, ["connect", "codex", "--config", str(config), "--base-url", url, "--yes"]
    )
    assert result.exit_code == 0, result.output
    written = (tmp_path / "vllm-sr.config.toml").read_text()
    # Codex sends its own sign-in only when its provider asks for it, and a key
    # variable would go in its place
    assert "requires_openai_auth = true" in written
    assert "env_key" not in written
    assert "Not probed: the gateway answers only on the login Codex" in result.output
    assert len(seen) == asked

    launch = ["--config", str(config), "--base-url", url, "--launch", "--", "exec"]
    result = CliRunner().invoke(main, ["connect", "codex", *launch, "hi"])
    assert result.exit_code == 3, result.output
    ran = json.loads(harness_bin.read_text())
    overrides = dict(arg.split("=", 1) for arg in ran["args"][1:-2:2] if "=" in arg)
    assert overrides["model_providers.vllm-sr.requires_openai_auth"] == "true"
    assert "model_providers.vllm-sr.env_key" not in overrides
    assert len(seen) == asked


@pytest.mark.parametrize(
    ("harness", "provider"), [("codex", "anthropic"), ("claude-code", "openai")]
)
def test_cli_refuses_a_caller_gateway_of_another_provider(
    tmp_path, gateway, monkeypatch, harness, provider
):
    url, seen, _ = gateway
    config = _caller_config(tmp_path, provider)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    asked = len(seen)
    result = CliRunner().invoke(
        main, ["connect", harness, "--config", str(config), "--base-url", url, "--yes"]
    )
    assert result.exit_code != 0
    assert f"own {provider} login upstream" in result.output
    assert not (tmp_path / "vllm-sr.config.toml").exists()
    assert not (tmp_path / "claude").exists()
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


BUNDLED_GPT_55 = {
    "slug": "gpt-5.5",
    "display_name": "GPT-5.5",
    "supports_search_tool": True,
    "apply_patch_tool_type": "freeform",
    "context_window": 272000,
    "max_context_window": 272000,
    "upgrade": {"model": "gpt-6-sol", "migration_markdown": "Meet GPT-6 Sol"},
    "service_tiers": [{"id": "priority", "name": "Fast"}],
    "additional_speed_tiers": ["fast"],
}


@pytest.fixture
def harness_bin(tmp_path, monkeypatch):
    """A claude and a codex on PATH that record their arguments, the settings file
    or model catalog they were given, and the key in their environment, then exit
    3. Asked for its bundled models, the codex lists a gpt-5.5 entry."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "ran.json"
    for name in ("claude", "codex"):
        script = bin_dir / name
        script.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "args = sys.argv[1:]\n"
            "if args[:2] == ['debug', 'models']:\n"
            f"    print(json.dumps({{'models': [{BUNDLED_GPT_55!r}]}}))\n"
            "    sys.exit(0)\n"
            "given = args[args.index('--settings') + 1] if '--settings' in args else None\n"
            "catalog = [json.loads(a.split('=', 1)[1]) for a in args if a.startswith('model_catalog_json=')]\n"
            f"json.dump({{'args': args, 'settings': open(given).read() if given else None,"
            " 'catalog': json.load(open(catalog[0])) if catalog else None,"
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
    settings = json.loads(ran["settings"])
    assert settings["env"]["ANTHROPIC_BASE_URL"] == url
    assert settings["env"]["ANTHROPIC_AUTH_TOKEN"] == ""
    assert settings["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "65536"
    # Claude Code reads the key from its own environment: it never reaches the
    # command line or the settings file, which is gone afterwards
    assert settings["apiKeyHelper"] == "printenv VLLM_SR_API_KEY"
    assert ran["key"] == "sk-secret"
    assert "sk-secret" not in " ".join(ran["args"]) + result.output + ran["settings"]
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
    assert [m["slug"] for m in ran["catalog"]["models"]] == ["vllm-sr/auto"]
    assert not home.exists()


@pytest.mark.parametrize("signum", [signal.SIGHUP, signal.SIGTERM])
def test_launch_passes_a_hangup_on_and_removes_its_files(tmp_path, gateway, signum):
    url, _, _ = gateway
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "ran.json"
    # a Claude Code whose terminal closes: connect is signalled, and the harness
    # exits on the signal connect passes on
    (bin_dir / "claude").write_text(
        f"#!{sys.executable}\n"
        "import json, os, signal, sys, time\n"
        "args = sys.argv[1:]\n"
        "settings = args[args.index('--settings') + 1]\n"
        f"def stop(signum, _):\n"
        f"    json.dump({{'settings': settings, 'signal': signum}}, open({str(record)!r}, 'w'))\n"
        "    sys.exit(128 + signum)\n"
        "signal.signal(signal.SIGHUP, stop)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        f"os.kill(os.getppid(), {int(signum)})\n"
        "time.sleep(10)\n"
    )
    (bin_dir / "claude").chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "CLAUDE_CONFIG_DIR": str(tmp_path / "home"),
        "VLLM_SR_API_KEY": "sk-secret",
    }
    launch = ["connect", "claude-code", "--base-url", url, "--launch"]
    done = subprocess.run(
        [sys.executable, "-m", "cli.main", *launch],
        cwd=tmp_path,
        env={**env, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
        capture_output=True,
        timeout=60,
        check=False,
    )
    ran = json.loads(record.read_text())
    assert ran["signal"] == signum
    assert done.returncode == 128 + signum, done.stderr
    assert not Path(ran["settings"]).parent.exists()


def test_codex_gets_a_catalog_entry_for_the_gateway_model(
    tmp_path, gateway, harness_bin, monkeypatch
):
    url, _, _ = gateway
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    args = ["connect", "codex", "--base-url", url, "--context-window", "65536"]
    result = CliRunner().invoke(main, [*args, "--no-vsr", "--yes"])
    assert result.exit_code == 0, result.output
    catalog = tmp_path / "vllm-sr.models.json"
    assert (
        f"model_catalog_json = {json.dumps(str(catalog))}"
        in (tmp_path / "vllm-sr.config.toml").read_text()
    )
    # Only the served model, adapted from the bundled base for the gateway.
    assert json.loads(catalog.read_text()) == {
        "models": [
            {
                **BUNDLED_GPT_55,
                "slug": "vllm-sr/auto",
                "display_name": "vllm-sr/auto",
                "visibility": "list",
                "supports_search_tool": False,
                "apply_patch_tool_type": None,
                "supported_reasoning_levels": [],
                "default_reasoning_level": None,
                "default_reasoning_summary": "auto",
                "support_verbosity": False,
                "default_verbosity": None,
                "upgrade": None,
                "service_tiers": [],
                "additional_speed_tiers": [],
                "context_window": 65536,
                "max_context_window": 65536,
            },
        ]
    }
    assert "does not know" not in result.output
    assert CliRunner().invoke(main, ["connect", "codex", "--undo"]).exit_code == 0
    assert not catalog.exists()

    # A bundled-name collision still gets gateway metadata and a restricted catalog.
    result = CliRunner().invoke(
        main, [*args, "--model", "gpt-5.5", "--no-vsr", "--yes"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(catalog.read_text())["models"][0]["slug"] == "gpt-5.5"
    assert json.loads(catalog.read_text())["models"][0]["supports_search_tool"] is False
    assert "does not know" not in result.output
    assert CliRunner().invoke(main, ["connect", "codex", "--undo"]).exit_code == 0

    monkeypatch.setattr("shutil.which", lambda _: None)
    result = CliRunner().invoke(main, [*args, "--no-vsr", "--yes"])
    assert result.exit_code == 0, result.output
    assert not catalog.exists()
    assert "model_catalog_json" not in (tmp_path / "vllm-sr.config.toml").read_text()
    assert "bundles no gpt-5.5 entry" in result.output


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
    assert not any(argv[1] == "plugin" for argv in codex.ran)
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
        "managementUrl": "http://localhost:8080",
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


def test_the_codex_plugin_reads_turns_from_the_management_api_the_stack_publishes(
    tmp_path, monkeypatch
):
    _codex_env(tmp_path, monkeypatch, FakeCodexPlugins(installed=True))
    (tmp_path / "config.yaml").write_text(
        "listeners:\n  - port: 8899\n    apis:\n      systemone:\n        models: [vela3-lite]\n"
        "global:\n  services:\n    management_api:\n      port: 9080\n"
    )
    result = CliRunner().invoke(main, ["connect", "codex", "--yes"])
    assert result.exit_code == 0, result.output
    held = json.loads((tmp_path / "vsr.json").read_text())
    assert held["managementUrl"] == "http://localhost:9080"
    # the stack's port offset moves it as it moves the listener
    monkeypatch.setenv("VLLM_SR_PORT_OFFSET", "100")
    command = importlib.import_module("cli.commands.connect")
    assert command._management_url("http://[::1]:8999", {}) == "http://[::1]:8180"


@pytest.mark.parametrize("harness", ["claude-code", "codex"])
def test_a_keyed_listener_without_its_key_is_refused_before_writing(
    tmp_path, monkeypatch, harness
):
    home = tmp_path / "home"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(home))
    monkeypatch.setenv("CODEX_HOME", str(home))
    monkeypatch.delenv("VLLM_SR_API_KEY", raising=False)
    config = tmp_path / "router.yaml"
    config.write_text(
        "version: v0.5\nlisteners:\n"
        "  - {name: http, address: 127.0.0.1, port: 8899, api_keys: [listener-key]}\n"
    )
    args = ["connect", harness, "--config", str(config), "--no-vsr", "--yes"]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 1, result.output
    assert "takes an API key and $VLLM_SR_API_KEY is not set" in result.output
    # a key written before stays: nothing is replaced with a placeholder
    assert not home.exists()


def test_a_second_run_with_nothing_to_change_writes_nothing(
    tmp_path, gateway, monkeypatch
):
    url, _, _ = gateway
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    args = ["connect", "claude-code", "--base-url", url, "--context-window", "65536"]
    args += ["--no-vsr", "--yes"]
    assert CliRunner().invoke(main, args).exit_code == 0
    files = [tmp_path / "settings.json", tmp_path / hc.MANIFEST]
    before = [path.stat().st_mtime_ns for path in files]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert "Nothing to change" in result.output
    assert "Written" not in result.output
    assert [path.stat().st_mtime_ns for path in files] == before


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


@pytest.mark.parametrize("model", ["vllm-sr/auto", "gpt-5.5", "codex-auto-review"])
@pytest.mark.parametrize("review", [None, "vllm-sr/review"])
def test_codex_catalog_exposes_only_gateway_models(model, review):
    levels = [{"effort": "low", "description": "Fast review"}]
    bundled = {
        "models": [
            {
                **BUNDLED_GPT_55,
                "supported_reasoning_levels": levels,
                "auto_review_model_override": "unserved-reviewer",
            },
            {"slug": "codex-auto-review"},
            {"slug": "gpt-6-sol"},
        ]
    }
    before = json.dumps(bundled)
    catalog = hc.codex_catalog(bundled, model, 64000, review, 16000)
    entries = catalog["models"]
    assert [e["slug"] for e in entries] == [model, *([review] if review else [])]
    main = entries[0]
    assert main["visibility"] == "list"
    assert main["supports_search_tool"] is False
    assert main["context_window"] == main["max_context_window"] == 64000
    assert main["supported_reasoning_levels"] == []
    if review:
        assert main["auto_review_model_override"] == review
        reviewer = entries[1]
        assert reviewer["visibility"] == "hide"
        assert reviewer["default_reasoning_level"] == "low"
        assert reviewer["supported_reasoning_levels"] == levels
        assert reviewer["context_window"] == reviewer["max_context_window"] == 16000
        assert "auto_review_model_override" not in reviewer
    else:
        assert "auto_review_model_override" not in main
    assert json.dumps(bundled) == before


@pytest.mark.parametrize("launch", [False, True])
@pytest.mark.parametrize("review", [False, True])
def test_codex_connect_review_catalog_and_login_hint(
    tmp_path, monkeypatch, harness_bin, launch, review
):
    monkeypatch.chdir(tmp_path)
    home = tmp_path / "codex-home"
    monkeypatch.setenv("CODEX_HOME", str(home))
    backend = {
        "endpoint": "https://chatgpt.com/backend-api/codex",
        "provider": "openai",
        "auth": {"credential": "caller"},
    }
    doc = {
        "version": "v0.5",
        "providers": {
            "defaults": {"model": "session"},
            "models": [
                {
                    "name": "session",
                    "context_window_size": 131072,
                    "backend_refs": [backend],
                },
                {
                    "name": "codex-auto-review",
                    "context_window_size": 16384,
                    "backend_refs": [backend],
                },
            ],
        },
        "entrypoints": [
            {
                "model_names": ["vllm-sr/auto"],
                "recipe": "main",
                "default": True,
                **({"review_model": " vllm-sr/review "} if review else {}),
            },
            {"model_names": ["vllm-sr/review"], "recipe": "review"},
        ],
        "recipes": [
            {
                "name": name,
                "decisions": [
                    {
                        "name": name,
                        "priority": 1,
                        "rules": {"operator": "AND", "conditions": []},
                        "modelRefs": [{"model": model}],
                    }
                ],
            }
            for name, model in [("main", "session"), ("review", "codex-auto-review")]
        ],
    }
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(doc))
    args = [
        "connect",
        "codex",
        "--config",
        str(config),
        "--base-url",
        "http://localhost:8899",
        "--context-window",
        "8000",
    ]
    args += ["--launch", "--", "exec", "hi"] if launch else ["--yes", "--no-vsr"]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == (3 if launch else 0), result.output
    if launch:
        catalog = json.loads(harness_bin.read_text())["catalog"]
        assert not home.exists()
    else:
        catalog = json.loads((home / "vllm-sr.models.json").read_text())
        assert not (home / "config.toml").exists()
    assert [e["slug"] for e in catalog["models"]] == [
        "vllm-sr/auto",
        *(["vllm-sr/review"] if review else []),
    ]
    assert catalog["models"][0]["context_window"] == 8000
    destination = "vllm-sr/review" if review else "vllm-sr/auto"
    assert (
        f"Approve for me reviews go to {destination} through the gateway."
        in result.output
    )
    if review:
        assert catalog["models"][1]["context_window"] == 16384
        assert catalog["models"][0]["auto_review_model_override"] == destination
        assert "To use the ChatGPT reviewer" not in result.output
    else:
        assert "auto_review_model_override" not in catalog["models"][0]
        for part in [
            "review_model: vllm-sr/review",
            "recipe: review",
            "name: codex-auto-review",
            "auth: {credential: caller}",
        ]:
            assert part in result.output
    if not launch:
        assert CliRunner().invoke(main, ["connect", "codex", "--undo"]).exit_code == 0
        assert not (home / "vllm-sr.models.json").exists()


def test_a_recipe_that_drops_what_no_longer_fits_advertises_each_decisions_largest():
    connect_command = importlib.import_module("cli.commands.connect")
    doc = yaml.safe_load(ROUTER)
    doc["providers"]["models"].append(
        {"name": "local", "api_format": "chat_completions", "context_window_size": 8192}
    )
    doc["recipes"][0]["decisions"].insert(
        0, {"name": "easy", "modelRefs": [{"model": "local"}, {"model": "big"}]}
    )
    groups = rc.entrypoint_candidates(doc, "vllm-sr/auto")
    assert groups == [(["local", "big"], 1), (["big", "card"], 1), (["small"], 1)]
    window, _ = connect_command._context_window(doc, "vllm-sr/auto", None)
    assert window == 8192
    doc["recipes"][0]["candidate_requirements"] = {"context": "known_limits"}
    window, notes = connect_command._context_window(doc, "vllm-sr/auto", None)
    # a long request moves from local to big, so only the default's 32,768 binds
    assert window == 32768
    assert "the most every decision still serves" in notes[0]
    # a decision that needs two candidates keeps them only up to its second
    # largest window, and one that calls every candidate up to its smallest
    easy = doc["recipes"][0]["decisions"][0]
    easy["algorithm"] = {"type": "multi_factor", "minimum_candidates": 2}
    assert connect_command._context_window(doc, "vllm-sr/auto", None)[0] == 8192
    easy["algorithm"] = {"type": "fusion"}
    assert rc.entrypoint_candidates(doc, "vllm-sr/auto")[0] == (["local", "big"], 2)
    assert connect_command._context_window(doc, "vllm-sr/auto", None)[0] == 8192


# What claude -p --output-format stream-json and codex exec --json print for a
# check that passes: the command's call, its output, and the answer to it.
CLAUDE_EVENTS = [
    {"type": "system", "subtype": "init", "model": "vllm-sr/auto"},
    {
        "type": "assistant",
        "message": {
            "model": "qwen36",
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "Bash",
                    "input": {"command": "echo vsr-check"},
                }
            ],
        },
    },
    {
        "type": "user",
        "message": {
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": "vsr-check",
                    "is_error": False,
                }
            ]
        },
    },
    {
        "type": "assistant",
        "message": {"model": "qwen36", "content": [{"type": "text", "text": "ok"}]},
    },
    {"type": "result", "subtype": "success", "is_error": False, "result": "vsr-check"},
]
CODEX_EVENTS = [
    {"type": "thread.started", "thread_id": "th1"},
    {"type": "turn.started"},
    {
        "type": "item.completed",
        "item": {
            "type": "command_execution",
            "command": "bash -lc 'echo vsr-check'",
            "aggregated_output": "vsr-check\n",
            "exit_code": 0,
            "status": "completed",
        },
    },
    {"type": "item.completed", "item": {"type": "agent_message", "text": "vsr-check"}},
    {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
]


@pytest.fixture
def checked_harness(tmp_path, monkeypatch):
    """A claude and a codex that answer --version, list a bundled gpt-5.5, and
    otherwise act out one --check run: they record how they were started, send a
    review request to $FAKE_REVIEW_URL when Codex reviews approvals, then print
    $FAKE_EVENTS and exit with $FAKE_EXIT."""
    bin_dir = tmp_path / "check-bin"
    bin_dir.mkdir()
    record = tmp_path / "check-ran.json"
    versions = {"claude": "2.1.325 (Claude Code)", "codex": "codex-cli 0.166.0"}
    for name, version in versions.items():
        script = bin_dir / name
        script.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys, time, urllib.request\n"
            "args = sys.argv[1:]\n"
            "if args == ['--version']:\n"
            f"    print({version!r})\n"
            "    sys.exit(0)\n"
            "if args[:2] == ['debug', 'models']:\n"
            f"    print(json.dumps({{'models': [{BUNDLED_GPT_55!r}]}}))\n"
            "    sys.exit(0)\n"
            "given = args[args.index('--settings') + 1] if '--settings' in args else None\n"
            "catalog = [a.split('=', 1)[1] for a in args if a.startswith('model_catalog_json=')]\n"
            "json.dump({'args': args, 'cwd': os.getcwd(),"
            " 'settings': open(given).read() if given else None,"
            " 'catalog': json.load(open(json.loads(catalog[0]))) if catalog else None,"
            " 'stdin_closed': os.path.samestat(os.fstat(0), os.stat(os.devnull)),"
            f" 'key': os.environ.get('VLLM_SR_API_KEY')}}, open({str(record)!r}, 'w'))\n"
            "if os.environ.get('FAKE_REVIEW_URL') and '--approve-for-me' in args:\n"
            "    body = json.dumps({'model': 'vllm-sr/review', 'input': 'review'}).encode()\n"
            "    urllib.request.urlopen(urllib.request.Request(os.environ['FAKE_REVIEW_URL'],"
            " body, {'content-type': 'application/json'})).read()\n"
            "time.sleep(float(os.environ.get('FAKE_SLEEP', '0')))\n"
            "for event in json.loads(os.environ.get('FAKE_EVENTS', '[]')):\n"
            "    print(json.dumps(event))\n"
            "sys.exit(int(os.environ.get('FAKE_EXIT', '0')))\n"
        )
        script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    return record


def _flat(output: str) -> str:
    return " ".join(output.split())


def test_check_runs_claude_code_on_the_settings_connect_wrote(
    tmp_path, gateway, checked_harness, monkeypatch
):
    url, _, _ = gateway
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("VLLM_SR_API_KEY", "sk-secret")
    monkeypatch.setenv("FAKE_EVENTS", json.dumps(CLAUDE_EVENTS))
    args = ["connect", "claude-code", "--base-url", url, "--yes", "--check"]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    output = _flat(result.output)
    assert "spends a few short model calls on your login or key" in output
    assert "Harness ok 2.1.325 (Claude Code) at " in output
    assert "Login ok the gateway answered, as qwen36" in output
    assert (
        "Tool round trip ok ran echo vsr-check and answered with its output" in output
    )
    assert "sk-secret" not in result.output
    ran = json.loads(checked_harness.read_text())
    settings = tmp_path / "claude" / "settings.json"
    assert ran["args"][:2] == ["-p", rc_prompt()]
    assert ran["args"][ran["args"].index("--settings") + 1] == str(settings)
    assert "Bash(echo vsr-check)" in ran["args"]
    assert "--no-session-persistence" in ran["args"]
    # its own temporary directory, gone afterwards, and no stdin to wait on
    assert ran["stdin_closed"] and not Path(ran["cwd"]).exists()
    assert ran["key"] == "sk-secret"  # the harness's own environment, unchanged

    # a dry run checks the settings it would write, and writes none
    settings.unlink()
    result = CliRunner().invoke(
        main, ["connect", "claude-code", "--base-url", url, "--dry-run", "--check"]
    )
    assert result.exit_code == 0, result.output
    ran = json.loads(checked_harness.read_text())
    assert json.loads(ran["settings"])["env"]["ANTHROPIC_BASE_URL"] == url
    assert not Path(ran["args"][ran["args"].index("--settings") + 1]).exists()
    assert not settings.exists()


def rc_prompt():
    return importlib.import_module("cli.harness_check").PROMPT


def test_a_failed_check_names_the_step_and_hides_the_key(
    tmp_path, gateway, checked_harness, monkeypatch
):
    url, _, _ = gateway
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("VLLM_SR_API_KEY", "sk-secret")
    refused = "API Error: 401 the key sk-secret is not valid, Bearer sk-other-123"
    events = [
        {
            "type": "assistant",
            "message": {
                "model": "<synthetic>",
                "content": [{"type": "text", "text": refused}],
            },
        },
        {"type": "result", "subtype": "success", "is_error": True, "result": refused},
    ]
    monkeypatch.setenv("FAKE_EVENTS", json.dumps(events))
    monkeypatch.setenv("FAKE_EXIT", "1")
    args = ["connect", "claude-code", "--base-url", url, "--yes", "--check"]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 1
    output = _flat(result.output)
    assert "Login failed API Error: 401 the key <hidden> is not valid" in output
    assert "Bearer <hidden>" in output and "sk-" not in output
    assert "Tool round trip not run" in output
    assert "The changes above stay written" in output

    # a model that answers without running the command fails the round trip
    monkeypatch.setenv("FAKE_EXIT", "0")
    monkeypatch.setenv(
        "FAKE_EVENTS", json.dumps([CLAUDE_EVENTS[3], {**CLAUDE_EVENTS[4]}])
    )
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 1
    assert "answered without running echo vsr-check" in _flat(result.output)


def test_check_without_the_harness_or_in_time_fails(
    tmp_path, gateway, checked_harness, monkeypatch
):
    url, _, _ = gateway
    hk = importlib.import_module("cli.harness_check")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    args = ["connect", "claude-code", "--base-url", url, "--yes", "--check"]
    monkeypatch.setenv("FAKE_SLEEP", "30")
    monkeypatch.setattr(hk, "TIMEOUT", 1)
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 1
    assert "Login failed claude did not finish in 1 s" in _flat(result.output)

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 1
    output = _flat(result.output)
    assert "Harness failed claude is not on PATH" in output
    assert "Login not run" in output

    for other in (["--launch"], ["--undo"]):
        result = CliRunner().invoke(main, ["connect", "codex", "--check", *other])
        assert result.exit_code == 2 and "does not take --check" in result.output


REVIEW_DOC = {
    "version": "v0.5",
    "providers": {
        "defaults": {"model": "session"},
        "models": [
            {"name": "session", "context_window_size": 131072},
            {"name": "reviewer", "context_window_size": 16384},
        ],
    },
    "entrypoints": [
        {
            "model_names": ["vllm-sr/auto"],
            "recipe": "main",
            "default": True,
            "review_model": "vllm-sr/review",
        },
        {"model_names": ["vllm-sr/review"], "recipe": "review"},
    ],
    "recipes": [
        {"name": name, "decisions": [{"name": name, "modelRefs": [{"model": model}]}]}
        for name, model in [("main", "session"), ("review", "reviewer")]
    ],
}


def test_check_sees_codex_send_its_approval_review_to_the_review_entrypoint(
    tmp_path, gateway, checked_harness, monkeypatch
):
    url, _, _ = gateway
    hk = importlib.import_module("cli.harness_check")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(REVIEW_DOC))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.setenv("VLLM_SR_API_KEY", "sk-secret")
    monkeypatch.setenv("FAKE_EVENTS", json.dumps(CODEX_EVENTS))
    monkeypatch.setenv("FAKE_REVIEW_URL", url + "/v1/responses")
    monkeypatch.setattr(hk, "management_url", lambda gateway: url)
    args = ["connect", "codex", "--config", str(config), "--base-url", url]
    result = CliRunner().invoke(main, [*args, "--no-vsr", "--yes", "--check"])
    assert result.exit_code == 0, result.output
    output = _flat(result.output)
    assert "Harness ok codex-cli 0.166.0 at " in output
    assert "Tool round trip ok ran echo vsr-check" in output
    assert "Approval review ok went to vllm-sr/review (replay r" in output
    ran = json.loads(checked_harness.read_text())
    assert ran["args"][:3] == ["exec", "--profile", "vllm-sr"]
    assert "--approve-for-me" in ran["args"] and "--ephemeral" in ran["args"]
    assert ran["args"][-1] == hk.REVIEW_PROMPT and ran["stdin_closed"]

    # without the Router's replay list the review is not observed, which passes
    monkeypatch.setattr(hk, "management_url", lambda gateway: "http://127.0.0.1:9")
    result = CliRunner().invoke(main, [*args, "--no-vsr", "--yes", "--check"])
    assert result.exit_code == 0, result.output
    assert "Approval review not observed the replay list at" in _flat(result.output)

    # a dry run passes the profile's settings as overrides, with its catalog
    home = tmp_path / "codex-dry"
    monkeypatch.setenv("CODEX_HOME", str(home))
    result = CliRunner().invoke(main, [*args, "--no-vsr", "--dry-run", "--check"])
    assert result.exit_code == 0, result.output
    ran = json.loads(checked_harness.read_text())
    overrides = dict(a.split("=", 1) for a in ran["args"] if "=" in a)
    assert overrides["model_providers.vllm-sr.base_url"] == json.dumps(url + "/v1")
    assert [m["slug"] for m in ran["catalog"]["models"]] == [
        "vllm-sr/auto",
        "vllm-sr/review",
    ]
    assert not home.exists()


def test_status_reports_each_connection_and_what_changed_since(
    tmp_path, gateway, checked_harness, monkeypatch
):
    url, seen, _ = gateway
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(REVIEW_DOC))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.setenv("VLLM_SR_API_KEY", "sk-secret")
    result = CliRunner().invoke(main, ["connect", "status"])
    assert result.exit_code == 0, result.output
    assert "claude-code: not connected (no record in" in result.output
    assert "codex: not connected" in result.output

    args = ["--config", str(config), "--base-url", url, "--no-vsr", "--yes"]
    for harness in ("claude-code", "codex"):
        result = CliRunner().invoke(main, ["connect", harness, *args])
        assert result.exit_code == 0, result.output
    asked = len(seen)
    result = CliRunner().invoke(main, ["connect", "status"])
    assert result.exit_code == 0, result.output
    output = _flat(result.output)
    settings = tmp_path / "claude" / "settings.json"
    catalog = tmp_path / "codex" / "vllm-sr.models.json"
    profile = tmp_path / "codex" / "vllm-sr.config.toml"
    for part in [
        "claude-code (user scope)",
        f"gateway {url}",
        "model vllm-sr/auto",
        "context window 131,072 tokens",
        "login gateway key, in the settings",
        f"files {settings}: as written",
        "gateway now ok: model agent-model",
        "codex (profile vllm-sr)",
        "review model vllm-sr/review",
        "login gateway key from $VLLM_SR_API_KEY",
        f"{catalog}: as written",
    ]:
        assert part in output, part
    assert "sk-secret" not in result.output
    # one short request each, with the key each harness sends
    assert [(p, auth) for p, _, auth in seen[asked:]] == [
        ("/v1/messages", "Bearer sk-secret"),
        ("/v1/responses", "Bearer sk-secret"),
    ]

    held = json.loads(settings.read_text())
    held["env"]["ANTHROPIC_MODEL"] = "opus"
    settings.write_text(json.dumps(held))
    catalog.unlink()
    # the Codex TUI's own tables leave the profile as connect wrote it, as for --undo
    profile.write_text(profile.read_text() + "\n[tui]\nscreen_reader = true\n")
    monkeypatch.delenv("VLLM_SR_API_KEY")
    result = CliRunner().invoke(main, ["connect", "status", "--json"])
    assert result.exit_code == 0, result.output
    claude, codex = json.loads(result.output)["connections"]
    assert claude["files"] == [
        {"path": str(settings), "state": "changed", "changed": ["ANTHROPIC_MODEL"]}
    ]
    assert codex["files"] == [
        {"path": str(profile), "state": "as written"},
        {"path": str(catalog), "state": "missing"},
    ]
    assert (codex["review_model"], codex["context_window"]) == (
        "vllm-sr/review",
        131072,
    )
    assert codex["probe"] == {
        "state": "not probed",
        "detail": "$VLLM_SR_API_KEY is not set, and Codex sends the key from it",
    }
    assert "sk-secret" not in result.output

    assert CliRunner().invoke(main, ["connect", "codex", "--undo"]).exit_code == 0
    result = CliRunner().invoke(main, ["connect", "status", "--json"])
    assert json.loads(result.output)["connections"][1]["connected"] is False


def test_a_file_changed_after_the_preview_is_not_written(
    tmp_path, gateway, monkeypatch
):
    url, _, _ = gateway
    command = importlib.import_module("cli.commands.connect")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr("shutil.which", lambda _: None)
    settings = tmp_path / "settings.json"
    settings.write_text('{"theme": "dark"}')

    def edited_while_asking(*args, **kwargs):
        settings.write_text('{"theme": "light"}')
        return True

    monkeypatch.setattr(command.click, "confirm", edited_while_asking)
    result = CliRunner().invoke(main, ["connect", "claude-code", "--base-url", url])
    assert result.exit_code == 1
    assert "changed after the changes above were worked out" in _flat(result.output)
    assert settings.read_text() == '{"theme": "light"}'
    assert not (tmp_path / hc.MANIFEST).exists()


@pytest.mark.parametrize("failing", [2, 3])
def test_a_write_that_fails_part_way_puts_back_what_it_wrote(
    tmp_path, monkeypatch, failing
):
    profile = tmp_path / "p.config.toml"
    profile.write_text("old = 1\n")
    profile.chmod(0o640)
    created = tmp_path / "vsr.json"
    changes = [
        hc.Change(profile, "file", {"file": "new = 1\n"}),
        hc.Change(created, "json", {"runtimeUrl": "http://gw"}),
    ]
    replace, calls = hc.os.replace, []

    def crash(*args):
        calls.append(args)
        if len(calls) == failing:  # the second file, or the record after both
            raise OSError("disk full")
        return replace(*args)

    monkeypatch.setattr(hc.os, "replace", crash)
    with pytest.raises(OSError, match="every file was left as it was"):
        hc.apply_changes(changes, tmp_path)
    assert profile.read_text() == "old = 1\n" and _mode(profile) == 0o640
    assert sorted(p.name for p in tmp_path.iterdir()) == ["p.config.toml"]

    monkeypatch.setattr(hc.os, "replace", replace)
    hc.apply_changes(changes, tmp_path)
    assert profile.read_text() == "new = 1\n" and _mode(profile) == 0o640


def test_status_fails_when_a_gateway_check_fails(tmp_path, gateway, monkeypatch):
    url, _, replies = gateway
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.setenv("VLLM_SR_API_KEY", "sk-secret")
    args = ["connect", "claude-code", "--base-url", url, "--context-window", "65536"]
    assert CliRunner().invoke(main, [*args, "--no-vsr", "--yes"]).exit_code == 0
    replies["/v1/messages"].pop("plain")
    for shown in ([], ["--json"]):
        result = CliRunner().invoke(main, ["connect", "status", *shown])
        assert result.exit_code == 1, result.output
        assert "failed" in result.output


def test_status_sends_no_key_to_a_gateway_the_settings_no_longer_name(
    tmp_path, gateway, checked_harness, monkeypatch
):
    url, seen, _ = gateway
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(REVIEW_DOC))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.setenv("VLLM_SR_API_KEY", "sk-secret")
    args = ["--config", str(config), "--base-url", url, "--no-vsr", "--yes"]
    for harness in ("claude-code", "codex"):
        result = CliRunner().invoke(main, ["connect", harness, *args])
        assert result.exit_code == 0, result.output
    settings = tmp_path / "claude" / "settings.json"
    profile = tmp_path / "codex" / "vllm-sr.config.toml"

    # a key replaced in place still goes to the gateway the settings name
    held = json.loads(settings.read_text())
    held["env"][hc.TOKEN_KEY] = "sk-rotated"
    settings.write_text(json.dumps(held))
    asked = len(seen)
    result = CliRunner().invoke(main, ["connect", "status", "--json"])
    assert result.exit_code == 0, result.output
    assert [auth for _, _, auth in seen[asked:]] == [
        "Bearer sk-rotated",
        "Bearer sk-secret",
    ]

    # moved to another gateway with that gateway's key: nothing goes to the one
    # recorded, and the status says why
    held["env"] |= {"ANTHROPIC_BASE_URL": "http://127.0.0.1:9", hc.TOKEN_KEY: "sk-b"}
    settings.write_text(json.dumps(held))
    profile.write_text(
        profile.read_text().replace(f"{url}/v1", "http://127.0.0.1:9/v1")
    )
    asked = len(seen)
    result = CliRunner().invoke(main, ["connect", "status", "--json"])
    assert result.exit_code == 0, result.output
    assert seen[asked:] == []
    claude, codex = json.loads(result.output)["connections"]
    assert claude["probe"] == {
        "state": "not probed",
        "detail": "the settings no longer hold the ANTHROPIC_BASE_URL connect"
        f" wrote, so their key is not sent to {url}",
    }
    assert codex["probe"] == {
        "state": "not probed",
        "detail": "the profile no longer holds what connect wrote, so no key is"
        f" sent to {url}",
    }
    assert "sk-b" not in result.output


def test_a_file_edited_while_the_changes_are_written_is_kept(tmp_path, monkeypatch):
    first, second = tmp_path / "p.config.toml", tmp_path / "vsr.json"
    first.write_text("old = 1\n")
    second.write_text('{"theme": "dark"}\n')
    changes = [
        hc.Change(first, "file", {"file": "new = 1\n"}),
        hc.Change(second, "json", {"runtimeUrl": "http://gw"}),
    ]
    write, written = hc.write_private, []

    def editor_saves(path, text, **kwargs):
        # an editor saves the next file once the first is written
        write(path, text, **kwargs)
        written.append(path)
        if len(written) == 1:
            second.write_text('{"theme": "light"}\n')

    monkeypatch.setattr(hc, "write_private", editor_saves)
    with pytest.raises(hc.StaleWriteError, match=f"{second} changed while"):
        hc.apply_changes(
            changes, tmp_path, {first: "old = 1\n", second: '{"theme": "dark"}\n'}
        )
    assert second.read_text() == '{"theme": "light"}\n'
    assert first.read_text() == "old = 1\n"
    assert not (tmp_path / hc.MANIFEST).exists()

    # putting back a write leaves alone a file edited after it was written
    second.write_text('{"theme": "dark"}\n')
    saved = []

    def edited_then_full(path, text, **kwargs):
        if path == second:
            raise OSError("disk full")
        write(path, text, **kwargs)
        if not saved:  # an editor saves the first file once it is written
            saved.append(first.write_text("mine = 1\n"))

    monkeypatch.setattr(hc, "write_private", edited_then_full)
    with pytest.raises(OSError, match="changed since it was written"):
        hc.apply_changes(changes, tmp_path)
    assert first.read_text() == "mine = 1\n"
    assert second.read_text() == '{"theme": "dark"}\n'


@pytest.mark.parametrize(
    "harness, events, code, timed_out, why",
    [
        (
            "codex",
            [
                *CODEX_EVENTS[:-1],
                {"type": "turn.failed", "error": {"message": "stream closed"}},
            ],
            1,
            False,
            "codex exited with status 1 and reported a failure: stream closed",
        ),
        ("codex", CODEX_EVENTS, 0, True, "codex did not finish in 120 s"),
        ("claude-code", CLAUDE_EVENTS, 2, False, "claude exited with status 2: boom"),
    ],
)
def test_a_run_that_ends_badly_fails_the_check_after_steps_that_passed(
    tmp_path, checked_harness, monkeypatch, harness, events, code, timed_out, why
):
    hk = importlib.import_module("cli.harness_check")
    out = "\n".join(json.dumps(event) for event in events)
    monkeypatch.setattr(hk, "_run_harness", lambda *_: (out, "boom\n", code, timed_out))
    name = "claude" if harness == "claude-code" else "codex"
    rows = hk.run(harness, [name, "exec"], dict(hk.os.environ), tmp_path, [])
    states = {row.name: (row.state, row.detail) for row in rows}
    assert states["Login"][0] == states["Tool round trip"][0] == "ok"
    assert states["Exit"] == ("failed", why)

    # the same harness that finishes well passes
    passed = CLAUDE_EVENTS if harness == "claude-code" else CODEX_EVENTS
    out = "\n".join(json.dumps(event) for event in passed)
    monkeypatch.setattr(hk, "_run_harness", lambda *_: (out, "", 0, False))
    rows = hk.run(harness, [name, "exec"], dict(hk.os.environ), tmp_path, [])
    assert [row.state for row in rows] == ["ok"] * 4
    assert rows[-1].detail == f"{name} exited with status 0"
