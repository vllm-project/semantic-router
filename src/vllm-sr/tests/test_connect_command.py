import difflib
import importlib
import json
import os
import re
import shlex
import signal
import stat
import subprocess
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import click
import pytest
import yaml
from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

main = importlib.import_module("cli.main").main
hc = importlib.import_module("cli.harness_connect")
hp = importlib.import_module("cli.harness_pi")
rc = importlib.import_module("cli.harness_router_check")
hs = importlib.import_module("cli.harness_session")
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
CHAT_EVENTS = [
    {
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "get_time", "arguments": ""},
                        }
                    ],
                },
            }
        ],
    },
    {
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]
                },
                "finish_reason": "tool_calls",
            }
        ],
    },
]
STAGES = {
    "/v1/chat/completions": {
        "plain": {
            "object": "chat.completion",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        },
        "tool": CHAT_EVENTS,
        "result": [
            {"choices": [{"index": 0, "delta": {"content": "It is 12:00."}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ],
    },
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
    answered = ("tool_result", "function_call_output", '"tool_call_id"')
    return "result" if any(word in sent for word in answered) else "tool"


UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"

# A translation loss whose reason holds a comma, as the Router writes it: an RFC 8941 list.
PLAIN_WARNINGS = (
    'protocol;action=approximated;reason="stop_sequence, absent, read as null";'
    'field="stop_sequence", truncated;count=2'
)


@pytest.fixture(autouse=True)
def pi_home(tmp_path, monkeypatch):
    """Pi's agent directory, in the test's own folder: connect status reads it."""
    home = tmp_path / "pi-agent"
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(home))
    return home


@pytest.fixture(autouse=True)
def router_answers(monkeypatch):
    """A Router answers at every configuration's listener, so connect starts none;
    the tests of starting one say otherwise."""
    monkeypatch.setattr(
        importlib.import_module("cli.commands.connect"), "_answers", lambda _: True
    )


@pytest.fixture(autouse=True)
def state_home(tmp_path_factory, monkeypatch):
    """The CLI's state directory, where --launch records its sessions."""
    state = tmp_path_factory.mktemp("state")
    monkeypatch.setenv("XDG_STATE_HOME", str(state))
    return state


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


def test_claude_code_names_its_request_class_for_the_session_title(tmp_path):
    # Router Replay names an agent run by the title Claude Code asks for on an
    # auxiliary request, which it marks only with its gateway hint headers on.
    settings = json.loads(_claude(tmp_path / "settings.json").new_text())
    assert settings["env"]["CLAUDE_CODE_GATEWAY_HINT_HEADERS"] == "1"


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


@pytest.mark.parametrize("kind", ["toml", "json-env"])
def test_undo_gives_back_the_file_byte_for_byte(tmp_path, kind):
    pytest.importorskip("tomllib")
    if kind == "toml":
        # Codex's config.toml whose last line has no newline
        path = tmp_path / "config.toml"
        before = 'model = "gpt-5"\n\n[projects."/src/app"]\ntrust_level = "trusted"'
        mine, theirs = '"trusted"', '"untrusted"'

        def change(gateway):
            return hc.codex_app_change(tmp_path, gateway, "vllm-sr/auto", "K")

    else:
        # Claude Code's settings as another tool wrote them, compact
        path = tmp_path / "settings.json"
        before = '{"theme": "dark", "env": {"FOO": "1"}}'
        mine, theirs = '"dark"', '"light"'

        def change(gateway):
            return _claude(path, gateway)

    path.write_text(before)
    hc.apply_changes([change("http://a")], tmp_path)
    hc.apply_changes([change("http://b")], tmp_path)
    assert path.read_text() != before
    hc.undo_changes(tmp_path)
    assert path.read_bytes() == before.encode()

    # an edit after connect wrote the file, or between two runs, is kept
    for edit_before_second_run in (False, True):
        path.write_text(before)
        hc.apply_changes([change("http://a")], tmp_path)
        if edit_before_second_run:
            path.write_text(path.read_text().replace(mine, theirs))
            hc.apply_changes([change("http://b")], tmp_path)
        else:
            path.write_text(path.read_text().replace(mine, theirs))
        hc.undo_changes(tmp_path)
        assert theirs in path.read_text() and "http://" not in path.read_text()


def test_undo_does_not_recreate_a_file_deleted_since(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{}")
    hc.apply_changes([_claude(path)], tmp_path)
    path.unlink()
    assert any("no longer exists" in line for line in hc.undo_changes(tmp_path))
    assert not path.exists()


# the fallback the router check asks for, with room for a long agent call
AGENT_FALLBACK = {
    "enabled": True,
    "total_timeout": "1200s",
    "per_attempt_timeout": "1200s",
}


def _router(
    lossy=(),
    learning=None,
    services=None,
    api_format="anthropic",
    plugins=None,
    reliable=True,
):
    """A configuration with one decision; reliable gives it what the reliability
    findings ask for: fallback to a second model, and no total timeout."""
    model = {"name": "m"}
    if api_format:
        model["api_format"] = api_format
    models = [model]
    if reliable:
        model["reliability"] = {"total_timeout": "0s"}
        models.append({**model, "name": "m2"})
    refs = [{"model": m["name"]} for m in models]
    decision = {"name": "d", "modelRefs": refs, "plugins": plugins or []}
    doc = {
        "providers": {"models": models},
        "recipes": [{"decisions": [decision]}],
        "global": {"router": {"allow_lossy_translation": list(lossy)}},
    }
    if reliable:
        doc["global"]["router"]["fallback"] = dict(AGENT_FALLBACK)
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


def test_router_check_needs_no_identity_headers_the_router_reads_by_default():
    session = {"enabled": True, "protection": {"enabled": True, "scope": "session"}}
    max_tokens = [
        {"type": "request_params", "configuration": {"default_max_tokens": 32000}}
    ]
    assert (
        rc.check_router_config(_router(["tools.defer_loading"], session), "claude-code")
        == []
    )
    codex = _router(["reasoning.signature"], session, plugins=max_tokens)
    assert rc.check_router_config(codex, "codex") == []
    assert rc.check_router_config(_router(["reasoning.signature"], session), "pi") == []
    # without protection, the advice turns it on and names no identity headers
    (finding,) = rc.check_router_config(_router(["tools.defer_loading"]), "claude-code")
    assert "scope: session" in finding.snippet and "identity" not in finding.snippet
    # identity headers that name others replace the defaults, so the advice names
    # the harness's own
    other = _router(["tools.defer_loading"], _learning({"agent": "x-agent-id"}))
    (finding,) = rc.check_router_config(other, "claude-code")
    assert "agent: x-claude-code-agent-id" in finding.snippet
    # Claude Code's session is read from session_id_headers when identity names none
    narrowed = _router(["tools.defer_loading"], session)
    narrowed["global"]["router"]["headers"] = {"session_id_headers": ["x-session-id"]}
    (finding,) = rc.check_router_config(narrowed, "claude-code")
    assert "session: x-claude-code-session-id" in finding.snippet


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


def test_probes_mark_themselves_so_no_run_is_listed_for_them():
    # Router Replay records a request with x-vsr-probe as a check, and a
    # session of only checks is no run in Insights
    for harness in ("claude-code", "codex", "pi"):
        _, headers, _ = rc._probe_target(harness, "m", "http://gw", "tok")
        assert headers[rc.PROBE_HEADER] == "connect"


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


def test_probe_reads_chat_completions_for_pi(gateway):
    url, _, replies = gateway
    chat = replies["/v1/chat/completions"]
    chat["plain"] = {"object": "response", "output": []}
    (plain,) = rc.probe_gateway(url, "pi", "m", "t")
    assert not plain.ok and "not a Chat Completions reply" in plain.detail

    chat["plain"] = STAGES["/v1/chat/completions"]["plain"]
    chat["tool"] = CHAT_EVENTS[:1]  # no finish reason
    loop = rc.probe_gateway(url, "pi", "m", "t")[1]
    assert not loop.ok and "did not finish" in loop.detail
    # a stream that fails says so in an error object of its own
    chat["tool"] = [CHAT_EVENTS[0], {"error": {"message": "lossy_translation"}}]
    loop = rc.probe_gateway(url, "pi", "m", "t")[1]
    assert not loop.ok and "lossy_translation" in loop.detail
    # a model that runs out of output before it calls the tool names the limit
    length = {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}
    chat["tool"] = [length]
    loop = rc.probe_gateway(url, "pi", "m", "t")[1]
    assert loop.ok and "output limit before calling the tool" in loop.detail
    chat["tool"] = CHAT_EVENTS
    chat["result"] = [length]
    loop = rc.probe_gateway(url, "pi", "m", "t")[1]
    assert not loop.ok and f"{rc.PROBE_TOKENS}-token output limit" in loop.detail


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


def test_claude_code_sends_the_gateway_hint_headers(tmp_path, gateway, monkeypatch):
    # Claude Code names a subagent's type and a request's class, which the
    # agent signal routes on, only with this variable set (2.1.325)
    url, _, _ = gateway
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_SR_API_KEY", "sk-secret")
    monkeypatch.setattr("shutil.which", lambda _: None)
    args = ["connect", "claude-code", "--base-url", url]
    preview = CliRunner().invoke(main, [*args, "--dry-run"])
    assert '"CLAUDE_CODE_GATEWAY_HINT_HEADERS": "1"' in preview.output
    assert CliRunner().invoke(main, [*args, "--yes"]).exit_code == 0
    settings = tmp_path / "settings.json"
    env = json.loads(settings.read_text())["env"]
    assert env["CLAUDE_CODE_GATEWAY_HINT_HEADERS"] == "1"
    env["CLAUDE_CODE_GATEWAY_HINT_HEADERS"] = "0"
    settings.write_text(json.dumps({"env": env}))
    status = CliRunner().invoke(main, ["connect", "status", "--json"])
    assert "CLAUDE_CODE_GATEWAY_HINT_HEADERS" in status.output
    # a dry run says what --undo would take back and changes nothing
    written = settings.read_text()
    result = CliRunner().invoke(main, ["connect", "claude-code", "--undo", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert f"would restore {settings}" in result.output
    assert "Dry run: nothing was undone." in result.output
    assert settings.read_text() == written
    assert CliRunner().invoke(main, ["connect", "claude-code", "--undo"]).exit_code == 0
    assert json.loads(settings.read_text()) == {
        "env": {"CLAUDE_CODE_GATEWAY_HINT_HEADERS": "0"}
    }


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
    own = "anthropic" if harness == "claude-code" else "openai"
    assert f"sends its {own} login, so every call" in _flat(result.output)
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


def test_connect_starts_the_router_only_when_nothing_answers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr("shutil.which", lambda _: None)
    (tmp_path / "config.yaml").write_text(ROUTER)
    answering = []
    monkeypatch.setattr(
        importlib.import_module("cli.commands.connect"),
        "_answers",
        lambda _: bool(answering),
    )
    started = []

    def serve(config):
        started.append(config)

    monkeypatch.setattr(
        importlib.import_module("cli.commands.runtime"),
        "serve",
        click.Command("serve", params=[click.Option(["--config"])], callback=serve),
    )
    args = ["connect", "claude-code", "--no-vsr", "--yes"]
    result = CliRunner().invoke(main, [*args, "--dry-run"])
    assert result.exit_code == 0, result.output
    assert (
        "Router: nothing answers at http://localhost:8899; would start it with"
        " vllm-sr serve --config config.yaml." in result.output
    )
    # --no-serve, and a gateway named by --base-url, never start one
    for extra in (["--no-serve"], ["--base-url", "http://localhost:8899"]):
        result = CliRunner().invoke(main, [*args, *extra])
        assert result.exit_code == 0, result.output
    assert started == []
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert started == ["config.yaml"]
    # a Router that answers is left alone
    answering.append(True)
    assert CliRunner().invoke(main, ["connect", "codex", "--yes"]).exit_code == 0
    assert started == ["config.yaml"]


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
    or model catalog they were given, the key and the homes in their environment
    and the directory they run in, then exit 3. Asked for its bundled models, the
    codex lists a gpt-5.5 entry."""
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
            f" 'key': os.environ.get('VLLM_SR_API_KEY'), 'cwd': os.getcwd(),"
            " 'homes': [os.environ.get('CLAUDE_CONFIG_DIR'), os.environ.get('CODEX_HOME')]"
            f"}}, open({str(record)!r}, 'w'))\n"
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
    assert ran["args"][-2:] == ["--resume", "x"] and "--session-id" not in ran["args"]
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
                "description": "vllm-sr/auto through vLLM Semantic Router, which picks"
                " the model for each request.",
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
    # Codex runs no hook a person has not trusted, and connect never trusts one
    trust = importlib.import_module("cli.commands.connect").CODEX_HOOK_TRUST
    assert trust in result.output
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
    assert "trust the vSR hooks" not in result.output
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


def test_vsr_asks_the_listener_the_gateway_url_names(monkeypatch):
    doc = {
        "listeners": [
            {"port": 8899, "apis": {"systemone": {"models": ["first"]}}},
            {"port": 9900, "apis": {"systemone": {"models": ["second"]}}},
        ]
    }
    assert rc.system_one_models(doc) == ["first"]
    assert rc.system_one_models(doc, "http://localhost:9900") == ["second"]
    assert rc.system_one_models(doc, "http://localhost:7000") == []
    # the local stack publishes each listener VLLM_SR_PORT_OFFSET ports higher
    monkeypatch.setenv("VLLM_SR_PORT_OFFSET", "100")
    assert rc.system_one_models(doc, "http://localhost:10000") == ["second"]
    assert rc.system_one_models(doc, "http://127.0.0.1:8999") == ["first"]
    assert rc.system_one_models(doc, "http://localhost:9900") == []
    # a gateway on another host is no port of the local stack
    assert rc.system_one_models(doc, "http://gw.example:9900") == ["second"]


def test_vsr_is_set_up_on_a_local_stack_with_a_port_offset(tmp_path, monkeypatch):
    _codex_env(tmp_path, monkeypatch, FakeCodexPlugins(installed=True))
    monkeypatch.setenv("VLLM_SR_PORT_OFFSET", "100")
    (tmp_path / "config.yaml").write_text(
        "version: v0.5\nlisteners:\n  - port: 8899\n    apis:\n      systemone:\n"
        "        models: [vela3-lite]\n"
    )
    result = CliRunner().invoke(main, ["connect", "codex", "--yes"])
    assert result.exit_code == 0, result.output
    assert "publishes no model" not in result.output
    assert json.loads((tmp_path / "vsr.json").read_text()) == {
        "runtimeUrl": "http://localhost:8999/v1/decisions",
        "runtimeModel": "vela3-lite",
        "managementUrl": "http://localhost:8180",
    }


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
    # a description of its own, not the base entry's
    assert main["description"] == (
        f"{model} through vLLM Semantic Router, which picks the model for each request."
    )
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


@pytest.mark.parametrize("review", [None, "vllm-sr/review"])
@pytest.mark.parametrize("memory", ["vllm-sr/memory", "vllm-sr/review"])
def test_codex_catalog_hides_the_memory_entrypoint(review, memory):
    levels = [{"effort": "low", "description": "Fast"}]
    bundled = {"models": [{**BUNDLED_GPT_55, "supported_reasoning_levels": levels}]}
    catalog = hc.codex_catalog(
        bundled, "vllm-sr/auto", 64000, review, 16000, memory, 8000
    )
    entries = catalog["models"]
    # one entrypoint that takes reviews and memories is listed once
    assert [e["slug"] for e in entries] == list(
        dict.fromkeys(["vllm-sr/auto", *([review] if review else []), memory])
    )
    assert entries[0].get("auto_review_model_override") == review
    found = entries[-1]
    assert found["slug"] == memory and found["visibility"] == "hide"
    assert found["supported_reasoning_levels"] == levels
    window = 16000 if memory == review else 8000
    assert found["context_window"] == found["max_context_window"] == window
    assert "auto_review_model_override" not in found


@pytest.mark.parametrize("mode", ["profile", "app", "launch"])
def test_codex_connect_sends_memories_to_the_memory_entrypoint(
    tmp_path, monkeypatch, harness_bin, mode
):
    tomllib = pytest.importorskip("tomllib")
    monkeypatch.chdir(tmp_path)
    home = tmp_path / "codex-home"
    monkeypatch.setenv("CODEX_HOME", str(home))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    login = {
        "endpoint": "https://chatgpt.com/backend-api/codex",
        "provider": "openai",
        "auth": {"credential": "caller"},
    }
    local = {"endpoint": "http://localhost:8000/v1", "provider": "vllm"}
    doc = {
        "version": "v0.5",
        "providers": {
            "defaults": {"model": "session"},
            "models": [
                {
                    "name": "session",
                    "context_window_size": 131072,
                    "backend_refs": [login],
                },
                {
                    "name": "local",
                    "api_format": "chat_completions",
                    "context_window_size": 32768,
                    "backend_refs": [local],
                },
            ],
        },
        "entrypoints": [
            {
                "model_names": ["vllm-sr/auto"],
                "recipe": "main",
                "default": True,
                "memory_model": " vllm-sr/memory ",
            },
            {"model_names": ["vllm-sr/memory"], "recipe": "memory"},
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
            for name, model in [("main", "session"), ("memory", "local")]
        ],
    }
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(doc))
    args = ["connect", "codex", "--config", str(config)]
    args += ["--base-url", "http://localhost:8899"]
    args += {
        "profile": ["--yes", "--no-vsr"],
        "app": ["--app", "--yes", "--no-vsr"],
        "launch": ["--launch", "--", "exec", "hi"],
    }[mode]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == (3 if mode == "launch" else 0), result.output
    output = _flat(result.output)
    assert "Codex memories go to vllm-sr/memory through the gateway." in output
    assert "Approve for me reviews go to vllm-sr/auto" in output
    wanted = {
        "extract_model": "vllm-sr/memory",
        "consolidation_model": "vllm-sr/memory",
    }
    if mode == "launch":
        ran = json.loads(harness_bin.read_text())
        catalog = ran["catalog"]
        for key, value in wanted.items():
            assert f"memories.{key}={json.dumps(value)}" in ran["args"]
        assert not home.exists()
    else:
        name = "vllm-sr-app" if mode == "app" else "vllm-sr"
        catalog = json.loads((home / f"{name}.models.json").read_text())
        written = home / ("config.toml" if mode == "app" else "vllm-sr.config.toml")
        assert tomllib.loads(written.read_text())["memories"] == wanted
    assert [
        (e["slug"], e["visibility"], e["context_window"]) for e in catalog["models"]
    ] == [("vllm-sr/auto", "list", 131072), ("vllm-sr/memory", "hide", 32768)]
    assert "auto_review_model_override" not in catalog["models"][0]
    if mode == "launch":
        return

    result = CliRunner().invoke(main, ["connect", "status"])
    assert result.exit_code == 0, result.output
    assert "memory model vllm-sr/memory" in _flat(result.output)
    result = CliRunner().invoke(main, ["connect", "status", "--json"])
    codex = json.loads(result.output)["connections"][1]
    assert (codex["model"], codex["memory_model"]) == (
        "vllm-sr/auto",
        "vllm-sr/memory",
    )
    assert CliRunner().invoke(main, ["connect", "codex", "--undo"]).exit_code == 0
    assert not (home / f"{name}.models.json").exists()


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


# What the check's file holds in every run here, and what claude -p
# --output-format stream-json and codex exec --json print for a check that
# passes: the command's call, its output, and the answer to it.
TOKEN = "vsr-check-0a1b2c3d"


@pytest.fixture(autouse=True)
def check_token(monkeypatch):
    monkeypatch.setattr(
        importlib.import_module("cli.harness_check"), "_token", lambda: TOKEN
    )


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
                    "input": {"command": "cat vsr-check.txt"},
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
                    "content": TOKEN,
                    "is_error": False,
                }
            ]
        },
    },
    {
        "type": "assistant",
        "message": {"model": "qwen36", "content": [{"type": "text", "text": "ok"}]},
    },
    {"type": "result", "subtype": "success", "is_error": False, "result": TOKEN},
]
CODEX_EVENTS = [
    {"type": "thread.started", "thread_id": "th1"},
    {"type": "turn.started"},
    {
        "type": "item.completed",
        "item": {
            "type": "command_execution",
            "command": "bash -lc 'cat vsr-check.txt'",
            "aggregated_output": TOKEN + "\n",
            "exit_code": 0,
            "status": "completed",
        },
    },
    {"type": "item.completed", "item": {"type": "agent_message", "text": TOKEN}},
    {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
]


@pytest.fixture
def checked_harness(tmp_path, monkeypatch):
    """A claude, a codex and a pi that answer --version, list a bundled gpt-5.5,
    and otherwise act out one --check run: they record how they were started and
    the models.json in Pi's agent directory, send a review request to
    $FAKE_REVIEW_URL when Codex reviews approvals, then print $FAKE_EVENTS, and
    $FAKE_STDERR on stderr, and exit with $FAKE_EXIT."""
    bin_dir = tmp_path / "check-bin"
    bin_dir.mkdir()
    record = tmp_path / "check-ran.json"
    versions = {
        "claude": "2.1.325 (Claude Code)",
        "codex": "codex-cli 0.166.0",
        "pi": "1.5.2",
    }
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
            "agent = os.environ.get('PI_CODING_AGENT_DIR') or ''\n"
            "models = os.path.join(agent, 'models.json')\n"
            "json.dump({'args': args, 'cwd': os.getcwd(), 'agent': agent,"
            " 'file': open('vsr-check.txt').read() if os.path.exists('vsr-check.txt') else None,"
            " 'models': open(models).read() if os.path.exists(models) else None,"
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
            "sys.stderr.write(os.environ.get('FAKE_STDERR', ''))\n"
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
        "Tool round trip ok ran cat vsr-check.txt and answered with its output"
        in output
    )
    assert "sk-secret" not in result.output
    ran = json.loads(checked_harness.read_text())
    settings = tmp_path / "claude" / "settings.json"
    assert ran["args"][:2] == ["-p", rc_prompt("Bash")]
    # each harness is told the name its own shell tool has
    assert "with your Bash tool," in ran["args"][1]
    # a copy of the settings connect wrote, gone afterwards, that also marks the
    # run as a check, so Router Replay lists no run for it; the settings Claude
    # Code uses from now on carry no such mark
    checked = json.loads(ran["settings"])
    assert checked["env"]["ANTHROPIC_BASE_URL"] == url
    assert checked["env"]["ANTHROPIC_CUSTOM_HEADERS"] == "x-vsr-probe: connect"
    assert not Path(ran["args"][ran["args"].index("--settings") + 1]).exists()
    assert "ANTHROPIC_CUSTOM_HEADERS" not in json.loads(settings.read_text())["env"]
    assert "Bash(cat vsr-check.txt)" in ran["args"]
    # the file it shows holds a token the prompt does not, so a model cannot
    # answer without running the command
    assert ran["file"] == TOKEN + "\n" and TOKEN not in ran["args"][1]
    assert "--no-session-persistence" in ran["args"]
    # its own temporary directory, gone afterwards, and no stdin to wait on
    assert ran["stdin_closed"] and not Path(ran["cwd"]).exists()
    assert ran["key"] == "sk-secret"  # the harness's own environment, unchanged

    # a dry run checks the settings it would write, and writes none; the mark
    # joins the custom headers Claude Code sends anyway
    settings.unlink()
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "x-team: blue")
    result = CliRunner().invoke(
        main, ["connect", "claude-code", "--base-url", url, "--dry-run", "--check"]
    )
    assert result.exit_code == 0, result.output
    ran = json.loads(checked_harness.read_text())
    checked = json.loads(ran["settings"])["env"]
    assert checked["ANTHROPIC_BASE_URL"] == url
    assert checked["ANTHROPIC_CUSTOM_HEADERS"] == "x-team: blue\nx-vsr-probe: connect"
    assert not Path(ran["args"][ran["args"].index("--settings") + 1]).exists()
    assert not settings.exists()


def rc_prompt(tool):
    return importlib.import_module("cli.harness_check").PROMPT.format(tool=tool)


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
    assert "answered without running cat vsr-check.txt" in _flat(result.output)


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


def test_a_harness_that_stops_before_its_run_is_no_login_failure(
    tmp_path, gateway, checked_harness, monkeypatch
):
    # Codex 0.166 on a config.toml it cannot load prints this and no event
    url, _, _ = gateway
    _homes(tmp_path, monkeypatch)
    monkeypatch.setenv(
        "FAKE_STDERR",
        "WARNING: proceeding, even though we could not create PATH aliases\n"
        "Error loading config.toml: invalid transport\n"
        "in `mcp_servers.other`\n\n",
    )
    monkeypatch.setenv("FAKE_EXIT", "1")
    args = ["connect", "codex", "--app", "--base-url", url, "--no-vsr", "--yes"]
    result = CliRunner().invoke(main, [*args, "--check"])
    assert result.exit_code == 1
    output = _flat(result.output)
    assert (
        "Harness failed codex exited with status 1 before it ran the task: Error"
        " loading config.toml: invalid transport in `mcp_servers.other`" in output
    )
    assert "Login not run" in output and "Tool round trip not run" in output
    assert "Login failed" not in output


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
    # the replay list is read from the management port this config names
    management = {"management_api": {"port": urllib.parse.urlsplit(url).port}}
    config.write_text(
        yaml.safe_dump({**REVIEW_DOC, "global": {"services": management}})
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.setenv("VLLM_SR_API_KEY", "sk-secret")
    monkeypatch.setenv("FAKE_EVENTS", json.dumps(CODEX_EVENTS))
    monkeypatch.setenv("FAKE_REVIEW_URL", url + "/v1/responses")
    args = ["connect", "codex", "--config", str(config), "--base-url", url]
    result = CliRunner().invoke(main, [*args, "--no-vsr", "--yes", "--check"])
    assert result.exit_code == 0, result.output
    output = _flat(result.output)
    assert "Harness ok codex-cli 0.166.0 at " in output
    assert "Tool round trip ok ran cat vsr-check.txt" in output
    assert "Approval review ok went to vllm-sr/review (replay r" in output
    ran = json.loads(checked_harness.read_text())
    assert ran["args"][:3] == ["exec", "--profile", "vllm-sr"]
    assert "--approve-for-me" in ran["args"] and "--ephemeral" in ran["args"]
    assert ran["args"][-1] == hk.REVIEW_PROMPT.format(tool="exec_command")
    assert "with your exec_command tool outside" in ran["args"][-1]
    assert ran["stdin_closed"]

    # without the Router's replay list the review is not observed, which passes
    management["management_api"]["port"] = 9
    config.write_text(
        yaml.safe_dump({**REVIEW_DOC, "global": {"services": management}})
    )
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
        "memory model vllm-sr/auto",
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
    claude, codex, pi = json.loads(result.output)["connections"]
    assert pi["connected"] is False
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
    # without a memory_model, memories go to the session model
    assert codex["memory_model"] == "vllm-sr/auto"
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


def test_status_names_the_project_settings_that_win_in_this_directory(
    tmp_path, gateway, monkeypatch
):
    url, _, _ = gateway
    project = tmp_path / "project"
    (project / ".claude").mkdir(parents=True)
    monkeypatch.chdir(project)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.setattr("shutil.which", lambda _: None)
    for harness in ("claude-code", "codex"):
        args = ["connect", harness, "--base-url", url, "--no-vsr", "--yes"]
        assert CliRunner().invoke(main, args).exit_code == 0
    local = project / ".claude" / "settings.local.json"
    local.write_text('{"env": {"ANTHROPIC_MODEL": "opus", "OTHER": "1"}}')
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    (project / ".codex").mkdir()
    toml = project / ".codex" / "config.toml"
    toml.write_text('model = "gpt-x"\n')

    # reported, as connect warns, but the gateway still answers
    result = CliRunner().invoke(main, ["connect", "status"])
    assert result.exit_code == 0, result.output
    output = _flat(result.output)
    assert f"overridden here {local} sets ANTHROPIC_MODEL" in output
    assert f"overridden here {toml} sets model, once Codex trusts" in output
    result = CliRunner().invoke(main, ["connect", "status", "--json"])
    assert result.exit_code == 0, result.output
    claude, codex, _ = json.loads(result.output)["connections"]
    assert claude["overrides"] == [{"path": str(local), "keys": ["ANTHROPIC_MODEL"]}]
    assert codex["overrides"] == [{"path": str(toml), "keys": ["model"]}]
    assert claude["probe"]["state"] == "ok"

    # elsewhere the user settings hold
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(main, ["connect", "status", "--json"])
    claude, codex, _ = json.loads(result.output)["connections"]
    assert claude["overrides"] == codex["overrides"] == []


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
    claude, codex = json.loads(result.output)["connections"][:2]
    assert claude["probe"] == {
        "state": "not probed",
        "detail": "the settings no longer hold the ANTHROPIC_BASE_URL connect"
        f" wrote, so their key is not sent to {url}",
    }
    profile = tmp_path / "codex" / "vllm-sr.config.toml"
    assert codex["probe"] == {
        "state": "not probed",
        "detail": f"{profile} no longer holds what connect wrote, so no key is"
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


def _git_state(repo: Path) -> list[str]:
    """The branches, worktrees and checked-out branch of repo."""
    return [
        subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
        ).stdout
        for args in (
            ["for-each-ref", "refs/heads"],
            ["worktree", "list", "--porcelain"],
            ["symbolic-ref", "HEAD"],
        )
    ]


def test_launch_starts_the_harness_in_a_worktree_with_its_home_or_another(
    tmp_path, gateway, harness_bin, monkeypatch
):
    url, _, _ = gateway
    tmp_path = tmp_path.resolve()
    repo, linked = tmp_path / "repo", tmp_path / "linked"
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "gitconfig"))
    git = [
        "git",
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.com",
        "-C",
        str(repo),
    ]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "start"], check=True)
    subprocess.run([*git, "worktree", "add", "-q", str(linked)], check=True)
    before = _git_state(repo)
    own, other = tmp_path / "own-home", tmp_path / "other-home"
    other.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(own))
    monkeypatch.setenv("CODEX_HOME", str(own))
    for index, harness in enumerate(["claude-code", "codex"]):
        args = ["connect", harness, "--base-url", url, "--launch"]
        result = CliRunner().invoke(main, [*args, "--worktree", str(linked)])
        assert result.exit_code == 3, result.output
        ran = json.loads(harness_bin.read_text())
        # the user's own home, so the harness keeps its own login
        assert ran["cwd"] == str(linked) and ran["homes"][index] == str(own)
        header = " ".join(result.output.split())
        assert f"harness {harness} model vllm-sr/auto gateway {url} " in header
        assert f"worktree {linked} branch linked home {own} started " in header

        result = CliRunner().invoke(
            main, [*args, "--worktree", str(linked), "--home", str(other)]
        )
        assert result.exit_code == 3, result.output
        assert json.loads(harness_bin.read_text())["homes"][index] == str(other)
        assert f"home {other} " in " ".join(result.output.split())
    # nothing was put in either home, and no branch or worktree changed
    assert list(other.iterdir()) == [] and not own.exists()
    assert _git_state(repo) == before

    launch = ["connect", "codex", "--base-url", url, "--launch"]
    result = CliRunner().invoke(main, [*launch, "--worktree", str(other)])
    assert result.exit_code == 2 and "is not inside a git work tree" in result.output
    for flag in ("--worktree", "--home"):
        result = CliRunner().invoke(main, ["connect", "codex", flag, str(linked)])
        assert result.exit_code == 2 and f"{flag} goes with --launch" in result.output


@pytest.mark.parametrize("harness", ["claude-code", "codex"])
def test_launch_runs_the_key_command_in_the_worktree_and_home_it_starts(
    tmp_path, gateway, harness_bin, monkeypatch, harness
):
    url, seen, _ = gateway
    tmp_path = tmp_path.resolve()
    repo, ran_in = tmp_path / "repo", tmp_path / "ran-in"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    own, other = tmp_path / "own-home", tmp_path / "other-home"
    for home, key in [(own, "sk-own"), (other, "sk-other")]:
        home.mkdir()
        (home / "key").write_text(f"{key}\n")
    variable = {"claude-code": "CLAUDE_CONFIG_DIR", "codex": "CODEX_HOME"}[harness]
    helper = f'pwd > {ran_in}; cat "${variable}/key"'
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(own))
    monkeypatch.setenv("CODEX_HOME", str(own))
    args = ["connect", harness, "--base-url", url, "--key-command", helper]
    result = CliRunner().invoke(
        main, [*args, "--launch", "--worktree", str(repo), "--home", str(other)]
    )
    assert result.exit_code == 3, result.output
    # the probe sent the key the harness's own run of the command gets
    assert {auth for _, _, auth in seen} == {"Bearer sk-other"}
    assert ran_in.read_text() == f"{repo}\n"
    assert json.loads(harness_bin.read_text())["cwd"] == str(repo)
    assert "sk-" not in result.output


def _committed_repo(tmp_path, monkeypatch) -> tuple[Path, list[str], str]:
    """A repository on main with one commit, the git command that commits in
    it, and that commit."""
    repo = tmp_path / "repo"
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "gitconfig"))
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-C"]
    start = [*git, str(repo), "commit", "-q", "--allow-empty", "-m", "start"]
    subprocess.run(start, check=True)
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return repo, git, head


@pytest.mark.parametrize("harness", ["claude-code", "codex"])
def test_launch_in_a_new_worktree_branches_from_head_and_tags_every_request(
    tmp_path, gateway, harness_bin, monkeypatch, harness
):
    url, _, _ = gateway
    tmp_path = tmp_path.resolve()
    repo, _, head = _committed_repo(tmp_path, monkeypatch)
    (repo / "uncommitted.txt").write_text("stays in the checkout\n")
    monkeypatch.chdir(repo)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "home"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "x-team: red\nx-vsr-session-id: old")
    # the new worktree has no settings of its own yet: the checkout's are read
    local = {"env": {"ANTHROPIC_CUSTOM_HEADERS": "x-team: blue\nx-vsr-session-id: x"}}
    (repo / ".claude").mkdir()
    (repo / ".claude" / "settings.local.json").write_text(json.dumps(local))
    args = ["connect", harness, "--base-url", url, "--launch", "--worktree"]
    linked, before = tmp_path / "repo-fix", _git_state(repo)

    # a dry run, and a gateway that fails the probe, make nothing
    result = CliRunner().invoke(main, [*args, "new:fix", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert f"git -C {repo} worktree add -b vllm-sr/fix {linked} {head}" in result.output
    result = CliRunner().invoke(main, [*args[:3], url + "/nope", *args[4:], "new:fix"])
    assert result.exit_code == 1, result.output
    assert _git_state(repo) == before and not linked.exists()
    assert not harness_bin.exists()

    result = CliRunner().invoke(main, [*args, "new:fix"])
    assert result.exit_code == 3, result.output
    ran = json.loads(harness_bin.read_text())
    assert ran["cwd"] == str(linked)
    assert not (linked / "uncommitted.txt").exists()
    listed = subprocess.run(
        ["git", "-C", str(repo), "worktree", "list", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert f"worktree {linked}\nHEAD {head}\nbranch refs/heads/vllm-sr/fix\n" in listed
    session = re.search(rf"^Session ({UUID})$", result.output, re.M)[1]
    header = " ".join(result.output.split())
    assert f"harness {harness} model vllm-sr/auto gateway {url} " in header
    made = f"worktree {linked} (made by connect from {head[:12]}) branch vllm-sr/fix "
    assert made in header
    # every request carries the session: the Router pins it from
    # x-vsr-session-id, and Claude Code's own session id is the same one
    if harness == "claude-code":
        sent = json.loads(ran["settings"])["env"]["ANTHROPIC_CUSTOM_HEADERS"]
        assert sent == f"x-team: blue\nx-vsr-session-id: {session}"
        assert ran["args"][2:4] == ["--session-id", session]
        carried = "x-vsr-session-id, x-claude-code-session-id"
    else:
        tag = f"model_providers.vllm-sr.http_headers.x-vsr-session-id={session!r}"
        assert tag.replace("'", '"') in ran["args"]
        carried = "x-vsr-session-id"
    assert header.endswith(f"requests {carried}: {session}")
    # the docs say where Router Replay files the calls; the header does not
    # guess at the Router's identity settings, given or not
    learning = {"protection": {"identity": {"headers": {"session": "session-id"}}}}
    config = tmp_path / "router.yaml"
    config.write_text(yaml.safe_dump(_router(learning=learning)))
    result = CliRunner().invoke(main, [*args[:-1], "--config", str(config)])
    assert result.exit_code == 3, result.output
    assert "replay" not in result.output

    # its worktree removed, the branch is still taken, and git says so; a
    # directory in the way is refused first, since git would make the branch
    # before it refuses
    harness_bin.unlink()
    remove = ["git", "-C", str(repo), "worktree", "remove", str(linked)]
    subprocess.run(remove, check=True)
    result = CliRunner().invoke(main, [*args, "new:fix"])
    assert result.exit_code == 1, result.output
    name = "claude" if harness == "claude-code" else harness
    assert result.output.endswith(
        "git worktree add failed: a branch named 'vllm-sr/fix' already exists,"
        f" so {name} was not started.\n"
    )
    (tmp_path / "repo-taken").mkdir()
    (tmp_path / "repo-taken" / "kept.txt").write_text("kept\n")
    before = _git_state(repo)
    result = CliRunner().invoke(main, [*args, "new:taken"])
    assert result.exit_code == 2 and "repo-taken already exists" in result.output
    assert _git_state(repo) == before and not harness_bin.exists()
    result = CliRunner().invoke(main, [*args, "new"])
    assert result.exit_code == 3, result.output
    made = re.search(rf"branch\s+vllm-sr/({harness}-\d{{8}}-\d{{6}})", result.output)
    assert json.loads(harness_bin.read_text())["cwd"] == f"{repo}-{made[1]}"
    for bad in ("new:../up", "new:"):
        result = CliRunner().invoke(main, [*args, bad])
        assert result.exit_code == 2 and "is not a worktree name" in result.output
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(main, [*args, "new"])
    assert result.exit_code == 2 and "not inside a git checkout" in result.output


def test_a_conversation_claude_code_resumes_keeps_its_own_session_id():
    assert hs.claude_session_args(["-p", "hi"], "s") == ["--session-id", "s"]
    for picked in (["--resume", "x"], ["-c"], ["--session-id=x"], ["--fork-session"]):
        assert hs.claude_session_args(picked, "s") == []


def test_connect_sessions_lists_each_launch_and_how_to_clean_it_up(
    tmp_path, gateway, harness_bin, monkeypatch
):
    url, _, _ = gateway
    tmp_path = tmp_path.resolve()
    repo, git, head = _committed_repo(tmp_path, monkeypatch)
    linked = tmp_path / "repo-fix"
    monkeypatch.chdir(repo)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "home"))
    launch = ["connect", "claude-code", "--base-url", url, "--launch"]
    result = CliRunner().invoke(main, launch)
    assert result.exit_code == 3, result.output
    first = re.search(r"^Session (\S+)$", result.output, re.M)[1]
    result = CliRunner().invoke(main, [*launch, "--worktree", "new:fix"])
    assert result.exit_code == 3, result.output
    made = re.search(r"^Session (\S+)$", result.output, re.M)[1]

    def listed():
        result = CliRunner().invoke(main, ["connect", "sessions"])
        assert result.exit_code == 0, result.output
        return " ".join(result.output.split())

    remove = f"git -C {repo} worktree remove {linked}"
    delete = f"git -C {repo} branch -d vllm-sr/fix"
    text = listed()
    # the newest first; a launch in a checkout of the user's own is not cleaned up
    assert text.index(f"Session {made}") < text.index(f"Session {first}")
    assert f"worktree {linked} (made by connect from {head[:12]})" in text
    assert f"state no uncommitted changes, no unmerged commits cleanup {remove}" in text
    assert f"then {delete}" in text
    assert f"worktree {repo} branch main" in text
    assert "none: this launch did not make the worktree" in text

    # connect never suggests removing work: a change, then a commit, stop it
    (linked / "notes.txt").write_text("work\n")
    text = listed()
    assert "state 1 uncommitted change, no unmerged commits cleanup" in text
    assert "commit or discard the changes in it first" in text and remove not in text
    subprocess.run([*git, str(linked), "add", "notes.txt"], check=True)
    subprocess.run([*git, str(linked), "commit", "-q", "-m", "work"], check=True)
    text = listed()
    assert f"state no uncommitted changes, 1 unmerged commit cleanup {remove}" in text
    assert (
        "then push or merge vllm-sr/fix before deleting it" in text
        and delete not in text
    )

    subprocess.run(
        ["git", "-C", str(repo), "worktree", "remove", str(linked)], check=True
    )
    text = listed()
    assert "state worktree removed, 1 unmerged commit cleanup push or merge" in text
    # merged into main, the branch holds nothing main lacks, and the step works
    merge = [*git, str(repo), "merge", "-q", "--ff-only", "vllm-sr/fix"]
    subprocess.run(merge, check=True)
    text = listed()
    assert f"state worktree removed, no unmerged commits cleanup {delete}" in text
    subprocess.run(delete.split(), check=True, capture_output=True)
    assert "state worktree removed, branch deleted cleanup nothing left" in listed()

    result = CliRunner().invoke(main, ["connect", "sessions", "--json"])
    assert result.exit_code == 0, result.output
    sessions = json.loads(result.output)["sessions"]
    assert sessions[0]["id"] == made
    assert sessions[0] | {"state": None} == {
        **sessions[0],
        "harness": "claude-code",
        "worktree": str(linked),
        "branch": "vllm-sr/fix",
        "repository": str(repo),
        "base": head,
        "made_by_connect": True,
        "state": None,
    }
    assert sessions[0]["state"]["worktree"] == "removed"


def test_a_claude_code_launch_keeps_the_custom_headers_its_settings_set(
    tmp_path, gateway, harness_bin, monkeypatch
):
    # Claude Code 2.1.325 takes ANTHROPIC_CUSTOM_HEADERS whole from its local,
    # project or user settings, the first that sets it, else from the shell,
    # and the launch's --settings would replace them all
    url, _, _ = gateway
    home, project = tmp_path / "home", tmp_path / "project"
    (project / ".claude").mkdir(parents=True)
    monkeypatch.chdir(project)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(home))
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "x-from: shell")
    launch = ["connect", "claude-code", "--base-url", url, "--launch"]

    def sent(*args):
        result = CliRunner().invoke(main, [*launch, *args])
        assert result.exit_code == 3, result.output
        session = re.search(rf"^Session ({UUID})$", result.output, re.M)[1]
        settings = json.loads(json.loads(harness_bin.read_text())["settings"])
        return settings["env"]["ANTHROPIC_CUSTOM_HEADERS"], session

    def write(path, headers):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"env": {"ANTHROPIC_CUSTOM_HEADERS": headers}}))

    headers, session = sent()
    assert headers == f"x-from: shell\nx-vsr-session-id: {session}"
    write(home / "settings.json", "x-from: user\nx-vsr-session-id: old")
    headers, session = sent()
    assert headers == f"x-from: user\nx-vsr-session-id: {session}"
    write(project / ".claude" / "settings.json", "x-from: project")
    assert sent()[0].startswith("x-from: project\n")
    write(project / ".claude" / "settings.local.json", "x-from: local")
    assert sent()[0].startswith("x-from: local\n")
    # another home is read in place of the user's
    other = tmp_path / "other"
    write(other / "settings.json", "x-from: other")
    (project / ".claude" / "settings.json").unlink()
    (project / ".claude" / "settings.local.json").unlink()
    assert sent("--home", str(other))[0].startswith("x-from: other\n")


def test_connect_sessions_offers_no_cleanup_that_loses_detached_or_ignored_work(
    tmp_path, gateway, harness_bin, monkeypatch
):
    url, _, _ = gateway
    tmp_path = tmp_path.resolve()
    repo, git, _ = _committed_repo(tmp_path, monkeypatch)
    (repo / ".gitignore").write_text(".env\n")
    subprocess.run([*git, str(repo), "add", ".gitignore"], check=True)
    subprocess.run([*git, str(repo), "commit", "-q", "-m", "ignore"], check=True)
    linked = tmp_path / "repo-fix"
    monkeypatch.chdir(repo)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "home"))
    launch = ["connect", "claude-code", "--base-url", url, "--launch"]
    result = CliRunner().invoke(main, [*launch, "--worktree", "new:fix"])
    assert result.exit_code == 3, result.output

    def listed():
        result = CliRunner().invoke(main, ["connect", "sessions"])
        assert result.exit_code == 0, result.output
        return " ".join(result.output.split())

    remove = f"git -C {repo} worktree remove {linked}"
    # a file git ignores is deleted by the removal: the list says so first
    (linked / ".env").write_text("TOKEN=kept\n")
    text = listed()
    assert "state no uncommitted changes, 1 ignored path, no unmerged commits" in text
    assert (
        "cleanup keep what you need of the 1 ignored path there, which the"
        f" removal deletes, then {remove}, then git -C {repo} branch -d" in text
    )
    # a commit on a detached HEAD is on no branch: no removal is offered
    subprocess.run(["git", "-C", str(linked), "checkout", "-q", "--detach"], check=True)
    (linked / "work.txt").write_text("work\n")
    subprocess.run([*git, str(linked), "add", "work.txt"], check=True)
    subprocess.run([*git, str(linked), "commit", "-q", "-m", "work"], check=True)
    text = listed()
    assert "1 ignored path, 1 detached commit, no unmerged commits" in text
    assert "cleanup put the commits on its detached HEAD on a branch first" in text
    assert remove not in text
    # on a branch, the commits outlive the worktree
    subprocess.run(["git", "-C", str(linked), "switch", "-q", "-c", "kept"], check=True)
    text = listed()
    assert "1 ignored path, no unmerged commits cleanup keep what" in text
    assert remove in text


def test_the_sessions_directory_is_private(tmp_path, state_home):
    sessions = state_home / "vllm-sr" / "sessions"
    hs.record("s1", "codex", hs.Workspace(tmp_path), tmp_path, "http://gw", "m")
    assert stat.S_IMODE(sessions.stat().st_mode) == 0o700
    assert stat.S_IMODE((sessions / "s1.json").stat().st_mode) == 0o600
    # one an earlier version made readable is made private
    sessions.chmod(0o755)
    hs.record("s2", "codex", hs.Workspace(tmp_path), tmp_path, "http://gw", "m")
    assert stat.S_IMODE(sessions.stat().st_mode) == 0o700


def test_a_launch_needs_git_only_for_a_worktree(
    tmp_path, gateway, harness_bin, monkeypatch
):
    url, _, _ = gateway
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PATH", str(harness_bin.parent / "bin"))
    launch = ["connect", "codex", "--base-url", url, "--launch"]
    result = CliRunner().invoke(main, launch)
    assert result.exit_code == 3, result.output
    assert json.loads(harness_bin.read_text())["cwd"] == str(tmp_path)
    for worktree in ("new", str(tmp_path)):
        result = CliRunner().invoke(main, [*launch, "--worktree", worktree])
        assert result.exit_code == 2, result.output
        assert "a worktree needs git, which is not on PATH" in result.output


def test_connect_sessions_lists_a_launch_without_git_and_its_state_as_unknown(
    tmp_path, gateway, harness_bin, monkeypatch
):
    url, _, _ = gateway
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "home"))
    # a worktree connect made while git was there, then a launch without git
    made = hs.Workspace(tmp_path / "repo-fix", "vllm-sr/fix", tmp_path / "repo", "a1")
    hs.record("s-made", "codex", made, tmp_path / "home", url, "vllm-sr/auto")
    monkeypatch.setenv("PATH", str(harness_bin.parent / "bin"))
    launch = ["connect", "codex", "--base-url", url, "--launch"]
    assert CliRunner().invoke(main, launch).exit_code == 3
    result = CliRunner().invoke(main, ["connect", "sessions"])
    assert result.exit_code == 0, result.output
    text = _flat(result.output)
    unknown = "git state unknown: git is not on PATH, and the state needs it"
    assert f"worktree {tmp_path} home" in text
    assert f"state {unknown} cleanup none: this launch did not make the" in text
    assert (
        f"state worktree removed, {unknown} cleanup unknown: git is not on PATH,"
        " and the cleanup steps need it" in text
    )
    result = CliRunner().invoke(main, ["connect", "sessions", "--json"])
    states = [s["state"] for s in json.loads(result.output)["sessions"]]
    assert states == [
        {"worktree": "present", "git": "not on PATH"},
        {"worktree": "removed", "git": "not on PATH"},
    ]


def test_a_key_command_gets_no_longer_than_the_harness_gives_it(
    tmp_path, gateway, monkeypatch
):
    url, seen, _ = gateway
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.setattr("shutil.which", lambda _: None)
    monkeypatch.setitem(hs.KEY_COMMAND_TIMEOUTS, "codex", 0.2)
    args = ["connect", "codex", "--base-url", url, "--no-vsr", "--yes"]
    result = CliRunner().invoke(main, [*args, "--key-command", "sleep 2; echo sk"])
    assert result.exit_code == 1, result.output
    assert "did not finish in 0.2 s, the time codex gives it" in result.output
    assert not seen and not (tmp_path / "codex").exists()


def test_claude_code_writes_a_key_command_as_its_helper_and_never_the_key(
    tmp_path, gateway, monkeypatch
):
    url, seen, _ = gateway
    runs, keyfile = tmp_path / "runs", tmp_path / "key"
    keyfile.write_text("sk-helper\n")
    helper = f"echo run >> {runs}; cat {keyfile}"
    settings = tmp_path / "claude" / "settings.json"
    settings.parent.mkdir()
    settings.write_text('{"theme": "dark"}')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(settings.parent))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.setenv("VLLM_SR_API_KEY", "sk-env")  # the command wins over it
    monkeypatch.setattr("shutil.which", lambda _: None)
    args = ["connect", "claude-code", "--base-url", url, "--key-command", helper]
    result = CliRunner().invoke(main, [*args, "--dry-run"])
    assert result.exit_code == 0, result.output
    assert not runs.exists()  # a dry run probes nothing, so needs no key

    result = CliRunner().invoke(main, [*args, "--yes"])
    assert result.exit_code == 0, result.output
    held = json.loads(settings.read_text())
    assert held["apiKeyHelper"] == helper and held["theme"] == "dark"
    # both win over the helper, so both are empty
    assert held["env"]["ANTHROPIC_AUTH_TOKEN"] == held["env"]["ANTHROPIC_API_KEY"] == ""
    record = (settings.parent / hc.MANIFEST).read_text()
    assert "sk-" not in settings.read_text() + record + result.output
    assert "Claude Code runs" in result.output and "(apiKeyHelper)" in result.output
    # run once, for the probe, which sent what it printed
    assert runs.read_text() == "run\n"
    assert {auth for _, _, auth in seen} == {"Bearer sk-helper"}

    asked = len(seen)
    result = CliRunner().invoke(main, ["connect", "status", "--json"])
    assert result.exit_code == 0, result.output
    claude = json.loads(result.output)["connections"][0]
    assert (claude["login"], claude["key_command"]) == ("key command", helper)
    assert claude["probe"]["state"] == "ok" and "sk-" not in result.output
    assert [auth for _, _, auth in seen[asked:]] == ["Bearer sk-helper"]
    result = CliRunner().invoke(main, ["connect", "status"])
    assert f"login gateway key from the command {helper}" in _flat(result.output)

    # a run with a key in its place takes the helper out
    result = CliRunner().invoke(main, [*args[:4], "--yes"])
    assert result.exit_code == 0, result.output
    held = json.loads(settings.read_text())
    assert "apiKeyHelper" not in held
    assert held["env"]["ANTHROPIC_AUTH_TOKEN"] == "sk-env"
    result = CliRunner().invoke(main, ["connect", "claude-code", "--undo"])
    assert result.exit_code == 0, result.output
    assert json.loads(settings.read_text()) == {"theme": "dark"}

    for broken, why in [("exit 4", "exited with status 4"), ("true", "printed no key")]:
        result = CliRunner().invoke(main, [*args[:4], "--key-command", broken, "--yes"])
        assert result.exit_code == 1
        assert f"--key-command {why}, so nothing was written" in _flat(result.output)
    assert json.loads(settings.read_text()) == {"theme": "dark"}


def test_codex_takes_a_key_command_as_its_providers_auth_command(
    tmp_path, gateway, harness_bin, monkeypatch
):
    tomllib = pytest.importorskip("tomllib")
    url, seen, _ = gateway
    keyfile = tmp_path / "key"
    keyfile.write_text("sk-helper\n")
    helper = f"cat {keyfile}"
    home = tmp_path / "codex"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CODEX_HOME", str(home))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    args = ["connect", "codex", "--base-url", url, "--key-command", helper]
    result = CliRunner().invoke(main, [*args, "--no-vsr", "--yes"])
    assert result.exit_code == 0, result.output
    profile = home / "vllm-sr.config.toml"
    provider = tomllib.loads(profile.read_text())["model_providers"]["vllm-sr"]
    # Codex refuses auth beside env_key or requires_openai_auth (0.166)
    assert provider["auth"] == {"command": "/bin/sh", "args": ["-c", helper]}
    assert "env_key" not in provider and "requires_openai_auth" not in provider
    assert {auth for _, _, auth in seen} == {"Bearer sk-helper"}
    assert "sk-" not in profile.read_text() + result.output
    assert "Codex runs" in result.output

    result = CliRunner().invoke(main, ["connect", "status", "--json"])
    assert result.exit_code == 0, result.output
    codex = json.loads(result.output)["connections"][1]
    assert (codex["login"], codex["key_command"]) == ("key command", helper)
    assert codex["probe"]["state"] == "ok" and "sk-" not in result.output

    # --launch gives the same table as overrides, and the key nowhere
    result = CliRunner().invoke(main, [*args, "--launch"])
    assert result.exit_code == 3, result.output
    ran = json.loads(harness_bin.read_text())
    overrides = dict(arg.split("=", 1) for arg in ran["args"] if "=" in arg)
    assert overrides["model_providers.vllm-sr.auth.command"] == '"/bin/sh"'
    assert json.loads(overrides["model_providers.vllm-sr.auth.args"]) == ["-c", helper]
    assert "model_providers.vllm-sr.env_key" not in overrides
    assert "sk-" not in " ".join(ran["args"]) + result.output
    assert CliRunner().invoke(main, ["connect", "codex", "--undo"]).exit_code == 0
    assert not profile.exists()


def test_a_check_hides_the_key_its_key_command_printed(
    tmp_path, gateway, checked_harness, monkeypatch
):
    url, _, _ = gateway
    keyfile = tmp_path / "key"
    keyfile.write_text("sk-helper-123\n")
    helper = f"cat {keyfile}"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    refused = "API Error: 401 the key sk-helper-123 is not valid"
    events = [
        {
            "type": "assistant",
            "message": {
                "model": "<synthetic>",
                "content": [{"type": "text", "text": refused}],
            },
        },
        {"type": "result", "is_error": True, "result": refused},
    ]
    monkeypatch.setenv("FAKE_EVENTS", json.dumps(events))
    monkeypatch.setenv("FAKE_EXIT", "1")
    args = ["connect", "claude-code", "--base-url", url, "--key-command", helper]
    result = CliRunner().invoke(main, [*args, "--dry-run", "--check"])
    assert result.exit_code == 1
    assert "Login failed API Error: 401 the key <hidden> is not valid" in _flat(
        result.output
    )
    assert "sk-helper" not in result.output
    # the harness runs the helper from the settings it is given
    ran = json.loads(checked_harness.read_text())
    assert json.loads(ran["settings"])["apiKeyHelper"] == helper


@pytest.mark.parametrize("harness", ["claude-code", "codex"])
def test_a_key_command_is_refused_for_a_caller_login_and_leaves_vsr_out(
    tmp_path, harness_bin, monkeypatch, harness
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    provider = "anthropic" if harness == "claude-code" else "openai"
    config = _caller_config(tmp_path, provider)
    args = ["connect", harness, "--base-url", "http://localhost:8899"]
    result = CliRunner().invoke(
        main, [*args, "--config", str(config), "--key-command", "printf k"]
    )
    assert result.exit_code == 2 and "own login instead" in result.output

    config.write_text(
        "listeners:\n  - port: 8899\n    apis:\n      systemone:\n        models: [vela3-lite]\n"
    )
    result = CliRunner().invoke(
        main, [*args, "--config", str(config), "--key-command", "printf k", "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "vSR is not set up: its plugin takes a stored key or a variable" in _flat(
        result.output
    )


# --- Pi and the Codex app ----------------------------------------------------------

PI_EVENTS = [
    {"type": "session", "version": 3, "id": "s1", "cwd": "/w"},
    {"type": "agent_start"},
    {
        "type": "message_end",
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "toolCall",
                    "id": "c1",
                    "name": "bash",
                    "arguments": {"command": "cat vsr-check.txt"},
                }
            ],
            "model": "vllm-sr/auto",
            "responseModel": "qwen36",
            "stopReason": "toolUse",
        },
    },
    {
        "type": "tool_execution_end",
        "toolCallId": "c1",
        "toolName": "bash",
        "result": {"content": [{"type": "text", "text": TOKEN + "\n"}]},
        "isError": False,
    },
    {
        "type": "message_end",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": TOKEN}],
            "model": "vllm-sr/auto",
            "responseModel": "qwen36",
            "stopReason": "stop",
        },
    },
    {"type": "agent_end", "messages": []},
]


def _homes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))


def test_pi_provider_keeps_the_other_providers_and_undo_puts_back_the_old_one(
    tmp_path,
):
    models = tmp_path / "models.json"
    theirs = {"baseUrl": "http://old/v1", "api": "openai-completions", "apiKey": "sk-1"}
    other = {"baseUrl": "http://localhost:11434/v1", "apiKey": "ollama"}
    models.write_text(json.dumps({"providers": {"ollama": other, "vllm-sr": theirs}}))
    entry = hp.provider("http://gw:8899/", "vllm-sr/auto", "VLLM_SR_API_KEY", 65536)
    change = hp.models_change(tmp_path, entry)
    # the provider it replaces may hold a key, which the diff never shows
    old = change.shown(change.old_text(), "")
    assert "sk-1" not in old and '"apiKey": "<hidden>"' in old
    assert '"apiKey": "$VLLM_SR_API_KEY"' in change.shown(change.new_text(), "")
    hc.apply_changes([change], tmp_path)
    providers = json.loads(models.read_text())["providers"]
    assert providers["ollama"] == other
    assert providers["vllm-sr"] == {
        "name": "vLLM Semantic Router",
        "baseUrl": "http://gw:8899/v1",
        "api": "openai-completions",
        "apiKey": "$VLLM_SR_API_KEY",
        "compat": {
            "sendSessionAffinityHeaders": True,
            "sessionAffinityFormat": "openrouter",
        },
        "models": [{"id": "vllm-sr/auto", "maxTokens": 32000, "contextWindow": 65536}],
    }
    hc.undo_changes(tmp_path)
    assert json.loads(models.read_text()) == {
        "providers": {"ollama": other, "vllm-sr": theirs}
    }

    models.unlink()
    hc.apply_changes([change], tmp_path)
    hc.undo_changes(tmp_path)
    assert not models.exists()
    # a listener without keys gets a placeholder, which Pi needs to list the model
    keyless = hp.provider("http://gw", "m", None, None)
    assert keyless["apiKey"] == hc.NO_KEY
    assert keyless["models"] == [{"id": "m", "maxTokens": 32000}]


def test_cli_connects_pi_through_chat_completions_and_undoes(
    tmp_path, gateway, pi_home, monkeypatch
):
    url, seen, _ = gateway
    _homes(tmp_path, monkeypatch)
    monkeypatch.setenv("VLLM_SR_API_KEY", "sk-secret")
    models = pi_home / "models.json"
    args = ["connect", "pi", "--base-url", url]
    result = CliRunner().invoke(main, [*args, "--dry-run"])
    assert result.exit_code == 0, result.output
    assert '"apiKey": "$VLLM_SR_API_KEY"' in result.output
    assert not models.exists() and not seen

    result = CliRunner().invoke(main, [*args, "--yes"])
    assert result.exit_code == 0, result.output
    output = _flat(result.output)
    assert "Basic reply: ok (model agent-model" in output
    assert "Streamed tool loop: ok" in output
    assert "Pi reads the key from $VLLM_SR_API_KEY" in output
    assert "Start Pi with: pi --provider vllm-sr --model vllm-sr/auto" in output
    assert "sk-secret" not in result.output + models.read_text()
    assert {(path, auth) for path, _, auth in seen} == {
        ("/v1/chat/completions", "Bearer sk-secret")
    }
    # the tool result goes back as Pi sends it, after the call it answers
    answer = seen[-1][1]["messages"]
    assert [m["role"] for m in answer] == ["user", "assistant", "tool"]
    assert answer[1]["tool_calls"][0]["function"] == {
        "name": "get_time",
        "arguments": "{}",
    }
    assert answer[2]["tool_call_id"] == "c1"

    result = CliRunner().invoke(main, ["connect", "status", "--json"])
    assert result.exit_code == 0, result.output
    (pi,) = [
        e for e in json.loads(result.output)["connections"] if e["harness"] == "pi"
    ]
    assert (pi["gateway"], pi["model"], pi["login"], pi["key_env"]) == (
        url,
        "vllm-sr/auto",
        "gateway key",
        "VLLM_SR_API_KEY",
    )
    assert pi["files"] == [{"path": str(models), "state": "as written"}]
    assert pi["probe"] == {"state": "ok", "detail": "model agent-model"}
    result = CliRunner().invoke(main, ["connect", "status"])
    assert "pi (provider vllm-sr)" in result.output

    result = CliRunner().invoke(main, ["connect", "pi", "--undo"])
    assert result.exit_code == 0, result.output
    assert not models.exists()
    # Pi reads providers only from models.json, so nothing launches it unwritten
    result = CliRunner().invoke(main, ["connect", "pi", "--launch"])
    assert result.exit_code == 2 and "No such option" in result.output


def test_pi_takes_a_key_command_as_its_api_key_and_never_the_key(
    tmp_path, gateway, pi_home, monkeypatch
):
    url, seen, _ = gateway
    _homes(tmp_path, monkeypatch)
    monkeypatch.delenv("VLLM_SR_API_KEY", raising=False)
    secret = tmp_path / "key"
    secret.write_text("sk-from-file\n")
    command = f"cat {secret}"
    args = ["connect", "pi", "--base-url", url, "--key-command", command, "--yes"]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    text = (pi_home / "models.json").read_text()
    assert json.loads(text)["providers"]["vllm-sr"]["apiKey"] == f"!{command}"
    assert "sk-from-file" not in result.output + text
    assert f"Pi runs {command}" in _flat(result.output)
    assert {auth for _, _, auth in seen} == {"Bearer sk-from-file"}


def _pi_logins(pi_home, **logins):
    """Pi's auth.json with a made-up OAuth login for each provider named."""
    pi_home.mkdir(parents=True, exist_ok=True)
    stored = {
        name: {"type": "oauth", "access": access, "refresh": "r", "expires": 4e12}
        for name, access in logins.items()
    }
    (pi_home / "auth.json").write_text(json.dumps(stored))
    return (pi_home / "auth.json").read_text()


@pytest.mark.parametrize(
    ("caller", "login", "api", "path"),
    [
        ("anthropic", "anthropic", "anthropic-messages", ""),
        ("openai", "openai-codex", "openai-completions", "/v1"),
    ],
)
def test_pi_sends_its_own_login_to_a_caller_gateway_of_its_provider(
    tmp_path, gateway, pi_home, monkeypatch, caller, login, api, path
):
    url, seen, _ = gateway
    _homes(tmp_path, monkeypatch)
    # a gateway key in the shell never replaces the login
    monkeypatch.setenv("VLLM_SR_API_KEY", "sk-secret")
    stored = _pi_logins(pi_home, **{login: "the-login-token"})
    config = _caller_config(tmp_path, caller)
    args = ["connect", "pi", "--config", str(config), "--base-url", url]
    result = CliRunner().invoke(main, [*args, "--key-command", "cat key", "--yes"])
    assert result.exit_code == 2
    assert "this gateway takes Pi's own login instead" in _flat(result.output)

    result = CliRunner().invoke(main, [*args, "--yes"])
    assert result.exit_code == 0, result.output
    models = pi_home / "models.json"
    held = json.loads(models.read_text())["providers"]["vllm-sr"]
    # Pi runs the command it offers external clients, which prints its stored
    # login and refreshes it, before each request
    assert held["apiKey"] == f"!pi auth print-bearer-token --provider {login}"
    assert (held["api"], held["baseUrl"]) == (api, url + path)
    if api == "anthropic-messages":
        # the Router refuses tools[].eager_input_streaming, Pi's default there
        assert held["compat"] == {"supportsEagerToolInputStreaming": False}
    output = _flat(result.output)
    assert f"Pi sends its own {login} login" in output
    assert "Not probed: the gateway answers only on the login Pi itself" in output
    for secret in ("the-login-token", "sk-secret"):
        assert secret not in result.output + models.read_text()
    assert not seen

    result = CliRunner().invoke(main, ["connect", "status", "--json"])
    assert result.exit_code == 0, result.output
    (pi,) = [
        e for e in json.loads(result.output)["connections"] if e["harness"] == "pi"
    ]
    assert (pi["login"], pi["pi_login"], pi["key_command"]) == (
        "own login",
        login,
        None,
    )
    assert pi["probe"] == {
        "state": "not probed",
        "detail": "the gateway answers only on the login Pi itself sends",
    }
    status = CliRunner().invoke(main, ["connect", "status"]).output
    assert f"Pi's own login ({login})" in status
    assert "the-login-token" not in result.output + status and not seen

    assert CliRunner().invoke(main, ["connect", "pi", "--undo"]).exit_code == 0
    assert not models.exists()
    assert (pi_home / "auth.json").read_text() == stored


@pytest.mark.parametrize(
    ("caller", "stored"),
    [
        # a login reaches only the provider that issued it
        ("anthropic", {"openai-codex": {"type": "oauth", "access": "the-login"}}),
        ("openai", {"anthropic": {"type": "oauth", "access": "the-login"}}),
        # a key Pi stored is no login
        ("anthropic", {"anthropic": {"type": "api_key", "key": "the-login"}}),
        ("openai", None),
    ],
)
def test_pi_refuses_a_caller_gateway_without_a_login_of_its_provider(
    tmp_path, gateway, pi_home, monkeypatch, caller, stored
):
    url, seen, _ = gateway
    _homes(tmp_path, monkeypatch)
    if stored is not None:
        pi_home.mkdir(parents=True)
        (pi_home / "auth.json").write_text(json.dumps(stored))
    config = _caller_config(tmp_path, caller)
    args = ["connect", "pi", "--config", str(config), "--base-url", url, "--yes"]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 1
    output = _flat(result.output)
    assert f"own {caller} login upstream, and Pi holds no {caller} login" in output
    login = {"anthropic": "anthropic", "openai": "openai-codex"}[caller]
    assert f"pi auth login {login}" in output
    assert "the-login" not in output
    assert not (pi_home / "models.json").exists() and not seen


def test_pi_takes_chatgpt_for_codex_before_chatgpt_for_the_api(tmp_path):
    _pi_logins(tmp_path, openai="api")
    assert hp.login(tmp_path, "openai") == "openai"
    _pi_logins(tmp_path, openai="api", **{"openai-codex": "codex"})
    assert hp.login(tmp_path, "openai") == "openai-codex"
    assert hp.login(tmp_path, "anthropic") is None


def test_pi_refuses_a_key_of_its_own_and_a_keyed_listener_without_one(
    tmp_path, gateway, pi_home, monkeypatch
):
    url, seen, _ = gateway
    _homes(tmp_path, monkeypatch)
    monkeypatch.delenv("VLLM_SR_API_KEY", raising=False)
    config = tmp_path / "pi.yaml"
    config.write_text("listeners:\n  - {port: 8899, api_keys: [listener-key]}\n")
    result = CliRunner().invoke(main, ["connect", "pi", "--config", str(config)])
    assert result.exit_code == 1
    assert "takes an API key and $VLLM_SR_API_KEY is not set" in result.output
    # Pi sends a key it stored for the provider before the one models.json names
    pi_home.mkdir(parents=True, exist_ok=True)
    (pi_home / "auth.json").write_text('{"vllm-sr": {"type": "api_key", "key": "old"}}')
    monkeypatch.setenv("VLLM_SR_API_KEY", "sk-secret")
    args = ["connect", "pi", "--base-url", url, "--yes"]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 1
    assert "Pi holds a key of its own for vllm-sr in auth.json" in _flat(result.output)
    assert not (pi_home / "models.json").exists() and not seen

    (pi_home / "auth.json").write_text("{}")
    assert CliRunner().invoke(main, args).exit_code == 0
    (pi_home / "auth.json").write_text('{"vllm-sr": {"type": "api_key", "key": "old"}}')
    result = CliRunner().invoke(main, ["connect", "status", "--json"])
    assert result.exit_code == 1, result.output
    (pi,) = [
        e for e in json.loads(result.output)["connections"] if e["harness"] == "pi"
    ]
    assert pi["probe"]["state"] == "failed"
    assert "auth.json" in pi["probe"]["detail"]


def test_pi_models_with_comments_are_refused_by_name(
    tmp_path, gateway, pi_home, monkeypatch
):
    url, _, _ = gateway
    _homes(tmp_path, monkeypatch)
    monkeypatch.setenv("VLLM_SR_API_KEY", "sk-secret")
    pi_home.mkdir(parents=True, exist_ok=True)
    (pi_home / "models.json").write_text('{\n  // mine\n  "providers": {}\n}\n')
    result = CliRunner().invoke(main, ["connect", "pi", "--base-url", url, "--yes"])
    assert result.exit_code == 1
    assert "is not plain JSON" in _flat(result.output)
    assert "remove them or edit the file yourself" in _flat(result.output)


def test_status_sends_no_key_once_pi_or_the_codex_app_points_elsewhere(
    tmp_path, gateway, pi_home, monkeypatch
):
    url, seen, _ = gateway
    _homes(tmp_path, monkeypatch)
    monkeypatch.setenv("VLLM_SR_API_KEY", "sk-secret")
    home = tmp_path / "codex"
    args = ["--base-url", url, "--no-vsr", "--yes"]
    assert CliRunner().invoke(main, ["connect", "codex", "--app", *args]).exit_code == 0
    assert (
        CliRunner()
        .invoke(main, ["connect", "pi", "--base-url", url, "--yes"])
        .exit_code
        == 0
    )
    config, models = home / "config.toml", pi_home / "models.json"
    config.write_text(config.read_text().replace(url, "http://other:1"))
    models.write_text(models.read_text().replace(url, "http://other:1"))
    asked = len(seen)
    result = CliRunner().invoke(main, ["connect", "status", "--json"])
    found = {e["harness"]: e for e in json.loads(result.output)["connections"]}
    for harness, path in (("codex", config), ("pi", models)):
        assert found[harness]["probe"] == {
            "state": "not probed",
            "detail": f"{path} no longer holds what connect wrote, so no key is"
            f" sent to {url}",
        }
    assert seen[asked:] == []


def test_the_codex_app_keeps_a_profiles_catalog(
    tmp_path, gateway, harness_bin, monkeypatch
):
    url, _, _ = gateway
    _homes(tmp_path, monkeypatch)
    home = tmp_path / "codex"
    args = ["connect", "codex", "--base-url", url, "--no-vsr", "--yes"]
    assert CliRunner().invoke(main, [*args, "--model", "vllm-sr/a"]).exit_code == 0
    catalog = home / "vllm-sr.models.json"
    before = catalog.read_text()
    assert (
        CliRunner().invoke(main, [*args, "--app", "--model", "vllm-sr/b"]).exit_code
        == 0
    )
    assert catalog.read_text() == before
    slugs = [
        m["slug"]
        for m in json.loads((home / "vllm-sr-app.models.json").read_text())["models"]
    ]
    assert slugs == ["vllm-sr/b"]


def test_exit_fails_on_a_pi_turn_that_ended_in_an_error():
    hk = importlib.import_module("cli.harness_check")
    events = [
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "stopReason": "error",
                "errorMessage": "HTTP 401",
            },
        },
    ]
    row = hk._exit_row("pi", events, "", 0, False)
    assert (row.state, row.detail) == ("failed", "pi reported a failure: HTTP 401")
    # a retry that answered leaves the run a success
    events.append(
        {"type": "message_end", "message": {"role": "assistant", "stopReason": "stop"}}
    )
    assert hk._exit_row("pi", events, "", 0, False).state == "ok"


def test_pi_checks_only_the_models_its_entrypoint_reaches(
    tmp_path, gateway, pi_home, monkeypatch
):
    url, _, _ = gateway
    _homes(tmp_path, monkeypatch)
    monkeypatch.setenv("VLLM_SR_API_KEY", "sk-secret")
    config = tmp_path / "pi.yaml"
    recipe = lambda name, model: {  # noqa: E731
        "name": name,
        "decisions": [{"name": name, "modelRefs": [{"model": model}]}],
    }
    config.write_text(
        yaml.safe_dump(
            {
                "providers": {
                    "models": [
                        {"name": "claude", "api_format": "messages"},
                        {"name": "local", "api_format": "chat_completions"},
                    ]
                },
                "entrypoints": [
                    {"model_names": ["vllm-sr/auto"], "recipe": "agent"},
                    {"model_names": ["vllm-sr/chat"], "recipe": "chat"},
                ],
                "recipes": [recipe("agent", "claude"), recipe("chat", "local")],
            }
        )
    )
    args = ["connect", "pi", "--config", str(config), "--base-url", url, "--yes"]
    chat = CliRunner().invoke(main, [*args, "--model", "vllm-sr/chat"])
    assert chat.exit_code == 0, chat.output
    assert "reasoning.signature" not in chat.output
    auto = CliRunner().invoke(main, [*args, "--model", "vllm-sr/auto"])
    assert "reasoning.signature" in auto.output


def _two_entrypoints(caller=False, limit=16384):
    local = {
        "name": "local",
        "api_format": "chat_completions",
        "max_output_tokens": limit,
    }
    login = {"name": "login", "api_format": "responses", "max_output_tokens": 128000}
    if caller:
        login["backend_refs"] = [
            {
                "endpoint": "https://chatgpt.com/backend-api/codex",
                "provider": "openai",
                "auth": {"credential": "caller"},
            }
        ]

    def recipe(name, models):
        refs = [{"model": model} for model in models]
        return {"name": name, "decisions": [{"name": name, "modelRefs": refs}]}

    return {
        "providers": {"models": [local, login]},
        "entrypoints": [
            {"model_names": ["vllm-sr/auto"], "recipe": "agent"},
            {"model_names": ["vllm-sr/chat"], "recipe": "chat"},
        ],
        "recipes": [recipe("agent", ["local", "login"]), recipe("chat", ["local"])],
    }


def test_pi_checks_the_login_and_limits_of_its_own_entrypoint(
    tmp_path, gateway, pi_home, monkeypatch
):
    url, _, _ = gateway
    _homes(tmp_path, monkeypatch)
    monkeypatch.setenv("VLLM_SR_API_KEY", "sk-secret")
    config = tmp_path / "pi.yaml"
    config.write_text(yaml.safe_dump(_two_entrypoints(caller=True)))
    args = ["connect", "pi", "--config", str(config), "--base-url", url, "--yes"]
    # a caller login only vllm-sr/auto reaches does not stop Pi on vllm-sr/chat
    chat = CliRunner().invoke(main, [*args, "--model", "vllm-sr/chat"])
    assert chat.exit_code == 0, chat.output
    held = json.loads((pi_home / "models.json").read_text())["providers"]["vllm-sr"]
    # the model refuses more output than its max_output_tokens
    assert held["models"][0]["maxTokens"] == 16384
    auto = CliRunner().invoke(main, [*args, "--model", "vllm-sr/auto"])
    assert auto.exit_code == 1 and "own openai login upstream" in _flat(auto.output)


@pytest.mark.parametrize(
    "harness, other", [("claude-code", "openai"), ("codex", "anthropic")]
)
def test_claude_code_and_codex_check_the_login_of_their_own_entrypoint(
    tmp_path, gateway, monkeypatch, harness, other
):
    url, _, _ = gateway
    _homes(tmp_path, monkeypatch)
    monkeypatch.setenv("VLLM_SR_API_KEY", "sk-secret")
    doc = _two_entrypoints(caller=True)
    # only vllm-sr/auto reaches the caller backend, on the other harness's login
    doc["providers"]["models"][1]["backend_refs"][0]["provider"] = other
    config = tmp_path / "router.yaml"
    config.write_text(yaml.safe_dump(doc))
    args = ["connect", harness, "--config", str(config), "--base-url", url]
    args += ["--no-vsr", "--yes"]
    chat = CliRunner().invoke(main, [*args, "--model", "vllm-sr/chat"])
    assert chat.exit_code == 0, chat.output
    auto = CliRunner().invoke(main, [*args, "--model", "vllm-sr/auto"])
    assert auto.exit_code == 1 and f"own {other} login upstream" in _flat(auto.output)


@pytest.mark.parametrize("role", ["review_model", "memory_model"])
def test_codex_checks_the_login_of_its_review_and_memory_entrypoints(
    tmp_path, gateway, harness_bin, monkeypatch, role
):
    url, _, _ = gateway
    _homes(tmp_path, monkeypatch)
    monkeypatch.setenv("VLLM_SR_API_KEY", "sk-secret")
    doc = _two_entrypoints(caller=True)
    # Codex asks vllm-sr/chat, whose models take a key, and calls vllm-sr/auto,
    # which reaches the caller backend, for its reviews or memories
    doc["entrypoints"][1][role] = "vllm-sr/auto"
    login = doc["providers"]["models"][1]["backend_refs"][0]
    login["provider"] = "anthropic"
    config = tmp_path / "router.yaml"
    config.write_text(yaml.safe_dump(doc))
    args = ["connect", "codex", "--config", str(config), "--base-url", url]
    args += ["--model", "vllm-sr/chat", "--no-vsr", "--yes"]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 1, result.output
    assert (
        f"own anthropic login upstream for vllm-sr/auto, the {role} of vllm-sr/chat,"
        " and Codex sends its openai login, so every call to it would be refused"
    ) in _flat(result.output)
    assert not (tmp_path / "codex" / "vllm-sr.config.toml").exists()
    # one that takes Codex's own login has Codex send it to every entrypoint
    login["provider"] = "openai"
    config.write_text(yaml.safe_dump(doc))
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    written = (tmp_path / "codex" / "vllm-sr.config.toml").read_text()
    assert "requires_openai_auth = true" in written and "env_key" not in written


def test_claude_code_asks_for_no_more_output_than_its_models_allow(
    tmp_path, gateway, monkeypatch
):
    url, _, _ = gateway
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("VLLM_SR_API_KEY", "sk-secret")
    config = tmp_path / "router.yaml"
    for limit, want in ((8192, "8192"), (64000, "32000")):
        config.write_text(yaml.safe_dump(_two_entrypoints(limit=limit)))
        args = ["connect", "claude-code", "--config", str(config), "--base-url", url]
        result = CliRunner().invoke(
            main, [*args, "--model", "vllm-sr/chat", "--no-vsr", "--yes"]
        )
        assert result.exit_code == 0, result.output
        env = json.loads((tmp_path / "claude" / "settings.json").read_text())["env"]
        assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == want


def test_one_model_needs_no_session_protection():
    doc = _two_entrypoints()

    def needs(model):
        return [f.need for f in rc.check_router_config(doc, "pi", model)]

    assert rc.PROTECTION_NEED in needs("vllm-sr/auto")
    assert rc.PROTECTION_NEED not in needs("vllm-sr/chat")
    # a recipe the document does not write shows no model, which is not one model
    doc["entrypoints"].append({"model_names": ["vllm-sr/x"], "recipe": "elsewhere"})
    assert rc.entrypoint_models(doc, "vllm-sr/x") == []
    assert rc.PROTECTION_NEED in needs("vllm-sr/x")


def test_a_preview_shows_a_key_variable_by_name():
    change = hc.Change(
        Path("models.json"), "json", {"providers/vllm-sr": {"apiKey": "!cat key"}}
    )
    shown = change.shown(
        '{"providers": {"vllm-sr": {"apiKey": "$VLLM_SR_API_KEY"}}}', ""
    )
    assert "$VLLM_SR_API_KEY" in shown
    secret = change.shown('{"providers": {"vllm-sr": {"apiKey": "sk-live"}}}', "")
    assert "sk-live" not in secret and "<hidden>" in secret


RELIABILITY_NEEDS = (
    "fallback",
    "a second model",
    "total_timeout",
    "reliability.total_timeout",
)


def _reliability_needs(doc, harness="claude-code", model=None):
    found = rc.check_router_config(doc, harness, model)
    return [f for f in found if f.need.startswith(RELIABILITY_NEEDS)]


def _need(doc, prefix, model=None):
    (found,) = [
        f for f in _reliability_needs(doc, model=model) if f.need.startswith(prefix)
    ]
    return found.need, yaml.safe_load(found.snippet)


def test_router_check_asks_every_harness_for_fallback_and_no_total_timeout():
    for harness in ("claude-code", "codex", "pi"):
        needs = _reliability_needs(_router(reliable=False), harness)
        assert [f.need.split(" ")[0] for f in needs] == ["fallback", "total_timeout:"]
        fallback, timeout = needs
        assert "fallback on decision d," in fallback.need and "10s" in fallback.need
        assert yaml.safe_load(fallback.snippet) == {
            "global": {"router": {"fallback": AGENT_FALLBACK}}
        }
        assert "m (300s, the listener's timeout)" in timeout.need
        assert yaml.safe_load(timeout.snippet) == {
            "providers": {
                "models": [{"name": "m", "reliability": {"total_timeout": "0s"}}]
            }
        }
        assert _reliability_needs(_router(), harness) == []


def test_fallback_finding_says_where_to_change_it():
    # fallback on with its default timeouts still ends a slow candidate
    doc = _router()
    doc["global"]["router"]["fallback"] = {"enabled": True}
    need, snippet = _need(doc, "fallback")
    assert "decision d," in need and snippet["global"]["router"]["fallback"] == (
        AGENT_FALLBACK
    )
    # fallback a decision turns on stays its own: the global block only adds room
    doc = _router()
    del doc["global"]["router"]["fallback"]
    doc["recipes"][0]["decisions"][0]["fallback"] = {"enabled": True}
    _, snippet = _need(doc, "fallback")
    assert snippet["global"]["router"]["fallback"] == {
        "total_timeout": "1200s",
        "per_attempt_timeout": "1200s",
    }
    # a recipe or decision that turns it off itself is where to turn it on
    doc = _router()
    doc["recipes"][0].update(name="agent", fallback={"enabled": False})
    need, snippet = _need(doc, "fallback")
    assert "recipe agent's own fallback" in need and snippet == {
        "fallback": AGENT_FALLBACK
    }
    decision = doc["recipes"][0]["decisions"][0]
    decision["fallback"] = {"enabled": False}
    need, snippet = _need(doc, "fallback")
    assert "decision d's own fallback" in need and snippet == {
        "fallback": AGENT_FALLBACK
    }
    decision["fallback"] = {"enabled": True, "per_attempt_timeout": "10s"}
    assert "decision d's own fallback" in _need(doc, "fallback")[0]
    decision["fallback"] = dict(AGENT_FALLBACK)
    assert _reliability_needs(doc) == []
    # one model leaves fallback nowhere to go
    decision["modelRefs"] = [{"model": "m"}]
    need, snippet = _need(doc, "a second model")
    assert "decision d names one" in need
    assert snippet == {"modelRefs": [{"model": "m"}, {"model": "<another model>"}]}
    # a request for a provider model by name takes no decision
    assert _reliability_needs(doc, model="m") == []
    # a decision that answers itself, such as a block, has no model to fall back
    # from and needs no fallback
    block = {"name": "block_marker", "plugins": [{"type": "fast_response"}]}
    doc["recipes"][0]["decisions"].append(block)
    assert not [f for f in _reliability_needs(doc) if "block_marker" in f.need]
    del decision["fallback"]
    doc["global"]["router"]["fallback"] = {"enabled": False}
    needs = _reliability_needs(doc)
    assert needs and not [f for f in needs if "block_marker" in f.need]


def test_timeout_finding_reads_the_decision_then_the_model_then_the_listener():
    doc = _router()
    doc["providers"]["models"][0]["reliability"] = {"total_timeout": "2m"}
    need, _ = _need(doc, "total_timeout")
    assert "m (2m, its total_timeout)" in need and "m2" not in need
    # a decision's override bounds the calls it routes instead
    decision = doc["recipes"][0]["decisions"][0]
    decision["reliability"] = {"total_timeout": "0s"}
    assert _reliability_needs(doc) == []
    decision["reliability"] = {"total_timeout": "120s"}
    need, snippet = _need(doc, "reliability.total_timeout")
    assert "decision d (120s)" in need and snippet == {
        "reliability": {"total_timeout": "0s"}
    }
    assert [f.need.split(" ")[0] for f in _reliability_needs(doc)] == [
        "reliability.total_timeout:"
    ]
    del decision["reliability"]
    doc["providers"]["models"][0]["reliability"] = {}
    doc["listeners"] = [{"name": "http", "port": 8899, "timeout": "900s"}]
    assert "m (900s" in _need(doc, "total_timeout")[0]
    for off in ("0s", 0):
        doc["listeners"][0]["timeout"] = off
        assert _reliability_needs(doc) == []


def test_router_check_for_pi():
    needs = [f.need for f in rc.check_router_config(_router(), "pi")]
    assert any("reasoning.signature" in n for n in needs)
    assert any("session protection" in n for n in needs)
    # Pi sends its own output limit, so no default is asked for
    assert not any("output limit" in n for n in needs)
    complete = _router(["reasoning.signature"], _learning(rc.PI_HEADERS))
    assert rc.check_router_config(complete, "pi") == []
    assert rc.PI_HEADERS == {"session": "x-session-id"}
    # on its own Anthropic login Pi speaks Messages, which keeps the signature,
    # and sends no session header
    login = _router()
    login["providers"]["models"][0]["backend_refs"] = [
        {"provider": "anthropic", "auth": {"credential": "caller"}}
    ]
    assert rc.check_router_config(login, "pi") == []


def test_check_runs_pi_on_the_provider_connect_wrote(
    tmp_path, gateway, checked_harness, pi_home, monkeypatch
):
    url, _, _ = gateway
    _homes(tmp_path, monkeypatch)
    monkeypatch.setenv("VLLM_SR_API_KEY", "sk-secret")
    monkeypatch.setenv("FAKE_EVENTS", json.dumps(PI_EVENTS))
    args = ["connect", "pi", "--base-url", url, "--check"]
    result = CliRunner().invoke(main, [*args, "--yes"])
    assert result.exit_code == 0, result.output
    output = _flat(result.output)
    assert "Harness ok 1.5.2 at " in output
    assert "Login ok the gateway answered, as qwen36" in output
    assert (
        "Tool round trip ok ran cat vsr-check.txt and answered with its output"
        in output
    )
    ran = json.loads(checked_harness.read_text())
    assert ran["file"] == TOKEN + "\n" and TOKEN not in ran["args"][-1]
    assert ran["args"][:7] == [
        "--mode",
        "json",
        "--provider",
        "vllm-sr",
        "--model",
        "vllm-sr/auto",
        "--tools",
    ]
    assert "--no-session" in ran["args"] and ran["args"][-1] == rc_prompt("bash")
    # Pi's only tool is bash: a model told of a shell tool answered without it
    assert "with your bash tool," in ran["args"][-1]
    # the user's own agent directory, with the provider written there; Pi takes
    # headers only from that file, which carries no check's mark
    assert ran["agent"] == str(pi_home)
    written = json.loads((pi_home / "models.json").read_text())
    assert "headers" not in written["providers"]["vllm-sr"]
    assert ran["stdin_closed"] and not Path(ran["cwd"]).exists()

    # a dry run checks the provider it would write, in an agent directory of its own
    (pi_home / "models.json").unlink()
    (pi_home / hc.MANIFEST).unlink()
    result = CliRunner().invoke(main, [*args, "--dry-run"])
    assert result.exit_code == 0, result.output
    ran = json.loads(checked_harness.read_text())
    assert ran["agent"] != str(pi_home) and not Path(ran["agent"]).exists()
    held = json.loads(ran["models"])["providers"]["vllm-sr"]
    assert held["baseUrl"] == url + "/v1" and held["apiKey"] == "$VLLM_SR_API_KEY"
    # the copy only the check reads marks its run as a check
    assert held["headers"] == {"x-vsr-probe": "connect"}
    assert not (pi_home / "models.json").exists()

    # a refused key is a failed login, with the key hidden
    refused = {
        "type": "message_end",
        "message": {
            "role": "assistant",
            "content": [],
            "stopReason": "error",
            "errorMessage": "401 the key sk-secret is not valid",
        },
    }
    monkeypatch.setenv("FAKE_EVENTS", json.dumps([refused]))
    result = CliRunner().invoke(main, [*args, "--dry-run"])
    assert result.exit_code == 1
    output = _flat(result.output)
    assert "Login failed 401 the key <hidden> is not valid" in output
    assert "Tool round trip not run" in output and "Nothing was written" in output


def test_check_runs_pi_on_its_own_login(
    tmp_path, gateway, checked_harness, pi_home, monkeypatch
):
    url, seen, _ = gateway
    _homes(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_EVENTS", json.dumps(PI_EVENTS))
    _pi_logins(pi_home, anthropic="the-login-token")
    config = _caller_config(tmp_path, "anthropic")
    args = ["connect", "pi", "--config", str(config), "--base-url", url, "--check"]
    command = "pi auth print-bearer-token --provider anthropic"
    # a dry run's agent directory holds only models.json, so its provider asks
    # Pi for the login stored in yours
    result = CliRunner().invoke(main, [*args, "--dry-run"])
    assert result.exit_code == 0, result.output
    ran = json.loads(checked_harness.read_text())
    assert ran["agent"] != str(pi_home)
    held = json.loads(ran["models"])["providers"]["vllm-sr"]
    pinned = f"PI_CODING_AGENT_DIR={shlex.quote(str(pi_home))} {command}"
    assert held["apiKey"] == f"!{pinned}"
    assert not (pi_home / "models.json").exists()

    result = CliRunner().invoke(main, [*args, "--yes"])
    assert result.exit_code == 0, result.output
    ran = json.loads(checked_harness.read_text())
    assert ran["agent"] == str(pi_home)
    assert json.loads(ran["models"])["providers"]["vllm-sr"]["apiKey"] == f"!{command}"
    assert "the-login-token" not in result.output and not seen


def test_codex_app_sets_the_shared_defaults_and_undo_keeps_the_rest(
    tmp_path, gateway, harness_bin, monkeypatch
):
    tomllib = pytest.importorskip("tomllib")
    url, _, _ = gateway
    _homes(tmp_path, monkeypatch)
    home = tmp_path / "codex"
    home.mkdir()
    auth = home / "auth.json"
    auth.write_text('{"tokens": "the login"}')
    config = home / "config.toml"
    before = (
        "# my settings\n"
        'model = "gpt-5"\n'
        "model_context_window = 400000\n"
        'approval_policy = "on-request"\n'
        "\n"
        '[projects."/src/app"]\n'
        'trust_level = "trusted"\n'
        "\n"
        "[model_providers.other]\n"
        'name = "Other"\n'
        'base_url = "http://other/v1"\n'
    )
    config.write_text(before)
    args = ["connect", "codex", "--app", "--base-url", url, "--no-vsr"]
    result = CliRunner().invoke(main, [*args, "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "the Codex CLI included" in _flat(result.output)
    assert config.read_text() == before
    preview = result.output

    result = CliRunner().invoke(main, [*args, "--yes"])
    assert result.exit_code == 0, result.output
    output = _flat(result.output)
    assert "The gateway is now Codex's default" in output
    # the preview is the change written, where it is written
    lines = [text.splitlines(keepends=True) for text in (before, config.read_text())]
    diff = difflib.unified_diff(*lines, str(config), str(config), n=0)
    assert "".join(diff) in preview
    assert result.output.rstrip().endswith("Start Codex with: codex")
    held = tomllib.loads(config.read_text())
    assert held["model"] == "vllm-sr/auto" and held["model_provider"] == "vllm-sr"
    assert "model_context_window" not in held  # the gateway's is unknown here
    assert held["model_catalog_json"] == str(home / "vllm-sr-app.models.json")
    assert held["approval_policy"] == "on-request"
    assert held["projects"] == {"/src/app": {"trust_level": "trusted"}}
    assert held["model_providers"]["other"]["name"] == "Other"
    assert held["model_providers"]["vllm-sr"]["base_url"] == url + "/v1"
    assert config.read_text().startswith("# my settings\n")
    assert not (home / "vllm-sr.config.toml").exists()
    assert auth.read_text() == '{"tokens": "the login"}'

    result = CliRunner().invoke(main, ["connect", "status", "--json"])
    (codex,) = [
        e for e in json.loads(result.output)["connections"] if e["harness"] == "codex"
    ]
    assert (codex["profile"], codex["gateway"], codex["model"]) == (
        None,
        url,
        "vllm-sr/auto",
    )
    assert codex["files"][0] == {"path": str(config), "state": "as written"}
    assert (
        "codex (config.toml, its default)"
        in CliRunner().invoke(main, ["connect", "status"]).output
    )

    # Codex adds a table of its own, and the person changes a setting of theirs
    config.write_text(
        config.read_text().replace("on-request", "never")
        + "\n[tui]\nscreen_reader = true\n"
    )
    result = CliRunner().invoke(main, ["connect", "codex", "--undo"])
    assert result.exit_code == 0, result.output
    undone = config.read_text()
    expected = before.replace("on-request", "never") + "\n[tui]\nscreen_reader = true\n"
    # each key gets its value back; one taken out returns beside the others
    assert tomllib.loads(undone) == tomllib.loads(expected)
    assert undone.startswith('# my settings\nmodel = "gpt-5"\n')
    assert "vllm-sr" not in undone
    assert auth.read_text() == '{"tokens": "the login"}'

    result = CliRunner().invoke(main, ["connect", "codex", "--app", "--launch"])
    assert result.exit_code == 2 and "does not take --app" in result.output


# Secrets the shared config.toml holds where hiding by setting name misses them:
# an MCP server's env, a provider's headers, a password in a URL argument.
THEIR_SECRETS = (
    'model = "gpt-5.5"\n'
    "\n"
    "[mcp_servers.brave]\n"
    'env = { "BRAVE_API_KEY" = "BSA-live-secret-123" }\n'
    'command = "npx"\n'
    "\n"
    "[mcp_servers.pg]\n"
    'command = "npx"\n'
    'args = ["-y", "@modelcontextprotocol/server-postgres",'
    ' "postgresql://admin:hunter2@db/prod"]\n'
    "[model_providers.other]\n"
    'base_url = "https://x"\n'
    'http_headers = { "X-Api-Key" = "xk-secret-999" }\n'
)


@pytest.mark.parametrize("ending", ["\n", ""])
def test_a_codex_app_preview_shows_no_line_connect_does_not_own(
    tmp_path, gateway, monkeypatch, ending
):
    url, _, _ = gateway
    _homes(tmp_path, monkeypatch)
    config = tmp_path / "codex" / "config.toml"
    config.parent.mkdir()
    # without a newline at its end, connect adds one to a line of theirs
    before = THEIR_SECRETS.removesuffix("\n") + ending
    config.write_text(before)
    args = ["connect", "codex", "--app", "--base-url", url, "--no-vsr"]
    preview = CliRunner().invoke(main, [*args, "--dry-run"])
    assert preview.exit_code == 0, preview.output
    for secret in ("BSA-live-secret-123", "hunter2", "xk-secret-999"):
        assert secret not in preview.output
    assert CliRunner().invoke(main, [*args, "--yes"]).exit_code == 0
    if ending:
        # the preview is the whole change, line numbers included, so patch applies it
        lines = [
            text.splitlines(keepends=True) for text in (before, config.read_text())
        ]
        diff = difflib.unified_diff(*lines, str(config), str(config), n=0)
        assert "".join(diff) in preview.output


def test_codex_app_check_runs_codex_without_a_profile(
    tmp_path, gateway, checked_harness, monkeypatch
):
    url, _, _ = gateway
    _homes(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_EVENTS", json.dumps(CODEX_EVENTS))
    args = ["connect", "codex", "--app", "--base-url", url, "--no-vsr", "--yes"]
    result = CliRunner().invoke(main, [*args, "--check"])
    assert result.exit_code == 0, result.output
    ran = json.loads(checked_harness.read_text())
    assert ran["args"][0] == "exec" and "--profile" not in ran["args"]
    # the check's run marks itself a check, through Codex's own provider headers
    marked = 'model_providers.vllm-sr.http_headers.x-vsr-probe="connect"'
    assert ran["args"][ran["args"].index(marked) - 1] == "-c"
    assert "x-vsr-probe" not in (tmp_path / "codex" / "config.toml").read_text()
    assert "with your exec_command tool," in ran["args"][-1]
    assert (tmp_path / "codex" / "config.toml").exists()


def test_taking_out_a_table_with_its_subtable_gives_back_the_file():
    harness_toml = importlib.import_module("cli.harness_toml")
    mine = "\n".join(
        [
            "[model_providers.vllm-sr]",
            'name = "vLLM Semantic Router"',
            "",
            "[model_providers.vllm-sr.auth]",
            'command = "/bin/sh"',
        ]
    )
    for text in ['# mine\n[model_providers.other]\nname = "Other"\n', "a = 1\n"]:
        written = harness_toml.set_values(text, {"model_providers.vllm-sr": mine})
        back = harness_toml.set_values(written, {"model_providers.vllm-sr": None})
        assert back == text


def test_a_toml_layout_that_cannot_be_edited_cleanly_is_left_alone():
    harness_toml = importlib.import_module("cli.harness_toml")
    pytest.importorskip("tomllib")
    wanted = {"model": 'model = "vllm-sr/auto"'}
    # a value written over several lines would be cut in half
    with pytest.raises(ValueError, match="left as it is"):
        harness_toml.set_values('model = """\ngpt-5\n"""\n', wanted)
    with pytest.raises(ValueError, match="not valid TOML"):
        harness_toml.set_values("model = \n", wanted)
    # a provider written as an inline table is not one a header can replace
    inline = '[model_providers]\nvllm-sr = { name = "mine" }\n'
    table = {"model_providers.vllm-sr": '[model_providers.vllm-sr]\nname = "x"'}
    with pytest.raises(ValueError, match="its layout could not be edited"):
        harness_toml.set_values(inline, table)
    # a top-level key goes after the others, before the first table
    text = "a = 1\n\n[t]\nb = 2\n"
    assert harness_toml.set_values(text, wanted) == (
        'a = 1\nmodel = "vllm-sr/auto"\n\n[t]\nb = 2\n'
    )
    assert harness_toml.values(text, ["a", "t", "model"]) == {
        "a": "a = 1",
        "t": None,  # a bare name is a top-level key
        "model": None,
    }


def test_codex_memories_ask_the_gateway_for_their_models(
    tmp_path, gateway, harness_bin, monkeypatch
):
    # Unset, Codex's memories ask for gpt-5.6-luna and gpt-5.6-terra, which the
    # gateway does not serve; every way connect sets Codex up names its model.
    tomllib = pytest.importorskip("tomllib")
    url, _, _ = gateway
    _homes(tmp_path, monkeypatch)
    home = tmp_path / "codex"
    home.mkdir()
    wanted = {"extract_model": "vllm-sr/auto", "consolidation_model": "vllm-sr/auto"}

    args = ["connect", "codex", "--base-url", url, "--no-vsr", "--yes"]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    profile = tomllib.loads((home / "vllm-sr.config.toml").read_text())
    assert profile["memories"] == wanted

    config = home / "config.toml"
    before = "[memories]\ngenerate_memories = false\nmax_unused_days = 30\n"
    config.write_text(before)
    result = CliRunner().invoke(main, [*args, "--app"])
    assert result.exit_code == 0, result.output
    assert tomllib.loads(config.read_text())["memories"] == {
        "generate_memories": False,
        "max_unused_days": 30,
        **wanted,
    }
    result = CliRunner().invoke(main, ["connect", "codex", "--undo"])
    assert result.exit_code == 0, result.output
    assert tomllib.loads(config.read_text()) == tomllib.loads(before)

    launch = hc.codex_launch_args(url, "vllm-sr/auto", None, None)
    assert "-c" in launch and 'memories.extract_model="vllm-sr/auto"' in launch
    assert 'memories.consolidation_model="vllm-sr/auto"' in launch


@pytest.mark.parametrize(
    "error, client_key",
    [
        # the listener, on a key it does not hold
        (
            {
                "message": "Unauthorized: missing or invalid API key",
                "type": "authentication_error",
                "code": "invalid_api_key",
            },
            True,
        ),
        # the Router past it, holding no key for the backend
        (
            {
                "message": "authentication failed",
                "type": "authentication_error",
                "code": "authentication_failed",
            },
            False,
        ),
        # the backend itself, on the key the Router sent it
        (
            {
                "message": "Incorrect API key provided: sk-up***",
                "type": "invalid_request_error",
                "code": "invalid_api_key",
            },
            False,
        ),
    ],
)
def test_a_failed_probe_says_what_failed_in_words(error, client_key):
    body = json.dumps({"error": error}).encode()

    class Refuses(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["content-length"]))
            self.send_response(401)
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Refuses)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    refused = rc.basic_reply(url, "codex", "vllm-sr/auto", "sk-wrong")
    server.shutdown()
    server.server_close()
    assert f"HTTP 401 {body.decode()}" in refused.detail
    assert ("the gateway did not accept the key" in refused.detail) is client_key
    down = rc.basic_reply(url, "codex", "vllm-sr/auto", "sk-wrong")
    assert down.detail == (
        f"{url}/v1/responses: no connection (Connection refused)."
        " Is the gateway running there?"
    )


def test_a_branch_git_cannot_count_is_never_offered_for_deletion(tmp_path, monkeypatch):
    repo, _, head = _committed_repo(tmp_path.resolve(), monkeypatch)
    subprocess.run(["git", "-C", str(repo), "branch", "vllm-sr/x"], check=True)
    session = {
        "worktree": str(repo.parent / "gone"),
        "made_by_connect": True,
        "repository": str(repo),
        "branch": "vllm-sr/x",
        "base": head,
    }
    found = hs.state(session)
    assert found == {"worktree": "removed", "unmerged": 0}
    assert hs.cleanup(session, found) == [f"git -C {repo} branch -d vllm-sr/x"]
    # a start commit git does not have: the count fails, and no step deletes
    found = hs.state({**session, "base": "0" * 40})
    assert found == {"worktree": "removed", "unmerged": None}
    assert hs.describe(found) == "worktree removed, unmerged commits unknown"
    assert hs.cleanup(session, found) == ["push or merge vllm-sr/x before deleting it"]
