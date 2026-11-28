"""AG8: scripted multi-round agent fixtures through the mock backend.

The mock's `--agent-fixtures` engine scripts the model's turns (tool calls on
the fixture's steps, final answer afterwards, per-model completion rates and
token budgets, cache-read usage), and agentic_probe.py plays the agent client.
Together they must be deterministic: two runs of the same fixture produce
identical normalized trajectories — the AG8 completion criterion.
"""

from __future__ import annotations

import importlib.util
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

_SMOKE_DIR = Path(__file__).resolve().parents[1] / "examples" / "smoke"
_FIXTURE_FILE = _SMOKE_DIR / "agentic_fixtures.json"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mock():
    return _load_module("mock_backend_ag8", _SMOKE_DIR / "mock_backend.py")


@pytest.fixture(scope="module")
def fixtures():
    return json.loads(_FIXTURE_FILE.read_text(encoding="utf-8"))


@pytest.fixture()
def agent_state(mock, fixtures):
    """Populate the scripted-agent tables from the shared fixture file."""
    tasks = {script["task_prompt"]: script for script in fixtures["tasks"].values()}
    mock._AGENT_TASKS.clear()
    mock._AGENT_TASKS.update(tasks)
    mock._AGENT_MODELS.clear()
    mock._AGENT_MODELS.update(fixtures["models"])
    yield mock
    mock._AGENT_TASKS.clear()
    mock._AGENT_MODELS.clear()


def _turn(mock, model, messages):
    scripted = mock._scripted_completion(model, messages)
    assert scripted is not None
    message, finish_reason, usage = scripted
    return message, finish_reason, usage


def test_fixture_sessions_reference_known_tasks(fixtures):
    tasks = fixtures["tasks"]
    for session in fixtures["sessions"]:
        assert session["task"] in tasks
        if "subtask_task" in session:
            assert session["subtask_task"] in tasks


def test_first_turn_emits_first_scripted_tool_call(agent_state):
    script = agent_state._AGENT_TASKS[
        "Fix the failing test in src/limiter.py, then run the full suite."
    ]
    message, finish_reason, usage = _turn(
        agent_state,
        "google/gemini-2.5-flash-lite",
        [{"role": "user", "content": script["task_prompt"]}],
    )
    assert finish_reason == "tool_calls"
    call = message["tool_calls"][0]
    assert call["function"]["name"] == "read_file"
    assert json.loads(call["function"]["arguments"]) == {"path": "src/limiter.py"}
    assert call["id"] == "call-0"
    # fresh conversation: nothing cached yet, but the cache-read field exists
    assert usage["prompt_tokens_details"] == {"cached_tokens": 0}
    assert usage["completion_tokens"] == 2  # flash-lite lane's per-step budget


def test_mid_loop_turns_advance_and_cache_grows(agent_state):
    prompt = "Fix the failing test in src/limiter.py, then run the full suite."
    messages = [
        {"role": "user", "content": prompt},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-0",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-0", "content": "ok"},
    ]
    message, finish_reason, usage = _turn(
        agent_state, "google/gemini-2.5-flash-lite", messages
    )
    assert finish_reason == "tool_calls"
    assert message["tool_calls"][0]["function"]["name"] == "edit_file"
    assert usage["prompt_tokens_details"]["cached_tokens"] > 0


def test_final_turn_after_tool_steps_completes(agent_state):
    prompt = "Fix the failing test in src/limiter.py, then run the full suite."
    messages: list[dict] = [{"role": "user", "content": prompt}]
    for _step in range(4):  # four scripted tool steps
        message, finish_reason, _ = _turn(
            agent_state, "google/gemini-2.5-flash-lite", messages
        )
        assert finish_reason == "tool_calls"
        messages.append(message)
        for call in message["tool_calls"]:
            messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": "ok"}
            )
    message, finish_reason, _ = _turn(
        agent_state, "google/gemini-2.5-flash-lite", messages
    )
    assert finish_reason == "stop"
    assert message["content"] == (
        "Fixed the off-by-one boundary check in src/limiter.py; all 12 tests pass."
    )


def test_per_model_token_budgets_differ(agent_state):
    prompt = "Summarise the project README in one sentence."
    messages = [{"role": "user", "content": prompt}]
    _, _, cheap_usage = _turn(agent_state, "google/gemini-2.5-flash-lite", messages)
    _, _, strong_usage = _turn(agent_state, "google/gemini-3.1-pro", messages)
    assert cheap_usage["prompt_tokens"] < strong_usage["prompt_tokens"]
    assert cheap_usage["completion_tokens"] < strong_usage["completion_tokens"]


def test_completion_rate_is_deterministic(agent_state):
    profile = dict(agent_state._MODEL_PROFILE_DEFAULTS, completion_rate=0.0)
    assert agent_state._model_completes("m", "task", profile) is False
    profile["completion_rate"] = 1.0
    assert agent_state._model_completes("m", "task", profile) is True
    profile["completion_rate"] = 0.5
    draws = {agent_state._model_completes("m", f"task-{i}", profile) for i in range(50)}
    assert draws == {True, False}  # mid rates decide per task, deterministically
    assert agent_state._model_completes("m", "task-1", profile) == (
        agent_state._model_completes("m", "task-1", profile)
    )


def test_incomplete_task_serves_abandonment(agent_state):
    agent_state._AGENT_TASKS["doomed"] = {
        "task_prompt": "doomed",
        "tool_call_steps": [],
        "final_content": "done",
        "abandon_content": "gave up",
        "complete": False,
    }
    message, finish_reason, _ = _turn(
        agent_state,
        "google/gemini-2.5-flash-lite",
        [{"role": "user", "content": "doomed"}],
    )
    assert finish_reason == "stop"
    assert message["content"] == "gave up"


def test_unscripted_prompt_falls_back_to_legacy(mock):
    assert (
        mock._scripted_completion(
            "google/gemini-2.5-flash-lite",
            [{"role": "user", "content": "capital of france?"}],
        )
        is None
    )


@pytest.fixture()
def probe():
    return _load_module("agentic_probe_ag8", _SMOKE_DIR / "agentic_probe.py")


@pytest.fixture()
def live_mock_backend(mock, fixtures):
    """The real mock Handler on an ephemeral port, with AG8 tables loaded."""
    mock._AGENT_TASKS.clear()
    mock._AGENT_TASKS.update({s["task_prompt"]: s for s in fixtures["tasks"].values()})
    mock._AGENT_MODELS.clear()
    mock._AGENT_MODELS.update(fixtures["models"])
    server = ThreadingHTTPServer(("127.0.0.1", 0), mock.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()
    mock._AGENT_TASKS.clear()
    mock._AGENT_MODELS.clear()


def test_probe_repeat_two_trajectories_identical(
    probe, live_mock_backend, tmp_path, capsys
):
    import sys

    out = tmp_path / "out"
    sys.argv = [
        "agentic_probe.py",
        "--output-dir",
        str(out),
        "--router",
        live_mock_backend,
        "--mgmt",
        "http://127.0.0.1:1",  # unreachable: replay dump degrades to empty
        "--repeat",
        "2",
    ]
    assert probe.main() == 0
    transcript = json.loads((out / "transcript-run1.json").read_text("utf-8"))
    # simple=1, long-loop=5, sub-agent=3 main + 2 sub, correction=2
    assert len(transcript) == 13
    assert [e["finish_reason"] for e in transcript if e["fixture"] == "long-loop"] == [
        "tool_calls",
        "tool_calls",
        "tool_calls",
        "tool_calls",
        "stop",
    ]
    sub = [e for e in transcript if e["agent"] == "sub"]
    assert len(sub) == 2
    assert all("x-claude-code-agent-id" in e["headers_sent"] for e in sub)
    # Z9: every turn carries a deterministic per-request id, and the run
    # writes the sub-agent manifest the diagnosis joins the protection
    # traces with
    assert all(e["request_id"] for e in transcript)
    assert transcript[0]["request_id"] == "probe-simple-main-0-run1"
    manifest = json.loads((out / "agent-requests-run1.json").read_text("utf-8"))
    assert manifest["agent_requests"] == [
        {
            "request_id": "probe-sub-agent-sub-0-run1",
            "agent": "sub",
            "fixture": "sub-agent",
            "step": 0,
        },
        {
            "request_id": "probe-sub-agent-sub-1-run1",
            "agent": "sub",
            "fixture": "sub-agent",
            "step": 1,
        },
    ]
    correction = [e for e in transcript if e["fixture"] == "correction"]
    assert len(correction) == 2
    assert "DETERMINISM" not in capsys.readouterr().err


def test_probe_detects_trajectory_mismatch(probe):
    base = {
        "fixture": "simple",
        "agent": "main",
        "step": 0,
        "status": 200,
        "finish_reason": "stop",
        "model": "body:m",
        "tool_calls": [],
        "content": "ok",
        "headers_sent": {"x-session-id": "s1", "x-conversation-id": "s1"},
    }
    drifted = dict(base, content="different final answer")
    assert probe.compare_transcripts([base], [base]) == []
    assert probe.compare_transcripts([base], [drifted])
    assert probe.compare_transcripts([base], [base, base])


def test_probe_replay_rows_carry_session_identity(
    probe, live_mock_backend, tmp_path, monkeypatch
):
    """Replay rows for one session must be attributable by identity headers:
    run the probe twice with different tags and check the transcripts stayed
    per-session (the shape the H20 replay comparison relies on)."""
    import sys

    out = tmp_path / "out"
    sys.argv = [
        "agentic_probe.py",
        "--output-dir",
        str(out),
        "--router",
        live_mock_backend,
        "--mgmt",
        "http://127.0.0.1:1",
        "--fixtures",
        "simple",
        "--run-tag",
        "solo",
    ]
    assert probe.main() == 0
    transcript = json.loads((out / "transcript-solo.json").read_text("utf-8"))
    assert len(transcript) == 1
    identity = "probe-simple-solo"
    assert transcript[0]["headers_sent"]["x-session-id"] == identity
    assert transcript[0]["headers_sent"]["x-conversation-id"] == identity


def test_parse_headers_rejects_garbage(probe):
    assert probe.parse_headers(["A: b"]) == {"A": "b"}
    with pytest.raises(SystemExit):
        probe.parse_headers(["novalue"])
