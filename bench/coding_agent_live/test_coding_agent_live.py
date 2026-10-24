"""Offline checks for the Claude Code lane: command, alias mapping, report and join."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def _load(name: str):
    """Load a sibling script by path, so the tests run without Harbor installed."""
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"coding_agent_live_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run = _load("run")
report = _load("report")

SESSION = "4f0c2a7e-0000-4000-8000-000000000001"


def _result_event(**fields) -> str:
    event = {"type": "result", "subtype": "success", "session_id": SESSION}
    event.update(fields)
    return json.dumps(event, separators=(",", ":"))


def _write_trial(job: Path, name: str, result: dict, stream: list[str]) -> Path:
    trial = job / name
    (trial / "agent").mkdir(parents=True)
    (trial / "result.json").write_text(json.dumps(result))
    (trial / "agent" / "claude-code.txt").write_text("\n".join(stream) + "\n")
    return trial


def test_command_pins_the_version_and_maps_each_alias_to_an_entrypoint():
    args = run.parse_args(
        [
            "--dataset=tasks",
            "--task=fix-git",
            "--task=log-summary",
            "--router-url=http://172.17.0.1:8899",
            "--model=vllm-sr/parent",
            "--haiku-model=vllm-sr/small",
            "--subagent-model=",
            "--job-name=lane",
        ]
    )
    command = run.harbor_command(args)
    assert command[:2] == ["harbor", "run"]
    assert command[command.index("-a") + 1] == run.AGENT_IMPORT_PATH
    assert "version=2.1.304" in command
    assert "ANTHROPIC_BASE_URL=http://172.17.0.1:8899" in command
    assert [command[i + 1] for i, part in enumerate(command) if part == "-i"] == [
        "fix-git",
        "log-summary",
    ]
    # An alias without its own flag must still name a Router model: Claude
    # Code would otherwise send its own model id, which no entrypoint serves.
    assert run.lane_env(args) == {
        "LANE_ANTHROPIC_DEFAULT_OPUS_MODEL": "vllm-sr/parent",
        "LANE_ANTHROPIC_DEFAULT_SONNET_MODEL": "vllm-sr/parent",
        "LANE_ANTHROPIC_DEFAULT_HAIKU_MODEL": "vllm-sr/small",
        "LANE_CLAUDE_CODE_SUBAGENT_MODEL": "",
    }


def test_lane_env_overrides_harbors_single_model_aliases_in_the_container():
    # Harbor sets every alias and the subagent default to the run's model.
    harbor_env = {
        "ANTHROPIC_MODEL": "vllm-sr/parent",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "vllm-sr/parent",
        "ANTHROPIC_DEFAULT_OPUS_MODEL": "vllm-sr/parent",
        "CLAUDE_CODE_SUBAGENT_MODEL": "vllm-sr/parent",
    }
    environ = {
        "LANE_ANTHROPIC_DEFAULT_HAIKU_MODEL": "vllm-sr/small",
        "LANE_CLAUDE_CODE_SUBAGENT_MODEL": "",
    }
    assert run.apply_lane_env(harbor_env, environ) == {
        "ANTHROPIC_MODEL": "vllm-sr/parent",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "vllm-sr/small",
        "ANTHROPIC_DEFAULT_OPUS_MODEL": "vllm-sr/parent",
    }
    assert harbor_env["CLAUDE_CODE_SUBAGENT_MODEL"] == "vllm-sr/parent"


def test_dry_run_prints_the_command_without_running_harbor(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise AssertionError("dry run started Harbor")

    monkeypatch.setattr(run.subprocess, "run", fail)
    code = run.main(
        [
            "--dataset=tasks",
            "--router-url=http://172.17.0.1:8899",
            "--model=m",
            "--job-name=j",
            "--dry-run",
        ]
    )
    assert code == 0
    assert "harbor run" in capsys.readouterr().out


def test_last_result_event_skips_noise_and_keeps_the_final_result():
    lines = [
        "installing claude code",
        '{"type":"assistant","message":{"content":"the word \\"type\\":\\"result\\" in text"}}',
        "{not json",
        _result_event(terminal_reason="max_turns"),
        _result_event(terminal_reason="completed"),
    ]
    event = report.last_result_event(lines)
    assert event["terminal_reason"] == "completed"
    assert report.last_result_event(["no events here"]) == {}


def test_api_errors_lists_each_message_once_in_order():
    first = "API Error: 400 translation would lose tool_result.is_error"
    lines = [
        json.dumps({"type": "assistant", "text": first + "\nretry later"}),
        json.dumps({"type": "result", "result": first}),
        json.dumps({"text": "API Error: 400 System message must be at the beginning."}),
        "plain line",
    ]
    assert report.api_errors(lines) == [
        first,
        "API Error: 400 System message must be at the beginning.",
    ]


def test_replay_summary_splits_main_and_subagent_calls_only_with_agent_identity():
    def record(model, tokens, agent=None, action="allow_switch", status=200):
        identity = {"session": {"status": "present"}}
        if agent is not None:
            identity["agent"] = {"status": agent}
        return {
            "selected_model": model,
            "total_tokens": tokens,
            "response_status": status,
            "session_policy": {"action": action, "identity": identity},
        }

    with_identity = report.summarize_replays(
        [
            record("big", 100, agent="absent", action="establish_baseline"),
            record("small", 40, agent="present"),
            record("small", None, agent="present", status=400),
        ]
    )
    assert with_identity["calls"] == {"main:big": 1, "sub:small": 2}
    assert with_identity["tokens"] == {"main:big": 100, "sub:small": 40}
    assert with_identity["statuses"] == {"200": 2, "400": 1}
    assert with_identity["protection_actions"] == {
        "allow_switch": 2,
        "establish_baseline": 1,
    }
    # Without agent identity the Router cannot tell a subagent from its parent.
    assert report.summarize_replays([record("big", 5)])["calls"] == {"all:big": 1}


def test_fetch_replays_follows_pagination_for_one_session():
    pages = [
        {"data": [{"id": "a"}, {"id": "b"}], "has_more": True, "next_offset": 2},
        {"data": [{"id": "c"}], "has_more": False},
    ]
    seen = []

    def opener(request, timeout):
        seen.append(request)
        return io.BytesIO(json.dumps(pages[len(seen) - 1]).encode())

    records = report.fetch_replays(
        "http://127.0.0.1:8080/", SESSION, token="t", opener=opener
    )
    assert [r["id"] for r in records] == ["a", "b", "c"]
    queries = [parse_qs(urlparse(r.full_url).query) for r in seen]
    assert [q["offset"] for q in queries] == [["0"], ["2"]]
    assert all(q["session_id"] == [SESSION] for q in queries)
    assert urlparse(seen[0].full_url).path == "/api/v1/observability/replays"
    assert seen[0].get_header("Authorization") == "Bearer t"


def test_trial_row_keeps_the_reward_after_a_timeout_and_joins_replay(tmp_path):
    trial = _write_trial(
        tmp_path,
        "fix-git__abc",
        {
            "task_name": "fix-git",
            "trial_name": "fix-git__abc",
            "verifier_result": {"rewards": {"reward": 1}},
            "exception_info": {"exception_type": "AgentTimeoutError"},
        },
        [
            '{"type":"system","subtype":"init"}',
            _result_event(
                num_turns=12,
                duration_ms=90000,
                terminal_reason="completed",
                subagent_stats={"spawned": 2, "by_type": {"Explore": 1}},
            ),
        ],
    )
    asked = []

    def replays(session_id):
        asked.append(session_id)
        return [{"selected_model": "big", "total_tokens": 10, "response_status": 200}]

    row = report.trial_row(trial, replays)
    assert row["reward"] == 1
    assert row["exception"] == "AgentTimeoutError"
    assert (row["turns"], row["minutes"]) == (12, 1.5)
    assert (row["subagents_spawned"], row["subagents_by_type"]) == (2, {"Explore": 1})
    assert row["ended_on_api_error"] is False
    assert asked == [SESSION]
    assert row["router"]["calls"] == {"all:big": 1}


def test_run_that_ends_on_an_api_error_is_counted(tmp_path):
    message = "API Error: 400 translation would lose tool_result.is_error"
    _write_trial(
        tmp_path,
        "a__1",
        {"task_name": "a", "verifier_result": {"rewards": {"reward": 0}}},
        [json.dumps({"text": message}), _result_event(is_error=True, result=message)],
    )
    _write_trial(
        tmp_path,
        "b__1",
        {"task_name": "b", "verifier_result": {"rewards": {"reward": 1}}},
        [_result_event(terminal_reason="completed")],
    )
    # Harbor's job-level result.json names no task and is not a trial.
    (tmp_path / "result.json").write_text(json.dumps({"id": "job"}))
    rows = [report.trial_row(path, None) for path in report.trial_dirs(tmp_path)]
    assert [row["task"] for row in rows] == ["a", "b"]
    assert "router" not in rows[0]
    assert report.summarize_job(rows) == {
        "trials": 2,
        "passed": 1,
        "ended_on_api_error": 1,
        "exceptions": {},
        "subagents_spawned": 0,
    }


def test_trial_dirs_ignores_directories_without_a_trial_result(tmp_path):
    (tmp_path / "setup").mkdir()
    assert report.trial_dirs(tmp_path) == []
