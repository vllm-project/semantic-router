"""Offline checks for the Codex lane: command, settings, report, and join."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

SESSION = "4f0c2a7e-0000-4000-8000-000000000001"
THREAD = "9b1e6c0a-1111-4000-8000-000000000002"


def _load(name: str):
    """Load a sibling script by path, so the tests run without Harbor installed."""
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"coding_agent_live_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run = _load("codex_run")
report = _load("codex_report")


def _write_trial(job: Path, name: str, result: dict, stream: list[str]) -> Path:
    trial = job / name
    (trial / "agent").mkdir(parents=True)
    (trial / "result.json").write_text(json.dumps(result))
    (trial / "agent" / "codex.txt").write_text("\n".join(stream) + "\n")
    return trial


def test_command_pins_codex_and_points_at_the_responses_ingress():
    args = run.parse_args(
        [
            "--dataset=tasks",
            "--task=fix-git",
            "--task=log-summary",
            "--router-url=http://172.17.0.1:8899/",
            "--model=vllm-sr/parent",
            "--job-name=lane",
            "--jobs-dir=jobs",
        ]
    )
    command = run.harbor_command(args)
    assert command[:2] == ["harbor", "run"]
    assert command[command.index("-a") + 1] == run.AGENT_IMPORT_PATH
    assert "version=0.163.0" in command
    assert "web_search=disabled" in command
    assert "OPENAI_BASE_URL=http://172.17.0.1:8899/v1" in command
    config = command[command.index("--ak") + 1 :]
    assert any(
        part.startswith("config=") and part.endswith("lane.codex.toml")
        for part in config
    )
    assert [command[i + 1] for i, part in enumerate(command) if part == "-i"] == [
        "fix-git",
        "log-summary",
    ]
    text = run.render_config(args.model)
    assert text == (
        'model = "vllm-sr/parent"\nweb_search = "disabled"\n\n[features]\nmulti_agent = false\n'
    )


def test_responses_base_url_is_not_doubled():
    assert (
        run.responses_base_url("http://172.17.0.1:8899/v1")
        == "http://172.17.0.1:8899/v1"
    )
    assert (
        run.responses_base_url("http://172.17.0.1:8899/v1/")
        == "http://172.17.0.1:8899/v1"
    )


def test_merge_maps_every_model_setting_and_forces_the_protocol_guide():
    original = {
        "model": "gpt-5.1-codex",
        "review_model": "gpt-5.1-codex",
        "web_search": "live",
        "openai_base_url": "http://router/v1",
        "features": {"multi_agent": True, "unified_exec": True},
        "profiles": {"fast": {"model": "gpt-5"}},
        "model_providers": {"openai": {"base_url": "https://api.openai.com/v1"}},
    }
    merged = run.merge_router_settings(original, "vllm-sr/parent")
    assert merged["model"] == "vllm-sr/parent"
    assert merged["review_model"] == "vllm-sr/parent"
    assert merged["web_search"] == "disabled"
    assert merged["features"] == {"multi_agent": False, "unified_exec": True}
    assert merged["profiles"]["fast"]["model"] == "vllm-sr/parent"
    assert merged["openai_base_url"] == "http://router/v1"
    assert merged["model_providers"] == original["model_providers"]
    assert original["web_search"] == "live"
    assert original["features"]["multi_agent"] is True


def test_harbor_model_flag_keeps_the_router_entrypoint():
    command = "codex exec --model parent --json -- "
    assert run.apply_router_model(command, "vllm-sr/parent") == (
        "codex exec --model vllm-sr/parent --json -- "
    )
    assert run.apply_router_model(command, "parent") == command


def test_dry_run_prints_the_command_without_running_harbor(
    monkeypatch, capsys, tmp_path
):
    def fail(*args, **kwargs):
        raise AssertionError("dry run started Harbor")

    monkeypatch.setattr(run.subprocess, "run", fail)
    code = run.main(
        [
            "--dataset=tasks",
            "--router-url=http://172.17.0.1:8899",
            "--model=vllm-sr/parent",
            "--job-name=lane",
            f"--jobs-dir={tmp_path}",
            "--dry-run",
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "multi_agent = false" in printed
    assert "harbor run" in printed
    assert "version=0.163.0" in printed
    assert not (tmp_path / "lane.codex.toml").exists()


def test_api_errors_ignore_tool_text_and_a_later_success_clears_the_terminal_flag():
    tool = json.dumps(
        {"type": "item.completed", "item": {"text": "API Error: ignored"}}
    )
    failure = json.dumps(
        {"type": "error", "message": "unexpected status 400 Bad Request"}
    )
    recovered = [
        tool,
        failure,
        "ERROR: unexpected status 401 Unauthorized",
        json.dumps({"type": "turn.completed"}),
    ]
    assert report.api_errors(recovered) == [
        "unexpected status 400 Bad Request",
        "ERROR: unexpected status 401 Unauthorized",
    ]
    assert report.ended_on_api_error(recovered) is False
    assert report.ended_on_api_error([failure]) is True


def test_replay_summary_keeps_a_present_agent_id_in_the_run():
    def record(model, tokens, agent=None, action="allow", status=200):
        identity = {"session": {"status": "present"}}
        if agent is not None:
            identity["agent"] = {"status": agent}
        return {
            "selected_model": model,
            "total_tokens": tokens,
            "response_status": status,
            "session_policy": {"action": action, "identity": identity},
        }

    summary = report.summarize_replays(
        [
            record("big", 100, agent="present", action="establish_baseline"),
            record("small", 40, agent="present", status=400),
        ]
    )
    assert summary["calls"] == {"big": 1, "small": 1}
    assert summary["tokens"] == {"big": 100, "small": 40}
    assert summary["statuses"] == {"200": 1, "400": 1}
    assert all(
        not key.startswith("sub:") and not key.startswith("main:")
        for key in summary["calls"]
    )


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
    assert [record["id"] for record in records] == ["a", "b", "c"]
    queries = [parse_qs(urlparse(request.full_url).query) for request in seen]
    assert [query["offset"] for query in queries] == [["0"], ["2"]]
    assert all(query["session_id"] == [SESSION] for query in queries)
    assert urlparse(seen[0].full_url).path == "/api/v1/observability/replays"
    assert seen[0].get_header("Authorization") == "Bearer t"


def test_trial_row_joins_the_rollout_session_and_keeps_a_timeout_reward(tmp_path):
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
            json.dumps({"type": "thread.started", "thread_id": THREAD}),
            json.dumps(
                {
                    "type": "session_meta",
                    "payload": {"cli_version": "0.163.0", "thread_id": THREAD},
                }
            ),
        ],
    )
    rollout = trial / "agent" / "sessions" / "2026" / "10" / "28"
    rollout.mkdir(parents=True)
    (rollout / f"rollout-2026-10-28T12-00-00-{SESSION}.jsonl").write_text("{}\n")
    asked = []

    def replays(session_id):
        asked.append(session_id)
        return [{"selected_model": "big", "total_tokens": 10, "response_status": 200}]

    row = report.trial_row(trial, replays)
    assert row["reward"] == 1
    assert row["exception"] == "AgentTimeoutError"
    assert row["session_id"] == SESSION
    assert row["session_id"] != THREAD
    assert row["codex_version"] == "0.163.0"
    assert row["ended_on_api_error"] is False
    assert asked == [SESSION]
    assert row["router"]["calls"] == {"big": 1}


def test_thread_id_alone_is_not_the_replay_join_key(tmp_path):
    trial = _write_trial(
        tmp_path,
        "fix-git__abc",
        {"task_name": "fix-git", "verifier_result": {"rewards": {"reward": 0}}},
        [json.dumps({"type": "thread.started", "thread_id": THREAD})],
    )
    row = report.trial_row(trial, lambda session_id: [])
    assert row["session_id"] is None
    assert "router" not in row


def test_run_that_ends_on_an_api_error_is_counted(tmp_path):
    _write_trial(
        tmp_path,
        "a__1",
        {"task_name": "a", "verifier_result": {"rewards": {"reward": 0}}},
        [
            json.dumps(
                {"type": "turn.failed", "error": {"message": "stream disconnected"}}
            )
        ],
    )
    _write_trial(
        tmp_path,
        "b__1",
        {"task_name": "b", "verifier_result": {"rewards": {"reward": 1}}},
        [json.dumps({"type": "turn.completed"})],
    )
    (tmp_path / "result.json").write_text(json.dumps({"id": "job"}))
    (tmp_path.parent / f"{tmp_path.name}.codex-lane.json").write_text(
        json.dumps({"codex_version": "0.163.0"})
    )
    version = report.requested_version(tmp_path)
    rows = [
        report.trial_row(path, None, version) for path in report.trial_dirs(tmp_path)
    ]
    assert [row["task"] for row in rows] == ["a", "b"]
    assert rows[0]["api_errors"] == ["stream disconnected"]
    assert rows[0]["codex_version"] == "0.163.0"
    assert "router" not in rows[0]
    assert report.summarize_job(rows) == {
        "trials": 2,
        "passed": 1,
        "ended_on_api_error": 1,
        "exceptions": {},
        "codex_versions": ["0.163.0"],
    }


def test_trial_dirs_ignores_directories_without_a_trial_result(tmp_path):
    (tmp_path / "setup").mkdir()
    assert report.trial_dirs(tmp_path) == []
