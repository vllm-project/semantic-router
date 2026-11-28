"""AG2 (Z14): the agentic driver and judge over a mock stack.

The mock stack is three fakes:

- an Anthropic mock backend (records every received header, replies a
  valid non-streaming message JSON);
- a fake `claude` executable (one main-agent request with the native
  W2 session header, one sub-agent request adding the agent header,
  then a CC-style result JSON and a real edit to the task's tracked file);
- a fake judge command (reads {predictions_path}, writes {report_path}
  with `resolved` = the patch is non-empty).

Through them the driver end-to-end contract is exercised without cloning
SWE-bench repos: per-task journals with injected request ids, the Z9
sub-agent manifest, sr-bench-shaped task rows, and the judge merge.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest
from recipe_lab.agentic_driver import HeaderForwardProxy, task_result_rows
from recipe_lab.cli import main
from recipe_lab.swebench_judge import JudgeError, parse_report

MOCK_MESSAGE = {
    "id": "msg_mock",
    "type": "message",
    "role": "assistant",
    "model": "mock-model",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 3, "output_tokens": 2},
}

FAKE_CLAUDE = r'''
"""Fake headless claude: one main request, one sub-agent request, an edit."""
import json
import os
import subprocess
import urllib.request
from pathlib import Path

base = os.environ["ANTHROPIC_BASE_URL"]
headers = {
    "content-type": "application/json",
    "anthropic-version": "2023-06-01",
    "x-api-key": os.environ.get("ANTHROPIC_API_KEY", "dummy"),
    "x-claude-code-session-id": "fake-session-" + Path.cwd().name,
}
body = json.dumps({
    "model": "mock-model",
    "max_tokens": 32,
    "messages": [{"role": "user", "content": "fix the bug"}],
}).encode()
for extra in ({}, {"x-claude-code-agent-id": "fake-agent-1"}):
    request = urllib.request.Request(
        base + "/v1/messages", data=body,
        headers={**headers, **extra}, method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        response.read()

tracked = Path("file.txt")
tracked.write_text(tracked.read_text(encoding="utf-8") + "fixed\n", encoding="utf-8")
subprocess.run(
    ["git", "-c", "user.name=t", "-c", "user.email=t@example.com",
     "commit", "-qm", "agent edit", "--only", "file.txt"], check=True,
)
Path("new.txt").write_text("long addition\n" * 800, encoding="utf-8")
print(json.dumps({
    "type": "result", "subtype": "success", "is_error": False,
    "result": "done", "session_id": headers["x-claude-code-session-id"],
    "num_turns": 2, "total_cost_usd": 0.01,
    "usage": {"input_tokens": 6, "output_tokens": 4},
}))
'''

FAKE_JUDGE = r'''
"""Fake harness: report.json with resolved = the patch is non-empty."""
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--predictions")
parser.add_argument("--report")
args = parser.parse_args()
rows = [json.loads(line) for line in open(args.predictions, encoding="utf-8")
        if line.strip()]
report = {
    row["instance_id"]: {"resolved": bool(row["model_patch"].strip())}
    for row in rows
}
with open(args.report, "w", encoding="utf-8") as handle:
    json.dump(report, handle)
print("fake harness done")
'''


class MockHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    received: ClassVar[list[dict]] = []
    lock = threading.Lock()

    def log_message(self, *args):  # quiet
        pass

    def do_POST(self):  # quiet
        length = int(self.headers.get("content-length") or 0)
        self.rfile.read(length)
        with MockHandler.lock:
            MockHandler.received.append({k.lower(): v for k, v in self.headers.items()})
        payload = json.dumps(MOCK_MESSAGE).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture()
def mock_backend():
    MockHandler.received.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _git(*args: str, cwd: str) -> str:
    proc = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def _prepare_task_repo(out: Path, instance_id: str) -> str:
    """A local 'SWE-bench repo': one tracked file, one commit; its sha is
    the task's base_commit (prepare_repo sees .git and skips the clone)."""
    repo = out / "tasks" / instance_id
    repo.mkdir(parents=True)
    _git("init", "-q", cwd=str(repo))
    (repo / "file.txt").write_text("original\n", encoding="utf-8")
    _git("add", "file.txt", cwd=str(repo))
    _git("commit", "-q", "-m", "base", cwd=str(repo))
    return _git("rev-parse", "HEAD", cwd=str(repo))


def _fake_claude_launcher(tmp_path: Path) -> str:
    script = tmp_path / "fake_claude.py"
    script.write_text(FAKE_CLAUDE, encoding="utf-8")
    if os.name == "nt":
        launcher = tmp_path / "fake_claude.cmd"
        launcher.write_text(
            f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n',
            encoding="utf-8",
        )
    else:
        launcher = tmp_path / "fake_claude.sh"
        launcher.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n',
            encoding="utf-8",
        )
        launcher.chmod(0o755)
    return str(launcher)


@pytest.fixture()
def task_run(mock_backend, tmp_path):
    """Two driven tasks through the real CLI, fake claude in between."""
    out = tmp_path / "run"
    shas = {
        inst: _prepare_task_repo(out, inst)
        for inst in ("local__task-a", "local__task-b")
    }
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        "".join(
            json.dumps(
                {
                    "instance_id": inst,
                    "repo": "local/test",
                    "base_commit": sha,
                    "problem_statement": "Fix the bug.",
                }
            )
            + "\n"
            for inst, sha in shas.items()
        ),
        encoding="utf-8",
    )
    code = main(
        [
            "run-agentic",
            "--tasks",
            str(tasks_path),
            "--target",
            mock_backend,
            "--out",
            str(out),
            "--claude-bin",
            _fake_claude_launcher(tmp_path),
            "--model",
            "mock-model",
            "--timeout-s",
            "120",
        ]
    )
    assert code == 0
    return out


def test_driver_end_to_end(task_run):
    results = {
        p.stem: json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((task_run / "results").glob("*.json"))
    }
    assert set(results) == {"local__task-a", "local__task-b"}
    for inst, result in results.items():
        assert result["status"] == "done"
        assert result["exit_code"] == 0
        assert "fixed" in result["model_patch"]
        # W2: the driver recorded the client's native session header
        assert result["session_ids"] == [f"fake-session-{inst}"]
        assert result["claude"]["total_cost_usd"] == 0.01

    journals = {
        p.stem: [
            json.loads(line)
            for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for p in sorted((task_run / "requests").glob("*.jsonl"))
    }
    for inst, rows in journals.items():
        # one journal row per request outcome, agents marked by the client's
        # own agent header (W2), request ids injected deterministically
        main_rows = [r for r in rows if r["agent"] == "main"]
        sub_rows = [r for r in rows if r["agent"] == "sub"]
        assert main_rows and sub_rows
        assert all(r["request_id_injected"] for r in rows)
        assert any(r["request_id"] == f"{inst}-0001" for r in main_rows)
        assert all(
            r.get("response_status") == 200
            for r in rows
            if "ts" in r and r.get("response_status") is not None
        )

    manifest = json.loads(
        (task_run / "agent-requests.json").read_text(encoding="utf-8")
    )
    entries = manifest["agent_requests"]
    assert {e["instance_id"] for e in entries} == set(journals)
    for entry in entries:
        assert entry["agent"] == "sub"
        assert entry["request_id"] in {
            r["request_id"]
            for r in journals[entry["instance_id"]]
            if r["agent"] == "sub"
        }

    rows = {r["case_id"]: r for r in task_result_rows(task_run)}
    for inst in journals:
        row = rows[inst]
        assert row["status"] == "completed"
        assert row["correct"] is None  # never judged, never guessed
        assert row["score"] is None
        assert row["call_count"] == 2
        assert row["cost_usd"] == 0.01


def test_driver_patch_applies_to_base_with_committed_and_untracked_files(
    task_run, tmp_path
):
    inst = "local__task-a"
    task = next(
        row
        for row in map(json.loads, (tmp_path / "tasks.jsonl").read_text().splitlines())
        if row["instance_id"] == inst
    )
    driven = task_run / "tasks" / inst
    clean = tmp_path / "clean"
    _git(
        "clone", "--quiet", "--no-checkout", str(driven), str(clean), cwd=str(tmp_path)
    )
    _git("checkout", "--quiet", task["base_commit"], cwd=str(clean))
    patch = json.loads((task_run / "results" / f"{inst}.json").read_text())[
        "model_patch"
    ]
    assert len(patch) > 4000
    subprocess.run(
        ["git", "apply", "-"], cwd=clean, input=patch.encode("utf-8"), check=True
    )
    assert (clean / "file.txt").read_text() == "original\nfixed\n"
    assert (clean / "new.txt").read_text() == "long addition\n" * 800
    assert _git("status", "--porcelain", cwd=str(driven)) == "?? new.txt"


def test_proxy_forwards_identity_and_injects_request_id(task_run):
    # every request reached the mock with the client's identity headers and
    # an injected x-request-id (the Z9 join key the replays record)
    assert len(MockHandler.received) == 4  # 2 requests x 2 tasks
    for headers in MockHandler.received:
        assert headers["x-claude-code-session-id"].startswith("fake-session-")
        assert "x-request-id" in headers
    subs = [h for h in MockHandler.received if "x-claude-code-agent-id" in h]
    mains = [h for h in MockHandler.received if "x-claude-code-agent-id" not in h]
    assert len(subs) == 2 and len(mains) == 2


def test_resume_keeps_existing_results(task_run, mock_backend, tmp_path):
    before = len(MockHandler.received)
    tasks_path = tmp_path / "tasks.jsonl"  # the fixture's task file
    assert (
        main(
            [
                "run-agentic",
                "--tasks",
                str(tasks_path),
                "--target",
                mock_backend,
                "--out",
                str(task_run),
                "--claude-bin",
                "claude-should-never-run",
                "--timeout-s",
                "60",
            ]
        )
        == 0
    )
    assert len(MockHandler.received) == before  # everything resumed, none re-run


def test_judge_end_to_end(task_run, tmp_path):
    judge = tmp_path / "fake_judge.py"
    judge.write_text(FAKE_JUDGE, encoding="utf-8")
    # a sabotaged patch: the task's tracked file edited back to empty diff
    result = json.loads(
        (task_run / "results" / "local__task-b.json").read_text(encoding="utf-8")
    )
    result["model_patch"] = ""
    (task_run / "results" / "local__task-b.json").write_text(
        json.dumps(result), encoding="utf-8"
    )
    assert (
        main(
            [
                "judge-agentic",
                "--run-dir",
                str(task_run),
                "--judge-command",
                f'"{sys.executable}" "{judge}" --predictions {{predictions_path}}'
                " --report {report_path}",
                "--run-id",
                "rl-test",
            ]
        )
        == 0
    )

    summary = json.loads((task_run / "judge-summary.json").read_text(encoding="utf-8"))
    assert summary["total"] == 2
    assert summary["judged"] == 2
    assert summary["resolved"] == 1
    assert summary["accuracy"] == 0.5

    rows = {
        json.loads(line)["case_id"]: json.loads(line)
        for line in (task_run / "judge-results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    }
    a = rows["local__task-a"]
    b = rows["local__task-b"]
    assert (a["correct"], a["score"]) == (True, 1.0)
    assert (b["correct"], b["score"]) == (False, 0.0)
    # sr-bench row shape: cost and call counts survive the judge merge
    assert a["cost_usd"] == 0.01 and a["call_count"] == 2
    assert b["status"] == "completed"


def test_default_judge_reads_current_aggregate_and_isolates_rejudging(
    tmp_path, monkeypatch
):
    # Match the installed 5.0.2 CLI, including its cache keyed by cwd + run_id.
    harness = tmp_path / "swebench" / "harness"
    harness.mkdir(parents=True)
    (harness.parent / "__init__.py").touch()
    (harness / "__init__.py").touch()
    (harness / "run_evaluation.py").write_text(
        """import argparse, json
from pathlib import Path
parser = argparse.ArgumentParser()
for name in ("predictions_path", "dataset_name", "run_id", "max_workers", "report_dir"):
    parser.add_argument("--" + name, required=True)
args = parser.parse_args()
assert args.dataset_name == "SWE-bench/SWE-bench_Verified"
row = json.loads(Path(args.predictions_path).read_text())
inst = row["instance_id"]
cached = Path("logs") / args.run_id / inst / "report.json"
cached.parent.mkdir(parents=True, exist_ok=True)
if not cached.exists():
    cached.write_text(json.dumps({inst: bool(row["model_patch"])}))
resolved = json.loads(cached.read_text())[inst]
name = row["model_name_or_path"].replace("/", "__") + "." + args.run_id + ".json"
(Path(args.report_dir) / name).write_text(json.dumps({
    "resolved_ids": [inst] if resolved else [],
    "unresolved_ids": [] if resolved else [inst],
}))
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    run = tmp_path / "run with spaces"
    (run / "results").mkdir(parents=True)
    for patch in ("original patch", ""):
        (run / "results" / "inst.json").write_text(
            json.dumps({"instance_id": "inst", "status": "done", "model_patch": patch})
        )
        assert main(["judge-agentic", "--run-dir", str(run), "--run-id", "repeat"]) == 0
        summary = json.loads((run / "judge-summary.json").read_text())
        assert summary["judged"] == 1
        assert summary["resolved"] == int(bool(patch))
        assert Path(summary["report_path"]).name == "recipe-lab.repeat.json"


def test_unreported_task_stays_unjudged(task_run, tmp_path):
    judge = tmp_path / "fake_judge.py"
    judge.write_text(FAKE_JUDGE, encoding="utf-8")
    # the judge sees only task-a's prediction: task-b is missing from the
    # report and must stay correct=None, counted unjudged
    assert (
        main(
            [
                "judge-agentic",
                "--run-dir",
                str(task_run),
                "--judge-command",
                f'"{sys.executable}" "{judge}" --predictions {{predictions_path}}'
                " --report {report_path}",
                "--predictions-path",
                str(_only_a_predictions(task_run)),
                "--run-id",
                "rl-test",
            ]
        )
        == 0
    )
    summary = json.loads((task_run / "judge-summary.json").read_text(encoding="utf-8"))
    assert summary["unjudged"] == 1
    rows = {
        json.loads(line)["case_id"]: json.loads(line)
        for line in (task_run / "judge-results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    }
    assert rows["local__task-a"]["correct"] is True
    assert rows["local__task-b"]["correct"] is None


def _only_a_predictions(task_run: Path) -> Path:
    rows = task_result_rows(task_run)
    only_a = [
        {"instance_id": r["case_id"], "model_patch": "x"}
        for r in rows
        if r["case_id"] == "local__task-a"
    ]
    path = task_run / "judge" / "predictions-a.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in only_a), encoding="utf-8")
    return path


def test_parse_report_shapes():
    assert parse_report({"psf__x-1": {"resolved": True, "report": {}}}) == {
        "psf__x-1": {"resolved": True, "report": {"resolved": True, "report": {}}}
    }
    assert parse_report({"psf__x-1": False}) == {"psf__x-1": {"resolved": False}}
    assert parse_report({"resolved_ids": ["a"], "unresolved_ids": ["b", ""]}) == {
        "a": {"resolved": True},
        "b": {"resolved": False},
    }
    assert parse_report([{"instance_id": "a", "resolved": True}]) == {
        "a": {"resolved": True}
    }
    with pytest.raises(JudgeError):
        parse_report({"nonsense": "shape"})
    with pytest.raises(JudgeError):
        parse_report([])


def test_proxy_records_target_outage(tmp_path):
    """A dead target is evidence (502 records), never a hung client."""
    proxy = HeaderForwardProxy("http://127.0.0.1:1", tmp_path / "reqs")
    proxy.start()
    proxy.begin_task("inst")
    import urllib.error
    import urllib.request

    body = json.dumps({"model": "m", "messages": []}).encode()
    request = urllib.request.Request(
        proxy.url() + "/v1/messages",
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(request, timeout=10)
        assert excinfo.value.code == 502
    finally:
        proxy.end_task()
        proxy.stop()
    rows = [
        json.loads(line)
        for line in (tmp_path / "reqs" / "inst.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert any(r.get("response_status") == 502 for r in rows)
    assert any(r.get("response_error") for r in rows)
