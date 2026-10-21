#!/usr/bin/env python3
"""Replay recorded agent runs through a running Router and report what it did.

Each step of a recorded run is sent to the Router as the agent sent it, with
one session and conversation id per run, and a backend started by this script
answers with the recorded reply. The same content can be sent in two wire
forms: ``text``, where the agent's commands and their outputs travel as
assistant and user messages (mini-SWE-agent, terminus), and ``tool``, where
they travel as tool calls and tool results. For every run the report keeps the
model the Router selected on each step, the switches inside the run, the
session phases and the protection actions it reported.

``learn`` streams tasks recorded for several models: the Router's choice on the
first request decides which recording the run follows, and the run's recorded
outcome can be posted to Router Learning afterwards.

Usage:
  replay.py fetch-tbench --out runs.jsonl --models deepseek-v3.2,gpt-5-mini
  replay.py run --recordings runs.jsonl --wire text --out out/text
  replay.py learn --recordings runs.jsonl --models deepseek-v3.2,gpt-5-mini --out out/learn
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "agent-run-recording.v1"
# yoonholee/terminalbench-trajectories, Apache-2.0, scraped from the
# Terminal-Bench 2.0 leaderboard.
TBENCH_DATASET = "yoonholee/terminalbench-trajectories"
TBENCH_REVISION = "04e8940f5b6736a7ce8d22224fe2f2af74163ed2"
TBENCH_FILES = (
    "data/train-00000-of-00002.parquet",
    "data/train-00001-of-00002.parquet",
)
# The scrape leaves some strings as unresolved references such as "$123".
UNRESOLVED = re.compile(r"^\$\d+$")
NOT_RECORDED = "[not recorded in the source]"
OUTCOME_RATE_PER_MINUTE = 60  # the management API's outcome ingest limit
OUTCOME_RATE_WINDOW_SECONDS = 60
HTTP_OK = 200
HTTP_BAD_REQUEST = 400
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run shell commands in the task environment and return their output.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}
REPORTED_HEADERS = (
    "x-vsr-selected-model",
    "x-vsr-selected-decision",
    "x-vsr-session-phase",
    "x-vsr-learning-actions",
    "x-vsr-learning-reasons",
    "x-vsr-replay-id",
    "x-vsr-cost",
)


class RecordingError(ValueError):
    pass


def parse_recording(raw: dict[str, Any]) -> dict[str, Any]:
    """Check one recorded run against the recording schema and return it."""
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise RecordingError(f"schema_version must be {SCHEMA_VERSION!r}")
    for key in ("run_id", "task_id", "model", "task"):
        if not isinstance(raw.get(key), str) or not raw[key]:
            raise RecordingError(f"{key} must be a nonempty string")
    if not isinstance(raw.get("system", ""), str):
        raise RecordingError("system must be a string")
    steps = raw.get("steps")
    if not isinstance(steps, list) or not steps:
        raise RecordingError(f"{raw['run_id']}: steps must be a nonempty list")
    for i, step in enumerate(steps):
        if not isinstance(step, dict) or not isinstance(step.get("reply"), str):
            raise RecordingError(f"{raw['run_id']}: step {i} needs a reply string")
        for key in ("command", "observation"):
            if step.get(key) is not None and not isinstance(step[key], str):
                raise RecordingError(
                    f"{raw['run_id']}: step {i} {key} must be a string"
                )
    if raw.get("outcome") is not None and not isinstance(raw["outcome"], bool):
        raise RecordingError(f"{raw['run_id']}: outcome must be true, false or null")
    return raw


def load_recordings(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [parse_recording(json.loads(line)) for line in f if line.strip()]


def assistant_message(step: dict[str, Any], index: int, wire: str) -> dict[str, Any]:
    """The recorded reply of one step as the agent sent it in the given wire form."""
    command = step.get("command")
    if wire == "text" or not command:
        content = step["reply"]
        if command:
            content += f"\n\n```bash\n{command.rstrip()}\n```"
        return {"role": "assistant", "content": content}
    call = {
        "id": f"call_{index}",
        "type": "function",
        "function": {"name": "bash", "arguments": json.dumps({"command": command})},
    }
    return {"role": "assistant", "content": step["reply"] or None, "tool_calls": [call]}


def request_body(run: dict[str, Any], k: int, wire: str, model: str) -> dict[str, Any]:
    """The request the agent sent for step k: everything before reply k."""
    messages = []
    if run.get("system"):
        messages.append({"role": "system", "content": run["system"]})
    messages.append({"role": "user", "content": run["task"]})
    for i, step in enumerate(run["steps"][:k]):
        reply = assistant_message(step, i, wire)
        messages.append(reply)
        output = step.get("observation") or ""
        if reply.get("tool_calls"):
            messages.append(
                {"role": "tool", "tool_call_id": f"call_{i}", "content": output}
            )
        else:
            messages.append({"role": "user", "content": output})
    body = {"model": model, "messages": messages}
    if wire == "tool":
        body["tools"] = [BASH_TOOL]
    return body


class RecordedBackend:
    """An OpenAI Chat backend that answers each request with a recorded reply.

    The driver names the recording to answer from before each run, per served
    model (or ``*`` for any model). The reply is step k of that recording,
    where k is the number of assistant messages already in the request.
    """

    def __init__(self, port: int = 0):
        self.recordings: dict[str, dict[str, Any]] = {}
        self.requests = 0
        backend = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args) -> None:
                pass

            def do_GET(self) -> None:
                models = [{"id": m, "object": "model"} for m in backend.recordings]
                self._send(HTTP_OK, {"object": "list", "data": models})

            def do_POST(self) -> None:
                size = int(self.headers.get("Content-Length") or 0)
                code, body = backend.answer(json.loads(self.rfile.read(size) or b"{}"))
                self._send(code, body)

            def _send(self, code: int, body: dict[str, Any]) -> None:
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def answer(self, request: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        self.requests += 1
        model = request.get("model")
        run = self.recordings.get(model) or self.recordings.get("*")
        if run is None:
            return HTTP_BAD_REQUEST, {
                "error": {"message": f"no recording for model {model!r}"}
            }
        if request.get("stream"):
            return HTTP_BAD_REQUEST, {
                "error": {
                    "message": "the recorded backend answers buffered requests only"
                }
            }
        k = sum(
            1 for m in request.get("messages") or [] if m.get("role") == "assistant"
        )
        if k >= len(run["steps"]):
            return HTTP_BAD_REQUEST, {
                "error": {"message": f"step {k} is past the recording"}
            }
        wire = "tool" if request.get("tools") else "text"
        reply = assistant_message(run["steps"][k], k, wire)
        prompt = sum(len(str(m.get("content") or "")) for m in request["messages"]) // 4
        completion = len(json.dumps(reply)) // 4
        return HTTP_OK, {
            "id": f"chatcmpl-replay-{run['run_id']}-{k}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": reply,
                    "finish_reason": (
                        "tool_calls" if reply.get("tool_calls") else "stop"
                    ),
                }
            ],
            "usage": {
                "prompt_tokens": max(1, prompt),
                "completion_tokens": max(1, completion),
                "total_tokens": max(1, prompt) + max(1, completion),
            },
        }


def http_json(
    method: str, url: str, body: Any = None, headers: dict[str, str] | None = None
):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        url, data=data, method=method, headers=headers or {}
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=300) as response:
            return (
                response.status,
                {k.lower(): v for k, v in response.headers.items()},
                response.read(),
            )
    except urllib.error.HTTPError as error:
        return (
            error.code,
            {k.lower(): v for k, v in error.headers.items()},
            error.read(),
        )


def send_step(router: str, session: str, body: dict[str, Any]) -> dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "x-session-id": session,
        "x-conversation-id": session,
        "x-vsr-debug": "true",
    }
    status, response_headers, raw = http_json(
        "POST", router + "/v1/chat/completions", body, headers
    )
    step = {"status": status, **{h: response_headers.get(h) for h in REPORTED_HEADERS}}
    if status >= HTTP_BAD_REQUEST:
        step["error"] = raw[:300].decode(errors="replace")
    return step


def summarize_run(run: dict[str, Any], steps: list[dict[str, Any]]) -> dict[str, Any]:
    """One report row: what the Router selected on each step of a run."""
    models = [s["x-vsr-selected-model"] for s in steps if s["status"] == HTTP_OK]
    switches = [i for i in range(1, len(models)) if models[i] != models[i - 1]]
    return {
        "run_id": run["run_id"],
        "task_id": run["task_id"],
        "recorded_model": run["model"],
        "steps": len(steps),
        "errors": sum(1 for s in steps if s["status"] != HTTP_OK),
        "selected_models": dict(collections.Counter(models)),
        "first_model": models[0] if models else None,
        "switch_steps": switches,
        "phases": dict(collections.Counter(s["x-vsr-session-phase"] for s in steps)),
        "learning_actions": dict(
            collections.Counter(s["x-vsr-learning-actions"] for s in steps)
        ),
        "first_replay_id": steps[0]["x-vsr-replay-id"] if steps else None,
        "router_cost": sum(float(s["x-vsr-cost"] or 0) for s in steps),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, collections.Counter] = {
        k: collections.Counter()
        for k in ("phases", "learning_actions", "selected_models")
    }
    for row in rows:
        for key, counter in totals.items():
            counter.update(row[key])
    return {
        "runs": len(rows),
        "requests": sum(r["steps"] for r in rows),
        "errors": sum(r["errors"] for r in rows),
        "runs_with_a_switch": sum(1 for r in rows if r["switch_steps"]),
        "switches": sum(len(r["switch_steps"]) for r in rows),
        **{key: dict(counter) for key, counter in totals.items()},
    }


def replay_runs(runs, wire, router, backend, model="vllm-sr/auto"):
    """Replay every run in full and return one report row per run."""
    rows = []
    for run in runs:
        backend.recordings = {"*": run}
        steps = [
            send_step(router, run["run_id"], request_body(run, k, wire, model))
            for k in range(len(run["steps"]))
        ]
        rows.append(summarize_run(run, steps))
    return rows


class OutcomePoster:
    """Post run outcomes to Router Learning within the API's per-minute limit."""

    def __init__(self, api: str, run_name: str):
        self.url = api + "/api/v1/observability/outcomes"
        self.run_name = run_name
        self.sent: collections.deque[float] = collections.deque()

    def post(self, row: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        while self.sent and now - self.sent[0] > OUTCOME_RATE_WINDOW_SECONDS:
            self.sent.popleft()
        if len(self.sent) >= OUTCOME_RATE_PER_MINUTE:
            time.sleep(OUTCOME_RATE_WINDOW_SECONDS + 0.05 - (now - self.sent[0]))
        outcome = {
            "replay_id": row["first_replay_id"],
            "target": "model",
            "target_ref": row["served_model"],
            "verdict": "good_fit" if run["outcome"] else "underpowered",
            "reason": (
                "recorded_run_succeeded" if run["outcome"] else "recorded_run_failed"
            ),
            "metadata": {"run_id": run["run_id"], "task_id": run["task_id"]},
        }
        headers = {
            "Content-Type": "application/json",
            "Idempotency-Key": f"{self.run_name}:{run['task_id']}",
        }
        status, _, raw = http_json("POST", self.url, outcome, headers)
        self.sent.append(time.time())
        return {
            "status": status,
            "verdict": outcome["verdict"],
            "body": raw[:300].decode(errors="replace"),
        }


def learn_tasks(
    runs: list[dict[str, Any]], models: list[str]
) -> list[dict[str, dict[str, Any]]]:
    """Tasks recorded for every model, with the first recording per model."""
    by_task: dict[str, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for run in runs:
        if run["model"] in models:
            by_task[run["task_id"]].setdefault(run["model"], run)
    return [by_task[t] for t in sorted(by_task) if len(by_task[t]) == len(models)]


def learn(tasks, wire, router, backend, poster=None, model="vllm-sr/auto"):
    """Stream tasks: the first request's selected model decides which recording a run follows."""
    rows = []
    for recordings in tasks:
        first = next(iter(recordings.values()))
        backend.recordings = dict(recordings)
        steps = [
            send_step(router, first["task_id"], request_body(first, 0, wire, model))
        ]
        served = steps[0]["x-vsr-selected-model"]
        run = recordings.get(served)
        if run is not None:
            backend.recordings = {"*": run}
            for k in range(1, len(run["steps"])):
                steps.append(
                    send_step(
                        router, first["task_id"], request_body(run, k, wire, model)
                    )
                )
        row = summarize_run(run or first, steps)
        row["served_model"] = served
        row["outcome"] = run["outcome"] if run else None
        if (
            poster
            and run is not None
            and row["first_replay_id"]
            and run["outcome"] is not None
        ):
            row["outcome_post"] = poster.post(row, run)
        rows.append(row)
    return rows


def learn_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    half = rows[len(rows) // 2 :]
    return {
        **summarize(rows),
        "served": dict(collections.Counter(r["served_model"] for r in rows)),
        "served_second_half": dict(
            collections.Counter(r["served_model"] for r in half)
        ),
        "outcome_posts": dict(
            collections.Counter(
                str((r.get("outcome_post") or {}).get("status")) for r in rows
            )
        ),
    }


def clean(value: Any) -> str | None:
    if value is None or value == "None":
        return None
    text = str(value)
    return NOT_RECORDED if UNRESOLVED.match(text) else text


def tbench_command(tools: Any) -> str | None:
    """The commands of one terminus step, from the scrape's Python-literal tool list."""
    if tools in (None, "None"):
        return None
    try:
        calls = ast.literal_eval(tools) if isinstance(tools, str) else tools
    except (ValueError, SyntaxError):
        return clean(tools)
    commands = [c.get("cmd") or "" for c in calls if isinstance(c, dict)]
    return "".join(commands) or None


def tbench_recording(row: dict[str, Any]) -> dict[str, Any] | None:
    """Convert one Terminal-Bench 2.0 trajectory row to the recording schema."""
    if row.get("steps") in (None, "null"):
        return None
    events = json.loads(row["steps"])
    system = next((clean(e["msg"]) for e in events if e.get("src") == "system"), "")
    task = next((clean(e["msg"]) for e in events if e.get("src") == "user"), None)
    if not task or task == NOT_RECORDED:
        task = f"{NOT_RECORDED} Task: {row['task_name']}"
    steps = [
        {
            "reply": clean(e.get("msg")) or "",
            "command": tbench_command(e.get("tools")),
            "observation": clean(e.get("obs")),
        }
        for e in events
        if e.get("src") == "agent"
    ]
    if not steps:
        return None
    return parse_recording(
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": row.get("trial_name") or row["trial_id"],
            "task_id": row["task_name"],
            "model": row["model"].split("@")[0],
            "outcome": bool(row["reward"]) if row.get("reward") is not None else None,
            "cost_usd": (row["cost_cents"] or 0) / 100,
            "system": system or "",
            "task": task,
            "steps": steps,
        }
    )


def fetch_tbench(
    out: Path, models: list[str], agent: str, tasks: int, revision: str
) -> int:
    """Write the recordings of the named models from the pinned Terminal-Bench 2.0 scrape."""
    # Only fetching needs these, so the replay itself runs on the standard library.
    import pandas as pd  # noqa: PLC0415
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    columns = [
        "task_name",
        "agent",
        "model",
        "reward",
        "cost_cents",
        "trial_name",
        "trial_id",
        "steps",
    ]
    frames = [
        pd.read_parquet(
            hf_hub_download(TBENCH_DATASET, f, repo_type="dataset", revision=revision),
            columns=columns,
        )
        for f in TBENCH_FILES
    ]
    data = pd.concat(frames)
    data = data[(data.agent == agent) & data.model.str.split("@").str[0].isin(models)]
    keep = sorted(data.task_name.unique())[:tasks] if tasks else None
    written = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in data.to_dict("records"):
            if keep is not None and row["task_name"] not in keep:
                continue
            recording = tbench_recording(row)
            if recording:
                f.write(json.dumps(recording) + "\n")
                written += 1
    return written


def write_report(
    out: Path, rows: list[dict[str, Any]], summary: dict[str, Any]
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "runs.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser(
        "fetch-tbench",
        help="convert pinned Terminal-Bench 2.0 trajectories to recordings",
    )
    fetch.add_argument("--out", type=Path, required=True)
    fetch.add_argument(
        "--models",
        required=True,
        help="comma-separated model names without the @provider suffix",
    )
    fetch.add_argument("--agent", default="terminus-2")
    fetch.add_argument(
        "--tasks",
        type=int,
        default=0,
        help="keep the first N task names in sorted order (0: all)",
    )
    fetch.add_argument("--revision", default=TBENCH_REVISION)
    for name in ("run", "learn"):
        p = sub.add_parser(name)
        p.add_argument("--recordings", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument(
            "--wire",
            choices=["text", "tool"],
            default="tool" if name == "learn" else "text",
        )
        p.add_argument("--router", default="http://127.0.0.1:18899")
        p.add_argument("--backend-port", type=int, default=18771)
        p.add_argument("--limit", type=int, default=0)
    sub.choices["learn"].add_argument("--models", required=True)
    sub.choices["learn"].add_argument("--api", default="http://127.0.0.1:18080")
    sub.choices["learn"].add_argument("--post-outcomes", action="store_true")
    sub.choices["learn"].add_argument("--run-name", default="agent-run-replay")
    args = parser.parse_args(argv)

    if args.command == "fetch-tbench":
        models = args.models.split(",")
        print(
            f"wrote {fetch_tbench(args.out, models, args.agent, args.tasks, args.revision)} recordings"
        )
        return 0
    runs = load_recordings(args.recordings)
    backend = RecordedBackend(args.backend_port)
    try:
        if args.command == "run":
            runs = runs[: args.limit] if args.limit else runs
            rows = replay_runs(runs, args.wire, args.router, backend)
            write_report(args.out, rows, summarize(rows))
        else:
            tasks = learn_tasks(runs, args.models.split(","))
            tasks = tasks[: args.limit] if args.limit else tasks
            poster = (
                OutcomePoster(args.api, args.run_name) if args.post_outcomes else None
            )
            rows = learn(tasks, args.wire, args.router, backend, poster)
            write_report(args.out, rows, learn_summary(rows))
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
