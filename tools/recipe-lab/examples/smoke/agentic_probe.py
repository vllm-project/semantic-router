"""Scripted multi-round agent probe for the recipe-lab smoke stack (AG8).

Drives deterministic agent conversations through the router so session
protection observes a real tool-loop shape, and reports per-turn routing
decisions plus a normalized per-run trajectory.

Each session in agentic_fixtures.json describes one conversation:

- user task prompt -> the mock backend scripts the model's turns (tool calls
  on the fixture's configured steps, final answer afterwards);
- this probe plays the agent client: it replays the growing message history,
  fabricates the tool results, and (for the sub-agent fixture) opens a nested
  loop under a second agent identity header;
- identity headers are configurable: x-session-id / x-conversation-id by
  default, plus any `--header Name: value` extras and a `--sub-agent-header`
  applied only to sub-agent turns;
- every request carries a deterministic `x-request-id` (recorded in the
  transcript), and each run writes `agent-requests-<tag>.json` naming the
  sub-agent request ids - Z9's client-side join key for judging sub-agent
  retention from the replay protection traces without router changes.

Determinism contract (AG8 completion criterion): the same fixture run twice
must produce identical session trajectories. `--repeat N` runs every fixture
N times and compares the normalized trajectories turn by turn; `--compare-with`
compares against a previous run directory. A mismatch exits non-zero.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_ROUTER = "http://127.0.0.1:8899"
DEFAULT_MGMT = "http://127.0.0.1:8080"
_HTTP_OK = 200

# Declared tool schemas, keyed by the tool names the fixtures use. Unknown
# names fall back to a parameterless spec so new fixtures need no probe edit.
TOOL_SPECS: dict[str, dict] = {
    "read_file": {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the workspace",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    "edit_file": {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Apply an edit to a workspace file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "change": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                    "target": {"type": "string"},
                },
            },
        },
    },
    "run_tests": {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run the test suite",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    "delegate_search": {
        "type": "function",
        "function": {
            "name": "delegate_search",
            "description": "Delegate a search subtask to a sub-agent",
            "parameters": {
                "type": "object",
                "properties": {"subtask": {"type": "string"}},
                "required": ["subtask"],
            },
        },
    },
    "apply_fix": {
        "type": "function",
        "function": {
            "name": "apply_fix",
            "description": "Apply a fix to the reported problem",
            "parameters": {
                "type": "object",
                "properties": {"target": {"type": "string"}},
            },
        },
    },
}
_GENERIC_SPEC = {
    "type": "function",
    "function": {
        "name": "?",
        "description": "Fixture tool",
        "parameters": {"type": "object", "properties": {}},
    },
}


def tool_specs(names: list[str]) -> list[dict]:
    specs = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue  # duplicate function names fail request validation
        seen.add(name)
        spec = TOOL_SPECS.get(name)
        if spec is None:
            spec = json.loads(json.dumps(_GENERIC_SPEC))
            spec["function"]["name"] = name
        specs.append(spec)
    return specs


def post(router: str, body: dict, headers: dict) -> tuple[int, dict, dict]:
    request = urllib.request.Request(
        router + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return (
                response.status,
                json.loads(response.read().decode("utf-8")),
                dict(response.headers),
            )
    except urllib.error.HTTPError as exc:
        return (
            exc.code,
            {"error": exc.read().decode("utf-8", "replace")[:400]},
            dict(exc.headers),
        )


def model_from(headers: dict, body: dict) -> str:
    usage = headers.get("x-vsr-model-usage") or ""
    if usage:
        return "hdr:" + usage[:120]
    return "body:" + str(body.get("model", ""))


def dump_replays(mgmt: str, output: Path) -> list[dict]:
    try:
        with urllib.request.urlopen(
            mgmt + "/api/v1/observability/replays?showDetails=true", timeout=30
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"replay dump failed: {exc}", file=sys.stderr)
        return []
    rows = payload.get("replays") or payload.get("records") or payload
    if isinstance(rows, dict):
        rows = list(rows.values())
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return rows


def trajectory_key(entry: dict) -> tuple:
    """Comparable projection of one turn: the fields that must match when
    the same fixture runs twice. Session/conversation ids are names only."""
    return (
        entry["fixture"],
        entry["agent"],
        entry["step"],
        entry["status"],
        entry["finish_reason"],
        entry["model"],
        tuple(entry["tool_calls"]),
        entry["content"] or "",
        tuple(sorted(entry["headers_sent"])),
    )


def compare_transcripts(a: list[dict], b: list[dict]) -> list[str]:
    """Turn-by-turn diff of two normalized transcripts (empty means equal)."""
    diffs = []
    if len(a) != len(b):
        diffs.append(f"turn count differs: {len(a)} vs {len(b)}")
    for i, (left, right) in enumerate(zip(a, b, strict=False)):
        ka, kb = trajectory_key(left), trajectory_key(right)
        if ka != kb:
            diffs.append(f"turn {i} differs: {ka} vs {kb}")
    return diffs


def agent_request_manifest(transcript: list[dict]) -> dict:
    """Z9: which request ids the client drove as sub-agent turns - the join
    key the diagnosis needs, because the router does not log agent headers
    into its replay records. Consumed by `recipe-lab diagnose
    --agent-requests` to judge sub-agent retention from the protection
    traces' identity hashes."""
    return {
        "agent_requests": [
            {
                "request_id": entry["request_id"],
                "agent": entry["agent"],
                "fixture": entry["fixture"],
                "step": entry["step"],
            }
            for entry in transcript
            if entry.get("agent") == "sub" and entry.get("request_id")
        ]
    }


class AgentRunner:
    """Plays the agent client for one run tag: drives every requested fixture
    session and records a normalized transcript with sent identity headers."""

    def __init__(
        self,
        *,
        router: str,
        mgmt: str,
        fixtures: dict,
        run_tag: str,
        session_prefix: str,
        sub_agent_header: str,
        extra_headers: dict[str, str],
        max_tokens: int,
        max_steps: int,
    ) -> None:
        self.router = router
        self.mgmt = mgmt
        self.fixtures = fixtures
        self.run_tag = run_tag
        self.session_prefix = session_prefix
        self.sub_agent_header = sub_agent_header
        self.extra_headers = extra_headers
        self.max_tokens = max_tokens
        self.max_steps = max_steps
        self.transcript: list[dict] = []

    def base_headers(self, fixture: str) -> dict[str, str]:
        identity = f"{self.session_prefix}-{fixture}-{self.run_tag}"
        headers = {
            "x-session-id": identity,
            "x-conversation-id": identity,
        }
        headers.update(self.extra_headers)
        return headers

    def record(self, entry: dict) -> None:
        self.transcript.append(entry)
        print(json.dumps(entry, ensure_ascii=False))

    def run_session(self, session: dict) -> None:
        fixture = str(session["fixture"])
        task = self.fixtures["tasks"][session["task"]]
        headers = self.base_headers(fixture)
        self.run_task(
            fixture=fixture,
            task=task,
            headers=headers,
            session=session,
            agent="main",
        )

    def run_task(
        self,
        *,
        fixture: str,
        task: dict,
        headers: dict,
        session: dict,
        agent: str,
    ) -> str:
        """Drive one scripted task to completion; returns the final content."""
        messages: list[dict] = [{"role": "user", "content": task["task_prompt"]}]
        tool_results = self.fixtures.get("tool_results") or {}
        sub_tool = session.get("sub_agent_tool")
        correction_after = session.get("user_correction_after_step")
        final_content = ""
        for step in range(self.max_steps):
            # Z9: a deterministic per-request id the client records locally -
            # the router does not log agent headers in replays, so the join
            # that judges sub-agent retention rides the request ids instead
            request_id = (
                f"{self.session_prefix}-{fixture}-{agent}-{step}-{self.run_tag}"
            )
            body: dict = {
                "model": "vllm-sr/auto",
                "max_tokens": self.max_tokens,
                "messages": messages,
            }
            names = list(task.get("tool_names") or [])
            if names:
                body["tools"] = tool_specs(names)
            status, data, resp_headers = post(
                self.router, body, {**headers, "x-request-id": request_id}
            )
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            calls = message.get("tool_calls") or []
            entry = {
                "fixture": fixture,
                "agent": agent,
                "step": step,
                "request_id": request_id,
                "status": status,
                "finish_reason": (
                    choice.get("finish_reason")
                    if status == _HTTP_OK
                    else f"error:{status}"
                ),
                "model": model_from(resp_headers, data),
                "tool_calls": [
                    str(c.get("function", {}).get("name", "?")) for c in calls
                ],
                "content": message.get("content"),
                "headers_sent": dict(headers),
                "usage": data.get("usage"),
            }
            self.record(entry)
            if status != _HTTP_OK:
                break
            if choice.get("finish_reason") == "tool_calls" and calls:
                messages.append(message)
                for call in calls:
                    name = str(call.get("function", {}).get("name", ""))
                    if name == sub_tool and agent == "main":
                        result = self.run_subagent(session, headers)
                    else:
                        result = str(tool_results.get(name, "ok"))
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": str(call.get("id", "")),
                            "content": result,
                        }
                    )
                if agent == "main" and correction_after == step:
                    messages.append(
                        {
                            "role": "user",
                            "content": str(session.get("correction_text", "")),
                        }
                    )
                continue
            final_content = str(message.get("content") or "")
            break
        return final_content

    def run_subagent(self, session: dict, headers: dict) -> str:
        """Nested tool loop for the delegated subtask, under a second agent
        identity header (the W2 finding: sub-agents add an agent id header)."""
        subtask = self.fixtures["tasks"][session["subtask_task"]]
        sub_headers = dict(headers)
        sub_headers[self.sub_agent_header] = (
            f"{self.session_prefix}-{session['fixture']}-subagent-{self.run_tag}"
        )
        return self.run_task(
            fixture=str(session["fixture"]),
            task=subtask,
            headers=sub_headers,
            session={k: v for k, v in session.items() if k != "sub_agent_tool"},
            agent="sub",
        )


def load_fixtures(path: Path) -> dict:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(fixture.get("tasks"), dict) or not fixture["tasks"]:
        raise SystemExit(f"{path}: missing non-empty `tasks` object")
    if not isinstance(fixture.get("sessions"), list) or not fixture["sessions"]:
        raise SystemExit(f"{path}: missing non-empty `sessions` list")
    return fixture


def parse_headers(pairs: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for pair in pairs:
        name, sep, value = pair.partition(":")
        if not sep or not name.strip():
            raise SystemExit(f"--header expects 'Name: value', got {pair!r}")
        headers[name.strip()] = value.strip()
    return headers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--fixture-file",
        default=str(Path(__file__).with_name("agentic_fixtures.json")),
        help="AG8 fixture JSON (sessions/tasks/tool_results/models)",
    )
    parser.add_argument(
        "--fixtures",
        default=None,
        help="comma-separated fixture names to run (default: every session)",
    )
    parser.add_argument(
        "--router", default=os.environ.get("RECIPE_LAB_PROBE_ROUTER", DEFAULT_ROUTER)
    )
    parser.add_argument(
        "--mgmt", default=os.environ.get("RECIPE_LAB_PROBE_MGMT", DEFAULT_MGMT)
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="run every fixture this many times (run1..runN) and compare the "
        "trajectories turn by turn; N >= 2 exercises the determinism contract",
    )
    parser.add_argument(
        "--compare-with",
        default=None,
        help="previous output dir: compare its transcript-run1.json against "
        "this run's run1 transcript",
    )
    parser.add_argument("--run-tag", default="run1")
    parser.add_argument("--session-prefix", default="probe")
    parser.add_argument(
        "--sub-agent-header",
        default="x-claude-code-agent-id",
        help="identity header applied to sub-agent turns only",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        help="extra identity header 'Name: value' sent on every request",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=12)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fixtures = load_fixtures(Path(args.fixture_file))
    sessions = fixtures["sessions"]
    if args.fixtures:
        wanted = [s.strip() for s in args.fixtures.split(",") if s.strip()]
        sessions = [s for s in sessions if s["fixture"] in wanted]
        missing = set(wanted) - {s["fixture"] for s in sessions}
        if missing:
            raise SystemExit(f"unknown fixtures: {sorted(missing)}")

    common: dict = {
        "router": args.router,
        "mgmt": args.mgmt,
        "fixtures": fixtures,
        "session_prefix": args.session_prefix,
        "sub_agent_header": args.sub_agent_header,
        "extra_headers": parse_headers(args.header),
        "max_tokens": args.max_tokens,
        "max_steps": args.max_steps,
    }
    failures: list[str] = []
    baseline: list[dict] | None = None
    for i in range(1, max(1, args.repeat) + 1):
        tag = f"run{i}" if args.repeat > 1 else args.run_tag
        runner = AgentRunner(run_tag=tag, **common)
        for session in sessions:
            runner.run_session(session)
        (out / f"transcript-{tag}.json").write_text(
            json.dumps(runner.transcript, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        (out / f"agent-requests-{tag}.json").write_text(
            json.dumps(
                agent_request_manifest(runner.transcript), ensure_ascii=False, indent=1
            ),
            encoding="utf-8",
        )
        rows = dump_replays(args.mgmt, out / f"replays-{tag}.json")
        summary = []
        for row in rows:
            details = row.get("details", row) if isinstance(row, dict) else {}
            summary.append(
                {
                    "session": details.get("session_id") or details.get("x-session-id"),
                    "conversation": details.get("conversation_id")
                    or details.get("x-conversation-id"),
                    "selected_model": details.get("selected_model")
                    or details.get("routed"),
                    "decision": details.get("decision"),
                    "body_bytes_captured": len(json.dumps(details, ensure_ascii=False)),
                }
            )
        (out / f"replay-summary-{tag}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"[{tag}] turns={len(runner.transcript)} replay records={len(rows)}")
        if args.repeat > 1:
            if baseline is None:
                baseline = runner.transcript
            else:
                diffs = compare_transcripts(baseline, runner.transcript)
                if diffs:
                    failures.extend(f"[{tag}] {d}" for d in diffs)
                else:
                    print(f"[{tag}] trajectories identical to run1")
    if args.compare_with:
        previous = json.loads(
            (Path(args.compare_with) / "transcript-run1.json").read_text(
                encoding="utf-8"
            )
        )
        current = json.loads((out / "transcript-run1.json").read_text(encoding="utf-8"))
        diffs = compare_transcripts(previous, current)
        if diffs:
            failures.extend(f"[compare-with] {d}" for d in diffs)
        else:
            print("[compare-with] trajectories identical")

    if failures:
        for failure in failures:
            print(f"DETERMINISM-FAIL {failure}", file=sys.stderr)
        return 2
    print("probe done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
