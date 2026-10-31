"""Report a Claude Code lane job, one row per trial, joined to Router Replay.

For each Harbor trial the row carries the verifier reward, Harbor's exception
type (a timeout still keeps the reward), Claude Code's own result event (session
id, terminal reason, turns, duration, subagents spawned by type), the API error
messages Claude Code printed, and the Router's replay records for that session:
calls and tokens per selected model, split into main agent and subagent calls
when the records carry agent identity, response statuses, and protection
actions. Replay records come from the management API, filtered by the session
id, which the Router takes from ``x-claude-code-session-id``.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

PAGE_LIMIT: int = 100
API_ERROR_MARKER: str = "API Error"


def last_result_event(lines: Iterable[str]) -> dict[str, Any]:
    """Claude Code's final stream-json ``result`` event, or an empty dict."""
    found: dict[str, Any] = {}
    for line in lines:
        if not line.startswith("{") or '"type":"result"' not in line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") == "result":
            found = event
    return found


def api_errors(lines: Iterable[str]) -> list[str]:
    """Distinct API error messages Claude Code reported, in order of appearance."""
    seen: list[str] = []
    for line in lines:
        start = line.find(API_ERROR_MARKER)
        if start < 0:
            continue
        message = line[start:].split('"', 1)[0].split("\\n", 1)[0][:200]
        if message not in seen:
            seen.append(message)
    return seen


def agent_kind(record: dict[str, Any]) -> str:
    """main or sub when the record carries agent identity, else all."""
    identity = (record.get("session_policy") or {}).get("identity") or {}
    agent = identity.get("agent")
    if not isinstance(agent, dict):
        return "all"
    return "sub" if agent.get("status") == "present" else "main"


def summarize_replays(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    calls: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    for record in records:
        key = f"{agent_kind(record)}:{record.get('selected_model') or '?'}"
        calls[key] += 1
        tokens[key] += record.get("total_tokens") or 0
        statuses[str(record.get("response_status") or "none")] += 1
        actions[(record.get("session_policy") or {}).get("action") or "none"] += 1
    return {
        "calls": dict(sorted(calls.items())),
        "tokens": dict(sorted(tokens.items())),
        "statuses": dict(sorted(statuses.items())),
        "protection_actions": dict(sorted(actions.items())),
    }


def fetch_replays(
    base_url: str,
    session_id: str,
    token: str | None = None,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> list[dict[str, Any]]:
    """Every replay record of one session, following the API's pagination."""
    records: list[dict[str, Any]] = []
    offset = 0
    while True:
        query = urllib.parse.urlencode(
            {"session_id": session_id, "limit": PAGE_LIMIT, "offset": offset}
        )
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/v1/observability/replays?{query}"
        )
        if token:
            request.add_header("Authorization", "Bearer " + token)
        with opener(request, timeout=30) as response:
            page = json.load(response)
        records += page.get("data") or []
        if not page.get("has_more"):
            return records
        offset = page["next_offset"]


def trial_row(
    trial_dir: Path,
    replays: Callable[[str], list[dict[str, Any]]] | None,
) -> dict[str, Any]:
    result = json.loads((trial_dir / "result.json").read_text())
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    exception = result.get("exception_info") or {}
    stream = trial_dir / "agent" / "claude-code.txt"
    lines = stream.read_text(errors="replace").splitlines() if stream.exists() else []
    event = last_result_event(lines)
    stats = event.get("subagent_stats") or {}
    final = str(event.get("result") or "")
    row: dict[str, Any] = {
        "task": result.get("task_name"),
        "trial": result.get("trial_name") or trial_dir.name,
        "reward": rewards.get("reward"),
        "exception": exception.get("exception_type"),
        "session_id": event.get("session_id"),
        "terminal_reason": event.get("terminal_reason"),
        "turns": event.get("num_turns"),
        "minutes": round((event.get("duration_ms") or 0) / 60000, 1),
        "subagents_spawned": stats.get("spawned"),
        "subagents_by_type": stats.get("by_type"),
        "api_errors": api_errors(lines),
        "ended_on_api_error": final.startswith(API_ERROR_MARKER)
        or event.get("api_error_status") is not None,
    }
    if replays is not None and row["session_id"]:
        row["router"] = summarize_replays(replays(row["session_id"]))
    return row


def trial_dirs(job_dir: Path) -> list[Path]:
    """Trial directories: those whose result.json names a task."""
    found = []
    for path in sorted(job_dir.glob("*/result.json")):
        if "task_name" in json.loads(path.read_text()):
            found.append(path.parent)
    return found


def summarize_job(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    exceptions = Counter(row["exception"] for row in rows if row["exception"])
    return {
        "trials": len(rows),
        "passed": sum(1 for row in rows if row["reward"] == 1),
        "ended_on_api_error": sum(1 for row in rows if row["ended_on_api_error"]),
        "exceptions": dict(sorted(exceptions.items())),
        "subagents_spawned": sum(row["subagents_spawned"] or 0 for row in rows),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("job_dir", type=Path, help="Harbor job directory")
    parser.add_argument(
        "--replay-url",
        help="Router management API base URL, for example http://127.0.0.1:8080; "
        "without it the rows carry no Router fields",
    )
    parser.add_argument(
        "--replay-token", help="Bearer token for a management API with auth on"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    replays = None
    if args.replay_url:

        def replays(session_id: str) -> list[dict[str, Any]]:
            return fetch_replays(args.replay_url, session_id, args.replay_token)

    rows = [trial_row(path, replays) for path in trial_dirs(args.job_dir)]
    for row in rows:
        print(json.dumps(row, sort_keys=True))
    print(json.dumps({"summary": summarize_job(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
