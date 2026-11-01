"""Report a Codex lane job, one row per trial, joined to Router Replay.

For each Harbor trial the row carries the verifier reward, Harbor's exception
type (a timeout still keeps the reward), the API errors Codex printed, and the
Router's replay records for that run. The join key is Codex's session id, which
the Router records when ``identity.headers.session`` is ``session-id``. Records
stay grouped by that run. Codex's root also sends ``thread-id``, so a record
with an agent id is not treated as a subagent; splitting threads waits on the
agent id in replay records.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

PAGE_LIMIT = 100
REPLAY_TIMEOUT_SECONDS = 30
API_ERROR_LIMIT = 200
_ROLLOUT_SESSION = re.compile(
    r"rollout-\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-(?P<session>.+)\.jsonl$"
)
_ERROR_EVENT_TYPES = frozenset({"error", "turn.failed"})


def _json_object(line: str) -> dict[str, Any] | None:
    stripped = line.strip()
    if not stripped.startswith("{"):
        return None
    try:
        event = json.loads(stripped)
    except ValueError:
        return None
    if isinstance(event, dict):
        return event
    return None


def _message_of(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        for key in ("message", "error"):
            found = _message_of(value.get(key))
            if found:
                return found
    return None


def api_error_message(line: str) -> str | None:
    """One API error from a Codex log line, or None when the line is not one."""
    event = _json_object(line)
    if event is not None:
        if event.get("type") not in _ERROR_EVENT_TYPES:
            return None
        message = _message_of(event.get("message")) or _message_of(event.get("error"))
        return None if message is None else message[:API_ERROR_LIMIT]
    stripped = line.strip()
    if (
        stripped.startswith("ERROR:")
        or "API Error" in stripped
        or "unexpected status" in stripped
    ):
        return stripped[:API_ERROR_LIMIT]
    return None


def api_errors(lines: Iterable[str]) -> list[str]:
    """Distinct API error messages, in order of appearance."""
    seen: list[str] = []
    for line in lines:
        message = api_error_message(line)
        if message and message not in seen:
            seen.append(message)
    return seen


def ended_on_api_error(lines: Iterable[str]) -> bool:
    """True when the log's last failure was not followed by a completed turn."""
    failed = False
    for line in lines:
        event = _json_object(line)
        if event is not None and event.get("type") == "turn.completed":
            failed = False
            continue
        if api_error_message(line):
            failed = True
    return failed


def _session_from_event(event: dict[str, Any]) -> str | None:
    if event.get("type") != "session_meta":
        return None
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return None
    session = payload.get("session_id") or payload.get("id")
    if isinstance(session, str) and session:
        return session
    return None


def _version_from_event(event: dict[str, Any]) -> str | None:
    if event.get("type") != "session_meta":
        return None
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return None
    version = payload.get("cli_version")
    if isinstance(version, str) and version:
        return version
    return None


def session_id_from_lines(lines: Iterable[str]) -> str | None:
    """Codex session id from ``session_meta``. A thread id is not a session id."""
    found = None
    for line in lines:
        event = _json_object(line)
        if event is None:
            continue
        session = _session_from_event(event)
        if session:
            found = session
    return found


def cli_version_from_lines(lines: Iterable[str]) -> str | None:
    found = None
    for line in lines:
        event = _json_object(line)
        if event is None:
            continue
        version = _version_from_event(event)
        if version:
            found = version
    return found


def rollout_paths(trial_dir: Path) -> list[Path]:
    sessions = trial_dir / "agent" / "sessions"
    if not sessions.is_dir():
        return []
    return sorted(path for path in sessions.rglob("rollout-*.jsonl") if path.is_file())


def session_id_from_rollouts(trial_dir: Path) -> str | None:
    found = None
    for path in rollout_paths(trial_dir):
        match = _ROLLOUT_SESSION.match(path.name)
        if match:
            found = match.group("session")
    return found


def requested_version(job_dir: Path) -> str | None:
    path = job_dir.parent / f"{job_dir.name}.codex-lane.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except ValueError:
        return None
    version = payload.get("codex_version") if isinstance(payload, dict) else None
    if isinstance(version, str) and version:
        return version
    return None


def summarize_replays(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Calls, tokens, statuses, and protection actions for one Codex run.

    Keys are selected models. An agent id on a record does not make it a
    subagent: the Codex root sends ``thread-id`` too, and replay does not yet
    carry a separate agent id to split on.
    """
    calls: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    for record in records:
        model = str(record.get("selected_model") or "?")
        calls[model] += 1
        tokens[model] += record.get("total_tokens") or 0
        statuses[str(record.get("response_status") or "none")] += 1
        action = (record.get("session_policy") or {}).get("action") or "none"
        actions[str(action)] += 1
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
        with opener(request, timeout=REPLAY_TIMEOUT_SECONDS) as response:
            page = json.load(response)
        records += page.get("data") or []
        if not page.get("has_more"):
            return records
        next_offset = page.get("next_offset")
        if not isinstance(next_offset, int) or next_offset <= offset:
            return records
        offset = next_offset


def trial_dirs(job_dir: Path) -> list[Path]:
    """Trial directories: those whose result.json names a task."""
    found = []
    for path in sorted(job_dir.glob("*/result.json")):
        try:
            result = json.loads(path.read_text())
        except ValueError:
            continue
        if isinstance(result, dict) and "task_name" in result:
            found.append(path.parent)
    return found


def trial_row(
    trial_dir: Path,
    replays: Callable[[str], list[dict[str, Any]]] | None,
    version: str | None = None,
) -> dict[str, Any]:
    result = json.loads((trial_dir / "result.json").read_text())
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    exception = result.get("exception_info") or {}
    stream = trial_dir / "agent" / "codex.txt"
    lines = stream.read_text(errors="replace").splitlines() if stream.exists() else []
    rollout_lines: list[str] = []
    paths = rollout_paths(trial_dir)
    if paths:
        rollout_lines = paths[-1].read_text(errors="replace").splitlines()
    session_id = session_id_from_rollouts(trial_dir) or session_id_from_lines(lines)
    if session_id is None:
        session_id = session_id_from_lines(rollout_lines)
    logged_version = cli_version_from_lines(lines) or cli_version_from_lines(
        rollout_lines
    )
    row: dict[str, Any] = {
        "task": result.get("task_name"),
        "trial": result.get("trial_name") or trial_dir.name,
        "reward": rewards.get("reward"),
        "exception": exception.get("exception_type"),
        "session_id": session_id,
        "codex_version": logged_version or version,
        "api_errors": api_errors(lines),
        "ended_on_api_error": ended_on_api_error(lines),
    }
    if replays is not None and row["session_id"]:
        row["router"] = summarize_replays(replays(row["session_id"]))
    return row


def summarize_job(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    exceptions = Counter(row["exception"] for row in rows if row["exception"])
    versions = sorted({row["codex_version"] for row in rows if row["codex_version"]})
    return {
        "trials": len(rows),
        "passed": sum(1 for row in rows if row["reward"] == 1),
        "ended_on_api_error": sum(1 for row in rows if row["ended_on_api_error"]),
        "exceptions": dict(sorted(exceptions.items())),
        "codex_versions": versions,
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

    version = requested_version(args.job_dir)
    rows = [trial_row(path, replays, version) for path in trial_dirs(args.job_dir)]
    for row in rows:
        print(json.dumps(row, sort_keys=True))
    print(json.dumps({"summary": summarize_job(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
