"""Tool timing for Router Replay's run view, from Codex's own tool hooks.

Codex runs `vsr.py tool` as an async hook (it does not wait for it) on PreToolUse,
PermissionRequest and PostToolUse for every tool. Each event writes a line to a small file per
call in the Codex home, and PostToolUse sends the call to the Router's management API: the tool's
name and call id, when it started and ended, and when it was put to approval. Nothing of a tool's
input or output leaves the machine; the file keeps a hash of a shell command only, to find the call
a PermissionRequest is about, since Codex 0.166 gives that event no call id.

Codex runs PostToolUse only for a tool that returned, so a call that was refused or failed to run
sends nothing, and Router Replay keeps the gateway's timing for it. Codex's hooks do not say when
a person answered an approval: a call that asked for one sends when Codex took it up, which then
includes the wait, beside when it asked.

A call is failed when its command exited with a code other than 0, else succeeded. Codex 0.166
gives PostToolUse the command's output alone; the output it gives the model, which it writes to
the session's transcript a moment later, starts with how the command ended. The hook reads that
line's header from the transcript on this machine and sends only the status.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import time
import urllib.request
from pathlib import Path

PATH = "/api/v1/observability/tool-events"
SPOOL = "vsr-tools"
# How long PostToolUse waits for its PreToolUse line (both run at once for a quick tool), how long
# its request may take, and how old a call's file gets before a later call removes it.
FIND_SECONDS = 1.0
SEND_SECONDS = 2.0
KEEP_SECONDS = 3600
_NAME = re.compile(r"[^A-Za-z0-9._-]")
# How a command ended, in the header of the output Codex gives the model: unified exec's line, or
# the shell tool's. The header ends where the command's own output starts.
_EXITED = re.compile(r"^(?:Process exited with code|Exit code:) (-?\d+)$", re.M)
OUTPUT_MARK = "\nOutput:"
# How much of the transcript's end is read for the call's output, which Codex has just written.
TAIL_BYTES = 1 << 20


def iso(seconds: float) -> str:
    return (
        time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(seconds))
        + f".{int(seconds * 1000) % 1000:03d}Z"
    )


def _safe(name: str) -> str:
    return _NAME.sub("_", name)[:200]


def _command_hash(event: dict) -> str:
    """A hash of the shell command a call runs, to match its PermissionRequest; empty for other
    tools, which match on the name alone."""
    tool_input = event.get("tool_input")
    command = tool_input.get("command") if isinstance(tool_input, dict) else None
    if not isinstance(command, str):
        return ""
    return hashlib.sha256(command.encode()).hexdigest()[:16]


def _folder(event: dict, opts: dict) -> Path | None:
    session = event.get("session_id")
    if not opts.get("managementUrl") or not isinstance(session, str) or not session:
        return None
    return Path(opts["home"]) / SPOOL / _safe(session)


def _append(path: Path, line: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(line) + "\n")


def _lines(path: Path) -> list[dict]:
    with contextlib.suppress(OSError):
        return [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    return []


def _sweep(folder: Path, now: float) -> None:
    """Removes the files of calls that never ended, such as one Codex refused."""
    for path in folder.parent.glob("*/*"):
        with contextlib.suppress(OSError):
            if now - path.stat().st_mtime > KEEP_SECONDS:
                path.unlink()


def _asked(folder: Path, event: dict, now: float) -> None:
    """Marks the earliest open call of the same tool, and the same command, as put to approval."""
    tool, digest = event.get("tool_name"), _command_hash(event)
    with contextlib.suppress(OSError):
        for path in sorted(folder.iterdir(), key=lambda p: p.stat().st_mtime):
            lines = _lines(path)
            called = next((line for line in lines if "at" in line), None)
            if called is None or any("asked" in line for line in lines):
                continue
            if called.get("tool") == tool and (
                not digest or not called.get("hash") or called["hash"] == digest
            ):
                _append(path, {"asked": now})
                return


def _timing(call_id: str, lines: list[dict], ended: float) -> dict | None:
    """The call as the Router takes it, from its lines; None without its PreToolUse line."""
    called = next((line for line in lines if "at" in line), None)
    if called is None:
        return None
    asked = next((line["asked"] for line in lines if "asked" in line), None)
    checked = max(
        [called["at"], *(line["checked"] for line in lines if "checked" in line)]
    )
    timing = {
        "tool_call_id": call_id,
        "tool": called["tool"],
        "status": "succeeded",
        # Without approval the tool started once the hooks Codex waits for were done; with one,
        # when Codex took the call up, as Codex does not say when the answer came.
        "started_at": iso(called["at"] if asked is not None else checked),
        "ended_at": iso(ended),
    }
    if asked is not None:
        timing["approval_requested_at"] = iso(asked)
    return timing


def _tail(path: str) -> list[str]:
    with contextlib.suppress(OSError), open(path, "rb") as transcript:
        transcript.seek(max(0, transcript.seek(0, os.SEEK_END) - TAIL_BYTES))
        return transcript.read().decode("utf-8", "replace").splitlines()
    return []


def _exit_code(event: dict, deadline: float) -> int | None:
    """The exit code the output Codex gave the model names for the call, read from the
    transcript until deadline; None for a call whose output names none, such as a tool that is
    no command or one still running, or without a transcript."""
    path, call_id = event.get("transcript_path"), event.get("tool_use_id")
    if not isinstance(path, str) or not path:
        return None
    while True:
        for line in reversed(_tail(path)):
            if call_id not in line:
                continue
            with contextlib.suppress(ValueError, KeyError, TypeError, AttributeError):
                payload = json.loads(line)["payload"]
                if (
                    payload["type"] == "function_call_output"
                    and payload["call_id"] == call_id
                ):
                    head, mark, _ = payload["output"].partition(OUTPUT_MARK)
                    found = _EXITED.search(head) if mark else None
                    return int(found.group(1)) if found else None
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.1)


def _send(timing: dict, session: str, opts: dict, opener) -> None:
    body = json.dumps(
        {"harness": "codex", "session_id": session, "events": [timing]}
    ).encode()
    headers = {"content-type": "application/json"}
    if opts.get("managementToken"):
        headers["Authorization"] = f"Bearer {opts['managementToken']}"
    url = opts["managementUrl"].rstrip("/") + PATH
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with contextlib.suppress(Exception), opener.open(request, timeout=SEND_SECONDS):
        pass


def record(event: dict, opts: dict, opener, now: float | None = None) -> None:
    """Takes one hook event; never raises, and never prints, since Codex does not wait for it."""
    now = time.time() if now is None else now
    folder = _folder(event, opts)
    name = event.get("hook_event_name")
    call_id = event.get("tool_use_id")
    if folder is None:
        return
    if name == "PermissionRequest":
        _asked(folder, event, now)
        return
    if not isinstance(call_id, str) or not call_id:
        return
    path = folder / _safe(call_id)
    if name == "PreToolUse":
        _append(
            path,
            {
                "at": now,
                "tool": str(event.get("tool_name")),
                "hash": _command_hash(event),
            },
        )
        _sweep(folder, now)
    elif name == "PostToolUse":
        deadline = time.monotonic() + FIND_SECONDS
        while (timing := _timing(call_id, _lines(path), now)) is None:
            if time.monotonic() >= deadline:
                return
            time.sleep(0.05)
        with contextlib.suppress(OSError):
            path.unlink()
        if _exit_code(event, time.monotonic() + FIND_SECONDS) not in (None, 0):
            timing["status"] = "failed"
        session = os.environ.get("VSR_SESSION_ID") or event["session_id"]
        _send(timing, session, opts, opener)


def checked(event: dict, opts: dict) -> None:
    """The guard's PreToolUse is done: a tool that asks for no approval starts after it."""
    folder = _folder(event, opts)
    call_id = event.get("tool_use_id")
    if folder is not None and isinstance(call_id, str) and call_id:
        with contextlib.suppress(OSError):
            _append(folder / _safe(call_id), {"checked": time.time()})
