"""Run the harness itself once through the gateway, as `vllm-sr connect --check` does.

The run is one small task in a temporary directory: the harness shows a file
the check writes there with a harmless shell command and answers with what it
holds. The harness reads its own settings and credentials; this module only
reads what the harness prints, and hides any secret in a line before showing it.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from secrets import token_hex

from cli.harness_connect import SECRET_NAME
from cli.router_management_client import (
    OBSERVABILITY_REPLAYS_PATH,
    RouterManagementClient,
)

WORD = "vsr-check"
# The file holds a token new for each run, so a model cannot answer without
# running the command: told to run echo vsr-check, qwen36 in Pi answered
# vsr-check without a tool call in most runs.
FILE = f"{WORD}.txt"
COMMAND = f"cat {FILE}"
# {tool} is the harness's own name for its shell tool: Bash in Claude Code,
# exec_command in Codex (0.166) and bash in Pi.
PROMPT = (
    f"Run the shell command `{COMMAND}` with your {{tool}} tool, then reply with"
    " its output and nothing else."
)
# A reviewer sees a command only once it asks to leave the sandbox.
REVIEW_PROMPT = (
    f"Run the shell command `{COMMAND}` with your {{tool}} tool outside the"
    ' sandbox, asking for escalated permissions with the justification "vllm-sr'
    ' connect check", then reply with its output and nothing else.'
)
# Seconds for the whole run: a start, a tool call and the answer to its result.
TIMEOUT = 120
# A shorter value is no key, and hiding it would cut words apart.
SHORTEST_SECRET = 8
LONGEST_LINE = 240


@dataclass
class Row:
    name: str
    state: str  # ok, failed, not observed or not run
    detail: str = ""


def claude_argv(settings: Path) -> list[str]:
    """claude -p with the settings connect wrote, allowed to run only the command."""
    return [
        "claude",
        "-p",
        PROMPT.format(tool="Bash"),
        "--settings",
        str(settings),
        "--output-format",
        "stream-json",
        "--verbose",
        "--allowedTools",
        f"Bash({COMMAND})",
        "--max-turns",
        "3",
        "--no-session-persistence",
    ]


def codex_argv(config: list[str], workdir: Path, review: bool) -> list[str]:
    """codex exec with config, which selects the profile or carries its settings.
    With review, approvals go to Codex's automatic review (Approve for me)."""
    sandbox = ["--approve-for-me"] if review else ["-s", "read-only"]
    return [
        "codex",
        "exec",
        *config,
        *sandbox,
        "--json",
        "--ephemeral",
        "--skip-git-repo-check",
        "-C",
        str(workdir),
        (REVIEW_PROMPT if review else PROMPT).format(tool="exec_command"),
    ]


def pi_argv(provider: str, model: str) -> list[str]:
    """pi in JSON mode on the provider connect wrote, with its shell tool only
    and nothing of its own loaded: no session file, extension, skill or
    instruction file. Pi has no per-command allowlist, so it runs whatever the
    model asks for in the check's directory, which holds only its file."""
    return [
        "pi",
        "--mode",
        "json",
        "--provider",
        provider,
        "--model",
        model,
        "--tools",
        "bash",
        "--no-session",
        "--no-extensions",
        "--no-skills",
        "--no-context-files",
        PROMPT.format(tool="bash"),
    ]


def run(
    harness: str,
    argv: list[str],
    env: dict[str, str],
    workdir: Path,
    secrets: list[str],
    review_model: str | None = None,
    management: str = "",
) -> list[Row]:
    """The harness found and its version, its login accepted, the tool round trip,
    with review_model whether a review reached that entrypoint, and how it
    exited."""
    later = [Row("Login", "not run"), Row("Tool round trip", "not run")]
    if review_model:
        later.append(Row("Approval review", "not run"))
    found = shutil.which(argv[0], path=env.get("PATH"))
    if found is None:
        return [Row("Harness", "failed", f"{argv[0]} is not on PATH"), *later]
    harness_row = Row("Harness", "ok", f"{_version(found, env) or argv[0]} at {found}")
    before = _reviews(management, review_model) if review_model else None
    token = _token()
    (workdir / FILE).write_text(f"{token}\n", encoding="utf-8")
    try:
        out, err, code, timed_out = _run_harness([found, *argv[1:]], env, workdir)
    except OSError as exc:
        return [Row("Harness", "failed", f"{found} did not start: {exc}"), *later]
    events = _json_lines(out)
    hidden = [*secrets, *(v for k, v in env.items() if SECRET_NAME.search(k))]
    if not events and not timed_out:
        # no event: it stopped before it ran the task, so no login was tried,
        # as when Codex cannot load its config.toml
        why = f"{argv[0]} exited with status {code} before it ran the task"
        cause = _cause(err)
        failed = Row("Harness", "failed", f"{why}: {cause}" if cause else why)
        return _hide([failed, *later], hidden)
    rows_of = {"claude-code": _claude_rows, "pi": _pi_rows}.get(harness, _codex_rows)
    login, tool = rows_of(events, token)
    if login.state == "failed" and not login.detail:
        login.detail = _cause(err) or f"{argv[0]} exited with status {code}"
    rows = [harness_row, login, tool]
    if review_model:
        rows.append(_review_row(management, review_model, before))
    if timed_out:
        rows = [
            (
                row
                if row.state in ("ok", "not observed") or row is harness_row
                else Row(row.name, "failed", f"{argv[0]} did not finish in {TIMEOUT} s")
            )
            for row in rows
        ]
    rows.append(_exit_row(argv[0], events, err, code, timed_out))
    return _hide(rows, hidden)


def _hide(rows: list[Row], secrets: list[str]) -> list[Row]:
    for row in rows:
        row.detail = shown(row.detail, secrets)
    return rows


def _cause(err: str) -> str:
    """Why a harness stopped, from its stderr: the last line that starts with
    "error" and those after it, where Codex says what the error is in, else the
    last line."""
    lines = [line.strip() for line in err.splitlines() if line.strip()]
    errors = [i for i, line in enumerate(lines) if line.lower().startswith("error")]
    return " ".join(lines[errors[-1] if errors else -1 :]) if lines else ""


def _token() -> str:
    return f"{WORD}-{token_hex(4)}"


def shown(text: str, secrets: list[str]) -> str:
    """One line of what a harness printed, without a secret in it."""
    for secret in sorted(set(secrets), key=len, reverse=True):
        if len(secret) >= SHORTEST_SECRET:
            text = text.replace(secret, "<hidden>")
    text = re.sub(r"(?i)(bearer\s+)\S+", r"\1<hidden>", " ".join(text.split()))
    if len(text) <= LONGEST_LINE:
        return text
    return text[: LONGEST_LINE - 3] + "..."


def _version(path: str, env: dict[str, str]) -> str:
    try:
        result = subprocess.run(
            [path, "--version"],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            env=env,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    lines = result.stdout.strip().splitlines() if result.returncode == 0 else []
    return lines[0] if lines else ""


def _run_harness(argv, env, workdir) -> tuple[str, str, int, bool]:
    """Its output, exit status and whether it ran out of time. Its stdin is
    closed, since codex exec waits for more of the prompt on an open one, and it
    runs in a session of its own, so on a timeout or Ctrl-C the shells and
    servers it started stop with it."""
    with subprocess.Popen(
        argv,
        cwd=workdir,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
    ) as process:
        try:
            out, err = process.communicate(timeout=TIMEOUT)
            timed_out = False
        except BaseException as exc:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(process.pid, signal.SIGKILL)
            if not isinstance(exc, subprocess.TimeoutExpired):
                raise
            out, err = process.communicate()
            timed_out = True
    return out, err, process.returncode, timed_out


def _json_lines(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _claude_rows(events: list[dict], token: str) -> tuple[Row, Row]:
    """From claude -p's stream-json: a reply the gateway gave, and the command's
    call, its result and the answer to it."""
    replies = [e.get("message") or {} for e in events if e.get("type") == "assistant"]
    answered = [m for m in replies if m.get("model") not in (None, "", "<synthetic>")]
    result = next((e for e in reversed(events) if e.get("type") == "result"), {})
    if not answered:
        # Claude Code writes a refused login as a reply of its own
        why = result.get("result") or _text_of(replies)
        return Row("Login", "failed", why), Row("Tool round trip", "not run")
    # the reply names the backend's own model id, not the Router's model name
    login = Row("Login", "ok", f"the gateway answered, as {answered[0]['model']}")
    blocks = [
        b for m in answered for b in m.get("content") or [] if isinstance(b, dict)
    ]
    called = any(
        b.get("type") == "tool_use" and WORD in json.dumps(b.get("input"))
        for b in blocks
    )
    results = [
        b
        for e in events
        if e.get("type") == "user"
        for b in (e.get("message") or {}).get("content") or []
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    ran = any(
        token in json.dumps(b.get("content")) and not b.get("is_error") for b in results
    )
    answer = str(result.get("result") or "")
    final = not result.get("is_error") and token in answer
    return login, _tool_row(called, ran, final, results and results[0].get("content"))


def _codex_rows(events: list[dict], token: str) -> tuple[Row, Row]:
    """From codex exec --json: a turn the gateway answered, and the command run,
    its output and the answer after it."""
    items = [e.get("item") or {} for e in events if e.get("type") == "item.completed"]
    errors = [
        e.get("message") or (e.get("error") or {}).get("message") or ""
        for e in events
        if e.get("type") in ("error", "turn.failed")
    ]
    answered = any(e.get("type") == "turn.completed" for e in events) or any(
        i.get("type") in ("agent_message", "command_execution") for i in items
    )
    if not answered:
        why = next((e for e in reversed(errors) if e), "")
        return Row("Login", "failed", why), Row("Tool round trip", "not run")
    login = Row("Login", "ok", "the gateway answered")
    commands = [
        i
        for i in items
        if i.get("type") == "command_execution" and WORD in str(i.get("command"))
    ]
    ran = next(
        (
            i
            for i in commands
            if i.get("exit_code") == 0 and token in str(i.get("aggregated_output"))
        ),
        None,
    )
    after = items[items.index(ran) + 1 :] if ran else []
    final = any(
        i.get("type") == "agent_message" and token in str(i.get("text")) for i in after
    )
    first = commands[0].get("aggregated_output") if commands else None
    return login, _tool_row(bool(commands), ran is not None, final, first)


def _exit_row(name: str, events: list[dict], err: str, code: int, timed_out) -> Row:
    """How the run ended. Running out of time, a nonzero exit status and a failed
    turn or error result each fail the check, whatever the steps before did."""
    if timed_out:
        return Row("Exit", "failed", f"{name} did not finish in {TIMEOUT} s")
    failures = [
        (e.get("error") or {}).get("message") or str(e.get("result") or e["type"])
        for e in events
        if e.get("type") == "turn.failed"
        or (e.get("type") == "result" and e.get("is_error"))
    ]
    # Pi ends a failed turn with an assistant message and exits 0
    ended = [
        e.get("message") or {}
        for e in events
        if e.get("type") == "message_end"
        and (e.get("message") or {}).get("role") == "assistant"
    ]
    if ended and ended[-1].get("stopReason") in FAILED_STOPS:
        failures.append(ended[-1].get("errorMessage") or ended[-1]["stopReason"])
    why = [f"exited with status {code}"] if code else []
    why += [f"reported a failure: {text}" for text in failures[-1:]]
    if not why:
        return Row("Exit", "ok", f"{name} exited with status 0")
    cause = _cause(err)
    if code and not failures and cause:
        why[0] += f": {cause}"
    return Row("Exit", "failed", f"{name} {' and '.join(why)}")


# A Pi reply that ended this way is no answer from the gateway.
FAILED_STOPS = ("error", "aborted")


def _pi_rows(events: list[dict], token: str) -> tuple[Row, Row]:
    """From pi --mode json: a reply the gateway gave, and the command's call, its
    run and the answer after it."""
    ended = [
        (index, e.get("message") or {})
        for index, e in enumerate(events)
        if e.get("type") == "message_end"
        and (e.get("message") or {}).get("role") == "assistant"
    ]
    answered = [(i, m) for i, m in ended if m.get("stopReason") not in FAILED_STOPS]
    if not answered:
        why = next((m.get("errorMessage") for _, m in reversed(ended)), "")
        return Row("Login", "failed", why or ""), Row("Tool round trip", "not run")
    first = answered[0][1]
    model = first.get("responseModel")
    login = Row(
        "Login",
        "ok",
        f"the gateway answered, as {model}" if model else "the gateway answered",
    )
    called = any(
        block.get("type") == "toolCall" and WORD in json.dumps(block.get("arguments"))
        for _, message in answered
        for block in message.get("content") or []
        if isinstance(block, dict)
    )
    runs = [
        (index, e)
        for index, e in enumerate(events)
        if e.get("type") == "tool_execution_end" and e.get("toolName") == "bash"
    ]
    ran = next(
        (
            i
            for i, e in runs
            if not e.get("isError") and token in json.dumps(e.get("result"))
        ),
        None,
    )
    final = ran is not None and any(
        index > ran and token in _text_of(message.get("content"))
        for index, message in answered
    )
    output = (runs[0][1].get("result") or {}).get("content") if runs else None
    return login, _tool_row(called, ran is not None, final, output)


def _tool_row(called: bool, ran: bool, final: bool, output) -> Row:
    name = "Tool round trip"
    if not called:
        return Row(name, "failed", f"the model answered without running {COMMAND}")
    if not ran:
        return Row(name, "failed", f"{COMMAND} did not run: {_text_of(output)}")
    if not final:
        return Row(name, "failed", f"{COMMAND} ran, but no answer to its output came")
    return Row(name, "ok", f"ran {COMMAND} and answered with its output")


def _text_of(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(_text_of(v) for v in value)
    if isinstance(value, dict):
        return _text_of(value.get("text") or value.get("content") or "")
    return ""


def _reviews(url: str, review_model: str) -> tuple[set[str] | None, str]:
    """The ids of the replay records for review_model that the Router management
    API at url lists, or None and why the list could not be read."""
    try:
        payload = (
            RouterManagementClient(url, timeout=5)
            .request(
                "GET",
                OBSERVABILITY_REPLAYS_PATH,
                params={"model": review_model, "limit": "100"},
            )
            .payload
        )
    except ValueError as exc:  # RouterManagementError, or a URL it refuses
        return None, f"the replay list at {url} could not be read: {exc}"
    records = payload.get("data") if isinstance(payload, dict) else None
    return {
        str(r.get("id"))
        for r in records or []
        if isinstance(r, dict) and r.get("original_model") == review_model
    }, ""


def _review_row(management, review_model, before) -> Row:
    name = "Approval review"
    ids, why = before
    after = None
    if ids is not None:
        after, why = _reviews(management, review_model)
    if after is None:
        return Row(name, "not observed", why)
    new = sorted(after - ids)
    if new:
        return Row(name, "ok", f"went to {review_model} (replay {', '.join(new)})")
    return Row(
        name,
        "not observed",
        f"no new replay record for {review_model}: replay capture may be off for"
        " its recipe, or the model ran the command without asking for approval",
    )
