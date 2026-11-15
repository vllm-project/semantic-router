"""Where `vllm-sr connect --launch` starts a harness, and the key a --key-command prints.

The launcher starts the harness in an existing git worktree, with the harness's
own home or another one given for this session. It never creates, switches or
deletes a branch, a worktree or a home, and never copies a credential from one
home to another: each home keeps its own login.
"""

from __future__ import annotations

import datetime
import shlex
import subprocess
from pathlib import Path

from cli import harness_connect as hc

# The variable each harness reads its home from.
HOME_VARIABLES = {"claude-code": "CLAUDE_CONFIG_DIR", "codex": "CODEX_HOME"}
# Seconds connect waits for --key-command when it probes the gateway itself: no
# longer than the harness waits for it (Codex 0.166:
# model_providers.<id>.auth.timeout_ms), so a key the probe got reaches the
# harness too.
KEY_COMMAND_TIMEOUTS = {"codex": 5}
KEY_COMMAND_TIMEOUT = 30


def worktree(path: Path) -> Path:
    """path, when git says it is inside a work tree. Nothing is changed."""
    found = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
        check=False,
    )
    if found.stdout.strip() != "true":
        raise ValueError(f"{path} is not inside a git work tree")
    return path


def session_env(harness: str, env: dict[str, str], home: Path | None) -> dict[str, str]:
    """The harness's environment: env, with home as its home when one is given."""
    return {**env, HOME_VARIABLES[harness]: str(home)} if home else dict(env)


def home_of(harness: str, env: dict[str, str]) -> Path:
    if harness == "codex":
        return hc.codex_home(env)
    return hc.claude_settings_path("user", Path.cwd(), env).parent


def receipt(where: Path, home: Path, gateway: str, model: str) -> str:
    """One line to match the session to its calls in Insights later."""
    started = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    fields = {
        "started": started,
        "worktree": str(where),
        "home": str(home),
        "gateway": gateway,
        "model": model,
    }
    return "Session: " + " ".join(f"{k}={shlex.quote(v)}" for k, v in fields.items())


def key_from(
    command: str, env: dict[str, str], cwd: Path | None = None, harness: str = ""
) -> str:
    """What command prints, stripped as both harnesses strip it, run the way they
    run it: with the harness's environment, in its directory (default: here), and
    within the time it allows. A failure says how it failed, never what it
    printed."""
    limit = KEY_COMMAND_TIMEOUTS.get(harness, KEY_COMMAND_TIMEOUT)
    try:
        result = subprocess.run(
            [hc.KEY_SHELL, "-c", command],
            capture_output=True,
            text=True,
            env=env,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            timeout=limit,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(
            f"--key-command did not finish in {limit} s"
            + (
                f", the time {harness} gives it"
                if harness in KEY_COMMAND_TIMEOUTS
                else ""
            )
        ) from exc
    except OSError as exc:
        raise ValueError(f"--key-command did not start: {exc}") from exc
    if result.returncode != 0:
        raise ValueError(f"--key-command exited with status {result.returncode}")
    key = result.stdout.strip()
    if not key:
        raise ValueError("--key-command printed no key")
    return key
