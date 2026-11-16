"""Where `vllm-sr connect --launch` starts a harness, the record each launch
leaves, and the key a --key-command prints.

The launcher starts the harness in a git worktree, with the harness's own home
or another one given for this session. For --worktree new it first makes a
branch and a worktree from the checkout's HEAD with git worktree add. It never
switches or deletes a branch, a worktree or a home, and never copies a
credential from one home to another: each home keeps its own login.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import re
import shlex
import shutil
import subprocess
import uuid
from pathlib import Path

from cli import harness_connect as hc
from cli.commands.runtime_paths import _create_or_harden_private_directory
from cli.instance_paths import cli_state_directory

# The variable each harness reads its home from.
HOME_VARIABLES = {"claude-code": "CLAUDE_CONFIG_DIR", "codex": "CODEX_HOME"}
# Seconds connect waits for --key-command when it probes the gateway itself: no
# longer than the harness waits for it (Codex 0.166:
# model_providers.<id>.auth.timeout_ms), so a key the probe got reaches the
# harness too.
KEY_COMMAND_TIMEOUTS = {"codex": 5, "pi": 10}
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


# The header the Router pins a request's session with, ahead of the headers
# global.router.headers.session_id_headers names; every launch sends its own.
SESSION_HEADER = "x-vsr-session-id"
CLAUDE_HEADERS = "ANTHROPIC_CUSTOM_HEADERS"
# Claude Code arguments that pick the conversation, and so its session id.
CLAUDE_PICKS_SESSION = ("--session-id", "-r", "--resume", "-c", "--continue")
# A worktree --worktree new:NAME makes is on this branch, beside the checkout.
BRANCH_PREFIX = "vllm-sr/"
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclasses.dataclass(frozen=True)
class Workspace:
    """Where the harness runs and the branch checked out there; for a worktree
    connect makes at launch, also the checkout it branches from and the commit
    it starts at."""

    path: Path
    branch: str = ""
    repository: Path | None = None
    base: str = ""

    @property
    def new(self) -> bool:
        return self.repository is not None

    @property
    def here(self) -> Path:
        """The directory that exists before the launch."""
        return self.repository or self.path


def workspace(value: str | None, harness: str) -> Workspace:
    """--worktree as given: none is this directory, new or new:NAME a worktree
    to make, anything else an existing git worktree. Nothing is changed. Only
    a worktree needs git."""
    if shutil.which("git") is None:
        if value:
            raise ValueError("a worktree needs git, which is not on PATH")
        return Workspace(Path.cwd())
    if value == "new":
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        return _planned(Path.cwd(), f"{harness}-{stamp}")
    if value and value.startswith("new:"):
        return _planned(Path.cwd(), value[4:])
    path = Path(value).expanduser().resolve() if value else Path.cwd()
    if not path.is_dir():
        raise ValueError(f"Directory {value!r} does not exist.")
    if value:
        worktree(path)
    return Workspace(path, hc._git(path, "branch", "--show-current").stdout.strip())


def _planned(here: Path, name: str) -> Workspace:
    """The worktree new:name makes: branch vllm-sr/name from HEAD, in
    <checkout>-name beside the checkout."""
    branch = BRANCH_PREFIX + name
    if (
        not _NAME.fullmatch(name)
        or hc._git(here, "check-ref-format", f"refs/heads/{branch}").returncode
    ):
        raise ValueError(
            f"{name!r} is not a worktree name: use letters, digits, '.', '_' and '-'"
        )
    top = hc._git(here, "rev-parse", "--show-toplevel").stdout.strip()
    if not top:
        raise ValueError(
            f"new makes a worktree, and {here} is not inside a git checkout"
        )
    base = hc._git(here, "rev-parse", "--verify", "--quiet", "HEAD^{commit}")
    if not base.stdout.strip():
        raise ValueError(f"{top} has no commit for a new worktree to start from")
    repository = Path(top)
    path = repository.parent / f"{repository.name}-{name}"
    # git refuses a branch that exists itself, but makes the branch before it
    # refuses a directory in the way (2.43)
    if path.exists():
        raise ValueError(f"{path} already exists; give another new:NAME")
    return Workspace(path, branch, repository, base.stdout.strip())


def add_argv(where: Workspace) -> list[str]:
    target = ["-b", where.branch, str(where.path), where.base]
    return ["git", "-C", str(where.repository), "worktree", "add", *target]


def add(where: Workspace) -> None:
    made = subprocess.run(add_argv(where), capture_output=True, text=True, check=False)
    if made.returncode:
        said = made.stderr.strip().rsplit("\n", 1)[-1].removeprefix("fatal: ")
        raise ValueError(f"git worktree add failed: {said}")


def session_id() -> str:
    """A new session id, a UUID since Claude Code takes no other."""
    return str(uuid.uuid4())


def claude_session_args(args: list[str], session: str) -> list[str]:
    """--session-id, so Claude Code's own session header carries the session
    too, unless args pick a conversation, which keeps its own id."""
    picks = (*CLAUDE_PICKS_SESSION, "--fork-session")
    if any(arg.split("=", 1)[0] in picks for arg in args):
        return []
    return ["--session-id", session]


def claude_headers(env: dict[str, str], cwd: Path, session: str) -> str:
    """ANTHROPIC_CUSTOM_HEADERS: the lines Claude Code started in cwd would send,
    with this session's header in place of one they hold. Claude Code takes the
    variable whole from its local, project or user settings, the first that
    sets it, else from env, and the launch's --settings replaces them all."""
    given = env.get(CLAUDE_HEADERS)
    project = cwd / ".claude"
    for path in (
        hc.claude_settings_path("user", cwd, env),
        project / "settings.json",
        project / "settings.local.json",
    ):
        try:
            settings = hc._settings(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        value = settings.get("env", {}).get(CLAUDE_HEADERS)
        given = value if isinstance(value, str) else given
    kept = [
        line
        for line in (given or "").splitlines()
        if line.strip() and line.split(":", 1)[0].strip().lower() != SESSION_HEADER
    ]
    return "\n".join([*kept, f"{SESSION_HEADER}: {session}"])


def codex_headers(session: str) -> list[str]:
    """The override that has Codex send the header with each provider request."""
    key = f"model_providers.{hc.PROVIDER}.http_headers.{SESSION_HEADER}"
    return ["-c", f"{key}={json.dumps(session)}"]


def _records() -> Path:
    return cli_state_directory() / "sessions"


def record(session: str, harness: str, where: Workspace, home, gateway, model) -> dict:
    """Write down the launch, so connect sessions lists it later, in a directory
    only its owner reads."""
    started = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    found = {
        "id": session,
        "harness": harness,
        "model": model,
        "gateway": gateway,
        "started": started,
        "worktree": str(where.path),
        "branch": where.branch,
        "home": str(home),
        "made_by_connect": where.new,
        "repository": str(where.repository) if where.new else None,
        "base": where.base or None,
    }
    _records().parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    records = _create_or_harden_private_directory(_records())
    hc.write_private(records / f"{session}.json", json.dumps(found, indent=2) + "\n")
    return found


def recorded() -> list[dict]:
    """Every launch recorded, the newest first."""
    found = []
    for path in sorted(_records().glob("*.json"), key=lambda p: -p.stat().st_mtime_ns):
        try:
            found.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot read {path}: {exc}") from exc
    return found


def rows(session: dict) -> list[tuple[str, str]]:
    """The session as the launch header and the list show it."""
    where = session["worktree"]
    if session.get("made_by_connect"):
        where += f" (made by connect from {session['base'][:12]})"
    shown = [(k, session[k]) for k in ("harness", "model", "gateway")]
    shown.append(("worktree", where))
    if session.get("branch"):
        shown.append(("branch", session["branch"]))
    return [*shown, ("home", session["home"]), ("started", session["started"])]


def state(session: dict) -> dict:
    """What is left now of the launch's worktree: its uncommitted changes, the
    paths git ignores there, which git worktree remove deletes, and the commits
    on its HEAD that no branch and no remote has; and when connect made it, its
    branch and the commits on it that no other branch and no remote has."""
    path = Path(session["worktree"])
    if not path.is_dir():
        found = {"worktree": "removed"}
    elif hc._git(path, "rev-parse", "--is-inside-work-tree").stdout.strip() != "true":
        found = {"worktree": "not a git work tree"}
    else:
        status = hc._git(path, "status", "--porcelain", "--ignored")
        lines = status.stdout.splitlines()
        ignored = sum(line.startswith("!! ") for line in lines)
        heads = ["rev-list", "--count", "HEAD", "--not", "--branches", "--remotes"]
        only = hc._git(path, *heads)
        # a count git could not make is no count: nothing is offered then
        found = {
            "worktree": "present",
            "changes": None if status.returncode else len(lines) - ignored,
            "ignored": None if status.returncode else ignored,
            "detached": None if only.returncode else int(only.stdout),
        }
    if session.get("made_by_connect"):
        repository, branch = Path(session["repository"]), session["branch"]
        ref = f"refs/heads/{branch}"
        if hc._git(repository, "rev-parse", "--verify", "--quiet", ref).returncode:
            found["branch"] = "deleted"
        else:
            others = ["--not", session["base"], f"--exclude={branch}", "--branches"]
            only = hc._git(repository, "rev-list", "--count", ref, *others, "--remotes")
            # a count git could not make is no count: nothing is offered then
            found["unmerged"] = None if only.returncode else int(only.stdout)
    return found


def _count(n: int | None, thing: str) -> str:
    if n is None:
        return f"{thing}s unknown"
    return f"{n} {thing}{'' if n == 1 else 's'}" if n else f"no {thing}s"


def describe(found: dict) -> str:
    worktree = found["worktree"]
    if worktree == "present":
        shown = [_count(found["changes"], "uncommitted change")]
        for key, thing in (
            ("ignored", "ignored path"),
            ("detached", "detached commit"),
        ):
            if found[key] != 0:
                shown.append(_count(found[key], thing))
    else:
        shown = ["worktree removed" if worktree == "removed" else worktree]
    if "unmerged" in found:
        shown.append(_count(found["unmerged"], "unmerged commit"))
    elif "branch" in found:
        shown.append("branch deleted")
    return ", ".join(shown)


def cleanup(session: dict, found: dict) -> list[str]:
    """The steps that remove what connect made for a launch, and only steps that
    lose no change and no commit: none while the worktree has uncommitted
    changes, which git worktree remove refuses too, or commits on a detached
    HEAD, which it does not, and no branch -d while the branch has unmerged
    commits. The removal deletes what git ignores there, so the steps say that
    first."""
    if not session.get("made_by_connect"):
        return ["none: this launch did not make the worktree"]
    present = found["worktree"] == "present"
    if present and found["changes"] != 0:
        return ["commit or discard the changes in it first"]
    if present and found["detached"] != 0:
        return ["put the commits on its detached HEAD on a branch first"]
    repository, branch = session["repository"], session["branch"]
    steps = []
    if present:
        if found["ignored"]:
            ignored = _count(found["ignored"], "ignored path")
            steps.append(
                f"keep what you need of the {ignored} there, which the removal deletes"
            )
        remove = ["git", "-C", repository, "worktree", "remove", session["worktree"]]
        steps.append(shlex.join(remove))
    if found.get("unmerged") == 0:
        steps.append(shlex.join(["git", "-C", repository, "branch", "-d", branch]))
    elif "unmerged" in found:
        steps.append(f"push or merge {branch} before deleting it")
    return steps or ["nothing left"]


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
