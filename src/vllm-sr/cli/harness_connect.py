"""Point Claude Code or Codex CLI at a Router gateway, and take it back.

The command owns the env keys it sets in Claude Code's settings and the whole of
one Codex profile file. It records what each held before and what it wrote, so
--undo puts back only what still holds what it wrote and leaves every other edit
in place.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROVIDER = "vllm-sr"
MANIFEST = ".vllm-sr-connect.json"
MAX_OUTPUT_TOKENS = "32000"
TOKEN_KEY = "ANTHROPIC_AUTH_TOKEN"
NO_KEY = "vllm-sr-no-key"
# Claude Code settings the command may own outside "env"
TOP_LEVEL_KEYS = ("statusLine",)
STATUS_LINE_SCRIPT = "vllm-sr-status-line.py"


@dataclass
class Change:
    """One file and what the command sets in it.

    kind is json-env (keys of the "env" object of a JSON settings file, and
    TOP_LEVEL_KEYS at its top level) or file (the whole file, held as the one
    value under "file").
    """

    path: Path
    kind: str
    values: dict[str, str]

    def old_text(self) -> str:
        return self.path.read_text(encoding="utf-8") if self.path.exists() else ""

    def new_text(self, values: dict[str, Any] | None = None) -> str:
        """The file with values (default: the command's) set; None removes one."""
        values = self.values if values is None else values
        if self.kind == "json-env":
            return set_json_env(self.old_text(), values)
        return values.get("file", self.old_text()) or ""

    def raw(self, text: str) -> dict[str, Any]:
        """What each owned key holds in text, None when absent."""
        if self.kind == "json-env":
            settings = _settings(text)
            env = settings.get("env") or {}
            return {
                key: (settings if key in TOP_LEVEL_KEYS else env).get(key)
                for key in self.values
            }
        return {"file": text or None}

    def shown(self, text: str, key_label: str) -> str:
        """The part of text the command owns, for display, with no key in it."""
        if self.kind == "file":
            return text
        held = {k: v for k, v in self.raw(text).items() if v is not None}
        if held.get(TOKEN_KEY):
            written = held[TOKEN_KEY] == self.values[TOKEN_KEY]
            held[TOKEN_KEY] = key_label if written else "<hidden>"
        shown = {k: held.pop(k) for k in TOP_LEVEL_KEYS if k in held}
        if held:
            shown["env"] = held
        return json.dumps(shown, indent=2) + "\n" if shown else ""


# --- Claude Code -----------------------------------------------------------------


def claude_settings_path(scope: str, cwd: Path, env: dict[str, str]) -> Path:
    if scope == "project":
        # the local file: a project's settings.json is often committed, and this can hold a key
        return cwd / ".claude" / "settings.local.json"
    base = env.get("CLAUDE_CONFIG_DIR") or str(Path.home() / ".claude")
    return Path(base) / "settings.json"


def claude_env(gateway: str, model: str, token: str) -> dict[str, str]:
    return {
        "ANTHROPIC_BASE_URL": gateway.rstrip("/"),
        TOKEN_KEY: token,
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": MAX_OUTPUT_TOKENS,
    }


def _settings(text: str) -> dict:
    settings = json.loads(text) if text.strip() else {}
    if not isinstance(settings, dict):
        raise ValueError("the settings file does not hold a JSON object")
    if not isinstance(settings.get("env", {}), dict):
        raise ValueError('the settings file\'s "env" is not an object')
    return settings


def set_json_env(text: str, values: dict[str, Any]) -> str:
    """Set (or, for None, remove) env keys and TOP_LEVEL_KEYS; every other key
    stays as it was."""
    settings = _settings(text)
    env = dict(settings.get("env") or {})
    for key, value in values.items():
        target = settings if key in TOP_LEVEL_KEYS else env
        if value is None:
            target.pop(key, None)
        else:
            target[key] = value
    if env:
        settings["env"] = env
    else:
        settings.pop("env", None)
    return json.dumps(settings, indent=2) + "\n" if settings else ""


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=False
    )


def tracked_by_git(path: Path, cwd: Path) -> bool:
    """True when git already tracks path, which no exclude rule can keep out."""
    return _git(cwd, "ls-files", "--error-unmatch", "--", str(path)).returncode == 0


def keep_out_of_git(paths: list[Path], cwd: Path) -> list[Path]:
    """Add each path git would track to the repository's own exclude file, which is
    never committed, so a key in a project's settings stays out of its history."""
    # check-ignore exits 1 for a path git would track, 128 outside a repository
    tracked = [
        p
        for p in paths
        if _git(cwd, "check-ignore", "-q", str(p)).returncode == 1
        and not tracked_by_git(p, cwd)
    ]
    if tracked:
        top = Path(_git(cwd, "rev-parse", "--show-toplevel").stdout.strip()).resolve()
        exclude = (
            cwd / _git(cwd, "rev-parse", "--git-path", "info/exclude").stdout.strip()
        )
        exclude.parent.mkdir(parents=True, exist_ok=True)
        with exclude.open("a", encoding="utf-8") as handle:
            for path in tracked:
                handle.write(f"/{path.resolve().relative_to(top)}\n")
    return tracked


STATUS_LINE_SOURCE = """\
#!/usr/bin/env python3
# Claude Code status line written by vllm-sr connect: the model that answered
# the last reply of this session through the gateway, and the others it used.

import json
import sys
from pathlib import Path


def main() -> str:
    session = json.load(sys.stdin)
    transcript = Path(session.get("transcript_path") or "")
    # each subagent keeps its own transcript in a folder beside the session's
    paths = [transcript, *sorted(transcript.with_suffix("").glob("subagents/*.jsonl"))]
    replies = {}
    for path in paths:
        if not path.is_file():  # a session's transcript starts with its first reply
            continue
        with open(path, encoding="utf-8") as lines:
            for line in lines:
                if '"assistant"' not in line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                message = record.get("message") or {}
                model = message.get("model")
                if model and model != "<synthetic>":
                    key = message.get("id") or len(replies)
                    replies[key] = (record.get("timestamp") or "", model)
    if not replies:
        return "vllm-sr: no reply yet"
    models = [model for _, model in sorted(replies.values())]
    counts = {model: models.count(model) for model in dict.fromkeys(reversed(models))}
    return "vllm-sr ⇢ " + " · ".join(f"{model} ({n})" for model, n in counts.items())


try:
    print(main())
except (OSError, ValueError):
    print("vllm-sr")
"""


def status_line(settings_path: Path) -> tuple[Change, dict[str, str]]:
    """The script Claude Code's status line runs, written beside its settings, and
    the statusLine setting that runs it."""
    script = settings_path.parent / STATUS_LINE_SCRIPT
    command = {"type": "command", "command": f'python3 "{script}"'}
    return Change(script, "file", {"file": STATUS_LINE_SOURCE}), command


# --- Codex CLI ---------------------------------------------------------------------


def codex_home(env: dict[str, str]) -> Path:
    return Path(env.get("CODEX_HOME") or str(Path.home() / ".codex"))


def codex_change(
    home: Path,
    profile: str,
    gateway: str,
    model: str,
    key_env: str | None,
    context_window: int | None = None,
) -> Change:
    """The profile file. Codex layers it over config.toml with --profile, and it
    holds the provider too, so config.toml is not touched. key_env is left out
    for a listener without keys: Codex refuses to start while the variable a
    provider names is unset."""
    text = (
        f"model = {json.dumps(model)}\n"
        + (f"model_context_window = {context_window}\n" if context_window else "")
        + f'model_provider = "{PROVIDER}"\n'
        # the Router does not serve hosted web search
        'web_search = "disabled"\n'
        "\n"
        f"[model_providers.{PROVIDER}]\n"
        'name = "vLLM Semantic Router"\n'
        f"base_url = {json.dumps(gateway.rstrip('/') + '/v1')}\n"
        'wire_api = "responses"\n'
        + (f"env_key = {json.dumps(key_env)}\n" if key_env else "")
    )
    return Change(home / f"{profile}.config.toml", "file", {"file": text})


# --- writing and undo --------------------------------------------------------------


def write_private(path: Path, text: str, *, always: bool = False) -> None:
    """Write text; a file created here, and every file with always, is readable by its owner only.

    The text goes to a file beside it that then replaces it, so a crash never
    leaves the settings half written.
    """
    path = path.resolve()  # a settings file linked from elsewhere stays a link
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = path.exists() and not always
    mode = stat.S_IMODE(path.stat().st_mode) if keep else 0o600
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def apply_changes(changes: list[Change], manifest_dir: Path) -> Path:
    """Write each file and record what it owns for --undo.

    A second run keeps each key's first "before", so --undo still returns to
    what was there before vllm-sr connect first touched it.
    """
    manifest_path = manifest_dir / MANIFEST
    record = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {"files": []}
    )
    by_path = {item["path"]: item for item in record["files"]}
    for change in changes:
        old, new = change.old_text(), change.new_text()
        item = by_path.get(str(change.path))
        if item is None:
            item = {
                "path": str(change.path),
                "kind": change.kind,
                "created": not change.path.exists(),
                "keys": {},
            }
            record["files"].append(item)
        before, written = change.raw(old), change.raw(new)
        for key in change.values:
            entry = item["keys"].setdefault(key, {"before": before[key]})
            entry["written"] = written[key]
        # Claude Code's settings now hold the key, so they are made private too
        write_private(change.path, new, always=change.kind == "json-env")
    write_private(manifest_path, json.dumps(record, indent=2) + "\n", always=True)
    return manifest_path


def undo_changes(manifest_dir: Path) -> list[str]:
    """Return each owned key to what it held before, unless it was changed since."""
    manifest_path = manifest_dir / MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(f"nothing to undo: {manifest_path} does not exist")
    report = []
    for item in json.loads(manifest_path.read_text(encoding="utf-8"))["files"]:
        path = Path(item["path"])
        if not path.exists():
            report.append(f"skipped {path}: it no longer exists")
            continue
        probe = Change(path, item["kind"], dict.fromkeys(item["keys"], ""))
        current = probe.raw(probe.old_text())
        restore = {}
        for key, entry in item["keys"].items():
            now, written = current[key], entry["written"]
            if now == written:
                restore[key] = entry["before"]
            elif key == "file" and now and now.startswith(written):
                # the Codex TUI appends tables of its own, such as [tui], to the profile
                tail = now[len(written) :].lstrip("\n")
                before = entry["before"] or ""
                if before and not before.endswith("\n"):
                    before += "\n"
                restore[key] = before + tail
            else:
                what = path if key == "file" else f"{key} in {path}"
                report.append(f"kept {what}: it was changed after vllm-sr connect")
        if not restore:
            continue
        text = probe.new_text(restore)
        if item["created"] and not text.strip():
            path.unlink()
            report.append(f"removed {path}")
        else:
            write_private(path, text)
            report.append(f"restored {path}")
    manifest_path.unlink()
    return report


def token_from(env: dict[str, str], key_env: str) -> tuple[str, bool]:
    """The key from key_env, or a placeholder for a listener without keys."""
    value = env.get(key_env, "")
    return (value, True) if value else (NO_KEY, False)
