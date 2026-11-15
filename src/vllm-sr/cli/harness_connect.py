"""Point Claude Code, Codex CLI or Pi at a Router gateway, and take it back.

The command owns the env keys it sets in Claude Code's settings, the whole of
one Codex profile file or, for the Codex app, a few keys of Codex's shared
config.toml, and one provider of Pi's models.json. It records what each held
before and what it wrote, so --undo puts back only what still holds what it
wrote and leaves every other edit in place.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cli import harness_toml

PROVIDER = "vllm-sr"
MANIFEST = ".vllm-sr-connect.json"
MAX_OUTPUT_TOKENS = "32000"
TOKEN_KEY = "ANTHROPIC_AUTH_TOKEN"
# Claude Code sends a shell's ANTHROPIC_API_KEY as x-api-key beside the gateway's
# token unless the settings set it empty, which they win over (2.1.323)
API_KEY = "ANTHROPIC_API_KEY"
NO_KEY = "vllm-sr-no-key"
# The command Claude Code runs for its key. Its documented order puts both
# variables above before it, so with one written both are set empty.
KEY_HELPER = "apiKeyHelper"
# Claude Code settings the command may own outside "env"
TOP_LEVEL_KEYS = ("statusLine", KEY_HELPER)
# What runs a key command: Claude Code runs apiKeyHelper in the system shell, and
# a Codex auth command is a program with its arguments (0.166).
KEY_SHELL = "/bin/sh"
STATUS_LINE_SCRIPT = "vllm-sr-status-line.py"


@dataclass
class Change:
    """One file and what the command sets in it.

    kind is json-env (keys of the "env" object of a JSON settings file, and
    TOP_LEVEL_KEYS at its top level), json (keys of a JSON file whose other
    keys are not the command's; a key with / in it names a nested one, such as
    providers/vllm-sr), toml (entries of a TOML file, as harness_toml reads
    them) or file (the whole file, held as the one value under "file").
    """

    path: Path
    kind: str
    values: dict[str, str]

    def old_text(self) -> str:
        return read(self.path) or ""

    def new_text(
        self, values: dict[str, Any] | None = None, old: str | None = None
    ) -> str:
        """old (default: the file now) with values (default: the command's) set;
        None removes one."""
        values = self.values if values is None else values
        old = self.old_text() if old is None else old
        if self.kind == "json-env":
            return set_json_env(old, values)
        if self.kind == "json":
            held = _settings(old)
            for key, value in values.items():
                _put(held, key.split("/"), value)
            return json.dumps(held, indent=2) + "\n" if held else ""
        if self.kind == "toml":
            return harness_toml.set_values(old, values)
        return values.get("file", old) or ""

    def raw(self, text: str) -> dict[str, Any]:
        """What each owned key holds in text, None when absent."""
        if self.kind == "json-env":
            settings = _settings(text)
            env = settings.get("env") or {}
            return {
                key: (settings if key in TOP_LEVEL_KEYS else env).get(key)
                for key in self.values
            }
        if self.kind == "json":
            held = _settings(text)
            return {key: _at(held, key.split("/")) for key in self.values}
        if self.kind == "toml":
            return harness_toml.values(text, self.values)
        return {"file": text or None}

    def shown(self, text: str, key_label: str) -> str:
        """The part of text the command owns, for display, with no key in it."""
        if self.kind == "file":
            return hide_secrets(text)
        if self.kind == "toml":
            held = [value for value in self.raw(text).values() if value]
            keys = "\n".join(v for v in held if not v.startswith("["))
            tables = [v for v in held if v.startswith("[")]
            shown = "\n\n".join(part for part in [keys, *tables] if part)
            return hide_secrets(shown + "\n") if shown else ""
        if self.kind == "json":
            shown: dict[str, Any] = {}
            for key, value in self.raw(text).items():
                path = key.split("/")
                if value is not None:
                    _put(shown, path, _hidden(value, self.values[key], path[-1]))
            return json.dumps(shown, indent=2) + "\n" if shown else ""
        held = {k: v for k, v in self.raw(text).items() if v is not None}
        if held.get(TOKEN_KEY):
            written = held[TOKEN_KEY] == self.values[TOKEN_KEY]
            held[TOKEN_KEY] = key_label if written else "<hidden>"
        if held.get(API_KEY):
            held[API_KEY] = "<hidden>"
        if held.get(KEY_HELPER) and held[KEY_HELPER] != self.values[KEY_HELPER]:
            held[KEY_HELPER] = "<hidden>"  # a helper of yours may name its secret
        shown = {k: held.pop(k) for k in TOP_LEVEL_KEYS if k in held}
        if held:
            shown["env"] = held
        return json.dumps(shown, indent=2) + "\n" if shown else ""


# A setting whose name says it holds a credential, and an `env_key`, which only names
# the variable that does.
SECRET_NAME = re.compile(
    r"token|secret|password|passwd|api[_-]?key|authorization|credential", re.I
)
SETTING_LINE = re.compile(r'(\s*"?([\w.-]+)"?\s*[=:]\s*)(.*?)(,?\s*)$')


# Characters of a line a preview shows; Codex's model catalog holds its base
# instructions as one line of tens of thousands.
SHOWN_LINE = 160


def _at(held: dict, path: list[str]) -> Any:
    for part in path:
        held = held.get(part) if isinstance(held, dict) else None
    return held


def _put(held: dict, path: list[str], value: Any) -> None:
    """Set the value at path, creating the objects above it; None removes it,
    and the objects it leaves empty."""
    if value is None:
        harness_toml.pop_path(held, path)
        return
    for part in path[:-1]:
        held = held.setdefault(part, {})
        if not isinstance(held, dict):
            raise ValueError(f'"{part}" is not an object')
    held[path[-1]] = value


def _hidden(value: Any, ours: Any, name: str) -> Any:
    """value as shown: a string under a name that says it holds a credential is
    hidden, unless it is what the command writes there."""
    if isinstance(value, dict):
        mine = ours if isinstance(ours, dict) else {}
        return {k: _hidden(v, mine.get(k), k) for k, v in value.items()}
    if isinstance(value, list):
        mine = ours if isinstance(ours, list) else []
        return [
            _hidden(v, mine[i] if i < len(mine) else None, name)
            for i, v in enumerate(value)
        ]
    # a "$NAME" reference names the variable that holds the key, not the key
    reference = isinstance(value, str) and re.fullmatch(r"\$\{?\w+\}?", value)
    if isinstance(value, str) and SECRET_NAME.search(name) and value != ours:
        return value if reference else "<hidden>"
    return value


def hide_secrets(text: str) -> str:
    """A whole file as shown: a line that sets a credential, or a bearer header,
    keeps its name and loses its value, and a long line is cut short."""
    lines = []
    for line in text.splitlines(keepends=True):
        found = SETTING_LINE.match(line)
        shown = line
        if found and not found.group(2).lower().endswith("env_key"):
            name, value = found.group(2), found.group(3)
            if SECRET_NAME.search(name) or "bearer " in value.lower():
                shown = f'{found.group(1)}"<hidden>"{found.group(4)}'
        body = shown.rstrip("\n")
        if len(body) > SHOWN_LINE:
            cut = len(body) - SHOWN_LINE
            shown = f"{body[:SHOWN_LINE]}... ({cut:,} more characters)\n"
        lines.append(shown)
    return "".join(lines)


# --- Claude Code -----------------------------------------------------------------


def claude_settings_path(scope: str, cwd: Path, env: dict[str, str]) -> Path:
    if scope == "project":
        # the local file: a project's settings.json is often committed, and this can hold a key
        return cwd / ".claude" / "settings.local.json"
    base = env.get("CLAUDE_CONFIG_DIR") or str(Path.home() / ".claude")
    return Path(base) / "settings.json"


def claude_env(
    gateway: str, model: str, token: str, key_command: str | None = None
) -> dict[str, str | None]:
    """The settings, and with key_command the helper that gives the key, which
    token must then leave empty. Without one any helper is removed: it would
    send its key in place of a caller's own login."""
    return {
        "ANTHROPIC_BASE_URL": gateway.rstrip("/"),
        TOKEN_KEY: token,
        API_KEY: "",
        KEY_HELPER: key_command,
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": MAX_OUTPUT_TOKENS,
    }


def claude_overrides(cwd: Path, keys, user: Path) -> list[tuple[Path, list[str]]]:
    """The project settings in cwd that set any of keys, in their env or for
    TOP_LEVEL_KEYS at the top: Claude Code reads them only where it starts, and
    they win over the user's settings. Started in the home directory,
    ./.claude/settings.json is the user's own file."""
    found = []
    for name in ("settings.json", "settings.local.json"):
        path = cwd / ".claude" / name
        if path.resolve() == user.resolve():
            continue
        try:
            settings = _settings(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        env = settings.get("env") or {}
        if held := [
            key for key in keys if key in (settings if key in TOP_LEVEL_KEYS else env)
        ]:
            found.append((path, held))
    return found


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


# The bundled Codex model whose catalog entry describes a gateway model: one the
# Router accepts as Codex sends it, where the gpt-6 entries put tools in an input
# item the Router refuses (Codex CLI 0.166).
CODEX_CATALOG_BASE = "gpt-5.5"


def codex_catalog(
    bundled: Any,
    model: str,
    context_window: int | None,
    review_model: str | None = None,
    review_context_window: int | None = None,
    memory_model: str | None = None,
    memory_context_window: int | None = None,
) -> dict[str, Any] | None:
    """Describe only the gateway's selected entrypoint and its optional review
    and memory entrypoints, which are hidden from the model picker.

    Adapt the bundled base even when model already names a bundled slug. Never
    expose unrelated bundled models: Codex would select codex-auto-review even
    though the gateway does not serve that public model. Without an override,
    Codex reviews with the session model. The memory entry gives memory
    extraction the window it fits each session into.
    """
    models = bundled.get("models") if isinstance(bundled, dict) else None
    models = [entry for entry in models or [] if isinstance(entry, dict)]
    base = next((m for m in models if m.get("slug") == CODEX_CATALOG_BASE), None)
    if base is None:
        return None

    def adapted(name: str, window: int | None, *, hidden: bool = False) -> dict:
        entry = {
            **base,
            "slug": name,
            "display_name": name,
            "visibility": "hide" if hidden else "list",
            "supports_search_tool": False,
            "apply_patch_tool_type": None,
            "supported_reasoning_levels": (
                base.get("supported_reasoning_levels", []) if hidden else []
            ),
            "default_reasoning_level": "low" if hidden else None,
            "default_reasoning_summary": "auto",
            "support_verbosity": False,
            "default_verbosity": None,
            "upgrade": None,
            "service_tiers": [],
            "additional_speed_tiers": [],
        }
        entry.pop("auto_review_model_override", None)
        if window:
            entry |= {"context_window": window, "max_context_window": window}
        return entry

    main = adapted(model, context_window)
    entries = [main]
    if review_model:
        main["auto_review_model_override"] = review_model
    for name, window in (
        (review_model, review_context_window),
        (memory_model, memory_context_window),
    ):
        if name and all(entry["slug"] != name for entry in entries):
            entries.append(adapted(name, window, hidden=True))
    return {"models": entries}


def codex_settings(
    gateway: str,
    model: str,
    key_env: str | None,
    context_window: int | None,
    login: bool = False,
    catalog: Path | None = None,
    key_command: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The top-level settings, then the provider's. key_env is left out for a
    listener without keys: Codex refuses to start while the variable a provider
    names is unset. login has Codex send its own sign-in, for a gateway that
    presents it upstream. catalog is the model catalog file Codex reads.
    key_command becomes the provider's auth table, whose command output Codex
    sends as the bearer key; it takes neither key_env nor login (0.166)."""
    top = {
        "model": model,
        "model_context_window": context_window,
        "model_catalog_json": str(catalog) if catalog else None,
        "model_provider": PROVIDER,
        # the Router does not serve hosted web search
        "web_search": "disabled",
    }
    provider = {
        "name": "vLLM Semantic Router",
        "base_url": gateway.rstrip("/") + "/v1",
        "wire_api": "responses",
        "env_key": key_env,
        "requires_openai_auth": login,
        "auth": key_command and {"command": KEY_SHELL, "args": ["-c", key_command]},
    }
    return (
        {k: v for k, v in top.items() if v},
        {k: v for k, v in provider.items() if v},
    )


def codex_memories(model: str, memory_model: str | None = None) -> dict[str, str]:
    """The models Codex's memories feature asks for: the entrypoint's
    memory_model, else model. Unset, it asks the provider for gpt-5.6-luna to
    extract and gpt-5.6-terra to consolidate, which the gateway does not serve,
    so memories would never be written (0.166)."""
    memory = memory_model or model
    return {"extract_model": memory, "consolidation_model": memory}


def _memories_table(settings: dict[str, Any]) -> str:
    lines = ["[memories]"]
    for key, value in settings.items():
        if isinstance(value, dict):
            raise ValueError(
                f"[memories].{key} is a table, which connect cannot carry over;"
                " set extract_model and consolidation_model there yourself"
            )
        lines.append(f"{key} = {json.dumps(value)}")
    return "\n".join(lines)


def codex_change(
    home: Path,
    profile: str,
    gateway: str,
    model: str,
    key_env: str | None,
    context_window: int | None = None,
    login: bool = False,
    catalog: Path | None = None,
    key_command: str | None = None,
    memory_model: str | None = None,
) -> Change:
    """The profile file. Codex layers it over config.toml with --profile, and it
    holds the provider too, so config.toml is not touched."""
    top, provider = codex_settings(
        gateway, model, key_env, context_window, login, catalog, key_command
    )
    lines = [f"{k} = {json.dumps(v)}" for k, v in top.items()]
    memories = _memories_table(codex_memories(model, memory_model))
    text = (
        "\n".join(lines) + "\n\n" + _provider_table(provider) + "\n\n" + memories + "\n"
    )
    return Change(home / f"{profile}.config.toml", "file", {"file": text})


def _provider_table(provider: dict[str, Any]) -> str:
    """The provider's table, and its auth table under it, as TOML."""
    auth = provider.pop("auth", None)
    lines = [f"[model_providers.{PROVIDER}]"]
    lines += [f"{k} = {json.dumps(v)}" for k, v in provider.items()]
    if auth:
        lines += ["", f"[model_providers.{PROVIDER}.auth]"]
        lines += [f"{k} = {json.dumps(v)}" for k, v in auth.items()]
    return "\n".join(lines)


# The top-level keys of config.toml that --app owns, set or removed: a model
# context window or catalog of yours would describe another model.
CODEX_APP_KEYS = (
    "model",
    "model_context_window",
    "model_catalog_json",
    "model_provider",
    "web_search",
)


def codex_app_change(
    home: Path,
    gateway: str,
    model: str,
    key_env: str | None,
    context_window: int | None = None,
    login: bool = False,
    catalog: Path | None = None,
    key_command: str | None = None,
    memory_model: str | None = None,
) -> Change:
    """The same settings as the profile, as the defaults of the shared
    config.toml, which the Codex app reads with no profile (0.166). Only
    CODEX_APP_KEYS and the provider's table are the command's."""
    top, provider = codex_settings(
        gateway, model, key_env, context_window, login, catalog, key_command
    )
    values: dict[str, str | None] = {
        key: f"{key} = {json.dumps(top[key])}" if key in top else None
        for key in CODEX_APP_KEYS
    }
    values[f"model_providers.{PROVIDER}"] = _provider_table(provider)
    # the memory models join whatever [memories] already sets, which undo puts back
    path = home / "config.toml"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    held = harness_toml.values(text, ["[memories]"])["[memories]"]
    settings = harness_toml.loads(held).get("memories", {}) if held else {}
    memories = codex_memories(model, memory_model)
    values["[memories]"] = _memories_table({**settings, **memories})
    return Change(path, "toml", values)


def codex_launch_args(
    gateway: str,
    model: str,
    key_env: str | None,
    context_window: int | None,
    login: bool = False,
    key_command: str | None = None,
    memory_model: str | None = None,
) -> list[str]:
    """The same settings as command-line overrides, which win over every
    configuration file, a trusted project's included (0.166)."""
    top, provider = codex_settings(
        gateway, model, key_env, context_window, login, key_command=key_command
    )
    auth = provider.pop("auth", None) or {}
    pairs = [*top.items()]
    pairs += [(f"model_providers.{PROVIDER}.{k}", v) for k, v in provider.items()]
    pairs += [(f"model_providers.{PROVIDER}.auth.{k}", v) for k, v in auth.items()]
    memories = codex_memories(model, memory_model)
    pairs += [(f"memories.{k}", v) for k, v in memories.items()]
    return [arg for k, v in pairs for arg in ("-c", f"{k}={json.dumps(v)}")]


def codex_model_overrides(cwd: Path) -> list[Path]:
    """The project configurations from the repository's top down to cwd that set
    model: once Codex trusts the project, it asks for that model instead of the
    profile's (0.166). Outside a repository Codex reads only cwd's."""
    top = _git(cwd, "rev-parse", "--show-toplevel").stdout.strip()
    here = cwd.resolve()
    dirs = [here, *here.parents] if top else [here]
    stop = Path(top).resolve() if top else here
    found = []
    for directory in dirs:
        path = directory / ".codex" / "config.toml"
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            text = ""
        # a top-level key comes before the first table header
        if re.search(r"^\s*model\s*=", ("\n" + text).split("\n[", 1)[0], re.M):
            found.append(path)
        if directory == stop:
            break
    return found


# --- writing and undo --------------------------------------------------------------


def read(path: Path) -> str | None:
    """A file's text, None when it does not exist."""
    return path.read_text(encoding="utf-8") if path.exists() else None


def write_private(
    path: Path, text: str, *, always: bool = False, mode: int | None = None
) -> None:
    """Write text with mode, by default the file's own; a file created here, and
    every file with always, is readable by its owner only.

    The text goes to a file beside it that then replaces it, so a crash never
    leaves the settings half written.
    """
    path = path.resolve()  # a settings file linked from elsewhere stays a link
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode is None:
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


class StaleWriteError(ValueError):
    """A file changed after the changes to it were worked out and shown."""


def apply_changes(
    changes: list[Change],
    manifest_dir: Path,
    seen: dict[Path, str | None] | None = None,
) -> Path:
    """Write each file and record what it owns for --undo, all or nothing.

    seen holds each file as the preview read it, None when absent. A file that
    no longer holds that is refused before anything is written, since the
    changes shown were worked out from it. A second run keeps each key's first
    "before", so --undo still returns to what was there before vllm-sr connect
    first touched it.
    """
    now = {change.path: read(change.path) for change in changes}
    moved = [p for p, text in now.items() if seen is not None and text != seen.get(p)]
    if moved:
        raise StaleWriteError(
            f"{', '.join(map(str, moved))} changed after the changes above were"
            " worked out from it, so nothing was written. Run the command again to"
            " see the changes against what it holds now."
        )
    manifest_path = manifest_dir / MANIFEST
    held = read(manifest_path)
    record = (json.loads(held) if held is not None else None) or {"files": []}
    by_path = {item["path"]: item for item in record["files"]}
    writes = []
    for change in changes:
        old = now[change.path]
        new = change.new_text(old=old or "")
        item = by_path.get(str(change.path))
        if item is None:
            item = {
                "path": str(change.path),
                "kind": change.kind,
                "created": old is None,
                "keys": {},
            }
            record["files"].append(item)
        before, written = change.raw(old or ""), change.raw(new)
        for key in change.values:
            entry = item["keys"].setdefault(key, {"before": before[key]})
            entry["written"] = written[key]
        # Claude Code's settings now hold the key, so they are made private too
        writes.append((change.path, new, change.kind == "json-env", old))
    record_text = json.dumps(record, indent=2) + "\n"
    writes.append((manifest_path, record_text, True, held))
    _write_all(writes)
    return manifest_path


def _write_all(writes: list[tuple[Path, str, bool, str | None]]) -> None:
    """Write each (path, text, always private, expected) in turn, where expected
    is the file the text was worked out from, None when absent: a file that holds
    anything else by then was edited meanwhile, and is refused. When one fails,
    the ones already written get back their text and mode, and one created is
    removed, except a file edited again after it was written."""
    done = []
    try:
        for path, text, always, expected in writes:
            before = read(path)
            if before != expected:
                raise StaleWriteError(f"{path} changed while the changes were written")
            mode = stat.S_IMODE(path.stat().st_mode) if before is not None else None
            write_private(path, text, always=always)
            done.append((path, text, before, mode))
    except BaseException as exc:
        lost = []
        for path, text, before, mode in reversed(done):
            try:
                if read(path) != text:
                    lost.append(f"{path} (changed since it was written)")
                elif before is None:
                    path.resolve().unlink(missing_ok=True)
                else:
                    write_private(path, before, mode=mode)
            except OSError:
                lost.append(str(path))
        if not isinstance(exc, (OSError, StaleWriteError)):
            raise
        kept = (
            f"; {', '.join(lost)} could not be put back"
            if lost
            else ", so every file was left as it was"
        )
        error = StaleWriteError if isinstance(exc, StaleWriteError) else OSError
        raise error(f"{exc}{kept}") from exc


def _record(manifest_path: Path) -> dict | None:
    text = read(manifest_path)
    return json.loads(text) if text is not None else None


def record_plugin(
    manifest_dir: Path, plugin_id: str, before: dict, written: dict
) -> None:
    """Record a plugin's options as set through its harness's command: a second run
    keeps each option's first "before", as files do."""
    manifest_path = manifest_dir / MANIFEST
    record = _record(manifest_path) or {"files": []}
    keys = record.setdefault("plugins", {}).setdefault(plugin_id, {})
    for key, value in written.items():
        keys.setdefault(key, {"before": before.get(key)})["written"] = value
    write_private(manifest_path, json.dumps(record, indent=2) + "\n", always=True)


def plugin_undo(manifest_dir: Path, inputs) -> list[tuple[str, dict]]:
    """For each recorded plugin, the options to set back: those still as written,
    each to its value before. `inputs(plugin_id)` reads what a plugin holds now."""
    plugins = (_record(manifest_dir / MANIFEST) or {}).get("plugins", {})
    undo = []
    for plugin_id, keys in plugins.items():
        now = inputs(plugin_id)
        back = {
            key: entry["before"]
            for key, entry in keys.items()
            if now.get(key) == entry["written"] and entry["before"] is not None
        }
        undo.append((plugin_id, back))
    return undo


def undo_changes(manifest_dir: Path) -> list[str]:
    """Return each owned key to what it held before, unless it was changed since."""
    manifest_path = manifest_dir / MANIFEST
    record = _record(manifest_path)
    if record is None:
        raise FileNotFoundError(f"nothing to undo: {manifest_path} does not exist")
    report = []
    for item in record["files"]:
        path = Path(item["path"])
        if not path.exists():
            report.append(f"skipped {path}: it no longer exists")
            continue
        probe = Change(path, item["kind"], dict.fromkeys(item["keys"], ""))
        current = probe.raw(probe.old_text())
        restore = {}
        for key, entry in item["keys"].items():
            now, written = current[key], entry["written"]
            if not still_written(key, now, written):
                what = path if key == "file" else f"{key} in {path}"
                report.append(f"kept {what}: it was changed after vllm-sr connect")
                continue
            restore[key] = entry["before"]
            if now != written:
                tail = now[len(written) :].lstrip("\n")
                before = entry["before"] or ""
                if before and not before.endswith("\n"):
                    before += "\n"
                restore[key] = before + tail
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


def still_written(key: str, now: Any, written: Any) -> bool:
    """Whether an owned key, or a whole file, still holds what connect wrote. The
    Codex TUI appends tables of its own, such as [tui], to the profile, which
    leaves the profile connect's."""
    return now == written or (
        key == "file"
        and isinstance(written, str)
        and bool(now)
        and now.startswith(written)
    )


# --- status ------------------------------------------------------------------------


def connections(manifest_dir: Path) -> list[dict[str, Any]]:
    """What the undo record in manifest_dir says connect set up, one entry for each
    Claude Code settings file, Codex profile and Codex config.toml: the gateway,
    model and window written there, and whether each file written still holds
    them. No entry holds a secret."""
    record = _record(manifest_dir / MANIFEST) or {}
    items = record.get("files") or []
    found = []
    for item in items:
        path = Path(item["path"])
        written = {key: entry.get("written") for key, entry in item["keys"].items()}
        if item["kind"] == "json-env":
            token, helper = written.get(TOKEN_KEY), written.get(KEY_HELPER)
            window = str(written.get("CLAUDE_CODE_MAX_CONTEXT_TOKENS") or "")
            found.append(
                {
                    "harness": "claude-code",
                    "scope": (
                        "project" if path.name == "settings.local.json" else "user"
                    ),
                    "gateway": written.get("ANTHROPIC_BASE_URL"),
                    "model": written.get("ANTHROPIC_MODEL"),
                    "context_window": int(window) if window.isdigit() else None,
                    "login": _login(token == "", token not in (None, NO_KEY), helper),
                    "key_command": helper,
                    "settings": str(path),
                    "files": [
                        file_state(other)
                        for other in items
                        if other is item
                        or Path(other["path"]).name == STATUS_LINE_SCRIPT
                    ],
                    "plugins": sorted(record.get("plugins") or {}),
                }
            )
        elif item["kind"] == "toml" or (
            item["kind"] == "file" and path.name.endswith(".config.toml")
        ):
            # the profile, or the entries --app wrote into config.toml
            text = "\n".join(str(value) for value in written.values() if value)
            profile = profile_values(text)
            catalog = profile.get("model_catalog_json")
            # the auth table connect writes: the shell, then -c and the command
            helper = (
                (profile.get("args") or [None])[-1] if "command" in profile else None
            )
            paired = [
                other
                for other in items
                if other is item
                or other["path"] == catalog
                or Path(other["path"]).name == "vsr.json"
            ]
            found.append(
                {
                    "harness": "codex",
                    "profile": (
                        path.name[: -len(".config.toml")]
                        if item["kind"] == "file"
                        else None
                    ),
                    "gateway": str(profile.get("base_url") or "").removesuffix("/v1")
                    or None,
                    "model": profile.get("model"),
                    "review_model": _review_model(items, catalog),
                    "memory_model": profile.get("extract_model"),
                    "context_window": profile.get("model_context_window"),
                    "login": _login(
                        bool(profile.get("requires_openai_auth")),
                        bool(profile.get("env_key")),
                        helper,
                    ),
                    "key_env": profile.get("env_key"),
                    "key_command": helper,
                    "files": [file_state(other) for other in paired],
                }
            )
    return found


def _login(own: bool, keyed: bool, helper: str | None = None) -> str:
    if helper:
        return "key command"
    return "own login" if own else "gateway key" if keyed else "no key"


def _review_model(items: list[dict], catalog: str | None) -> str | None:
    """The reviewer the catalog connect wrote names for its main model."""
    item = next((i for i in items if catalog and i["path"] == catalog), None)
    try:
        text = item["keys"]["file"]["written"] if item else None
        models = json.loads(text or "{}").get("models") or [{}]
    except (KeyError, ValueError, AttributeError):
        return None
    return models[0].get("auto_review_model_override")


def file_state(item: dict) -> dict[str, Any]:
    """A recorded file now: as written, changed (naming the owned keys that no
    longer hold what connect wrote) or missing."""
    path = Path(item["path"])
    text = read(path)
    if text is None:
        return {"path": str(path), "state": "missing"}
    probe = Change(path, item["kind"], dict.fromkeys(item["keys"], ""))
    try:
        now = probe.raw(text)
        changed = [
            key
            for key, entry in item["keys"].items()
            if not still_written(key, now[key], entry["written"])
        ]
    except ValueError:  # no longer the JSON object connect wrote into
        changed = list(item["keys"])
    if not changed:
        return {"path": str(path), "state": "as written"}
    keys = {} if item["kind"] == "file" else {"changed": changed}
    return {"path": str(path), "state": "changed", **keys}


def profile_values(text: str) -> dict[str, Any]:
    """The settings of a profile connect wrote, one `key = <JSON>` a line. Its
    provider and auth tables repeat no key of another, so all share one dict."""
    values = {}
    for line in text.splitlines():
        key, sep, value = line.partition(" = ")
        if sep:
            try:
                values[key.strip()] = json.loads(value)
            except ValueError:
                continue
    return values


def token_from(env: dict[str, str], key_env: str) -> tuple[str, bool]:
    """The key from key_env, or a placeholder for a listener without keys."""
    value = env.get(key_env, "")
    return (value, True) if value else (NO_KEY, False)
