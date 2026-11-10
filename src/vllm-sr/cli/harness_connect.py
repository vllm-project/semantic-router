"""Point Claude Code or Codex CLI at a Router gateway, and take it back.

The command owns a few keys in each file it touches: the env keys of Claude
Code's settings, the fields of one Codex provider table, and the top-level keys
of one Codex profile. It records what each key held before and what it wrote, so
--undo puts back only the keys still holding what it wrote and leaves every other
edit in place.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        tomllib = None

PROVIDER = "vllm-sr"
MANIFEST = ".vllm-sr-connect.json"
BACKUP_SUFFIX = ".vllm-sr-connect.bak"
MAX_OUTPUT_TOKENS = "32000"
SECRET_KEY = re.compile(
    r"(auth|api)_?(token|key)|secret|password|authorization|token$", re.I
)
SECRET_VALUE = re.compile(
    r"""([\w.-]+)(["']?\s*[:=]\s*)("[^"]*"|'[^']*'|[^\s,{}\[\]]+)"""
)


@dataclass
class Change:
    """One file and the keys the command sets in it.

    kind is json-env (keys of the "env" object), toml-table (fields of the
    [table] TOML table) or toml-top (keys before the first TOML table).
    """

    path: Path
    kind: str
    values: dict[str, str]
    table: str = ""

    def old_text(self) -> str:
        return self.path.read_text(encoding="utf-8") if self.path.exists() else ""

    def new_text(self) -> str:
        old = self.old_text()
        if self.kind == "json-env":
            return set_json_env(old, self.values)
        if self.kind == "toml-table":
            new = set_table_keys(old, self.table, self.values)
        else:
            new = set_top_level_keys(old, self.values)
        check_toml(new, self.path, self.table, self.values)
        return new

    def raw(self, text: str) -> dict[str, Any]:
        """What each owned key holds in text: the JSON value or the TOML line."""
        if self.kind == "json-env":
            return {key: _json_env(text).get(key) for key in self.values}
        lines = text.splitlines()
        if self.kind == "toml-table":
            span = table_span(lines, self.table)
            lines = lines[span[0] + 1 : span[1]] if span else []
        else:
            lines = lines[: _first_table(lines)]
        return {key: _key_line(lines, key) for key in self.values}


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
        "ANTHROPIC_AUTH_TOKEN": token,
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


def _json_env(text: str) -> dict:
    return _settings(text).get("env") or {}


def set_json_env(text: str, values: dict[str, Any]) -> str:
    """Set (or, for None, remove) env keys; every other key stays as it was."""
    settings = _settings(text)
    env = dict(settings.get("env") or {})
    for key, value in values.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    if env:
        settings["env"] = env
    else:
        settings.pop("env", None)
    return json.dumps(settings, indent=2) + "\n" if settings else ""


# --- Codex CLI ---------------------------------------------------------------------


def codex_home(env: dict[str, str]) -> Path:
    return Path(env.get("CODEX_HOME") or str(Path.home() / ".codex"))


def codex_changes(
    home: Path, profile: str, gateway: str, model: str, key_env: str
) -> list[Change]:
    provider = {
        "name": "vLLM Semantic Router",
        "base_url": gateway.rstrip("/") + "/v1",
        "wire_api": "responses",
        "env_key": key_env,
    }
    profile_values = {
        "model": model,
        "model_provider": PROVIDER,
        "web_search": "disabled",
    }
    return [
        Change(
            home / "config.toml", "toml-table", provider, f"model_providers.{PROVIDER}"
        ),
        Change(home / f"{profile}.config.toml", "toml-top", profile_values),
    ]


def _first_table(lines: list[str]) -> int:
    return next(
        (i for i, line in enumerate(lines) if line.lstrip().startswith("[")), len(lines)
    )


def _header_pattern(table: str) -> re.Pattern:
    parts = [
        rf"""\s*(?:{re.escape(p)}|"{re.escape(p)}"|'{re.escape(p)}')\s*"""
        for p in table.split(".")
    ]
    return re.compile(r"^\s*\[" + r"\.".join(parts) + r"\]\s*(?:#.*)?$")


def table_span(lines: list[str], table: str) -> tuple[int, int] | None:
    """The header line of [table] and the line after its last key, or None."""
    header = _header_pattern(table)
    start = next((i for i, line in enumerate(lines) if header.match(line)), None)
    if start is None:
        return None
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].lstrip().startswith("[")),
        len(lines),
    )
    return start, end


def _key_pattern(key: str) -> re.Pattern:
    return re.compile(
        rf"""^\s*(?:{re.escape(key)}|"{re.escape(key)}"|'{re.escape(key)}')\s*="""
    )


def _key_line(lines: list[str], key: str) -> str | None:
    pattern = _key_pattern(key)
    return next((line for line in lines if pattern.match(line)), None)


def _set_lines(lines: list[str], values: dict[str, Any], raw: bool) -> list[str]:
    """Replace each key's line (a None value removes it); new keys go first.

    With raw, a value is a whole line as recorded before, put back as it was.
    """

    def line_of(key: str, value: Any) -> str:
        return value if raw else f"{key} = {json.dumps(value)}"

    out, missing = [], dict(values)
    for line in lines:
        key = next((k for k in values if _key_pattern(k).match(line)), None)
        if key is None or key not in missing:
            out.append(line)
        elif missing[key] is not None:
            out.append(line_of(key, missing.pop(key)))
        else:
            missing.pop(key)
    return [line_of(k, v) for k, v in missing.items() if v is not None] + out


def set_table_keys(
    text: str, table: str, values: dict[str, Any], raw: bool = False
) -> str:
    """Set fields of [table], keeping its other fields and the rest of the file."""
    lines = text.splitlines()
    span = table_span(lines, table)
    if span is None:
        if all(v is None for v in values.values()):
            return text
        while lines and not lines[-1].strip():
            lines.pop()
        lines = (
            lines
            + ([""] if lines else [])
            + [f"[{table}]"]
            + _set_lines([], values, raw)
        )
    else:
        start, end = span
        body = _set_lines(lines[start + 1 : end], values, raw)
        if not any(line.strip() and not line.lstrip().startswith("#") for line in body):
            # nothing left in a table the command added
            lines = lines[:start] + lines[end:]
        else:
            lines = lines[: start + 1] + body + lines[end:]
    return _joined(lines)


def set_top_level_keys(text: str, values: dict[str, Any], raw: bool = False) -> str:
    """Set keys before the first table, where TOML reads them as top-level keys."""
    lines = text.splitlines()
    first = _first_table(lines)
    head = _set_lines(lines[:first], values, raw)
    while head and not head[-1].strip():
        head.pop()
    tail = lines[first:]
    return _joined(head + ([""] if head and tail else []) + tail)


def _joined(lines: list[str]) -> str:
    while lines and not lines[0].strip():
        lines = lines[1:]
    return "\n".join(lines) + "\n" if any(line.strip() for line in lines) else ""


def check_toml(text: str, path: Path, table: str, values: dict[str, str]) -> None:
    """Refuse to write a file Codex could not read, or one where the line edit did not
    set what Codex will read (a dotted key, a line inside a multi-line string)."""
    if tomllib is None:
        raise ValueError(
            f"checking the edit to {path} needs Python 3.11 or the tomli package"
        )
    try:
        doc = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(
            f"{path} would no longer be valid TOML ({exc}); edit it by hand"
        ) from exc
    for part in table.split(".") if table else []:
        doc = doc.get(part) if isinstance(doc, dict) else None
    if not isinstance(doc, dict) or any(doc.get(k) != v for k, v in values.items()):
        raise ValueError(f"could not set {sorted(values)} in {path}; edit it by hand")


# --- writing, backups and undo -----------------------------------------------------


def write_private(path: Path, text: str, *, always: bool = False) -> None:
    """Write text; a file created here, and every file with always, is readable by its owner only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not always:
        path.write_text(text, encoding="utf-8")
        return
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.chmod(path, 0o600)


def apply_changes(changes: list[Change], manifest_dir: Path) -> Path:
    """Back up each file, write it, and record its owned keys for --undo.

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
            backup = ""
            if change.path.exists():
                backup = str(change.path) + BACKUP_SUFFIX
                write_private(Path(backup), old, always=True)
            item = {
                "path": str(change.path),
                "kind": change.kind,
                "table": change.table,
                "created": not change.path.exists(),
                "backup": backup,
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
        probe = Change(
            path, item["kind"], dict.fromkeys(item["keys"], ""), item.get("table", "")
        )
        current = probe.raw(probe.old_text())
        restore = {}
        for key, entry in item["keys"].items():
            if current[key] == entry["written"]:
                restore[key] = entry["before"]
            else:
                report.append(
                    f"kept {key} in {path}: it was changed after vllm-sr connect"
                )
        text = probe.old_text()
        if probe.kind == "json-env":
            text = set_json_env(text, restore)
        elif probe.kind == "toml-table":
            text = set_table_keys(text, probe.table, restore, raw=True)
        else:
            text = set_top_level_keys(text, restore, raw=True)
        if item["created"] and not text.strip():
            path.unlink(missing_ok=True)
            report.append(f"removed {path}")
        else:
            write_private(path, text)
            report.append(f"restored {path}")
        if item["backup"]:
            Path(item["backup"]).unlink(missing_ok=True)
    manifest_path.unlink()
    return report


def masked(text: str, secrets: dict[str, str]) -> str:
    """text with each known secret shown as its label and every other secret-looking
    value hidden, for display."""
    for secret in sorted(secrets, key=len, reverse=True):
        text = text.replace(secret, secrets[secret])
    labels = set(secrets.values())

    def hide(match: re.Match) -> str:
        key, sep, value = match.groups()
        if (
            key == "env_key"
            or not SECRET_KEY.search(key)
            or value.strip("\"'") in labels
        ):
            return match.group(0)
        return f"{key}{sep}<hidden>"

    # every value set under a secret-looking key, also inside an inline table such
    # as http_headers = { Authorization = '...' }
    return SECRET_VALUE.sub(hide, text)


def secrets_in(change: Change, token: str, key_env: str) -> dict[str, str]:
    """Secret-looking env values the settings already hold, hidden, and the key in use,
    named by its variable so the diff shows when it replaces them."""
    found = {}
    if change.kind == "json-env":
        found = {
            str(v): "<hidden>"
            for k, v in _json_env(change.old_text()).items()
            if SECRET_KEY.search(k) and v
        }
    if token:
        found[token] = f"<${key_env}>"
    return found


def token_from(env: dict[str, str], key_env: str) -> tuple[str, bool]:
    """The key from key_env, or a placeholder for a listener without keys."""
    value = env.get(key_env, "")
    return (value, True) if value else ("vllm-sr-no-key", False)


def environ() -> dict[str, str]:
    return dict(os.environ)
