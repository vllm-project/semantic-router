#!/usr/bin/env python3
"""Read, merge and write vSR's userConfig values in Claude Code's user settings.

Claude Code reads `pluginConfigs` only from user (and managed) settings, keyed by plugin id:
`vsr@<marketplace>` for an installed plugin, `vsr@inline` for one loaded with `--plugin-dir`.
The sensitive `runtimeToken` lives in Claude Code's secure storage, never in settings.json.
"""
from __future__ import annotations

import argparse
import copy
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

PLUGIN = "vsr"
MANIFEST = Path(__file__).resolve().parents[3] / ".claude-plugin" / "plugin.json"
# The options, their types, defaults and choices: the manifest is the one source.
USER_CONFIG: dict[str, dict] = json.loads(MANIFEST.read_text())["userConfig"]

# What the manifest cannot say about a number: its range, and whether it is whole.
# Every other number is a probability, 0 to 1.
DIFFICULTY_LINES = (
    "routerEasyBelow",
    "routerHardFrom",
    "routerStepsLow",
    "routerStepsHigh",
)
WHOLE_NUMBERS = ("calibrateEvery",)
# (lower, upper): the upper line is never below the lower one.
ORDERED = (("routerEasyBelow", "routerHardFrom"), ("routerStepsLow", "routerStepsHigh"))
# A path is absolute or starts with ~/, saved as written (see log_file); empty turns it off.
PATHS = ("logPath",)


class SettingsError(Exception):
    pass


def config_dir() -> Path:
    return Path(
        os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude"
    ).expanduser()


def settings_path() -> Path:
    return config_dir() / "settings.json"


def parse_value(name: str, raw: object) -> object:
    """One option value from the command line or a JSON file, checked against the manifest."""
    spec = USER_CONFIG.get(name)
    if spec is None:
        raise SettingsError(f"unknown option {name!r}; known: {', '.join(USER_CONFIG)}")
    if spec.get("sensitive"):
        raise SettingsError(
            f"{name} is sensitive: save it with `/plugin configure <plugin id>` so it goes to secure storage"
        )
    text = raw.strip() if isinstance(raw, str) else raw
    if spec["type"] == "boolean":
        return _boolean(name, text)
    if spec["type"] == "number":
        return _number(name, text)
    if "options" in spec:
        if text not in spec["options"]:
            raise SettingsError(f"{name} is one of {', '.join(spec['options'])}")
        return text
    if name in PATHS:
        if text and log_file(str(text)) is None:
            raise SettingsError(
                f"{name} must be absolute or start with ~/; a relative path would land in each project's folder"
            )
        return str(text)
    if name == "runtimeModel":
        # Empty means the runtime's only model; a hosted System One API needs one named.
        return str(text).strip()
    if not isinstance(text, str) or text == "":
        raise SettingsError(f"{name} takes a non-empty string")
    if name == "runtimeUrl" and not text.startswith(("http://", "https://")):
        raise SettingsError("runtimeUrl must start with http:// or https://")
    return text


def _boolean(name: str, text: object) -> bool:
    if isinstance(text, bool):
        return text
    if str(text).lower() in ("true", "on", "yes", "1"):
        return True
    if str(text).lower() in ("false", "off", "no", "0"):
        return False
    raise SettingsError(f"{name} takes true or false, not {text!r}")


def _number(name: str, text: object) -> float | int:
    try:
        number = float(text)
    except (TypeError, ValueError):
        raise SettingsError(f"{name} takes a number, not {text!r}") from None
    if name in WHOLE_NUMBERS:
        if number < 0 or number != int(number):
            raise SettingsError(f"{name} takes a whole number, 0 or more")
        return int(number)
    upper = 2.0 if name in DIFFICULTY_LINES else 1.0
    if not 0.0 <= number <= upper:
        raise SettingsError(f"{name} must be between 0 and {upper:g}")
    return round(number, 4)


def parse_values(pairs: list[str], values_file: str | None = None) -> dict[str, object]:
    raw: dict[str, object] = {}
    if values_file:
        loaded = json.loads(Path(values_file).read_text())
        if not isinstance(loaded, dict):
            raise SettingsError(
                f"{values_file} must hold a JSON object of option values"
            )
        raw.update(loaded)
    for pair in pairs:
        name, sep, value = pair.partition("=")
        if not sep:
            raise SettingsError(f"--set takes name=value, not {pair!r}")
        raw[name.strip()] = value
    return {name: parse_value(name, value) for name, value in raw.items()}


def check_order(options: dict) -> None:
    """Each upper line at or above its lower line, as the mod holds them."""
    for lower, upper in ORDERED:
        if options[upper] < options[lower]:
            raise SettingsError(
                f"{upper} ({options[upper]}) must not be below {lower} ({options[lower]})"
            )


def load(path: Path) -> tuple[str, dict]:
    """The settings file's text and its parsed object; a missing file is an empty one."""
    if not path.exists():
        return "", {}
    text = path.read_text()
    if text.strip() == "":
        return text, {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise SettingsError(
            f"{path} is not valid JSON ({error}); fix it by hand before vSR writes to it"
        ) from None
    if not isinstance(parsed, dict):
        raise SettingsError(f"{path} does not hold a JSON object")
    return text, parsed


def merge(settings: dict, plugin_id: str, values: dict[str, object]) -> dict:
    """The settings with `values` merged into pluginConfigs[plugin_id].options; nothing else changes.

    A value equal to its default is not saved, and clears a saved one, so the option follows
    the manifest's default when that changes. Raises SettingsError when the merged options put
    a line below the one it must not be under.
    """
    merged = copy.deepcopy(settings)
    configs = merged.setdefault("pluginConfigs", {})
    entry = configs.setdefault(plugin_id, {})
    options = entry.setdefault("options", {})
    for name, value in values.items():
        if value == parse_value(name, USER_CONFIG[name]["default"]):
            options.pop(name, None)
        else:
            options[name] = value
    check_order(effective_options(merged, plugin_id))
    # Leave no empty containers behind for an id that ends up with nothing saved.
    if not options:
        del entry["options"]
    if not entry:
        del configs[plugin_id]
    if not configs:
        del merged["pluginConfigs"]
    return merged


def saved_options(settings: dict, plugin_id: str) -> dict:
    entry = settings.get("pluginConfigs", {}).get(plugin_id, {})
    return dict(entry.get("options", {})) if isinstance(entry, dict) else {}


def defaults() -> dict:
    """The manifest's defaults, the token left out."""
    return {
        name: spec["default"]
        for name, spec in USER_CONFIG.items()
        if not spec.get("sensitive")
    }


def log_file(path: str) -> Path | None:
    """The file the mod logs to for a `logPath` value, placed as hooks/lib/env.ts places it.

    `~/.claude/` is the Claude config directory ($CLAUDE_CONFIG_DIR when set), any other `~/`
    the home directory; an absolute path is used as given. None where the mod keeps no log:
    an empty path, or a relative one, which would land in each project's working directory.
    """
    if path.startswith("/") or re.match(r"[A-Za-z]:[/\\]", path):
        return Path(path)
    if not path.startswith("~/"):
        return None
    config = os.environ.get("CLAUDE_CONFIG_DIR")
    if config and path.startswith("~/.claude/"):
        return Path(config.rstrip("/\\") + "/" + path[len("~/.claude/") :])
    home = os.environ.get("HOME") or os.environ.get("USERPROFILE")
    return Path(home.rstrip("/\\") + path[1:]) if home else None


def effective_options(settings: dict, plugin_id: str) -> dict:
    """What the mod runs with: the saved values over the manifest's defaults (the token left out)."""
    saved = saved_options(settings, plugin_id)
    return {**defaults(), **{k: v for k, v in saved.items() if k != "runtimeToken"}}


def options_for(plugin_id: str | None) -> dict:
    """The options vSR runs with under `plugin_id`, read from the settings; the defaults without one."""
    saved = load(settings_path())[1] if plugin_id else {}
    return effective_options(saved, plugin_id or "")


def other_ids(settings: dict, plugin_id: str) -> list[str]:
    return sorted(
        k
        for k in settings.get("pluginConfigs", {})
        if k.split("@")[0] == PLUGIN and k != plugin_id
    )


def render(settings: dict) -> str:
    return json.dumps(settings, indent=2, ensure_ascii=False) + "\n"


def diff(old: str, new: str, path: Path) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(True),
            new.splitlines(True),
            f"{path} (now)",
            f"{path} (after)",
        )
    )


def write(path: Path, text: str) -> Path | None:
    """Writes atomically, after copying the old file aside; returns the backup's path.

    A symlinked settings file is written through, so the link stays. The new file keeps the
    old one's permission bits: settings.json can hold API keys in `env`.
    """
    target = Path(os.path.realpath(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if target.exists():
        backup = _backup_path(target)
        shutil.copy2(target, backup)
    temporary = target.with_name(f".{target.name}.sr-{os.getpid()}")
    temporary.touch(mode=0o600)
    if backup:
        shutil.copymode(target, temporary)
    temporary.write_text(text)
    os.replace(temporary, target)
    return backup


def _backup_path(target: Path) -> Path:
    """A backup name no earlier backup holds, even two in the same second."""
    name = f"{target.name}.sr-backup-{time.strftime('%Y%m%d-%H%M%S')}"
    backup, n = target.with_name(name), 0
    while backup.exists():
        n += 1
        backup = target.with_name(f"{name}-{n}")
    return backup


def resolve_id(listed: list[dict], plugin_root: str | None) -> str:
    """The id Claude Code keys vSR's settings under, from `claude plugin list --json` and the running copy's root.

    A root inside Claude Code's plugins folder is an installed copy, keyed by its marketplace
    even when the list names another version of it; any other root was loaded with
    `--plugin-dir`, which Claude Code keys as `vsr@inline`.
    """
    ours = [
        entry for entry in listed if str(entry.get("id", "")).split("@")[0] == PLUGIN
    ]
    if plugin_root:
        root = Path(os.path.realpath(plugin_root))
        for entry in ours:
            for key in ("readFromFolder", "installPath"):
                if entry.get(key) and Path(os.path.realpath(entry[key])) == root:
                    return entry["id"]
        plugins = Path(os.path.realpath(config_dir() / "plugins"))
        if not root.is_relative_to(plugins):
            return f"{PLUGIN}@inline"
        parts = root.relative_to(plugins).parts
        # cache/<marketplace>/<plugin>/<version>
        if len(parts) >= 3 and parts[0] == "cache" and parts[2] == PLUGIN:
            return f"{PLUGIN}@{parts[1]}"
        if len(ours) == 1:
            return ours[0]["id"]
        raise SettingsError(
            f"cannot tell which vSR install runs from {root}; pass --id"
        )
    if len(ours) == 1:
        return ours[0]["id"]
    if not ours:
        return f"{PLUGIN}@inline"
    raise SettingsError(
        f"several vSR installs ({', '.join(e['id'] for e in ours)}); pass --id"
    )


def listed_plugins() -> list[dict]:
    try:
        done = subprocess.run(
            ["claude", "plugin", "list", "--json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return json.loads(done.stdout) if done.returncode == 0 else []
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return []


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("id", "show", "plan", "apply"):
        one = sub.add_parser(name)
        one.add_argument("--id", help="plugin id, such as vsr@vllm-sr or vsr@inline")
        one.add_argument(
            "--plugin-root", help="the running copy's ${CLAUDE_PLUGIN_ROOT}"
        )
        if name in ("plan", "apply"):
            one.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
            one.add_argument("--values", help="a JSON file of option values")
    args = parser.parse_args(argv)
    try:
        plugin_id = args.id or resolve_id(listed_plugins(), args.plugin_root)
        if args.command == "id":
            print(plugin_id)
            return 0
        path = settings_path()
        text, settings = load(path)
        if args.command == "show":
            print(
                json.dumps(
                    {
                        "id": plugin_id,
                        "settings": str(path),
                        "saved": saved_options(settings, plugin_id),
                        "effective": effective_options(settings, plugin_id),
                        "otherIds": other_ids(settings, plugin_id),
                    },
                    indent=2,
                )
            )
            return 0
        values = parse_values(args.set, args.values)
        merged = merge(settings, plugin_id, values)
        new_text = render(merged)
        change = diff(text, new_text, path)
        if merged == settings or not change:
            print(
                f"no change: {path} already holds these values under pluginConfigs[{plugin_id!r}]"
            )
            return 0
        print(change, end="")
        if args.command == "apply":
            backup = write(path, new_text)
            print(f"wrote {path}" + (f" (previous copy: {backup})" if backup else ""))
        return 0
    except SettingsError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
