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
import shutil
import subprocess
import sys
import time
from pathlib import Path

PLUGIN = "vsr"
APPS = ("router", "escalate", "redact", "shield", "guard", "check")
RUNTIMES = ("local-cpu", "local-gpu", "remote")
LOGOS = ("auto", "image", "smooth", "blocks", "off")

# The values each `choice` option takes, as the manifest lists them.
DISPLAYS = ("normal", "quiet", "verbose")
ROUTER_GOALS = ("quality", "balanced", "save")
CHOICES = {
    "runtime": RUNTIMES,
    "logo": LOGOS,
    "display": DISPLAYS,
    "routerGoal": ROUTER_GOALS,
}

# name: (kind, default). The contract the mod reads; `secret` is set through /plugin configure only.
OPTIONS: dict[str, tuple[str, object]] = {
    "runtime": ("choice", "local-cpu"),
    "runtimeUrl": ("string", "http://127.0.0.1:8790/v1/decisions"),
    "runtimeToken": ("secret", ""),
    **dict.fromkeys(APPS, ("bool", True)),
    "routerEasyBelow": ("number", 0.5),
    "routerHardFrom": ("number", 1.5),
    "routerMinConfidence": ("number", 0.3),
    "routerDowngradeConfidence": ("number", 0.6),
    "routerStepsLow": ("number", 0.5),
    "routerStepsHigh": ("number", 1.2),
    "routerStepsMinConfidence": ("number", 0.3),
    "escalateFeedback": ("number", 0.8),
    "escalateReask": ("number", 0.75),
    "redactPii": ("number", 0.7),
    "shieldInject": ("number", 0.9),
    "guardReview": ("number", 0.9),
    "checkUnsupported": ("number", 0.5),
    "display": ("choice", "normal"),
    "routerGoal": ("choice", "balanced"),
    "logo": ("choice", "auto"),
    "logPath": ("path", "~/.claude/vsr/decisions.jsonl"),
    "calibrateEvery": ("count", 200),
}


class SettingsError(Exception):
    pass


def config_dir() -> Path:
    return Path(
        os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude"
    ).expanduser()


def settings_path() -> Path:
    return config_dir() / "settings.json"


def parse_value(name: str, raw: object) -> object:
    """One option value from the command line or a JSON file, checked against the contract."""
    if name not in OPTIONS:
        raise SettingsError(f"unknown option {name!r}; known: {', '.join(OPTIONS)}")
    kind, _ = OPTIONS[name]
    text = raw.strip() if isinstance(raw, str) else raw
    if kind == "secret":
        raise SettingsError(
            "runtimeToken is sensitive: save it with `/plugin configure <plugin id>` so it goes to secure storage"
        )
    if kind == "bool":
        if isinstance(text, bool):
            return text
        if str(text).lower() in ("true", "on", "yes", "1"):
            return True
        if str(text).lower() in ("false", "off", "no", "0"):
            return False
        raise SettingsError(f"{name} takes true or false, not {raw!r}")
    if kind in ("number", "count"):
        try:
            number = float(text)
        except (TypeError, ValueError):
            raise SettingsError(f"{name} takes a number, not {raw!r}") from None
        if kind == "count":
            if number < 1 or number != int(number):
                raise SettingsError(
                    f"{name} takes a whole number of decisions, at least 1"
                )
            return int(number)
        upper = (
            2.0
            if name
            in (
                "routerEasyBelow",
                "routerHardFrom",
                "routerStepsLow",
                "routerStepsHigh",
            )
            else 1.0
        )
        if not 0.0 <= number <= upper:
            raise SettingsError(f"{name} must be between 0 and {upper:g}")
        return round(number, 4)
    if kind == "choice":
        if text not in CHOICES[name]:
            raise SettingsError(f"{name} is one of {', '.join(CHOICES[name])}")
        return text
    if kind == "path":
        return str(Path(str(text)).expanduser().resolve())
    if not isinstance(text, str) or text == "":
        raise SettingsError(f"{name} takes a non-empty string")
    if name == "runtimeUrl" and not text.startswith(("http://", "https://")):
        raise SettingsError("runtimeUrl must start with http:// or https://")
    return text


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
    values = {name: parse_value(name, value) for name, value in raw.items()}
    easy = values.get("routerEasyBelow")
    hard = values.get("routerHardFrom")
    if easy is not None and hard is not None and easy >= hard:
        raise SettingsError("routerEasyBelow must be below routerHardFrom")
    low = values.get("routerStepsLow")
    high = values.get("routerStepsHigh")
    if low is not None and high is not None and low > high:
        raise SettingsError("routerStepsLow must not be above routerStepsHigh")
    return values


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
    """The settings with `values` merged into pluginConfigs[plugin_id].options; nothing else changes."""
    merged = copy.deepcopy(settings)
    configs = merged.setdefault("pluginConfigs", {})
    entry = configs.setdefault(plugin_id, {})
    entry.setdefault("options", {}).update(values)
    return merged


def saved_options(settings: dict, plugin_id: str) -> dict:
    entry = settings.get("pluginConfigs", {}).get(plugin_id, {})
    return dict(entry.get("options", {})) if isinstance(entry, dict) else {}


def effective_options(settings: dict, plugin_id: str) -> dict:
    """What the mod runs with: the contract defaults under the saved values (the token left out)."""
    options = {
        name: default for name, (kind, default) in OPTIONS.items() if kind != "secret"
    }
    options["logPath"] = str(Path(str(options["logPath"])).expanduser())
    options.update(
        {
            k: v
            for k, v in saved_options(settings, plugin_id).items()
            if k != "runtimeToken"
        }
    )
    return options


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
    The new file keeps the old one's permission bits: settings.json can hold API keys in `env`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if path.exists():
        backup = path.with_name(
            f"{path.name}.sr-backup-{time.strftime('%Y%m%d-%H%M%S')}"
        )
        shutil.copy2(path, backup)
    temporary = path.with_name(f".{path.name}.sr-{os.getpid()}")
    temporary.touch(mode=0o600)
    if backup:
        shutil.copymode(path, temporary)
    temporary.write_text(text)
    os.replace(temporary, path)
    return backup


def resolve_id(listed: list[dict], plugin_root: str | None) -> str:
    """The id Claude Code keys vSR's settings under, from `claude plugin list --json` and the running copy's root."""
    ours = [
        entry for entry in listed if str(entry.get("id", "")).split("@")[0] == PLUGIN
    ]
    if plugin_root:
        root = os.path.realpath(plugin_root)
        for entry in ours:
            for key in ("readFromFolder", "installPath"):
                if entry.get(key) and os.path.realpath(entry[key]) == root:
                    return entry["id"]
        return f"{PLUGIN}@inline"
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
        new_text = render(merge(settings, plugin_id, values))
        change = diff(text, new_text, path)
        if not change:
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
